import json
from dataclasses import replace
from types import SimpleNamespace

import httpx

from tradingbot.analyst import Analyst, build_brief
from tradingbot.data.geckoterminal import GeckoTerminal, base_token
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.factors import evaluate
from tradingbot.journal import Journal
from tests.test_analyst_engine import FakeClient, resp

PASS_JSON = json.dumps({"action": "pass", "conviction": 0.1, "thesis": "no", "red_flags": [], "catalysts": [],
                        "social_signal": "none", "sources": []})


def pool(mint):
    return {"relationships": {"base_token": {"data": {"id": f"solana_{mint}"}}}}


def gecko_with(responses):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        window = request.url.params["duration"]
        calls.append(window)
        r = responses[window]
        return r if isinstance(r, httpx.Response) else httpx.Response(200, json={"data": [pool(m) for m in r]})

    return GeckoTerminal(httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x"),
                         min_interval_s=0), calls


def test_base_token_parsing():
    assert base_token(pool("ABC")) == "ABC"
    assert base_token({"relationships": {}}) is None
    assert base_token(pool("ABC"), network="eth") is None


def test_trending_dedupes_and_skips_non_memecoins():
    g, calls = gecko_with({"1h": ["AAA", "BBB", "So11111111111111111111111111111111111111112"],
                           "6h": ["BBB", "CCC", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"]})
    assert g.discover() == ["AAA", "BBB", "CCC"]
    assert calls == ["1h", "6h"]


def test_one_window_failing_does_not_stop_the_other():
    g, _ = gecko_with({"1h": httpx.Response(429), "6h": ["CCC"]})
    assert g.discover() == ["CCC"]


class TwoFeedDex:
    """DexScreener stand-in: lists one token itself and prices any token asked for."""

    def __init__(self, pairs):
        self.by_token = {p.token_address: p for p in pairs}

    def discover(self):
        return ["DEXONLY", "BOTH"], set()

    def pairs(self, addrs, boosted=frozenset()):
        return [replace(self.by_token[a]) for a in addrs if a in self.by_token]


def test_engine_merges_feeds_and_records_sources(cfg, good_pair, safe):
    pairs = [replace(good_pair, token_address=t, symbol=t) for t in ("DEXONLY", "BOTH", "GECKOONLY")]
    gecko = SimpleNamespace(discover=lambda: ["BOTH", "GECKOONLY"])
    client = FakeClient([resp("end_turn", PASS_JSON)] * 3)
    j = Journal(":memory:")
    eng = Engine(cfg, TwoFeedDex(pairs), SimpleNamespace(report=lambda m: safe), Analyst(cfg, client),
                 PaperExecutor(cfg), j, gecko=gecko)
    survivors = eng.screen()
    sources = {s.token_address: s.sources for s, _, _ in survivors}
    assert sources == {"DEXONLY": ["dexscreener_listings"],
                       "BOTH": ["dexscreener_listings", "geckoterminal_trending"],
                       "GECKOONLY": ["geckoterminal_trending"]}

    cfg.discovery = "geckoterminal"
    assert set(eng.discover()[0]) == {"BOTH", "GECKOONLY"}


def test_claude_is_told_where_a_token_came_from(cfg, good_pair, safe):
    snap = replace(good_pair, sources=["geckoterminal_trending"])
    brief = build_brief(snap, safe, evaluate(snap, safe, cfg))
    assert '"discovered_via": [\n    "geckoterminal_trending"\n  ]' in brief


def test_established_tokens_now_pass_the_age_filter(cfg, good_pair, safe):
    week_old = replace(good_pair, created_at_ms=int((good_pair.fetched_at - 7 * 86400) * 1000))
    assert not any("too old" in r for r in evaluate(week_old, safe, cfg).rejections)


def test_gecko_gate_serves_higher_priority_first_and_backs_off_on_429():
    import threading
    import time
    from tradingbot.data import geckoterminal as gt
    gate = gt._Gate()
    gate.next_free = time.monotonic() + 0.2  # busy for a moment, so all three queue up
    order = []
    threads = [threading.Thread(target=lambda p=p: (gate.acquire(p, 0.01), order.append(p)))
               for p in (gt.P_EXTRA, gt.P_INFO, gt.P_SNAPSHOT)]
    for t in threads:
        t.start()
        time.sleep(0.02)
    for t in threads:
        t.join(5)
    assert order == [gt.P_SNAPSHOT, gt.P_INFO, gt.P_EXTRA]
    before = gate.interval
    gate.throttled()
    assert gate.interval == before * 1.5
    for _ in range(100):
        gate.ok()
    assert gate.interval == gt.MIN_INTERVAL_S


def test_trending_discovery_is_reused_for_ten_minutes():
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json={"data": []})
    g = GeckoTerminal(httpx.Client(transport=httpx.MockTransport(handler), base_url="https://g"), min_interval_s=0)
    g.discover_ttl_s = 600
    g.discover()
    g.discover()
    assert len(calls) == 2  # two trending windows, fetched once


def test_pumpfun_pauses_when_cloudflare_blocks_it():
    from tradingbot.data.pumpfun import PumpFun
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(403, text="<title>Just a moment...</title>")
    pf = PumpFun(httpx.Client(transport=httpx.MockTransport(handler), base_url="https://p"), min_interval_s=0)
    assert pf.coin("M") is None and pf.coin("M2") is None
    assert len(calls) == 1
