import json

import pytest
from dataclasses import replace
from types import SimpleNamespace

from tradingbot.analyst import Analyst, parse_verdict
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal

BUY_JSON = json.dumps({
    "action": "buy", "conviction": 0.8, "thesis": "Organic meme with a dated catalyst.",
    "red_flags": [], "catalysts": ["listing on 2026-10-12"], "social_signal": "organic",
    "sources": ["https://example.com"],
})


def resp(stop_reason, text=None):
    content = [SimpleNamespace(type="text", text=text)] if text else [SimpleNamespace(type="server_tool_use")]
    return SimpleNamespace(stop_reason=stop_reason, content=content,
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_parse_verdict_clamps_conviction():
    v = parse_verdict([SimpleNamespace(type="text", text=BUY_JSON.replace("0.8", "1.7"))])
    assert v.conviction == 1.0


def test_parse_verdict_rejects_garbage():
    assert parse_verdict([SimpleNamespace(type="text", text="not json")]) is None


def test_analyst_resumes_pause_turn(cfg, good_pair, safe):
    from tradingbot.factors import evaluate
    client = FakeClient([resp("pause_turn"), resp("end_turn", BUY_JSON)])
    v = Analyst(cfg, client).analyse(good_pair, safe, evaluate(good_pair, safe, cfg))
    assert v.action == "buy"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "assistant"
    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["tools"][0]["type"] == "web_search_20260209"
    assert req["fallbacks"] == "default"


def test_analyst_refusal_returns_none(cfg, good_pair, safe):
    from tradingbot.factors import evaluate
    client = FakeClient([resp("refusal")])
    assert Analyst(cfg, client).analyse(good_pair, safe, evaluate(good_pair, safe, cfg)) is None


def test_analyst_hourly_budget(cfg, good_pair, safe):
    from tradingbot.factors import evaluate
    cfg.max_analyses_per_hour = 1
    client = FakeClient([resp("end_turn", BUY_JSON)])
    a = Analyst(cfg, client)
    f = evaluate(good_pair, safe, cfg)
    assert a.analyse(good_pair, safe, f) is not None
    assert a.analyse(good_pair, safe, f) is None
    assert len(client.calls) == 1


class FakeDex:
    def __init__(self, pair):
        self.pair = pair

    def discover(self):
        return [self.pair.token_address], set()

    def pairs(self, addrs, boosted=frozenset()):
        addrs = set(addrs)
        return [self.pair] if self.pair.token_address in addrs else []


def test_engine_buys_then_stops_out(cfg, good_pair, safe):
    dex = FakeDex(good_pair)
    rug = SimpleNamespace(report=lambda mint: safe)
    analyst = Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)]))
    journal = Journal(":memory:")
    eng = Engine(cfg, dex, rug, analyst, PaperExecutor(cfg), journal)

    eng.tick()
    [pos] = journal.open_positions()
    assert eng.cash < cfg.starting_cash_usd

    dex.pair = replace(good_pair, price_usd=good_pair.price_usd * 0.5)
    eng.tick()  # same token was analysed recently, so no second Claude call
    assert journal.open_positions() == []
    stats = journal.stats()
    assert stats["closed_trades"] == 1 and stats["net_realised_usd"] < 0
    # Loss is bounded near the risk budget (2% of equity, plus gap and costs)
    assert -stats["net_realised_usd"] < cfg.starting_cash_usd * 0.06


def test_engine_respects_daily_loss_limit(cfg, good_pair, safe):
    cfg.daily_loss_limit_pct = 0.0001
    journal = Journal(":memory:")
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)] * 3)), PaperExecutor(cfg), journal)
    eng.tick()
    eng.dex.pair = replace(good_pair, price_usd=good_pair.price_usd * 0.5)
    eng.tick()
    assert not eng.entries_allowed()


def test_failed_take_profit_sell_is_retried(cfg, good_pair, safe):
    journal = Journal(":memory:")
    paper = PaperExecutor(cfg)
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)])), paper, journal)
    eng.tick()
    [pos] = journal.open_positions()

    eng.executor = SimpleNamespace(sell=lambda *a: None)  # exchange outage
    eng.dex.pair = replace(good_pair, price_usd=pos.entry_price * 1.6)
    eng.manage_positions()
    [pos] = journal.open_positions()
    assert pos.tp_hits == 0  # ladder rolled back

    eng.executor = paper
    eng.manage_positions()
    [pos] = journal.open_positions()
    assert pos.tp_hits == 1 and pos.qty < pos.initial_qty


def test_refused_analysis_is_not_retried_every_tick(cfg, good_pair, safe):
    client = FakeClient([resp("refusal"), resp("refusal")])
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, client), PaperExecutor(cfg), Journal(":memory:"))
    eng.tick()
    eng.tick()
    assert len(client.calls) == 1


def test_monitor_thread_exits_while_research_is_blocked(cfg, good_pair, safe):
    import threading
    import time as _time

    cfg.monitor_interval_s = 0.02
    cfg.scan_interval_s = 0.02
    journal = Journal(":memory:")
    dex = FakeDex(good_pair)
    eng = Engine(cfg, dex, SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)])), PaperExecutor(cfg), journal)
    eng.tick()  # open a position
    [pos] = journal.open_positions()

    release = threading.Event()
    blocked = threading.Event()

    def stuck_research(*a, **k):
        blocked.set()
        release.wait(5)
        return None

    eng.analyst = SimpleNamespace(analyse=stuck_research, budget_ok=lambda: True)
    eng.journal.recently_seen = lambda *a: False  # make the scan loop pick the token up again
    eng.screen = lambda: [(good_pair, SimpleNamespace(score=1), safe)]
    runner = threading.Thread(target=eng.run_forever, daemon=True)
    runner.start()
    assert blocked.wait(2)

    dex.pair = replace(good_pair, price_usd=pos.entry_price * 0.5)  # crash through the stop
    deadline = _time.time() + 2
    while journal.open_positions() and _time.time() < deadline:
        _time.sleep(0.02)
    assert journal.open_positions() == []  # exited while research was still blocked
    assert not release.is_set()
    release.set()
    eng._stop.set()
    runner.join(2)

    row = journal.db.execute("SELECT trigger_price, price FROM fills WHERE side='sell'").fetchone()
    assert row["trigger_price"] == pytest.approx(pos.entry_price * 0.5)
