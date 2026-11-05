import httpx

from tradingbot.config import Config
from tradingbot.data import jupiter
from tradingbot.data.extras import Extras
from tradingbot.data.jupiter import RATE_LIMITED, Gate, Jupiter
from tradingbot.execution.sim import SimExecutor

MINT = "M1nt1111111111111111111111111111111111111"


def _jup(status):
    def handler(request):
        if request.url.path.endswith("/quote"):
            return httpx.Response(status, json={"error": "x"})
        return httpx.Response(200, json={})
    return Jupiter(http=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x"), gate=Gate(0.0))


def test_paid_base_and_header_with_key(monkeypatch):
    monkeypatch.delenv("JUPITER_API_KEY", raising=False)
    assert jupiter.base_url() == jupiter.LITE_BASE and jupiter.headers() == {}
    assert Gate().interval_s == jupiter.FREE_INTERVAL_S
    monkeypatch.setenv("JUPITER_API_KEY", "k")
    assert jupiter.base_url() == jupiter.PAID_BASE and jupiter.headers() == {"x-api-key": "k"}
    assert Gate().interval_s == 1 / jupiter.PAID_RPS
    # 8/s flat out for a 31-day month stays inside the 25M-request plan
    assert jupiter.PAID_RPS * 86400 * 31 < 25_000_000


def test_extras_jupiter_calls_use_key(monkeypatch):
    monkeypatch.setenv("JUPITER_API_KEY", "k")
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers.get("x-api-key")))
        return httpx.Response(200, json=[{"id": MINT, "holderCount": 5}])
    ex = Extras(client=httpx.Client(transport=httpx.MockTransport(handler)), pace_s=0)
    assert ex.jupiter_token(MINT)["holderCount"] == 5
    assert seen[0][0].startswith("https://api.jup.ag/tokens/v2/search") and seen[0][1] == "k"


def test_refused_quote_is_rate_limited_not_no_route(monkeypatch):
    monkeypatch.setattr(jupiter.time, "sleep", lambda s: None)
    jup = _jup(429)
    assert jup.quote(jupiter.USDC_MINT, MINT, 1000) is None
    assert jup.last_failure() == RATE_LIMITED
    jup = _jup(400)  # Jupiter's answer when there is no route
    assert jup.quote(jupiter.USDC_MINT, MINT, 1000) is None
    assert jup.last_failure() == "no route"


def test_sim_reports_rate_limit(monkeypatch):
    monkeypatch.setattr(jupiter.time, "sleep", lambda s: None)
    jup = _jup(429)
    jup._decimals[MINT] = 6
    jup._sol_price = (150.0, 1e18)
    ex = SimExecutor(Config(db_path=":memory:"), jup, latency_s=0)
    fill = ex.sell(MINT, 10.0, 1.0)
    assert not fill.ok and fill.usd == 0 and fill.note == RATE_LIMITED


def test_identical_requests_at_the_same_moment_share_one_call(monkeypatch):
    """11-05: ten books buying one token sent ten identical quotes through the 8/s pace and filled seconds apart."""
    import threading
    import time
    from tradingbot.data import jupiter as jup
    monkeypatch.setattr(jup, "SHARE_S", 1.0)
    monkeypatch.setattr(jup, "SHARED", jup._Shared())
    calls = []

    def handler(request):
        calls.append(str(request.url))
        time.sleep(0.2)
        return httpx.Response(200, json={"inAmount": "100", "outAmount": "5000", "otherAmountThreshold": "4900",
                                         "priceImpactPct": "0.01", "routePlan": []})
    j = Jupiter(http=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x"), gate=Gate(0.0))
    out = []
    threads = [threading.Thread(target=lambda: out.append(j.quote("A", "B", 100))) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(out) == 10 and all(q.out_raw == 5000 for q in out)
    assert j.quote("A", "B", 200).out_raw == 5000 and len(calls) == 2  # a different amount is its own request
    time.sleep(1.05)
    j.quote("A", "B", 100)
    assert len(calls) == 3  # older than SHARE_S: asked again


def test_a_failed_shared_request_is_not_reused(monkeypatch):
    from tradingbot.data import jupiter as jup
    monkeypatch.setattr(jup, "SHARE_S", 1.0)
    monkeypatch.setattr(jup, "SHARED", jup._Shared())
    codes = iter([400, 200])
    j = Jupiter(http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(next(codes), json={
        "inAmount": "1", "outAmount": "2", "otherAmountThreshold": "2", "routePlan": []})), base_url="https://x"),
        gate=Gate(0.0))
    assert j.quote("A", "B", 1) is None
    assert j.quote("A", "B", 1).out_raw == 2
