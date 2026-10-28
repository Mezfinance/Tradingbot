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
