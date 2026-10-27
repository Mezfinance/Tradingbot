import httpx
import pytest

from tradingbot.data.jupiter import USDC_MINT, Jupiter, Quote
from tradingbot.execution.sim import SimExecutor


def q(out_raw, min_out_raw, impact=0.5, in_raw=0):
    return Quote("a", "b", in_raw, out_raw, min_out_raw, impact, ["Pool"], {})


class FakeJupiter:
    """Returns queued quotes in order; SOL at $100; every token has 6 decimals."""

    def __init__(self, quotes):
        self.quotes = list(quotes)

    def quote(self, in_mint, out_mint, in_raw):
        return self.quotes.pop(0)

    def decimals(self, mint):
        return 6

    def sol_price(self):
        return 100.0


@pytest.fixture
def sim(cfg):
    def make(quotes, mode="midpoint"):
        cfg.sim_fill_mode = mode
        return SimExecutor(cfg, FakeJupiter(quotes), latency_s=0)
    return make


def test_buy_fills_between_quote_and_slippage_floor(sim):
    # $50 buy; fee = 0.001 SOL + 0.00203928 SOL rent = $0.303928
    ex = sim([q(1_000_000_000, 970_000_000), q(990_000_000, 960_000_000)])
    fill = ex.buy("TOK", 50, 0.00005)
    assert fill.ok
    assert fill.qty == pytest.approx((990_000_000 + 970_000_000) // 2 / 1e6)
    assert fill.usd == 50
    assert fill.fee_usd == pytest.approx(0.303928)


def test_fill_modes(sim):
    quotes = [q(1_000_000, 970_000), q(990_000, 960_000)]
    assert sim(list(quotes), "expected").buy("T", 50, 1).qty == pytest.approx(0.99)
    assert sim(list(quotes), "pessimistic").buy("T", 50, 1).qty == pytest.approx(0.97)


def test_price_moving_past_slippage_reverts_and_costs_fee(sim):
    ex = sim([q(1_000_000, 970_000), q(900_000, 870_000)])
    fill = ex.buy("T", 50, 1)
    assert not fill.ok and "reverted" in fill.note
    assert fill.usd == pytest.approx(0.1)  # 0.001 SOL priority fee only, no rent


def test_high_impact_entry_is_declined_for_free(sim):
    ex = sim([q(1_000_000, 970_000, impact=9.0)])
    assert ex.buy("T", 50, 1) is None
    assert ex.last_note == "price impact 9.00%"  # the engine logs this as the skip reason


def test_exit_ignores_impact_cap(sim):
    ex = sim([q(40_000_000, 39_000_000, impact=25.0), q(40_000_000, 39_000_000, impact=25.0)])
    fill = ex.sell("T", 1000, 0.05)
    assert fill.ok
    assert fill.usd == pytest.approx(39.5 - 0.1)


def test_exit_with_no_route_is_stuck_but_free(sim):
    fill = sim([None]).sell("T", 1000, 0.05)
    assert not fill.ok and fill.usd == 0 and fill.note == "no route"


def test_jupiter_client_parses_quote_and_round_trip():
    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        if request.url.path == "/swap/v1/quote":
            amount = int(p["amount"])
            if p["inputMint"] == USDC_MINT:  # buy: 1 USDC -> 1000 tokens
                out = amount * 1000
            else:  # sell back loses 20% (a "tax")
                out = int(amount / 1000 * 0.8)
            return httpx.Response(200, json={
                "inAmount": str(amount), "outAmount": str(out),
                "otherAmountThreshold": str(int(out * 0.97)), "priceImpactPct": "0.012",
                "routePlan": [{"swapInfo": {"label": "Raydium"}}],
            })
        return httpx.Response(404)

    jup = Jupiter(http=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x"))
    quote = jup.quote(USDC_MINT, "TOK", 5_000_000)
    assert quote.price_impact_pct == pytest.approx(1.2)
    assert quote.route == ["Raydium"]
    loss, impact = jup.round_trip("TOK", 5.0)
    assert loss == pytest.approx(20.0)


def test_token_too_new_for_the_price_api_gets_decimals_onchain_and_a_quote_price(monkeypatch):
    monkeypatch.delenv("HELIUS_API_KEY", raising=False)
    monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/price/v3":
            return httpx.Response(200, json={})  # not listed yet
        if request.method == "POST":  # getTokenSupply on the public RPC
            return httpx.Response(200, json={"result": {"value": {"decimals": 6, "amount": "1"}}})
        if request.url.path == "/swap/v1/quote":
            out = int(request.url.params["amount"]) * 1000  # 1 USDC -> 1000 tokens
            return httpx.Response(200, json={"inAmount": request.url.params["amount"], "outAmount": str(out),
                                             "otherAmountThreshold": str(out), "priceImpactPct": "0.001",
                                             "routePlan": []})
        return httpx.Response(404)

    jup = Jupiter(http=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x"))
    assert jup.decimals("NEWTOKEN") == 6
    assert jup.quote_price("NEWTOKEN") == pytest.approx(0.001)  # $20 bought 20,000 tokens
    assert calls.count("/") == 1  # decimals read on-chain once, then cached
