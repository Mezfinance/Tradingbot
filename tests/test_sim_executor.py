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
        if not self.quotes:
            return None
        nxt = self.quotes.pop(0)
        return nxt(in_raw) if callable(nxt) else nxt  # a callable describes a pool: quote for any size

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


def test_jupiter_unknown_impact_is_estimated_from_the_trigger_price(sim):
    # Jupiter says 100% (it can't value a seconds-old token) and the size probe fails (None): the quote's own
    # price is 2% above the trigger, so the estimate lets the buy through
    swap = 50 - 0.303928
    tokens = swap / (1.02 * 0.001)  # quote price = 1.02 x trigger
    raw = int(tokens * 1e6)
    ok = sim([q(raw, int(raw * 0.97), impact=100.0), lambda n: None,
              q(raw, int(raw * 0.97), impact=100.0)]).buy("T", 50, 0.001)
    assert ok is not None and ok.ok
    # a really bad price still gets declined: the quote is 20% worse than the trigger
    bad_raw = int(swap / (1.20 * 0.001) * 1e6)
    ex = sim([q(bad_raw, int(bad_raw * 0.97), impact=100.0)])
    assert ex.buy("T", 50, 0.001) is None and "price impact 20" in ex.last_note
    # no trigger price to compare with: the 100% stands
    ex = sim([q(raw, int(raw * 0.97), impact=100.0)])
    assert ex.buy("T", 50, 0) is None and "100.00%" in ex.last_note


def pool(tokens_per_usdc_raw, impact_per_full=0.0, reported=94.0):
    """A quote for any size from a pool: a $1-equivalent rate, minus `impact_per_full` x (size / 50 USDC)."""
    def quote(in_raw):
        rate = tokens_per_usdc_raw * (1 - impact_per_full * in_raw / 50e6)
        out = int(in_raw * rate)
        return q(out, int(out * 0.97), impact=reported, in_raw=in_raw)
    return quote


def test_bogus_jupiter_impact_is_replaced_by_our_own_size_probe(sim):
    # Jupiter reports 94% on a fresh token; quoting 1% of the size shows the real cost of our size is ~2%,
    # even though the price ran 30% since the trigger (the trigger estimate would have declined it)
    p = pool(1000, impact_per_full=0.02)
    ex = sim([p, p, p])
    fill = ex.buy("T", 50, 0.001 / 1.3)
    assert fill is not None and fill.ok
    # a really thin pool is still declined: our size costs ~20%
    thin = pool(1000, impact_per_full=0.20)
    ex = sim([thin, thin])
    assert ex.buy("T", 50, 0.001) is None and "price impact 19" in ex.last_note
