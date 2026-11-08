"""Short Only market analysis: price marks for every listing, picks vs skips vs short/long everything."""
import pytest
from shortbook import cli, market
from tests.test_shortbook import DAY, Clock, FakeListing, FakePerp, make, q


class PathListing(FakeListing):
    """Spot price path per symbol: price_at(symbol, ts) = path(ts - first trade)."""

    def __init__(self, markets):
        super().__init__(markets)
        self.paths, self.calls = {}, 0

    def price_at(self, symbol, ts):
        self.calls += 1
        f = self.paths.get(symbol)
        t0 = self.trades.get(symbol)
        return f(ts - t0) if f and t0 is not None and ts >= t0 else None


def _setup():
    clock = Clock()
    t0 = clock.t
    up = PathListing({"BTC": "KRW-BTC"})
    perp = FakePerp("binance", {"OLD": t0 - 30 * DAY, "MID": t0 - 5 * DAY}, {"OLD": q(10.0), "MID": q(10.0)})
    book, _ = make(clock, listing=up, perps={"binance": perp})
    book.detect()
    for b in ("OLD", "MID", "NOPE"):
        up.m[b] = f"KRW-{b}"
    book.detect()
    for b in ("OLD", "MID", "NOPE"):
        up.trades[f"KRW-{b}"] = t0
    # OLD (picked) falls 20% by 7 days, MID (between A and B) rises 10%, NOPE (no perp) falls 50%
    up.paths = {"KRW-OLD": lambda dt: 10.0 * (0.8 if dt > 3600 else 1.0),
                "KRW-MID": lambda dt: 10.0 * (1.1 if dt > 3600 else 1.0),
                "KRW-NOPE": lambda dt: 10.0 * (0.5 if dt > 3600 else 1.0)}
    book.check_pending()
    return clock, book, up, t0


def test_reasons_are_grouped():
    assert market.reason_of("trading began 3.1 h before we saw it") == "seen too late"
    assert market.reason_of("no perp on binance/bybit") == "no perp"
    assert market.reason_of("group A coinbase not traded") == "group not traded"
    assert market.reason_of("something new") == "other"


def test_marks_fill_as_horizons_come_due_and_compare_groups():
    clock, book, up, t0 = _setup()
    st = {r["base"]: r["status"] for r in book.s.q("SELECT base, status FROM listings")}
    assert st == {"OLD": "entered", "MID": "skipped", "NOPE": "skipped"}
    market.track(book)
    assert book.s.q("SELECT COUNT(*) n FROM marks")[0]["n"] == 0       # start (first trade + 5 min) not due yet
    clock.t = t0 + 2 * 3600
    market.track(book)
    hs = sorted({r["h"] for r in book.s.q("SELECT h FROM marks")})
    assert hs == [0, 1]                                                  # 24 h not due
    clock.t = t0 + 8 * DAY
    market.track(book)
    assert book.s.q("SELECT COUNT(*) n FROM marks")[0]["n"] == 3 * 5
    calls = up.calls
    market.track(book)
    assert up.calls == calls                                             # nothing fetched twice
    out = market.analyse(book.s, cost_pct=0.0)
    g = out["horizons"]["168"]["groups"]
    assert g["picks (shorted)"]["mean_pct"] == pytest.approx(20.0)
    assert g["skipped"]["n"] == 2 and g["skipped"]["mean_pct"] == pytest.approx((50 - 10) / 2)
    assert g["short every listing"]["mean_pct"] == pytest.approx((20 + 50 - 10) / 3)
    assert g["long every listing"]["mean_pct"] == pytest.approx(-(20 + 50 - 10) / 3)
    hz = out["horizons"]["168"]
    assert set(hz["skip_reasons"]) == {"between A and B", "no perp"}
    assert hz["kinds"]["B (traded elsewhere 7+ days)"]["n"] == 1
    assert "SHORT ONLY MARKET" in market.render(out)


def test_missing_price_is_retried_then_given_up():
    clock, book, up, t0 = _setup()
    up.paths["KRW-NOPE"] = lambda dt: None
    clock.t = t0 + 2 * 3600
    market.track(book)
    nope = book.s.q("SELECT id FROM listings WHERE base='NOPE'")[0]["id"]
    assert not book.s.q("SELECT 1 FROM marks WHERE listing_id=?", (nope,))
    clock.t = t0 + 3 * DAY
    market.track(book)
    r = book.s.q("SELECT h, px FROM marks WHERE listing_id=?", (nope,))
    assert [(x["h"], x["px"]) for x in r] == [(0, None)]                 # given up a day after it was due
    assert market.analyse(book.s)["horizons"]["24"]["groups"]["short every listing"]["n"] == 2


def test_short_loss_is_floored_and_feed_carries_the_analysis():
    clock, book, up, t0 = _setup()
    up.paths["KRW-MID"] = lambda dt: 10.0 * (4.0 if dt > 3600 else 1.0)  # +300%: a short can't lose more than it holds
    clock.t = t0 + 8 * DAY
    market.track(book)
    st = market.analyse(book.s, cost_pct=0.0)["horizons"]["168"]["skip_reasons"]["between A and B"]
    assert st["mean_pct"] == pytest.approx(-100.0)
    book.s.put("heartbeat", clock.t)
    cli._market_cache.clear()
    f = cli.feed(book, now=clock.t)
    assert f["market"]["horizons"]["168"]["groups"]["picks (shorted)"]["n"] == 1


def test_step_runs_the_marks_job():
    clock, book, up, t0 = _setup()
    clock.t = t0 + 2 * 3600
    book.step({})
    assert book.s.q("SELECT COUNT(*) n FROM marks")[0]["n"] == 6
