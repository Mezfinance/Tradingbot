"""Live leverage profiles: the book's own shorts scaled to 1x/2x/3x/4x, OI cap, isolated-margin liquidation."""
import pytest
from shortbook import cli, leverage
from tests.test_shortbook import DAY, Clock, FakeListing, FakePerp, make, q


def _book(oi=100e6):
    clock = Clock()
    t0 = clock.t
    perp = FakePerp("binance", {"XYZ": t0 - 30 * DAY}, {"XYZ": q(10.0, oi=oi, vol=1e15, fr=0.0)})
    book, up = make(clock, perps={"binance": perp}, fee_round_trip_pct=0.0)
    book.detect(); up.m["XYZ"] = "KRW-XYZ"; book.detect(); up.trades["KRW-XYZ"] = t0
    book.check_pending()
    return clock, book, perp, t0


def _p(book, L):
    return next(p for p in leverage.profiles(book)["profiles"] if p["lev"] == L)


def test_profiles_scale_the_same_short():
    clock, book, perp, t0 = _book()
    assert book.open_positions()[0]["size_usd"] == pytest.approx(100_000)
    perp.qs["XYZ"] = q(8.0, vol=1e15)                          # -20%: the short is up 20%
    clock.t = t0 + 7 * DAY + 1
    book.manage()
    for L in (1, 2, 3, 4):
        assert _p(book, L)["pnl_usd"] == pytest.approx(20_000 * L, rel=1e-3)
        assert _p(book, L)["liquidated"] == 0


def test_oi_cap_stops_a_capped_short_from_growing():
    clock, book, perp, t0 = _book(oi=6e6)                      # 2% of OI = $120k: 1x is $100k, 2x+ capped at $120k
    perp.qs["XYZ"] = q(9.0, oi=6e6, vol=1e15)
    book.manage()
    assert _p(book, 1)["open_usd"] == pytest.approx(100_000)
    assert _p(book, 2)["open_usd"] == pytest.approx(120_000)
    assert _p(book, 4)["open_usd"] == pytest.approx(120_000)


def test_squeeze_liquidates_high_leverage_only_and_caps_the_loss():
    clock, book, perp, t0 = _book()
    perp.qs["XYZ"] = q(13.0, vol=1e15)                         # +30% against: past 4x's line (25% - 2.5%), short of 3x's (33.3% - 2.5%)
    book.manage()
    perp.qs["XYZ"] = q(9.0, vol=1e15)                          # then it falls back
    book.manage()
    assert _p(book, 4)["liquidated"] == 1 and _p(book, 3)["liquidated"] == 0
    p4 = _p(book, 4)
    assert p4["pnl_usd"] == pytest.approx(-400_000 / 4 - 0.01 * 400_000)
    assert _p(book, 2)["pnl_usd"] == pytest.approx(200_000 * 0.1, rel=1e-3)   # still open, up 10%
    assert book.open_positions()[0]["max_px"] == pytest.approx(13.0)


def test_snapshot_writes_one_row_per_profile_and_feed_carries_them():
    clock, book, perp, t0 = _book()
    for h in range(8):
        clock.t = t0 + h * DAY
        perp.qs["XYZ"] = q(10.0 - 0.1 * h, vol=1e15)
        book.manage(); book.snapshot()
    n = book.s.q("SELECT COUNT(*) n FROM lev_equity")[0]["n"]
    assert n == 8 * 4
    p2 = _p(book, 2)
    assert p2["sharpe"] is not None and len(p2["equity"]) == 8
    book.s.put("heartbeat", clock.t)
    f = cli.feed(book, now=clock.t)
    assert [p["lev"] for p in f["leverage"]["profiles"]] == [1, 2, 3, 4]
    assert "LEVERAGE PROFILES" in leverage.render(f["leverage"])


def test_levels_parse_and_skip_bad_entries():
    class C: leverage_profiles = "4, 1,x,2,2,-1"
    assert leverage.levels(C) == [1.0, 2.0, 4.0]
