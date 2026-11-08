"""band-compare: the by-band books against the sized book, real and ideal, on the same pools."""
import json

import pytest

from tradingbot import band_compare as bc
from tradingbot import launch_live as ll
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

T0 = 1_700_006_400.0


def _pool(st, i, ts, liq, ret, rw_pct, rwb_pct):
    st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, status, ret, liq_usd, feat, rw_pct, "
               "rwb_pct) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (f"P{i}", f"M{i}", ts, 1, "done", ret, liq, json.dumps({}), rw_pct, rwb_pct))


def _buy(j, book, i, ts, cost, back):
    j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
              "VALUES (?,?,?,?,0,?,?,?)", (f"M{i}", f"M{i}", cost, back, ts, None if back is None else ts + 360, book))


def _setup():
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    # thin pools ($10k): band keeps rank < 80, rw50 keeps rw_pct < 50
    _pool(st, 1, T0 + 60, 10_000, 0.40, rw_pct=70, rwb_pct=50)    # band only: +40%
    _pool(st, 2, T0 + 120, 10_000, -1.0, rw_pct=20, rwb_pct=90)   # rw50 only: rug
    _pool(st, 3, T0 + 180, 10_000, 0.20, rw_pct=10, rwb_pct=10)   # both: +20%
    _pool(st, 4, T0 + 240, 20_000, -0.10, rw_pct=30, rwb_pct=60)  # $15-25k: rw50 only (band needs < 30)
    _pool(st, 5, T0 + 300, 50_000, 0.30, rw_pct=10, rwb_pct=None)  # above $40k: rw50 only, band never
    _buy(j, "launchrwband", 1, T0 + 70, 100, 140)
    _buy(j, "launchrwband", 3, T0 + 190, 100, 118)
    _buy(j, "launchrwband_sz", 1, T0 + 70, 250, 340)
    _buy(j, "launchrw50_sz", 2, T0 + 130, 250, 0)
    _buy(j, "launchrw50_sz", 3, T0 + 190, 250, 290)
    _buy(j, "launchrw50_sz", 4, T0 + 250, 25, None)               # never sold: -100%
    return st, j


def test_band_compare_splits_selection_and_execution():
    st, j = _setup()
    out = bc.compare(st, j, now=T0 + 3 * 3600)
    assert out["pools"] == 5 and out["from"] is not None
    p = out["picks"]
    assert p["band picks (launchrwband / _sz)"]["all"]["n"] == 2
    assert p["band picks (launchrwband / _sz)"]["all"]["mean_pct"] == pytest.approx(30.0)
    assert p["rw50 picks (launchrw50 / _sz)"]["all"]["n"] == 4
    assert p["band only"]["all"]["n"] == 1 and p["rw50 only"]["all"]["n"] == 3 and p["picked by both"]["all"]["n"] == 1
    # rw50 picks at sz sizes: $250 x (-100% + 20%) + $25 x -10% + $100 x +30% = -200 - 2.5 + 30
    assert p["rw50 picks (launchrw50 / _sz)"]["at launchrw50_sz sizes"]["usd_at_book_size"] == round(-200 - 2.5 + 30)
    r = out["real"]
    assert r["launchrwband"]["all"]["pnl_usd"] == 58 and r["launchrwband"]["all"]["n"] == 2
    assert r["launchrw50_sz"]["all"]["n"] == 3
    assert r["launchrw50_sz"]["all"]["pnl_usd"] == round(-250 + 40 - 25)
    assert r["launchrw50_sz"]["$15-25k"]["n"] == 1
    ex = out["execution"]["launchrwband"]
    assert ex["matched"] == 2 and ex["real minus ideal (pts)"] == pytest.approx(-1.0)
    text = bc.render(out)
    assert "BAND FAMILY vs SZ" in text and "EXECUTION" in text and "band only" in text


def test_window_starts_at_the_band_books_first_trade():
    st, j = _setup()
    _buy(j, "launchrw50_sz", 9, T0 - 7200, 250, 500)             # before the band books: left out
    out = bc.compare(st, j, now=T0 + 3 * 3600)
    assert out["real"]["launchrw50_sz"]["all"]["n"] == 3
    assert bc.compare(st, j, hours=10, now=T0 + 3 * 3600)["real"]["launchrw50_sz"]["all"]["n"] == 4
