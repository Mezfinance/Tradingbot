"""canary-rug-check: rug streaks and the bought-vs-skipped split."""
from tradingbot import canary_rugs
from tradingbot.config import Config
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore
from tradingbot import launch_live as ll


def _setup():
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    t0 = 1_700_000_000.0
    # 20 thin rw50 picks; the canary buys every 5th (4) and all 4 rug, 3 s after our fill; 2 of the 16 skipped rug
    for i in range(20):
        c, tok = t0 + i * 100, f"T{i}"
        bought = i % 5 == 0
        rug = bought or i in (3, 4)
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, ret, status, liq_usd, rw_pct, "
                   "pulled_ts) VALUES (?,?,?,?,?,?,?,?,?)",
                   (f"P{i}", tok, c, 1, -1.0 if rug else 0.2, "done", 12_000.0, 30.0, c + 13 if rug else None))
        for book, fill, back in (("launchrw50_sz", c + 9, 0.0 if rug else 120.0),
                                 ("canary_rw50_sz@2026-11-06-canary-20", c + 10, 0.0)) if bought else \
                                (("launchrw50_sz", c + 9, 0.0 if rug else 120.0),):
            pid = j.execute("INSERT INTO positions(token, symbol, cost_usd, opened_at, closed_at, realized_usd, strategy) "
                            "VALUES (?,?,?,?,?,?,?)", (tok, tok, 100.0, fill, c + 365, back, book))
            j.execute("INSERT INTO fills(position_id, side, price, usd, fee_usd, ts) VALUES (?,?,?,?,?,?)",
                      (pid, "buy", 1.0, -100.0, 0.0, fill))
    # left out: a deep pool and a pool in the riskier half
    st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, ret, status, liq_usd, rw_pct) "
               "VALUES ('PD','DEEP',?,1,-1.0,'done',40000,30)", (t0 + 5,))
    st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, ret, status, liq_usd, rw_pct) "
               "VALUES ('PR','RISKY',?,1,-1.0,'done',12000,80)", (t0 + 6,))
    return st, j


def test_bought_pools_rugging_far_more_than_skipped_ones_shows_up():
    st, j = _setup()
    out = canary_rugs.rug_check(st, j, Config())
    thin = out["thin picks"]
    assert thin["n"] == 20 and thin["rugged"] == 6 and thin["longest_streak"] == 3  # picks 3, 4 and 5
    split = out["since the canary started"]
    assert split["thin picks the canary bought"]["n"] == 4 and split["thin picks the canary bought"]["rugged"] == 4
    assert split["thin picks the canary skipped"]["n"] == 16 and split["thin picks the canary skipped"]["rugged"] == 2
    assert split["chance the bought ones rug this often if our buy changed nothing"] < 0.01
    rows = out["canary trades"]
    assert len(rows) == 4 and all(r["pulled s after our fill"] == 3.0 for r in rows)
    assert rows[0]["canary filled at pool second"] == 10.0 and rows[0]["twin filled at pool second"] == 9.0


def test_streaks_count_back_to_back_rugs_in_time_order():
    runs = canary_rugs._streaks([(1, True), (2, True), (3, False), (4, True), (5, True), (6, True)])
    assert [r["length"] for r in runs] == [3, 2]
    assert canary_rugs._fisher_greater(0, 4, 0, 16) == 1.0
