import json

import pytest

from tradingbot import launch_analysis as la
from tradingbot.config import Config
from tradingbot.watcher import WatchStore

T0 = 1_800_000_000.0


def add(st, pool, txs, seen_s, first_price, snap_price, first_liq=20_000.0, snap_liq=20_000.0, complete=True):
    st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts, first) VALUES(?,?,?,?,?,?,?)",
               (pool, pool + "Mint", "pumpswap", pool, T0, T0 + seen_s,
                json.dumps({"price": first_price, "liq": first_liq})))
    st.execute("INSERT INTO watch_enrich(pool, kind, ts, data) VALUES(?,?,?,?)",
               (pool, "launch", T0 + 400, json.dumps({"create_ts": T0, "txs": txs, "complete": complete})))
    st.execute("INSERT INTO watch_snaps(pool, age_min, ts, data) VALUES(?,?,?,?)",
               (pool, 5, T0 + 380, json.dumps({"price": snap_price, "liq": snap_liq})))


def row(t, tokens, sol):
    return [t, 1, "W", 0, 0, tokens, sol]


def test_entry_seconds_link_chain_prices_to_first_sight_and_the_5_minute_exit():
    st = WatchStore(":memory:")
    # chain price (SOL/token) 1.0 until second 30, 2.0 from second 30; first sight at second 60; exit = 1.5x
    add(st, "UP", [row(0, 100, 100), row(30, 100, 200)], 60, 1e-5, 1.5e-5)
    # pulled on-chain at second 40, before first sight at second 70
    add(st, "RUG", [row(0, 100, 100), row(40, 100, 1)], 70, 1e-5, 1e-5)
    add(st, "BUSY", [row(0, 100, 100)], 90, 1e-5, 1e-5, complete=False)  # 100 txs ended before first sight
    out = la.backtest(st, Config(), picks={"UP"})
    assert out["pools"] == 3 and out["dropped"] == {"over 100 txs before first sight": 1}
    up = {r["pool"]: r for r in la.rows(st, Config())}["UP"]["ret"]
    assert up["first sight"] == pytest.approx(0.5 - 0.06)
    assert up[10] == pytest.approx(1.5 * 2.0 - 1 - 0.06)  # bought at half the first-sight chain price
    assert up[30] == pytest.approx(0.5 - 0.06) and 90 not in up  # entries after first sight aren't scored
    rug = {r["pool"]: r for r in la.rows(st, Config())}["RUG"]["ret"]
    assert rug["first sight"] is None and rug[10] == -1.0  # the early buyer is caught; first sight skips it
    picks = out["early-book picks"]
    assert picks["first sight"]["n"] == 1 and picks["second 10"]["mean_pct"] == pytest.approx(194.0)
