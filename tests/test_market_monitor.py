import json
import math

from tradingbot import launch_live as ll
from tradingbot import market_monitor as mm
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

H = 3600.0
T0 = 1_700_006_400.0 - 1_700_006_400.0 % H  # a UTC hour


def _pool(st, name, ts, creator, seed, ret, rw_pct=None, liq=10_000.0, pulled_after=None, liquid=1, buy_sol=2.0):
    feat = {"log_seed_sol": math.log1p(seed), "log_buy_sol": math.log1p(buy_sol), "log_buyers": math.log1p(4),
            "repeat_creator": 1.0 if creator == "OP" else 0.0}
    st.execute("INSERT INTO watch_launch_live(pool, creator, create_ts, liquid, status, ret, pulled_ts, liq_usd, feat, "
               "rw_pct) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (name, creator, ts, liquid, "done", ret, ts + pulled_after if pulled_after else None, liq,
                json.dumps(feat), rw_pct))


def _store():
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    return st


def test_hour_row_measures_supply_operators_rugs_and_the_sized_book(cfg):
    st, j = _store(), Journal(":memory:")
    # 4 pools from one operator on the same 80 SOL seed (2 pulled at 30 s), 2 others, 1 not liquid
    for i in range(4):
        _pool(st, f"OP{i}", T0 + i * 60, "OP", 80.0, -1.0 if i < 2 else 0.6, rw_pct=70.0 if i < 2 else 20.0,
              pulled_after=30 if i < 2 else None)
    _pool(st, "A", T0 + 600, "A", 20.0, 0.1, rw_pct=10.0, liq=30_000.0)
    _pool(st, "B", T0 + 700, "B", 60.0, -1.0, rw_pct=90.0, pulled_after=200)
    _pool(st, "C", T0 + 800, "C", 5.0, None, liquid=0)
    j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
              "VALUES ('T','T',250,300,0,?,?, 'launchrw50_sz')", (T0 + 100, T0 + 460))
    j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
              "VALUES ('T','T',100,0,0,?,NULL, 'launchrw50_sz')", (T0 + 200,))  # never sold: -100%
    st.execute("INSERT INTO watch_market(ts, sol_usd) VALUES (?, 100), (?, 98)", (T0 + 10, T0 + 3000))
    st.executemany("INSERT INTO watch_macro(ts, key, value) VALUES (?,?,?)",
                   [(T0 + 10, "btc_usd", 100_000.0), (T0 + 3000, "btc_usd", 101_000.0), (T0 + 10, "fear_greed", 30.0)])
    m = mm.MarketMonitor(st, j, cfg)
    assert m.step(now=T0 + H + mm.HOUR_DELAY_S - 1) == 0  # the hour isn't finished long enough yet
    assert m.step(now=T0 + H + mm.HOUR_DELAY_S + 1) == 1
    row = mm.load(st)[0]
    assert row["pools"] == 7 and row["liquid"] == 6 and row["creators"] == 3
    assert row["top5_pct"] == 100.0 and row["template_pct"] == round(100 * 4 / 6, 1)
    assert row["rug_pct"] == 50.0 and row["early_pull_pct"] == round(100 * 2 / 6, 1)
    assert row["late_rug_pct"] == round(100 / 6, 1) and row["med_pull_s"] == 30.0
    assert row["big_win_pct"] == round(100 * 2 / 6, 1)
    assert row["rw_kept"] == round(100 * (0.6 + 0.6 + 0.1) / 3, 1) and row["rw_skipped"] == -100.0
    # sz picks at ideal: two $250 pools at +60% and one $100 pool (liq 30k) at +10%
    assert row["sz_picks_ideal"] == round(100 * (0.6 * 250 * 2 + 0.1 * 100) / 600, 1)
    assert row["sz_trades"] == 2 and row["sz_real"] == round(100 * (50 - 100) / 350, 1)
    assert row["sol_chg_pct"] == -2.0 and row["btc_chg_pct"] == 1.0 and row["fear_greed"] == 30.0
    # liquidity bands (sz's): five pools at $10k, one at $30k
    assert row["thin_pct"] == round(100 * 5 / 6, 1) and row["mid_pct"] == 0.0 and row["thick_pct"] == round(100 / 6, 1)
    assert row["thin_mean"] == -36.0 and row["thin_rug_pct"] == 60.0 and row["thick_mean"] == 10.0
    assert row["thin_sz_mean"] == 60.0 and row["mid_mean"] is None
    assert m.step(now=T0 + H + mm.HOUR_DELAY_S + 2) == 0  # stored once
    # a row stored by an older version is rebuilt
    old = {k: v for k, v in row.items() if k not in ("hour", "v", "thin_pct")}
    st.execute("UPDATE market_hours SET data=? WHERE hour=?", (json.dumps(old), T0))
    assert m.step(now=T0 + H + mm.HOUR_DELAY_S + 3) == 1 and mm.load(st)[0]["thin_pct"] == row["thin_pct"]


def test_compare_flags_a_shift_against_the_trailing_week():
    rows = [{"hour": T0 + i * H, "liquid": 30, "rug_pct": 20.0 + (i % 4), "med_pull_s": 30.0 + (i % 5)}
            for i in range(60)]
    rows += [{"hour": T0 + (60 + i) * H, "liquid": 30, "rug_pct": 21.0, "med_pull_s": 90.0} for i in range(3)]
    c = mm.compare(rows)
    assert c["med_pull_s"]["now"] == 90.0 and c["med_pull_s"]["z"] > 2
    assert abs(c["rug_pct"]["z"]) < 2
    assert mm.compare(rows[-10:])["med_pull_s"]["z"] is None  # too little baseline


def test_status_renders(cfg):
    st = _store()
    for h in range(5):
        _pool(st, f"P{h}", T0 + h * H + 60, "X", 50.0, 0.2, rw_pct=30.0)
    mm.MarketMonitor(st, None, cfg).step(now=T0 + 6 * H)
    out = mm.status(st, hours=24, now=T0 + 6 * H)
    assert len(out["rows"]) == 5
    text = mm.render(out)
    assert "MEME MARKET MONITOR" in text and "rug%" in text


def test_baseline_skips_early_hours_and_sz_before_its_first_trade():
    """11-07: hours before the Helius budget was uncapped saw half the pools, and sz read 0 trades before it
    existed; neither may count as a quiet market."""
    rows = [{"hour": T0 + i * H, "liquid": 15 + i % 3, "sz_trades": 0} for i in range(40)]
    rows += [{"hour": T0 + (40 + i) * H, "liquid": 30 + i % 4, "sz_trades": 30 + i % 5} for i in range(40)]
    rows += [{"hour": T0 + (80 + i) * H, "liquid": 31, "sz_trades": 32} for i in range(3)]
    plain = mm.compare(rows)
    assert plain["liquid"]["baseline"] < 25  # half-blind hours drag the baseline down
    fixed = mm.compare(rows, since=T0 + 40 * H)
    assert fixed["liquid"]["baseline"] > 30 and abs(fixed["liquid"]["z"]) < 2 and fixed["sz_trades"]["baseline"] > 30
    only_sz = mm.compare(rows)["sz_trades"]  # sz measures skip the hours before its first trade even without since
    assert only_sz["baseline"] > 30 and abs(only_sz["z"]) < 2
