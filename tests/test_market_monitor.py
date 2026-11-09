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


def _cmp(**zs):
    return {k: {"z": z, "now": 1.0, "baseline": 0.5} for k, z in zs.items()}


def test_flag_opens_after_two_hot_hours_and_closes_when_it_cools():
    st = {}
    assert mm.advance_flags(st, T0, _cmp(rug_pct=2.5)) == []  # one hot hour: no flag yet
    ev = mm.advance_flags(st, T0 + H, _cmp(rug_pct=2.2))
    assert [e["kind"] for e in ev] == ["open"] and ev[0]["hour"] == T0 and ev[0]["direction"] == "up"
    assert [e["kind"] for e in mm.advance_flags(st, T0 + 2 * H, _cmp(rug_pct=1.7))] == ["update"]  # above close
    assert [e["kind"] for e in mm.advance_flags(st, T0 + 3 * H, _cmp(rug_pct=None))] == []  # no z keeps it open
    ev = mm.advance_flags(st, T0 + 4 * H, _cmp(rug_pct=1.2))
    assert [e["kind"] for e in ev] == ["close"] and ev[0]["hour"] == T0 + 4 * H
    # a direction flip resets the streak; sz's own trade count never flags
    assert mm.advance_flags(st, T0 + 5 * H, _cmp(rug_pct=2.5, sz_trades=9)) == []
    assert mm.advance_flags(st, T0 + 6 * H, _cmp(rug_pct=-2.5, sz_trades=9, btc_usd=5.0)) == []
    assert [e["direction"] for e in mm.advance_flags(st, T0 + 7 * H, _cmp(rug_pct=-3.0))] == ["down"]
    s2 = {}  # price levels and fear & greed never flag
    for k in range(3):
        assert mm.advance_flags(s2, T0 + k * H, _cmp(btc_usd=5.0, sol_usd=-5.0, fear_greed=4.0)) == []


def test_flags_are_saved_with_szs_result_in_their_hours(cfg):
    st = _store()
    st.execute(mm.SCHEMA)
    rows = []
    for i in range(40):  # quiet week, then 4 hours of fast rugs while sz loses
        hot = i >= 34 and i < 38
        rows.append({"v": mm.VERSION, "liquid": 30, "rug_pct": (60.0 if hot else 20.0 + i % 4),
                     "market_mean": -10.0 if hot else 2.0, "sz_trades": 5, "sz_pnl": -50.0 if hot else 10.0,
                     "sz_cost": 500.0})
    for i, d in enumerate(rows):
        st.execute("INSERT INTO market_hours(hour, data, ts) VALUES (?,?,0)", (T0 + i * H, json.dumps(d)))
    m = mm.MarketMonitor(st, None, cfg)
    events = m.update_flags()
    assert any(e["kind"] == "open" and e["metric"] == "rug_pct" for e in events)
    flags = mm.read_flags(st.query)
    rug = [f for f in flags if f["metric"] == "rug_pct"]
    assert len(rug) == 1 and rug[0]["direction"] == "up" and rug[0]["opened_hour"] == T0 + 34 * H
    assert rug[0]["closed_hour"] is None  # still hot at the last stored hour
    summ = mm.flag_summary(mm.load(st), flags, now=T0 + 41 * H)
    o = next(f for f in summ["open"] if f["metric"] == "rug_pct")
    assert o["hours"] == 6 and o["sz_pct"] == round(100 * (-50 * 4 + 10 * 2) / 3000, 1)
    g = next(x for x in summ["by_metric"] if x["metric"] == "rug_pct")
    assert g["in"]["sz_pct"] < 0 < g["out"]["sz_pct"] and g["sz_gap"] < 0
    # quiet hours close it; running again does not double count
    for i in range(40, 46):
        st.execute("INSERT INTO market_hours(hour, data, ts) VALUES (?,?,0)",
                   (T0 + i * H, json.dumps({**rows[0], "rug_pct": 21.0})))
    m.update_flags()
    m.update_flags()
    rug = [f for f in mm.read_flags(st.query) if f["metric"] == "rug_pct"]
    assert len(rug) == 1 and rug[0]["closed_hour"] is not None
    text = mm.render({**mm.status(st, hours=24, now=T0 + 47 * H), "rows": []})
    assert "WHAT THE FLAGS MEANT FOR SZ" in text and "rugged by minute 6" in text


def test_step_records_its_status_and_errors(cfg, monkeypatch):
    st = _store()
    _pool(st, "P", T0 + 60, "X", 50.0, 0.2, rw_pct=30.0)
    m = mm.MarketMonitor(st, None, cfg)
    m.step(now=T0 + 3 * H)
    s = mm.read_status(st.query)
    assert s["error"] is None and s["stored"] >= 1 and s["ok_ts"] == s["ran"]
    assert mm.payload(mm.load(st), now=T0 + 3 * H, status=s)["status"] == s

    def boom(*a, **k):
        raise RuntimeError("table gone")
    monkeypatch.setattr(m, "_step", boom)
    try:
        m.step(now=T0 + 4 * H)
    except RuntimeError:
        pass
    s2 = mm.read_status(st.query)
    assert "table gone" in s2["error"] and s2["ok_ts"] == s["ok_ts"]  # the last success is kept
    assert mm.read_status(WatchStore(":memory:").query) == {}


def test_hour_rows_carry_eight_pool_size_bands(cfg):
    from tradingbot import pool_bands as pb
    st, j = _store(), Journal(":memory:")
    for i, (liq, ret, rw) in enumerate([(10_000, 0.5, 20.0), (12_000, -1.0, 80.0), (30_000, 0.1, 30.0),
                                        (30_000, -0.2, 70.0), (700_000, 0.05, 10.0)]):
        _pool(st, f"P{i}", T0 + 60 * i, f"C{i}", 50.0, ret, rw_pct=rw, liq=liq)
    st.execute("UPDATE watch_launch_live SET token_mint='M0' WHERE pool='P0'")
    j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
              "VALUES ('M0','M0',250,350,0,?,?, 'launchrw50_sz')", (T0 + 10, T0 + 400))
    mm.MarketMonitor(st, j, cfg).step(now=T0 + H + mm.HOUR_DELAY_S + 1)
    b = mm.load(st)[0]["bands"]
    assert set(b) == {"$5-15k", "$25-40k", "$600k+"}
    assert b["$5-15k"]["n"] == 2 and b["$5-15k"]["rugs"] == 1 and b["$5-15k"]["kept_n"] == 1
    assert b["$5-15k"]["sz_n"] == 1 and b["$5-15k"]["sz_pnl"] == 100.0
    agg = {r["band"]: r for r in pb.aggregate(mm.load(st))}
    assert agg["$5-15k"]["mean_pct"] == -25.0 and agg["$5-15k"]["gap"] == 150.0 and agg["$5-15k"]["sz_pct"] == 40.0
    assert agg["$25-40k"]["kept_pct"] == 10.0 and agg["$25-40k"]["skipped_pct"] == -20.0
    assert agg["$15-25k"]["pools"] == 0 and agg["$15-25k"]["mean_pct"] is None
    tot = agg[pb.TOTAL]
    assert tot["pools"] == 5 and tot["all_100_usd"] == round(100 * (0.5 - 1.0 + 0.1 - 0.2 + 0.05)) and tot["sz_pnl"] == 100.0
    assert agg["$5-15k"]["kept_100_usd"] == 50
    assert agg["$5-15k"]["win_pct"] == 50.0 and agg["$5-15k"]["kept_n"] == 1 and agg["$5-15k"]["kept_win_pct"] == 100.0
    assert agg["$5-15k"]["kept_vs_all"] == 75.0 and agg["$5-15k"]["kept_rugged_pct"] == 0.0
    assert pb.study(st, j, now=T0 + 2 * H)["bands"][-1]["band"] == pb.TOTALS[1][0]
    # a pool under $5k measured by the scorer (status thin_done) lands in its band and the under-$5k total only
    _pool(st, "TH", T0 + 900, "CT", 5.0, 0.3, liq=2_000.0, liquid=0)
    st.execute("UPDATE watch_launch_live SET status='thin_done' WHERE pool='TH'")
    st.execute("DELETE FROM market_hours")
    mm.MarketMonitor(st, j, cfg).step(now=T0 + H + mm.HOUR_DELAY_S + 1)
    agg = {r["band"]: r for r in pb.aggregate(mm.load(st))}
    assert agg["$1-2.5k"]["pools"] == 1 and agg["$1-2.5k"]["mean_pct"] == 30.0 and agg["$1-2.5k"]["kept_n"] == 0
    assert agg[pb.TOTALS[1][0]]["pools"] == 1 and agg[pb.TOTAL]["pools"] == 5
    assert mm.load(st)[0]["liquid"] == 5  # the thin pool stays out of every other measure
    s = mm.status(st, hours=24, now=T0 + 2 * H)
    assert "whole life" in s["bands"] and "RUG-VS-WIN BY POOL SIZE" in mm.render(s)
    assert pb.band_of(4_999) == "$2.5-5k" and pb.band_of(None) is None and pb.band_of(0) == "$0-1k" and pb.band_of(150_000) == "$150-300k" and pb.band_of(10**7) == "$600k+"
    out = pb.study(st, j, now=T0 + 2 * H)
    assert [x["band"] for x in out["bands"]] == [*pb.NAMES, *(t[0] for t in pb.TOTALS)]
    by = {x["band"]: x for x in out["bands"]}
    assert by["$5-15k"]["launchrw50_sz real"]["pnl_usd"] == 100 and by["$1-2.5k"]["pools"] == 1
    assert by[pb.TOTAL]["pools"] == 5 and by[pb.TOTALS[1][0]]["pools"] == 1
    assert "RUG-VS-WIN BY POOL SIZE" in pb.render(out)


def test_hours_when_the_scorer_was_impaired_are_left_out(cfg):
    """11-09 02:00: decisions ran 84 s late, so most pools were skipped 'too late' and the hour read 15 liquid of 82
    and a -55-pt rug-vs-win gap. Such hours (or hours with no pools at all) must not move the comparison or flags."""
    st = _store()
    for i in range(8):  # healthy hour: 8 pools, one not liquid (a market skip, not an impairment)
        _pool(st, f"H{i}", T0 + i * 60, "X", 50.0, 0.2, rw_pct=30.0, liquid=0 if i == 0 else 1)
    st.execute("UPDATE watch_launch_live SET status='skipped', reason='not liquid at second 4' WHERE pool='H0'")
    for i in range(10):  # stalled hour: 7 of 10 skipped too late, the 3 that got through all rugged
        _pool(st, f"S{i}", T0 + H + i * 60, "X", 50.0, -1.0, rw_pct=30.0)
    st.execute("UPDATE watch_launch_live SET status='skipped', reason='too late (pool older than 20 s)', liquid=0 "
               "WHERE pool IN ('S0','S1','S2','S3','S4','S5','S6')")
    m = mm.MarketMonitor(st, None, cfg)
    m.step(now=T0 + 3 * H + mm.HOUR_DELAY_S + 1)  # hour 2 has no pools at all
    ok, stall, empty = mm.load(st)
    assert not ok["impaired"] and ok["data_skips"] == 0
    assert stall["impaired"] and stall["data_skips"] == 7 and "7 of 10" in stall["impaired_why"] \
        and "too late" in stall["impaired_why"]
    assert empty["impaired"] and "no pools" in empty["impaired_why"]
    # the comparison ignores impaired hours: a stalled recent window has no value instead of a fake crash
    rows = [{"hour": T0 + i * H, "liquid": 30 + i % 3, "rug_pct": 20.0 + i % 4} for i in range(60)]
    rows += [{"hour": T0 + (60 + i) * H, "liquid": 3, "rug_pct": 100.0, "impaired": True} for i in range(3)]
    c = mm.compare(rows)
    assert c["liquid"]["now"] is None and c["liquid"]["z"] is None
    rows[30:33] = [{**r, "liquid": 0, "impaired": True} for r in rows[30:33]]  # nor do they drag the baseline
    assert mm.compare(rows[:-3])["liquid"]["baseline"] > 29
    # flags: impaired hours neither open nor extend one
    st2 = _store()
    st2.execute(mm.SCHEMA)
    data = [{"hour": T0 + i * H, "v": mm.VERSION, "liquid": 30 + i % 3} for i in range(40)]
    data += [{"hour": T0 + (40 + i) * H, "v": mm.VERSION, "liquid": 2, "impaired": True,
              "impaired_why": "x"} for i in range(4)]
    st2.executemany("INSERT INTO market_hours(hour, data, ts) VALUES(?,?,?)",
                    [(d["hour"], json.dumps({k: v for k, v in d.items() if k != "hour"}), 0) for d in data])
    m2 = mm.MarketMonitor(st2, None, cfg)
    assert m2.update_flags() == [] and not st2.query("SELECT * FROM market_flags")
    text = mm.render(mm.status(st2, hours=6, now=T0 + 44 * H))
    assert "scorer impaired" in text and "!" in text
