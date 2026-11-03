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


def test_launch_features_read_what_the_chain_shows_by_second_x():
    txs = [[0, 1, "OP", 1000, 50, 1000, 50], [0, 1, "OP", -10, 1, 990, 51],  # creator's own buy in the creation slot
           [3, 2, "A", -20, 2, 970, 53], [8, 3, "B", 15, -1, 985, 52], [20, 4, "C", -50, 5, 935, 57]]
    f = la.launch_features(txs, 10, "OP", prior_pools=2, seed_sol=50)
    assert f["log_buys"] == pytest.approx(__import__("math").log1p(2)) and f["log_sells"] > 0
    assert f["creator_buys"] == 1 and f["first_slot_txs"] == 2 and f["repeat_creator"] == 1
    assert f["sol_growth"] == pytest.approx(52 / 50)  # the trade at second 20 isn't visible yet


def test_ranking_walks_forward_and_its_top_picks_beat_buying_everything():
    import numpy as np
    st = WatchStore(":memory:")
    rng = np.random.default_rng(3)
    for i in range(360):  # three days; pools with many early buyers end up higher
        busy = rng.random() < 0.3
        pool = f"P{i}"
        created = T0 + (i // 120) * 86400 + (i % 120) * 600
        txs = [[0, 1, "OP", 1000, 50, 1000, 50]] + [[1 + j, 2 + j, f"W{j}", -1, 0.5, 1000 - j, 50 + 0.5 * j]
                                                     for j in range(8 if busy else 1)]
        st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts, first) VALUES(?,?,?,?,?,?,?)",
                   (pool, pool + "Mint", "pumpswap", "TOK", created, created + 60,  # same name: no drift
                    json.dumps({"price": 1e-5, "liq": 20_000.0})))
        st.execute("INSERT INTO watch_enrich(pool, kind, ts, data) VALUES(?,?,?,?)",
                   (pool, "launch", created + 400, json.dumps({"create_ts": created, "txs": txs, "complete": True,
                                                                "creator": "OP", "selection": "sample"})))
        st.execute("INSERT INTO watch_snaps(pool, age_min, ts, data) VALUES(?,?,?,?)",
                   (pool, 5, created + 380, json.dumps({"price": (1.6e-5 if busy else 0.8e-5), "liq": 20_000.0})))
    out = la.backtest(st, Config())
    r10 = out["ranking (walk-forward, on-chain features at second x)"]["second 10"]
    assert r10["days_scored"] == 2 and r10["top 10%"]["n"] > 0
    assert r10["top 10%"]["mean_pct"] > r10["every liquid pool (test days)"]["mean_pct"]
    fair = out["fair: liquid at second x"]["second 10"]
    assert fair["second x"]["n"] == 360 and "paired gain (second x - first sight)" in fair


def _three_days(st):
    import numpy as np
    rng = np.random.default_rng(3)
    for i in range(360):  # busy pools have more early buyers (seconds 1-8) and end higher
        busy = rng.random() < 0.3
        pool = f"P{i}"
        created = T0 + (i // 120) * 86400 + (i % 120) * 10  # one hour a day: no hour-of-day noise
        txs = [[0, 1, "OP", 1000, 50, 1000, 50]] + [[1 + j, 2 + j, f"W{j}", -1, 0.5, 1000 - j, 50 + 0.5 * j]
                                                     for j in range(8 if busy else 1)]
        st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts, first) VALUES(?,?,?,?,?,?,?)",
                   (pool, pool + "Mint", "pumpswap", "TOK", created, created + 60,
                    json.dumps({"price": 1e-5, "liq": 20_000.0})))
        st.execute("INSERT INTO watch_enrich(pool, kind, ts, data) VALUES(?,?,?,?)",
                   (pool, "launch", created + 400, json.dumps({"create_ts": created, "txs": txs, "complete": True,
                                                                "creator": "OP", "selection": "sample"})))
        st.execute("INSERT INTO watch_snaps(pool, age_min, ts, data) VALUES(?,?,?,?)",
                   (pool, 5, created + 380, json.dumps({"price": (1.6e-5 if busy else 0.8e-5), "liq": 20_000.0})))


def test_timing_decides_on_early_features_and_fills_later():
    import time as _t
    st = WatchStore(":memory:")
    _three_days(st)
    out = la.backtest(st, Config(), only_timing=True)["timing (walk-forward: decide at d, fill at f)"]
    assert set(out) == {f"decide {d}s, fill {f}s" for d, f in la.TIMING}
    cell = out["decide 4s, fill 6s"]
    assert cell["days_scored"] == 2 and len(cell["by day"]) == 2
    assert cell["top 5%"]["n"] > 0 and cell["top 5%"]["mean_pct"] > cell["every liquid pool"]["mean_pct"]
    day3 = _t.strftime("%Y-%m-%d", _t.gmtime(T0 + 2 * 86400))
    later = la.backtest(st, Config(), only_timing=True, test_from=day3)["timing (walk-forward: decide at d, fill at f)"]
    assert list(later["decide 5s, fill 7s"]["by day"]) == [day3]  # earlier days only train


def test_fill_return_uses_the_chain_price_at_the_fill_second():
    st = WatchStore(":memory:")
    # chain price 0.5 until second 5, 1.0 from second 6 (buyers piling in); first sight at second 60, exit 1.2x
    add(st, "X", [row(0, 100, 50), row(6, 100, 100)], 60, 1e-5, 1.2e-5)
    r = la.rows(st, Config())[0]["ret"]
    assert r[5] == pytest.approx(1.2 * 2.0 - 1 - 0.06) and r[6] == pytest.approx(1.2 - 1 - 0.06)
    assert 4 in r and 7 in r


def test_timing_without_name_features(monkeypatch):
    st = WatchStore(":memory:")
    _three_days(st)
    seen = []
    real = la.wa._matrix
    monkeypatch.setattr(la.wa, "_matrix", lambda feats, names, *a: seen.append(set(names)) or real(feats, names, *a))
    out = la.backtest(st, Config(), only_timing=True, no_name=True)["timing (walk-forward: decide at d, fill at f)"]
    assert seen and not any(n & set(la.LIVE_DROP) for n in seen)
    assert out["decide 4s, fill 6s"]["top 5%"]["n"] > 0


def test_exit_timing_check_compares_the_snapshot_with_minute_6():
    st = WatchStore(":memory:")
    _three_days(st)
    for r in st.query("SELECT pool, created_ts FROM watch_pools"):
        c = r["created_ts"]
        candles = [[c, 1, 1, 1, 1.0, 5], [c + 330, 1, 2, 1, 2.0, 5], [c + 370, 2, 3, 2, 3.0, 5]]
        st.execute("INSERT INTO watch_enrich(pool, kind, ts, data) VALUES(?,?,?,?)",
                   (r["pool"], "candles15", c + 900, json.dumps({"candles": candles})))
    out = la.backtest(st, Config(), only_timing=True)["timing (walk-forward: decide at d, fill at f)"]
    chk = out["decide 4s, fill 6s"]["exit timing check"]["every liquid pool"]
    assert chk["snapshot taken at pool minute"]["median"] == pytest.approx(380 / 60, abs=0.01)
    n = chk["candle price at pool minute 6"]["n"]
    assert n > 0 and chk["pools with candles"] == f"{n} of {n}"
    # candle price doubles by minute 6 and triples by the snapshot (both from c_seen = 1.0)
    assert chk["candle price at the snapshot time"]["mean_pct"] > chk["candle price at pool minute 6"]["mean_pct"]
    r = next(r for r in la.rows(st, Config()) if "cx" in r)
    assert la.candle_ret(r, 6, "c6") == pytest.approx(min(2.0 * r["cx"]["link"][6] - 1 - la.COST, 3.0))
    assert "exit timing check" not in out["decide 5s, fill 5s"]


def test_fetch_candles_gets_history_for_picks_and_a_sample():
    st = WatchStore(":memory:")
    _three_days(st)
    asked = []

    class Gecko:
        def ohlcv(self, pool, minutes=5, limit=300, before=None):
            asked.append(pool)
            c = before - la.HIST_CANDLE_WINDOW_S
            return [[c, 1, 1, 1, 1.0, 5], [c + 330, 1, 2, 1, 2.0, 5], [c + 370, 2, 3, 2, 3.0, 5]]
    out = la.backtest(st, Config(), only_timing=True, fetch_candles=10, gecko=Gecko())
    chk = out["timing (walk-forward: decide at d, fill at f)"]["decide 4s, fill 6s"]["exit timing check"]
    n_picks = int(chk["top 5%"]["pools with candles"].split(" of ")[1])
    assert len(asked) == len(set(asked)) == n_picks + 10
    assert chk["top 5%"]["pools with candles"] == f"{n_picks} of {n_picks}"
    la.backtest(st, Config(), only_timing=True, fetch_candles=10, gecko=Gecko())
    assert len(asked) == n_picks + 10  # saved: a rerun fetches nothing new


def test_export_exit_rows_lists_picks_and_a_sample():
    st = WatchStore(":memory:")
    _three_days(st)
    out = la.export_exit_rows(la.rows(st, Config()), None, True, 10)
    tops = [r for r in out["rows"] if r["top"]]
    assert tops and len(out["rows"]) == len(tops) + 10
    r = tops[0]
    assert r["link"] > 0 and r["snap_min"] == pytest.approx(380 / 60, abs=0.01) and r["created"] and r["day"]
    assert {r["top"] for r in tops} <= {2.0, 5.0}


def test_fit_score_caps_extreme_feature_values():
    import numpy as np
    rng = np.random.default_rng(1)
    train = [{"feat": {4: {"a": float(rng.normal()), "b": float(rng.normal())}}, "ret": {6: float(rng.normal())}}
             for _ in range(200)]
    test = [{"feat": {4: {"a": 0.0, "b": 1e9}}, "ret": {6: 0.0}}]  # a ratio blown up by a near-zero denominator
    tr, te = la._fit_score(train, test, 4, 6)
    xs = np.array([[r["feat"][4]["a"], r["feat"][4]["b"]] for r in train])
    assert abs(te[0]) <= np.abs(tr).max() + la.CLIP_Z * 10  # bounded, not ~1e9 times the weight
