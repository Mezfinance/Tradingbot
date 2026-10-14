import json
import random

import numpy as np

from tradingbot import watch_analysis as wa
from tradingbot.watcher import WatchStore

T0 = 1_800_000_000.0
NOW = T0 + 30 * 86400


def snap(price, buyers=10, sellers=10, liq=5000.0):
    return {"price": price, "liq": liq, "fdv": 20000.0, "buys_m5": buyers, "sells_m5": sellers,
            "buyers_m5": buyers, "sellers_m5": sellers, "vol_m5": 1000.0, "chg_m5": 0.0,
            "buys_h1": buyers, "sells_h1": sellers, "buyers_h1": buyers, "sellers_h1": sellers, "vol_h1": 1000.0}


def add_pool(store, pool, mint, created, dex="pump-fun", snaps=None, candles=None, info=None, info_ts=None):
    store.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts, first, info, info_ts, ohlcv,"
                  " closed) VALUES(?,?,?,?,?,?,?,?,?,?,1)",
                  (pool, mint, dex, f"{pool} / SOL", created, created, json.dumps(snap(1.0)),
                   json.dumps(info) if info is not None else None, info_ts,
                   json.dumps(candles) if candles is not None else "[]"))
    for age, d in (snaps or {}).items():
        store.execute("INSERT INTO watch_snaps(pool, age_min, ts, data) VALUES(?,?,?,?)",
                      (pool, age, created + age * 60, json.dumps(d)))


def candle_path(created, start_min, mults):
    """5-min candles after `start_min`, one per multiple (high = low = close = multiple)."""
    return [[created + (start_min + 5 * i) * 60, m, m, m, m, 100] for i, m in enumerate(mults)]


def synthetic(n=600, signal=True, seed=1):
    rng = random.Random(seed)
    store = WatchStore(":memory:")
    for i in range(n):
        created = T0 + i * 1800
        strong = rng.random() < 0.5
        buyers = rng.randint(40, 80) if strong else rng.randint(5, 30)
        if signal:
            pumps = rng.random() < (0.5 if strong else 0.05)
        else:
            pumps = rng.random() < 0.2
        path = [1.2, 1.5, 2.5, 1.8] if pumps else [0.9, 0.7, 0.4, 0.3]
        add_pool(store, f"P{i}", f"M{i}", created, snaps={15: snap(1.0, buyers=buyers, sellers=20)},
                 candles=candle_path(created, 15, path))
    return store


def test_auc_matches_definition():
    assert wa.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    assert wa.auc(np.array([1.0, 1.0]), np.array([0, 1])) == 0.5


def test_planted_signal_is_found_and_holds_out_of_sample():
    a = wa.analyse(synthetic(), age=15, now=NOW)
    assert a["rows"] == 600
    assert a["test_auc"] > 0.75 and a["test_lift"] > 1.5
    top_features = [u["feature"] for u in a["univariate"][:3]]
    assert any("buyer" in f for f in top_features)
    assert a["trade_top_decile"]["mean_ret"] > a["trade_all_test_mean"]


def test_noise_does_not_look_like_an_edge():
    a = wa.analyse(synthetic(signal=False), age=15, now=NOW)
    assert 0.35 < a["test_auc"] < 0.65


def test_pump_on_the_graduated_pool_counts_for_the_launch():
    store = WatchStore(":memory:")
    add_pool(store, "LAUNCH", "MINT", T0, snaps={15: snap(1.0)})  # bonding curve, no candles of its own
    add_pool(store, "GRAD", "MINT", T0 + 3600, dex="pumpswap", candles=candle_path(T0, 70, [1.5, 3.0, 2.0]))
    [row] = [r for r in wa.build_dataset(store, 15, now=NOW) if r["pool"] == "LAUNCH"]
    assert row["pump2x"] and row["best_mult"] == 3.0


def test_stop_before_target_is_not_a_clean_pump():
    store = WatchStore(":memory:")
    add_pool(store, "P", "M", T0, snaps={15: snap(1.0)}, candles=candle_path(T0, 15, [0.6, 0.45, 2.5]))
    [row] = wa.build_dataset(store, 15, now=NOW)
    assert row["pump2x"] and not row["clean2x"]
    assert abs(row["trade_ret"] - (-0.5 - wa.ROUND_TRIP_COST)) < 1e-9


def test_no_peeking_at_info_fetched_after_the_snapshot():
    store = WatchStore(":memory:")
    info = {"developer": "DEV", "twitter": True, "holders": 50}
    add_pool(store, "EARLY", "M1", T0, snaps={15: snap(1.0)}, info=info, info_ts=T0 + 14 * 60,
             candles=candle_path(T0, 15, [1.0]))
    add_pool(store, "LATE", "M2", T0, snaps={15: snap(1.0)}, info=info, info_ts=T0 + 60 * 60,
             candles=candle_path(T0, 15, [1.0]))
    rows = {r["pool"]: r for r in wa.build_dataset(store, 15, now=NOW)}
    assert rows["EARLY"]["has_twitter"] == 1.0
    assert "has_twitter" not in rows["LATE"]


def test_developer_history_uses_only_finished_earlier_tokens():
    store = WatchStore(":memory:")
    info = {"developer": "SERIAL"}
    for i in range(3):  # three earlier launches, all pumped, all finished
        c = T0 + i * 3600
        add_pool(store, f"OLD{i}", f"O{i}", c, snaps={15: snap(1.0)}, info=info, info_ts=c + 60,
                 candles=candle_path(c, 15, [3.0]))
    c = T0 + 3 * 86400
    add_pool(store, "NEW", "N", c, snaps={15: snap(1.0)}, info=info, info_ts=c + 60, candles=candle_path(c, 15, [1.0]))
    sibling = T0 + 3 * 86400 - 600  # same developer, earlier, but its outcome is not known yet at NEW's snapshot
    add_pool(store, "SIB", "S", sibling, snaps={15: snap(1.0)}, info=info, info_ts=sibling + 60,
             candles=candle_path(sibling, 15, [0.1]))
    rows = {r["pool"]: r for r in wa.build_dataset(store, 15, now=NOW)}
    assert rows["NEW"]["dev_prior_launches"] == np.log1p(4)
    assert rows["NEW"]["dev_prior_pump_rate"] == 1.0  # SIB is counted as a launch, not as a known outcome


def test_too_little_data_says_so(tmp_path):
    a = wa.analyse(WatchStore(":memory:"), age=15, now=NOW)
    assert a["rows"] == 0 and "need at least" in a["note"]
    assert "need at least" in wa.render(a, wa.summary(WatchStore(":memory:"), now=NOW))


def test_export_writes_features_and_labels(tmp_path):
    out = tmp_path / "rows.csv"
    assert wa.export_csv(synthetic(n=50), str(out), age=15, now=NOW) == 50
    header = out.read_text().splitlines()[0].split(",")
    assert "pump2x" in header and "buyer_ratio_m5" in header and "pool" in header
