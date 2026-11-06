import json
import random

from tradingbot import early_rw
from tradingbot.config import Config
from tradingbot.watcher import WatchStore
from tests.test_model_book import snap, T0


def _store(n=500, seed=4):
    """Hand-made pools seen at minute 2. Pools with few sellers at first sight get pulled before minute 5;
    busy ones (many buyers) often run +60%. The other pools drift."""
    rng = random.Random(seed)
    st = WatchStore(":memory:")
    for i in range(n):
        created = T0 + i * 600
        risky, busy = rng.random() < 0.3, rng.random() < 0.3
        first = snap(1.0, buyers=rng.randint(60, 120) if busy else rng.randint(5, 20), liq=20_000.0)
        first["sellers_m5"] = 1 if risky else 15
        st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts, first) VALUES(?,?,?,?,?,?,?)",
                   (f"P{i}", f"M{i}", "pumpswap", f"P{i} / SOL", created, created + 120, json.dumps(first)))
        pulled = risky and rng.random() < 0.7
        p5 = 1.7 if busy and rng.random() < 0.6 else rng.uniform(0.9, 1.15)
        for age, mult in ((5, p5), (15, p5 * 1.1), (30, p5 * 0.9)):
            st.execute("INSERT INTO watch_snaps(pool, age_min, ts, data) VALUES(?,?,?,?)",
                       (f"P{i}", age, created + age * 60 + 30,
                        json.dumps(snap(mult, liq=100.0 if pulled else 20_000.0))))
    return st


def test_early_rw_study_learns_rugs_and_reports_every_exit(monkeypatch):
    monkeypatch.setattr(early_rw, "MIN_TRAIN", 150)
    monkeypatch.setattr(early_rw, "RANDOM_DRAWS", 100)
    st = _store()
    cfg = Config(db_path=":memory:")
    out = early_rw.study(st, cfg, now=T0 + 500 * 600 + 7200)
    assert out["tested"] > 100 and out["blocks"] > 5
    assert out["AUC at the 5-min exit (score = riskiness)"]["rugs (want > 0.5)"] > 0.7
    pools = out["every eligible pool"]
    five = pools["snapshot 5 min"]["skip riskiest 30%"]
    assert five["kept"]["rugged_pct"] < five["skipped"]["rugged_pct"]
    assert five["kept minus skipped (pts)"] > 10
    assert five["random keeps: mean / share worse than kept"][1] > 0.9
    assert "5 min, hold to 15 if up" in pools and "snapshot 30 min" in pools
    assert pools["snapshot 180 min"]["all"]["rugged_pct"] == 100  # no 180-min snapshots: only pulled pools count
    # a pool pulled at minute 5 stays -100% at every later exit
    rows = early_rw.load_pools(st, cfg)
    pulled = [p for p in rows if p["rets"][5] == -1.0]
    assert pulled and all(p["rets"][30] == -1.0 and p["rets"]["5 min, hold to 15 if up"] == -1.0 for p in pulled)
