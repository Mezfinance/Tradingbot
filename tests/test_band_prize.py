import json

import numpy as np

from tradingbot import band_prize as bp
from tradingbot import launch_live as ll
from tradingbot.watcher import WatchStore

T0 = 1_700_006_400.0


def _store(n=2000, seed=2):
    """$40-80k pools where a planted `sig` marks rugs (high) and winners (low); other pools carry no signal."""
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        band = i % 2 == 0
        sig = float(rng.normal())
        p_rug = 0.3 + (0.25 * np.tanh(sig) if band else 0.0)
        u = rng.random()
        ret = -1.0 if u < p_rug else float(rng.uniform(-0.2, 0.6))
        feat = {"sig": sig, "noise": float(rng.normal()), "log_seed_sol": 3.0}
        rows.append((f"P{i}", f"C{i}", T0 + i * 90, "done", 1, ret, json.dumps(feat),
                     50_000.0 if band else 10_000.0, float(rng.uniform(0, 100)), 1e-6, 100.0, 120.0))
    st.executemany("INSERT INTO watch_launch_live(pool, creator, create_ts, status, liquid, ret, feat, liq_usd, rw_pct, "
                   "fill_px, fill_sol, peak_sol) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return st


def test_finds_a_planted_signal_in_the_band_and_shows_the_ceiling():
    out = bp.study(_store(), now=T0 + 2000 * 90 + 3600)
    assert out["pools"] == 1000 and out["share of all liquid pools"] == 50.0
    c = out["ceiling"]
    assert c["rugs removed (a perfect rug filter)"]["rugged_pct"] == 0
    assert c["best half in hindsight"]["mean_pct"] > c["every pool"]["mean_pct"]
    top = out["signals"][0]
    assert top["signal"] == "sig" and top["same sign both halves"]
    assert top["AUC"]["rug"] > 0.6
    j = out["filters"]["not rug (band), keep best 50%"]
    assert j["kept minus every pool (pts)"] > 5 and j["random keeps worse than kept"] > 0.9
    assert out["minute 10 (cached pools only)"]["pools"] == 0
    text = bp.render(out)
    assert "PRIZE IN THE $40-80k BAND" in text and "SINGLE SIGNALS" in text and "WALK-FORWARD FILTERS" in text


def test_minute_10_uses_cached_states():
    st = _store()
    ll.ensure_schema(st)
    st.execute(ll.PATH_SCHEMA)
    st.executemany("INSERT INTO watch_launch_path(pool, sec, sol, tok, ts) VALUES (?,?,?,?,?)",
                   [(f"P{i}", 600, 110.0, 100.0 / 1.1e-6, 0.0) for i in range(0, 2000, 2)])
    out = bp.study(st, now=T0 + 2000 * 90 + 3600)
    m = out["minute 10 (cached pools only)"]
    assert m["pools"] > 100 and "minute 10" in m
    assert "MINUTE 10" in bp.render(out)


def test_too_few_pools_says_so():
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    assert "note" in bp.study(st, now=T0)
