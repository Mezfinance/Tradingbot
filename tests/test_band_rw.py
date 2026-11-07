import json

import numpy as np

from tradingbot import band_rw as br
from tradingbot import launch_live as ll
from tradingbot.watcher import WatchStore

H = 3600.0
T0 = 1_700_006_400.0


def _store(n=2400, seed=1):
    """Two bands, opposite planted signals: in $5-15k a high `sig` marks big winners, in $25-40k it marks rugs."""
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        thin = i % 2 == 0
        sig = float(rng.normal())
        p_rug = 0.25 + (0.0 if thin else 0.2 * np.tanh(sig))
        p_big = 0.2 + (0.15 * np.tanh(sig) if thin else 0.0)
        u = rng.random()
        ret = -1.0 if u < p_rug else (rng.uniform(0.5, 1.5) if u < p_rug + p_big else rng.uniform(-0.3, 0.3))
        feat = {"sig": sig, "noise": float(rng.normal()), "log_seed_sol": 3.0}
        rows.append((f"P{i}", f"C{i}", T0 + i * 90, "done", 1, ret, json.dumps(feat),
                     10_000.0 if thin else 30_000.0, float(rng.uniform(0, 100))))
    st.executemany("INSERT INTO watch_launch_live(pool, creator, create_ts, status, liquid, ret, feat, liq_usd, rw_pct) "
                   "VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return st


def test_a_model_trained_on_each_band_finds_signals_the_global_model_blurs():
    out = br.study(_store(), now=T0 + 2400 * 90 + H)
    by = {b["band"]: b for b in out["bands"]}
    thin, mid = by["$5-15k"], by["$25-40k"]
    assert thin["with a band model"] > 300 and mid["with a band model"] > 300
    # the band models sort the right way in both bands: rugs high, big wins low
    assert thin["AUC band model"]["big wins (want < 0.5)"] < 0.4
    assert mid["AUC band model"]["rugs (want > 0.5)"] > 0.6
    for b in (thin, mid):
        j = b["skip riskiest 50%"]["band model"]
        assert j["kept minus every pool (pts)"] > 0 and j["random keeps worse than kept"] > 0.9
    # the global model mixes the two opposite signals and does worse than the band model in at least one band
    assert any(b["skip riskiest 50%"]["band model"]["kept"]["mean_pct"]
               > b["skip riskiest 50%"]["global model ranked in the band"]["kept"]["mean_pct"] + 2 for b in (thin, mid))
    assert by["$80-150k"]["pools tested"] == 0 and "note" in by["$80-150k"]
    text = br.render(out)
    assert "RUG-VS-WIN TRAINED PER BAND" in text and "band model, skip 50%" in text


def test_too_few_pools_says_so():
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    assert "note" in br.study(st, now=T0)
