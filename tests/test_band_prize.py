import json

import numpy as np

from tradingbot import band_prize as bp
from tradingbot import launch_live as ll
from tradingbot.data import pumpswap as ps
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
                     50_000.0 if band else 10_000.0, float(rng.uniform(0, 100)), 100.0 / 1_000_000, 100.0, 100.0,
                     "M" + f"P{i}"))
    st.executemany("INSERT INTO watch_launch_live(pool, creator, create_ts, status, liquid, ret, feat, liq_usd, rw_pct, "
                   "fill_px, fill_sol, peak_sol, token_mint) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
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
    assert out["later exits"]["covered"] == 0 and "by exit" not in out["later exits"]
    text = bp.render(out)
    assert "PRIZE IN THE $40-80k BAND" in text and "SINGLE SIGNALS" in text and "WALK-FORWARD FILTERS" in text


class _Helius:
    """Pool state after its last tx by a time: even pools climb to 1.2x by minute 10 and hold, odd pools are
    drained between minute 6 and 10."""

    def __init__(self):
        self.credits_used, self.calls = 0, 0

    def last_before(self, pool, ts):
        self.calls += 1
        self.credits_used += 10
        i = int(pool[1:])
        sol = 120.0 if i % 4 == 0 else 2.0
        tok = 1_000_000 if i % 4 == 0 else 50_000_000
        return {"meta": {"err": None, "postTokenBalances": [
            {"owner": pool, "mint": ps.WSOL, "uiTokenAmount": {"uiAmount": sol, "amount": str(int(sol * 1e9))}},
            {"owner": pool, "mint": "M" + pool, "uiTokenAmount": {"uiAmount": tok / 1e6, "amount": str(tok)}}]},
            "blockTime": ts}


def test_later_exits_fetch_states_and_judge_every_exit_on_the_same_pools():
    st, h, now = _store(), _Helius(), T0 + 2000 * 90 + 3600
    out = bp.study(st, now=now, helius=h)
    le = out["later exits"]
    alive = le["alive at minute 6"]
    assert le["alive covered"] == alive and le["covered"] > alive  # every live pool fetched, rugs added back
    assert le["fetch"]["call failed"] == 0 and h.calls > alive  # drained pools stop after minute 10
    by = le["by exit"]
    assert set(by) == {6, 10, 15, 30}
    # half the live pools are drained after minute 6: rugs rise, the paired gap vs minute 6 is negative
    assert by[10]["every pool"]["rugged_pct"] > by[6]["every pool"]["rugged_pct"]
    assert by[10]["rug causes"].get("SOL drained", 0) > 100
    assert by[10]["every pool"]["n"] == by[30]["every pool"]["n"] == by[6]["every pool"]["n"]
    assert "not rug (band), keep best 50%" in by[15]["filters"]
    text = bp.render(out)
    assert "LATER EXITS" in text and "FILTERS AT MINUTE 30" in text
    # a rerun fetches nothing new
    calls = h.calls
    bp.study(st, now=now, helius=h)
    assert h.calls == calls


def test_a_credit_cap_leaves_an_even_sample():
    st, h, now = _store(), _Helius(), T0 + 2000 * 90 + 3600
    out = bp.study(st, now=now, helius=h, max_credits=1500)
    le = out["later exits"]
    assert h.credits_used <= 1500 and "stopped at the credit cap" in " ".join(le["fetch"])
    assert 0 < le["alive covered"] < le["alive at minute 6"]
    # rugs by minute 6 are only counted inside the fetched prefix, so the minute-6 rug rate stays near the band's
    assert "by exit" in le
    assert abs(le["by exit"][6]["every pool"]["rugged_pct"] - out["ceiling"]["every pool"]["rugged_pct"]) < 15


def test_too_few_pools_says_so():
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    assert "note" in bp.study(st, now=T0)
