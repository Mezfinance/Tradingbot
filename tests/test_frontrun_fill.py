"""Realistic NET of the live fires, settled on Helius true reserves."""
from tradingbot import frontrun_fill as ff


def _row(slot, tokd, sold, tok_after, sol_after):
    # launch_record row: [sec, slot, payer, tokenDelta, solDelta, tok_after, sol_after]
    return [0.0, slot, "W", tokd, sold, tok_after, sol_after]


def _ramp_rec():
    # a rising pool: price (sol/tok) climbs slot 100 -> 105, every trade a buy (tokens out, SOL in)
    txs = [
        _row(100, -20, 2, 1000, 100),
        _row(101, -20, 2, 980, 102),
        _row(102, -10, 2, 970, 104),
        _row(103, -20, 6, 950, 110),
        _row(104, -10, 3, 940, 113),
        _row(105, -10, 3, 930, 116),
    ]
    return {"txs": txs}


def _rug_rec():
    txs = [
        _row(100, -20, 2, 1000, 100),
        _row(101, -10, 1, 990, 101),
        _row(102, -980, -100, 10, 1),      # tokDelta<0 AND solDelta<0 => an LP pull (the rug)
        _row(103, 0, 0, 10, 1),
    ]
    return {"txs": txs}


def test_rising_pool_nets_positive_and_pays():
    out = ff.measure([{"pool": "P", "slot": 100}], have={"P": _ramp_rec()})
    assert out["n_checked"] == 1
    # the N+1 small-size cell buys the climb and sells higher -> clearly positive after impact + fees
    c = out["agg"]["1|25.0"]
    assert c["n"] == 1 and c["mean"] > 0.02 and c["win"] == 1.0
    best = max(out["agg"].values(), key=lambda v: v["mean"])
    assert best["mean"] > 0


def test_a_pull_in_the_hold_is_minus_100():
    out = ff.measure([{"pool": "R", "slot": 100}], have={"R": _rug_rec()}, offsets=(1,), sizes=(25.0,))
    # fire 100, land N+1 (101), hold 2 -> exit 103; the pull at 102 is inside (101, 103] -> -100%
    assert out["agg"]["1|25.0"]["mean"] == -1.0
    assert out["agg"]["1|25.0"]["rug"] == 1.0


def test_box_sol_reserve_drift_is_decimals_free():
    # true SOL reserve at slot 101 is 102 (SOL). A box stamp of 100e9 lamports -> 100 SOL -> -1.96% drift.
    fire = {"pool": "P", "slot": 100, "sol": 100_000_000_000, "tok": 980, "reserve_slot": 101}
    out = ff.measure([fire], have={"P": _ramp_rec()})
    r = out["rows"][0]
    assert abs(r["sol_drift"] - (100.0 / 102.0 - 1.0)) < 1e-9
    assert out["sol_drift"]["n"] == 1


def test_unfetched_fire_without_helius_is_skipped_not_crashed():
    out = ff.measure([{"pool": "X", "slot": 5}])          # no rec, no helius
    assert out["n_checked"] == 0 and out["rows"][0]["status"] == "no helius"


def test_render_runs_and_flags_no_stamps():
    out = ff.measure([{"pool": "P", "slot": 100}], have={"P": _ramp_rec()})
    txt = ff.render(out)
    assert "FRONT-RUN NET" in txt and "No box reserve stamps" in txt and "BEST CELL" in txt
