"""The live HybridRunner mirrors flurry_study._hybrid_bank EXACTLY (the box == the study guarantee).

The study's front-run path forces CAUSAL_WALLET on (online, no look-ahead wallet ranks); the live banks are
causal by construction, so parity is checked with CAUSAL_WALLET on."""
import pytest

from tradingbot import flurry_study as fl
from tradingbot import frontrun_live as live
from tradingbot import precursor_gate as pg


@pytest.fixture(autouse=True)
def _causal():
    saved = fl.CAUSAL_WALLET
    fl.CAUSAL_WALLET = True          # the frontrun gate is fit on causal features; the live banks are causal
    yield
    fl.CAUSAL_WALLET = saved


def _row(slot, payer, tokd, sold, tok_after, sol_after, vis):
    return [0.0, slot, payer, tokd, sold, tok_after, sol_after, vis]


def _rec():
    # a mix of direct (shred-visible) and NON-direct (aggregator) trades, so the rebased history differs from the
    # shred-only view at several slots.
    return {"txs": [
        _row(10, "A", -20, 2, 1000, 100, True),
        _row(11, "A", -20, 4, 980, 104, False),    # non-direct: dead in shreds, dominant in full
        _row(12, "C", -10, 2, 970, 106, True),
        _row(13, "D", -10, 1, 965, 107, False),    # non-direct new wallet: "." in shreds, "N" in full
        _row(14, "C", -30, 5, 940, 112, True),
        _row(15, "E", -10, 2, 935, 114, True),
        _row(16, "A", -10, 1, 930, 115, False),
        _row(17, "F", -10, 2, 925, 117, True),
    ]}


def _feed_records(runner, rec):
    """Drive the runner the box's way: deliver every FULL slot (Helius WS) before scoring the matching DIRECT slot
    (shred). Returns the live hybrid record per direct slot."""
    for s in fl.slot_flow(rec, "full")["slots"]:
        runner.on_full_slot(s)
    out = []
    for s in fl.slot_flow(rec, "direct")["slots"]:
        runner.on_direct_slot(s)
        out.append(runner.last_rec)
    return out


def test_hybrid_runner_records_match_the_study_bank_field_for_field():
    tgt = fl._tgt("price", 3, 2, 1.0)
    rec = _rec()
    study = fl._hybrid_bank(rec, 0.0, tgt)
    gate = {"intercept": -10.0, "predictors": [{"coef": 20.0, "name": "sym == N"}], "threshold": {"1": 0.0}, "hold": 2}
    live_recs = _feed_records(live.HybridRunner(gate, "1"), rec)
    assert len(live_recs) == len(study)
    for g, w in zip(live_recs, study):
        assert g["sym"] == w["sym"]
        assert g["m2"] == w["m2"] and g["m3"] == w["m3"]
        assert set(g["f"]) == set(w["f"])
        for k in g["f"]:
            assert abs(g["f"][k] - w["f"][k]) < 1e-9, k


def test_hybrid_runner_fires_on_exactly_the_study_gate_slots():
    rec = _rec()
    tgt = fl._tgt("price", 3, 2, 1.0)
    study = fl._hybrid_bank(rec, 0.0, tgt)
    gate = pg.compile_gate({"intercept": -1.0, "predictors": [{"coef": 5.0, "name": "sym == N"}],
                            "threshold": {"1": 0.0}, "hold": 2})
    direct_slots = [s["slot"] for s in fl.slot_flow(rec, "direct")["slots"]]
    study_fires = [direct_slots[i] for i, r in enumerate(study) if pg.fires(r, gate, "1")]
    runner = live.HybridRunner(gate, "1")
    for s in fl.slot_flow(rec, "full")["slots"]:
        runner.on_full_slot(s)
    live_fires = [sig["signal_slot"] for s in fl.slot_flow(rec, "direct")["slots"]
                  if (sig := runner.on_direct_slot(s))]
    assert live_fires == study_fires and len(study_fires) >= 1


def test_hybrid_consumer_routes_feeds_per_pool_and_fires():
    rec = _rec()
    gate = pg.compile_gate({"intercept": -1.0, "predictors": [{"coef": 5.0, "name": "sym == N"}],
                            "threshold": {"1": 0.0}, "hold": 2})
    pools_seen, fires = [], []
    con = live.HybridConsumer(gate, "1", on_fire=fires.append, on_pool=pools_seen.append)
    # rebase first (Helius), then score (shred) -- the box's ordering
    for s in fl.slot_flow(rec, "full")["slots"]:
        con.on_helius_slot("POOL", s)
    for s in fl.slot_flow(rec, "direct")["slots"]:
        con.on_shred_slot("POOL", s)
    assert pools_seen == ["POOL"]                 # on_pool fired once, on first sight
    assert fires and all(f["pool"] == "POOL" for f in fires)
    # parity: the consumer fires on exactly the study's gate slots
    study = fl._hybrid_bank(rec, 0.0, fl._tgt("price", 3, 2, 1.0))
    direct_slots = [s["slot"] for s in fl.slot_flow(rec, "direct")["slots"]]
    want = [direct_slots[i] for i, r in enumerate(study) if pg.fires(r, gate, "1")]
    assert [f["signal_slot"] for f in fires] == want


def test_hybrid_consumer_fires_on_every_qualifying_slot_no_dedup():
    # a setup seen twice fires twice -- no per-pool dedup. The live rate is read raw against the study's ~1000/12h.
    rec = _rec()
    gate = pg.compile_gate({"intercept": -1.0, "predictors": [{"coef": 5.0, "name": "sym == N"}],
                            "threshold": {"1": 0.0}, "hold": 2})
    study = fl._hybrid_bank(rec, 0.0, fl._tgt("price", 3, 2, 1.0))
    n_fires = sum(1 for r in study if pg.fires(r, gate, "1"))
    assert n_fires >= 2                                       # the planted flow fires on more than one slot
    fired = []
    con = live.HybridConsumer(gate, "1", on_fire=fired.append)
    for s in fl.slot_flow(rec, "full")["slots"]:
        con.on_helius_slot("POOL", s)
    for s in fl.slot_flow(rec, "direct")["slots"]:
        con.on_shred_slot("POOL", s)
    assert len(fired) == n_fires                              # every qualifying slot fired, repeats included


def test_hybrid_consumer_freshness_gate_suppresses_mid_life_pools():
    gate = pg.compile_gate({"intercept": -1.0, "predictors": [{"coef": 9.0, "name": "sym == N"}],
                            "threshold": {"1": 0.0}, "hold": 2})
    fires = []
    con = live.HybridConsumer(gate, "1", on_fire=fires.append, fresh_window_slots=5)
    # a pool first seen at the very first slot (during the warm-up) never fires, even on an N slot
    con.on_shred_slot("OLD", {"slot": 100, "price": 0.0, "n_buys": 1, "buy_sol": 1.0, "max_buy": 1.0,
                              "n_sells": 0, "sell_sol": 0.0, "buyers": ["NEW"], "net_sol": 1.0})
    assert fires == []
    # a pool first seen well after the warm-up, firing within the window, is allowed
    con2 = live.HybridConsumer(gate, "1", on_fire=fires.append, fresh_window_slots=5)
    for s in range(100, 110):                       # advance the global start past the warm-up
        con2.on_shred_slot("WARM", {"slot": s, "price": 0.0, "n_buys": 0, "buy_sol": 0.0, "max_buy": 0.0,
                                    "n_sells": 1, "sell_sol": 1.0, "buyers": [], "net_sol": -1.0})
    con2.on_shred_slot("FRESH", {"slot": 108, "price": 0.0, "n_buys": 1, "buy_sol": 1.0, "max_buy": 1.0,
                                 "n_sells": 0, "sell_sol": 0.0, "buyers": ["NEWBUYER"], "net_sol": 1.0})
    assert fires and fires[-1]["pool"] == "FRESH"
