"""The live runner must rebuild the study's per-slot record ONLINE exactly, or a gate fitted on Fly fires on
different slots live than in the backtest. SlotBank parity with flurry_study._slot_bank is the guarantee."""
from tradingbot import flurry_study as fl
from tradingbot import frontrun_live as live
from tradingbot import precursor_gate as pg
from tests.test_flurry_study import _cre_front


def _study_recs(flow, flurry_min=3):
    saved = fl.CAUSAL_WALLET
    try:
        fl.CAUSAL_WALLET = True                              # the runner is always causal
        return fl._slot_bank(flow, 0.0, fl._tgt("price", flurry_min, 1, 30.0))
    finally:
        fl.CAUSAL_WALLET = saved


def test_slotbank_push_matches_slot_bank_exactly():
    for seed in range(6):
        flow = fl.slot_flow(_cre_front(seed))
        want = _study_recs(flow)
        bank = live.SlotBank(flurry_min=3)
        got = [bank.push(s) for s in flow["slots"]]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g["sym"] == w["sym"] and g["m2"] == w["m2"] and g["m3"] == w["m3"]
            assert set(g["f"]) == set(w["f"])
            for k in w["f"]:
                assert abs(g["f"][k] - w["f"][k]) < 1e-9, k


def test_runner_fires_when_precursor_gate_fires_and_names_entry_exit_slots():
    # a gate that fires on a new-wallet slot (sym == N), which _cre_front produces each episode
    gate = {"hold": 2, "intercept": 0.0, "threshold": {"1": 5.0},
            "predictors": [{"name": "sym == N  (new)", "coef": 10.0,
                            "predicate": {"kind": "symbol", "symv": "N"}}]}
    pg.compile_gate(gate)
    flow = fl.slot_flow(_cre_front(0))
    runner = live.Runner(gate, tier="1", flurry_min=3)
    bank2 = live.SlotBank(flurry_min=3)                       # mirror to score each rec independently
    fired = 0
    for s in flow["slots"]:
        rec = bank2.push(s)
        want_fire = pg.fires(rec, gate, "1")
        sig = runner.on_slot(s)
        assert (sig is not None) == want_fire                # runner fires iff the gate fires
        if sig:
            fired += 1
            assert sig["entry_slot"] == s["slot"] + 1        # enter front of N+1
            assert sig["exit_slot"] == s["slot"] + 2         # exit at N+hold (hold 2)
            assert sig["hold"] == 2 and sig["sym"] == "N"
    assert fired > 0, "new-wallet slots should fire this hand gate"


def test_net_return_is_fills_minus_cost():
    assert abs(live.net_return(1.0, 1.10, 0.01) - 0.09) < 1e-12   # +10% gross, -1% cost
    assert live.net_return(0.0, 1.0) == 0.0                       # no entry fill -> no return


def test_runner_reproduces_the_exported_gate_decision():
    # end to end: fit + export on planted flows, load the gate, and the runner's live decision must equal
    # scoring the study's own causal record with precursor_gate.
    saved = fl.USE_VQ
    try:
        fl.USE_VQ = False
        metas = [{"pool": "P%d" % i, "token_mint": "M%d" % i, "create_ts": float(i)} for i in range(6)]
        have = {"P%d" % i: _cre_front(i) for i in range(6)}
        out = fl.frontrun(have, metas, move=0.30, cost=0.01)
        spec = fl.export_gate(out, hold=2)
        assert spec is not None
        gate = pg.compile_gate(spec)
        flow = fl.slot_flow(_cre_front(0))
        want = _study_recs(flow)
        runner = live.Runner(gate, tier="2", flurry_min=3)
        for s, w in zip(flow["slots"], want):
            sig = runner.on_slot(s)
            assert (sig is not None) == pg.fires(w, gate, "2")
    finally:
        fl.USE_VQ = saved
