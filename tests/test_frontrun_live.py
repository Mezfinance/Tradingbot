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


def test_swap_return_is_the_price_move_with_no_cost():
    # swap % = the move a trade captured, fee-free -- so a $0.01 probe still shows what it WOULD have made at scale
    assert abs(live.swap_return(1.0, 1.10) - 0.10) < 1e-12
    assert abs(live.swap_return(2.0, 1.0) + 0.50) < 1e-12         # a -50% move
    assert live.swap_return(0.0, 1.0) == 0.0                      # no entry fill -> no return
    # net is exactly swap minus the round-trip cost
    for ef, xf, c in ((1.0, 1.10, 0.01), (1.0, 0.80, 0.06), (3.0, 4.5, 0.0)):
        assert abs(live.net_return(ef, xf, c) - (live.swap_return(ef, xf) - c)) < 1e-12


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


# --- WS-only consumer: one feed (Helius processed WS), per-pool routing + fresh gate --------------

def _fire_gate():                        # a gate that fires on sym == N (new wallet), as _cre_front produces
    g = {"hold": 3, "intercept": 0.0, "threshold": {"0.5": 5.0},
         "predictors": [{"name": "sym == N  (new)", "coef": 10.0,
                         "predicate": {"kind": "symbol", "symv": "N"}}]}
    pg.compile_gate(g)
    return g


def _always_gate():                      # eta = intercept (no predictors) >= 0 -> fires on every slot
    g = {"hold": 3, "intercept": 100.0, "threshold": {"0.5": 0.0}, "predictors": []}
    pg.compile_gate(g)
    return g


def _agg(slot):                          # a minimal complete per-slot aggregate (FullSlotAggregator shape)
    return {"slot": slot, "price": 1.0, "n_buys": 1, "buy_sol": 0.1, "max_buy": 0.1,
            "n_sells": 0, "sell_sol": 0.0, "buyers": ["w%d" % slot], "net_sol": 0.1}


def test_ws_only_consumer_matches_a_bare_runner_per_pool_and_stamps_the_pool():
    # fresh gate off: WsOnlyConsumer.on_slot(pool, s) must forward exactly the fires a bare Runner gives on the
    # same single feed, with the pool stamped and on_fire/on_pool called.
    gate = _fire_gate()
    flow = fl.slot_flow(_cre_front(0))
    runner = live.Runner(gate, tier="0.5", flurry_min=3)
    want = [runner.on_slot(s) for s in flow["slots"]]
    fires, opened = [], []
    con = live.WsOnlyConsumer(gate, tier="0.5", flurry_min=3,
                              on_fire=fires.append, on_pool=opened.append, fresh_window_slots=0)
    got = [con.on_slot("POOLX", s) for s in flow["slots"]]
    assert opened == ["POOLX"]                               # subscription opened once, on first sight
    for g, w in zip(got, want):
        assert (g is not None) == (w is not None)
        if g is not None:
            assert g["signal_slot"] == w["signal_slot"] and g["entry_slot"] == w["entry_slot"]
            assert g["exit_slot"] == w["exit_slot"] and g["hold"] == 3 and g["pool"] == "POOLX"
    assert [s["pool"] for s in fires] == ["POOLX"] * len(fires)
    assert len(fires) == sum(1 for g in got if g is not None) > 0


def test_ws_only_consumer_routes_pools_independently():
    gate = _always_gate()
    opened = []
    con = live.WsOnlyConsumer(gate, tier="0.5", on_pool=opened.append, fresh_window_slots=0)
    con.on_slot("A", _agg(0)); con.on_slot("B", _agg(0))
    assert set(opened) == {"A", "B"} and set(con.runners) == {"A", "B"}
    con.drop("A")
    assert set(con.runners) == {"B"}                         # aged-out pool dropped


def test_ws_only_consumer_fresh_window_gates_mid_life_and_late_fires():
    # new-pools-only: a pool first seen during the warm-up (first - start < w) is mid-life -> never forwarded,
    # even though the always-gate fires every slot; a pool first seen after warm-up fires only within w of its
    # first sight.
    gate = _always_gate()
    fires = []
    con = live.WsOnlyConsumer(gate, tier="0.5", on_fire=fires.append, fresh_window_slots=5)
    assert con.on_slot("MID", _agg(0)) is None               # start_slot=0, first[MID]=0 -> (0-0)<5 -> mid-life
    assert con.on_slot("MID", _agg(3)) is None               # still inside warm-up, suppressed
    assert con.on_slot("FRESH", _agg(10)) is not None        # (10-0)>=5 fresh, (10-10)=0<=5 -> fires
    assert con.on_slot("FRESH", _agg(15)) is not None        # (15-10)=5<=5 -> fires
    assert con.on_slot("FRESH", _agg(16)) is None            # (16-10)=6>5 -> aged past the window
    assert [s["pool"] for s in fires] == ["FRESH", "FRESH"]
