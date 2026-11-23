"""Hybrid feature bank: current slot scored from the DIRECT (shred) view, history rebased to FULL (Helius)."""
from tradingbot import flurry_study as fl


def _row(slot, payer, tokd, sold, tok_after, sol_after, vis):
    # launch_record row with the _VIS flag: [sec, slot, payer, tokDelta, solDelta, tok_after, sol_after, vis]
    return [0.0, slot, payer, tokd, sold, tok_after, sol_after, vis]


def _rec():
    # slot 10 direct buy A; slot 11 a NON-direct (aggregator) buy by A -> dead in the direct view, "B" in full;
    # slot 12 direct buy C. The price path (full) rises every slot.
    return {"txs": [
        _row(10, "A", -20, 2, 1000, 100, True),
        _row(11, "A", -20, 4, 980, 104, False),
        _row(12, "C", -10, 2, 970, 106, True),
    ]}


def test_hybrid_rebases_history_to_full_while_the_current_slot_stays_shred():
    tgt = fl._tgt("price", 3, 2, 1.0)
    rec = _rec()
    hyb = fl._hybrid_bank(rec, create_ts=0.0, tgt=tgt)
    direct = fl._slot_bank(fl.slot_flow(rec, "direct"), 0.0, tgt)   # a pure-shred bank, for contrast
    assert len(hyb) == 3
    h12, d12 = hyb[2], direct[2]
    # the prior slot (11) is dead in the pure-direct view but rebased to its FULL symbol "B" in the hybrid
    assert d12["m2"] == ".N"          # shred-only: prior slot looked dead
    assert h12["m2"] == "BN"          # rebased: prior slot's real (Helius) symbol
    assert h12["m3"] == "NBN"
    # lagged features come from the FULL prior slot (1 buy there), not the direct-dead 0
    assert h12["f"]["prev_buys"] == 1.0
    assert d12["f"]["prev_buys"] == 0.0
    assert h12["f"]["prev_abs_move"] > 0.0          # the rebased prior slot's move is available
    # the CURRENT slot stays shred: its own symbol from direct, and its own price move zeroed (no live reserves)
    assert h12["sym"] == "N"
    assert h12["f"]["abs_move"] == 0.0 and h12["f"]["move"] == 0.0


def test_hybrid_mode_switch_and_prev_abs_move_is_in_the_rule_bank():
    assert "prev_abs_move" in fl.NUMERIC_FIELDS and "prev_move" in fl.NUMERIC_FIELDS
    # _bank_for honours the module flag
    rec = _rec()
    tgt = fl._tgt("price", 3, 2, 1.0)
    flow = fl.slot_flow(rec, "full")
    saved = fl.HYBRID
    try:
        fl.HYBRID = False
        assert len(fl._bank_for(rec, flow, 0.0, tgt)) == len(fl._slot_bank(flow, 0.0, tgt))
        fl.HYBRID = True
        assert fl._bank_for(rec, flow, 0.0, tgt)[2]["m2"] == "BN"
    finally:
        fl.HYBRID = saved
