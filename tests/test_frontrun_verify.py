"""The feed verifier must (a) confirm a box fire that reproduces on ground truth, (b) flag one that does not, and
(c) catch a box record that disagrees field-by-field -- all without Helius, by injecting the ground-truth rec."""
from tradingbot import flurry_study as fl
from tradingbot import frontrun_live as live
from tradingbot import frontrun_verify as fv
from tradingbot import precursor_gate as pg
from tests.test_flurry_study import _cre_front

# a hand gate that fires on any new-wallet slot (sym == N), which _cre_front plants each episode
GATE = {"hold": 2, "intercept": 0.0, "threshold": {"1": 5.0},
        "predictors": [{"name": "sym == N  (new)", "coef": 10.0, "predicate": {"kind": "symbol", "symv": "N"}}]}


def _firing_slots(seed):
    """The slots the hand gate fires on, by running the parity-tested brain over the planted flow."""
    flow = fl.slot_flow(_cre_front(seed))
    bank = live.SlotBank(flurry_min=3)
    out = []
    for s in flow["slots"]:
        r = bank.push(s)
        if pg.fires(r, pg.compile_gate(dict(GATE)), "1"):
            out.append((s["slot"], r["sym"], r["m2"]))
    return out


def test_reproduced_fire_is_confirmed():
    seed = 0
    fires = _firing_slots(seed)
    assert fires, "the planted flow should fire the hand gate somewhere"
    slot, sym, m2 = fires[0]
    out = fv.verify([{"pool": "P", "slot": slot, "box": {"sym": sym, "m2": m2}}],
                    dict(GATE), have={"P": _cre_front(seed)})
    assert out["n_reproduced"] == 1 and out["n_diverged"] == 0
    row = out["rows"][0]
    assert row["in_flow"] and row["offline"]["fires"] and row["reproduced"]
    assert all(row["match"].values())                 # box sym/m2 agree with ground truth


def test_nonfiring_slot_is_flagged_as_diverged():
    # claim the box fired on a slot the gate does NOT fire on -> ground truth must say it does not reproduce
    flow = fl.slot_flow(_cre_front(0))
    firing = {s for s, _, _ in _firing_slots(0)}
    dead = next(s["slot"] for s in flow["slots"] if s["slot"] not in firing)
    out = fv.verify([{"pool": "P", "slot": dead, "box": {}}], dict(GATE), have={"P": _cre_front(0)})
    assert out["n_reproduced"] == 0 and out["n_diverged"] == 1
    assert out["rows"][0]["reproduced"] is False


def test_box_record_mismatch_is_caught():
    slot, sym, m2 = _firing_slots(0)[0]
    bad = "S" if sym != "S" else "N"                   # a box that recorded the wrong symbol
    out = fv.verify([{"pool": "P", "slot": slot, "box": {"sym": bad, "m2": m2}}],
                    dict(GATE), have={"P": _cre_front(0)})
    row = out["rows"][0]
    assert row["match"]["sym"] is False and row["match"]["m2"] is True


def _visrec():
    """A vis-tagged launch_record where slot 12 holds a DIRECT sell + an AGGREGATOR (box-blind) new-wallet buy.
    full view:   n_buys=1 (the new buyer) + n_sells=1 -> new buyer present -> sym 'N'
    direct view: n_buys=0              + n_sells=1 -> more sells        -> sym 'S'  (what the box actually sees)."""
    rows = [
        [0.0,  10, "w0", -1.0,  1.0, 1e9, 100.0, True],      # buy (warm; w0 first-seen)
        [0.4,  11, "w0", -1.0,  1.0, 1e9, 101.0, True],      # buy
        [0.8,  12, "op",  1.0, -1.0, 1e9, 100.5, True],      # DIRECT sell (box sees it)
        [0.85, 12, "newbie", -1.0, 1.0, 1e9, 101.0, False],  # AGGREGATOR new-wallet buy (box-blind)
        [1.2,  13, "w0", -1.0,  1.0, 1e9, 102.0, True],      # buy
        [1.6,  14, "w0", -1.0,  1.0, 1e9, 103.0, True],      # buy
    ]
    return {"create_ts": 0.0, "txs": rows}


# fires on any net-sell slot (sym == S). On _visrec, slot 12 is sym S in the DIRECT view but sym N in FULL.
GATE_S = {"hold": 2, "intercept": 0.0, "threshold": {"1": 5.0},
          "predictors": [{"name": "sym == S  (sellers)", "coef": 10.0, "predicate": {"kind": "symbol", "symv": "S"}}]}


def test_verify_replays_through_the_gate_visibility():
    rec = _visrec()
    g_direct = dict(GATE_S); g_direct["vis"] = "direct"
    g_full = dict(GATE_S); g_full["vis"] = "full"
    fire = [{"pool": "P", "slot": 12, "box": {}}]
    od = fv.verify(fire, g_direct, have={"P": rec})
    of = fv.verify(fire, g_full, have={"P": rec})
    # the box's own (direct) view: the aggregator buy is hidden, slot 12 is sym S, the gate reproduces the fire
    assert od["vis"] == "direct" and od["n_reproduced"] == 1 and od["rows"][0]["offline"]["sym"] == "S"
    # a gate replayed on the FULL view scores sym N on that slot -> it fires on the WRONG slot (the old divergence)
    assert of["vis"] == "full" and of["n_reproduced"] == 0 and of["rows"][0]["offline"]["sym"] == "N"


def test_hybrid_verify_scores_current_from_direct_and_differs_from_full():
    # HYBRID ground truth: the CURRENT slot from the DIRECT shred view, PRIOR slots rebased to FULL. On _visrec
    # slot 12 is sym S in direct (the box's view) but sym N in full, so a hybrid verify of a sym-S gate reproduces
    # the fire (like the box), while a pure-full gate scores N there and does NOT -- proving hybrid uses direct
    # for the current slot, not the full view.
    rec = _visrec()
    fire = [{"pool": "P", "slot": 12, "box": {"sym": "S"}}]
    oh = fv.verify(fire, dict(GATE_S), have={"P": rec}, hybrid=True)
    of = fv.verify(fire, {**GATE_S, "vis": "full"}, have={"P": rec})
    assert oh["hybrid"] is True
    assert oh["n_reproduced"] == 1 and oh["rows"][0]["offline"]["sym"] == "S"
    assert all(oh["rows"][0]["match"].values())                       # box sym S agrees with the hybrid ground truth
    assert of["n_reproduced"] == 0 and of["rows"][0]["offline"]["sym"] == "N"   # pure full fires on the wrong slot
    # the current-slot abs_move is zeroed in hybrid (shreds carry no price), by design
    assert oh["rows"][0]["offline"]["abs_move"] == 0.0


def test_hybrid_auto_detected_from_the_gate_flag():
    rec = _visrec()
    g = {**GATE_S, "hybrid": True}                                    # a hybrid-exported gate stamps its own flag
    out = fv.verify([{"pool": "P", "slot": 12, "box": {}}], g, have={"P": rec})
    assert out["hybrid"] is True and out["rows"][0]["offline"]["sym"] == "S"


def test_parse_fires_reads_jsonl_and_a_log_blob():
    jl = '{"pool": "AAA", "slot": 5, "sym": "N"}\n{"pool": "BBB", "slot": 9}'
    rows = fv.parse_fires(jl)
    assert [r["pool"] for r in rows] == ["AAA", "BBB"] and rows[0]["box"]["sym"] == "N"
    log = "foo\n  GATE FIRE CcDdEe slot 472079845 eta -3.9 B (NB)\nbar\n  GATE FIRE FfGg slot 100\n"
    rows = fv.parse_fires(log)
    assert [(r["pool"], r["slot"]) for r in rows] == [("CcDdEe", 472079845), ("FfGg", 100)]


def test_render_runs():
    out = fv.verify([{"pool": "P", "slot": _firing_slots(0)[0][0], "box": {}}], dict(GATE), have={"P": _cre_front(0)})
    assert "FRONT-RUN FEED VERIFY" in fv.render(out)


def test_truncated_fetch_is_classified_separately_from_a_divergence():
    flow = {"slots": [{"slot": 10}, {"slot": 20}, {"slot": 30}]}
    assert fv._is_truncated(40, flow, reached=False) is True    # capped before the slot -> truncated, not a feed bug
    assert fv._is_truncated(40, flow, reached=True) is False    # the fetch reached the end -> a real miss
    assert fv._is_truncated(25, flow, reached=False) is False   # within the fetched range -> a real dead-slot divergence
    assert fv._is_truncated(5, {"slots": []}, reached=False) is True


def test_injected_rec_is_never_truncated():
    # an injected rec (have=) is complete, so a slot the flow lacks is a real divergence, never "truncated"
    out = fv.verify([{"pool": "P", "slot": 999999999, "box": {}}], dict(GATE), have={"P": _cre_front(0)})
    assert out.get("n_truncated", 0) == 0
    assert out["rows"][0]["status"] == "ok" and out["rows"][0]["reproduced"] is False


# -- box-exact: replicate the live box's LUT drop in the ground truth, to tell a LUT gap from a deeper one --
from tradingbot.data import helius as he


def _lut_tx(pool, mint, lut_pool, slot=100, bt=0):
    """A minimal top-level pumpswap BUY tx (tokens out, SOL in) whose pool account is static or LUT-loaded."""
    bal = lambda tok, sol: [
        {"owner": pool, "mint": mint, "uiTokenAmount": {"uiAmount": tok}},
        {"owner": pool, "mint": he.SOL_MINT, "uiTokenAmount": {"uiAmount": sol}},
    ]
    return {
        "blockTime": bt, "slot": slot,
        "transaction": {"message": {"accountKeys": [
            {"pubkey": "PAYER", "source": "transaction"},
            {"pubkey": pool, "source": "lookupTable" if lut_pool else "transaction"},
        ]}},
        "meta": {"err": None,
                 "logMessages": ["Program %s invoke [1]" % he.PUMPSWAP_PROGRAM],
                 "preTokenBalances": bal(1000, 10), "postTokenBalances": bal(900, 11)},
    }


def test_box_exact_drops_lut_pooled_direct_swaps():
    pool, mint = "POOL", "MINT"
    # LUT-pooled top-level pumpswap swap: direct by the log bracket, but the box can't resolve its pool (drop_lut)
    lut = _lut_tx(pool, mint, lut_pool=True)
    assert he._top_level_pumpswap(lut) is True          # invoke [1] -> looks direct to the study
    assert he._lut_sourced(lut, pool) is True           # ...but the pool came from a lookup table
    kept = he.launch_record([lut], pool, mint, with_vis=True)["txs"]
    boxx = he.launch_record([lut], pool, mint, with_vis=True, vis_box_exact=True)["txs"]
    assert kept and kept[0][fl._VIS] is True             # default vis keeps it -> overstates the box's view
    assert boxx and boxx[0][fl._VIS] is False            # box-exact drops it -> matches the box's drop_lut

    # a STATIC-pooled swap is seen by the box, so both modes keep it
    stat = _lut_tx(pool, mint, lut_pool=False)
    assert he._lut_sourced(stat, pool) is False
    assert he.launch_record([stat], pool, mint, with_vis=True, vis_box_exact=True)["txs"][0][fl._VIS] is True
