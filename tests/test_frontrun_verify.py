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
