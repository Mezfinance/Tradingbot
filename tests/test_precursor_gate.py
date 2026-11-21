"""The live precursor-gate scorer must reproduce the study's rule evaluation and joint-logit score exactly,
or a gate fitted on Fly fires on different slots live than it did in the backtest. These pin that."""
import json

from tradingbot import flurry_study as fl
from tradingbot import precursor_gate as pg


def _recs():
    """Build _slot_bank records the way flurry_study.frontrun does, from planted precursor flows."""
    tgt = fl._tgt("price", fl.DEFAULT_FLURRY_MIN, 1, 30.0)
    recs = []
    for i in range(6):
        flow = fl.slot_flow(_cre_front(i))
        for r in fl._slot_bank(flow, float(i), tgt):
            recs.append(r)
    return recs


# reuse the planted-flow generator from the front-run study test (new wallet buys the slot before a spike)
from tests.test_flurry_study import _cre_front  # noqa: E402


def test_structured_predicate_matches_build_specs_for_every_rule_family():
    recs = _recs()
    specs = fl._build_specs(recs)
    assert specs, "the planted flows should yield a candidate bank"
    kinds = {s["kind"] for s in specs}
    assert {"symbol", "motif", "numeric", "bool"} <= kinds       # every family is exercised
    for spec in specs:
        meta = {k: spec.get(k) for k in ("kind", "field", "thr", "symv", "mkey", "mot")}
        pred = pg.predicate_from_meta(meta)                       # the exact path export_gate uses
        for r in recs:
            assert bool(pred(r)) == bool(spec["fn"](r)), spec["rule"]


def test_rule_string_parsing_is_exact_for_non_numeric_families():
    # the rule string rounds a numeric threshold to 3 sig figs (lossy), so the string fallback is only
    # guaranteed for symbol/motif/bool; those must still round-trip exactly.
    recs = _recs()
    for spec in fl._build_specs(recs):
        if spec["kind"] == "numeric":
            continue
        pred = pg.predicate_from_name(spec["rule"])
        for r in recs:
            assert bool(pred(r)) == bool(spec["fn"](r)), spec["rule"]


def test_eta_matches_the_inline_joint_logit_score():
    recs = _recs()
    specs = {s["rule"]: s["fn"] for s in fl._build_specs(recs)}
    # a hand-built gate over three real rules with arbitrary coefficients + intercept
    names = [n for n in ("sym == N  (a new wallet bought in N)", "sym == .  (a dead slot)") if n in specs]
    # fall back to whatever symbol rules exist so the test is robust to SYM_DESC wording
    if len(names) < 2:
        names = [r for r in specs if r.startswith("sym == ")][:2]
    coefs = [2.5, -1.3]
    gate = {"intercept": -4.0, "predictors": [{"name": n, "coef": c} for n, c in zip(names, coefs)]}
    terms = [(c, specs[n]) for n, c in zip(names, coefs)]
    for r in recs:
        want = -4.0 + sum(c for c, fn in terms if fn(r))
        assert abs(pg.eta(r, gate) - want) < 1e-12


def test_fires_uses_the_exported_threshold_and_fails_closed():
    gate = {"intercept": 0.0, "threshold": {"1": 5.0, "2": 2.0},
            "predictors": [{"name": "sym == N  (new)", "coef": 10.0}]}
    pg.compile_gate(gate)
    n_rec = {"f": {}, "sym": "N", "m2": ".N", "m3": "..N"}   # eta = 10 -> clears both tiers
    dot_rec = {"f": {}, "sym": ".", "m2": "..", "m3": "..."}  # eta = 0 -> clears neither
    assert pg.fires(n_rec, gate, "1") and pg.fires(n_rec, gate, "2")
    assert not pg.fires(dot_rec, gate, "1") and not pg.fires(dot_rec, gate, "2")
    assert not pg.fires(n_rec, gate, "99")                   # unknown tier -> fail closed


def test_export_gate_round_trips_through_the_scorer(tmp_path):
    saved = fl.USE_VQ
    try:
        fl.USE_VQ = False
        metas = [{"pool": "P%d" % i, "token_mint": "M%d" % i, "create_ts": float(i)} for i in range(6)]
        have = {"P%d" % i: _cre_front(i) for i in range(6)}
        out = fl.frontrun(have, metas, move=0.30, cost=0.01)
        spec = fl.export_gate(out, hold=2)
        assert spec is not None and spec["kind"] == "frontrun_precursor"
        assert spec["predictors"] and "intercept" in spec
        assert "1" in spec["threshold"] and "2" in spec["threshold"]
        assert spec["hold"] == 2 and spec["move"] == 0.30
        assert not any(fl._is_move_rule(p["name"]) for p in spec["predictors"])  # no backrun contaminant
        path = tmp_path / "gate.json"
        path.write_text(json.dumps(spec))
        g = pg.load(str(path))
        # a new-wallet precursor slot scores at least as high as a dead filler slot
        n_rec = {"f": {k: 0.0 for k in ("new_buyers", "sell_sol", "n_sells")}, "sym": "N", "m2": ".N", "m3": "..N"}
        dot_rec = {"f": {k: 0.0 for k in ("new_buyers", "sell_sol", "n_sells")}, "sym": ".", "m2": "..", "m3": "..."}
        assert pg.eta(n_rec, g) >= pg.eta(dot_rec, g)
        assert isinstance(pg.fires(n_rec, g, "1"), bool)
    finally:
        fl.USE_VQ = saved
