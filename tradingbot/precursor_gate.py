"""Score a live slot against an exported front-run precursor gate (flurry_study --frontrun).

The gate is the joint logistic fit on the wallet-SETUP predictors only -- abs_move/move/price_up are
excluded by the precursor fit, so it fires BEFORE the move = a front-run, not a backrun. flurry_study.
export_gate writes a small JSON; this module rebuilds the predicates from the rule strings and returns
eta / fire for a _slot_bank-style record, so LIVE scoring == STUDY scoring (a parity test in
tests/test_precursor_gate.py pins that). Pure python, no numpy, so the box ShredStream runner can import it.

Live use (per slot N, the signal slot):
    g = precursor_gate.load("gate.json")
    rec = {"f": {... numeric features ...}, "sym": "N", "m2": ".N", "m3": "..N"}   # _slot_bank shape at N
    if precursor_gate.fires(rec, g, tier="1"):        # top 1% of the joint logit score
        ...  # land a buy at the FRONT of slot N+1, hold g["hold"] slots, sell at N+hold

The rec fields a predicate may read (all from flurry_study._slot_bank):
    rec["f"]  numeric dict: n_buys, buy_sol, max_buy, net_sol, n_sells, sell_sol, distinct_buyers,
              new_buyers, mean_buy, slots_since_flurry, age_slots, outside_buy, dominant_bought,
              new_present, sell_present, prev_buys, prev_buy_sol, prev_new_buyers, prev_net_sol
              (abs_move/move/price_up exist on the rec but never appear in a precursor gate)
    rec["sym"]  one of F S N O B b .
    rec["m2"]   two-char motif prev_sym + sym
    rec["m3"]   three-char motif prev2_sym + prev_sym + sym
"""
import json


def predicate_from_meta(m):
    """Return callable(rec)->bool from the EXACT structured predicate params (kind + field/thr/symv/mkey/mot)
    that flurry_study.export_gate carries. Preferred over the rule string, whose numeric threshold is rounded
    to 3 sig figs and so cannot reproduce the fit's exact cut."""
    kind = m.get("kind")
    if kind == "numeric":
        return lambda r, _f=m["field"], _t=float(m["thr"]): float(r["f"].get(_f, 0.0)) >= _t
    if kind == "bool":
        return lambda r, _f=m["field"]: float(r["f"].get(_f, 0.0)) >= 1.0
    if kind == "symbol":
        return lambda r, _x=m["symv"]: r.get("sym") == _x
    if kind == "motif":
        return lambda r, _k=m["mkey"], _mo=m["mot"]: r.get(_k) == _mo
    raise ValueError("unknown predicate kind: %r" % kind)


def predicate_from_name(name):
    """Return callable(rec)->bool for a flurry_study._build_specs rule string. Mirrors _build_specs exactly:
    'sym == X  (desc)', 'N-1..N = <m2>', 'N-2..N = <m3>', '<field> >= <thr> (qNN)', or a bare '<field>'
    boolean (f[field] >= 1.0)."""
    s = name.strip()
    if s.startswith("sym == "):
        x = s[len("sym == "):].split("  (")[0].strip()        # one symbol char; the "  (desc)" is dropped
        return lambda r, _x=x: r.get("sym") == _x
    if s.startswith("N-1..N = "):
        mot = s[len("N-1..N = "):].strip()
        return lambda r, _m=mot: r.get("m2") == _m
    if s.startswith("N-2..N = "):
        mot = s[len("N-2..N = "):].strip()
        return lambda r, _m=mot: r.get("m3") == _m
    if " >= " in s:
        field = s.split(" >= ", 1)[0].strip()
        thr = float(s.split(" >= ", 1)[1].split(" (")[0])
        return lambda r, _f=field, _t=thr: float(r["f"].get(_f, 0.0)) >= _t
    field = s                                                 # bare field name -> boolean
    return lambda r, _f=field: float(r["f"].get(_f, 0.0)) >= 1.0


def _predicate(p):
    """Prefer the exact structured predicate; fall back to parsing the rule string (lossy for numeric cuts)."""
    m = p.get("predicate")
    if m and m.get("kind"):
        return predicate_from_meta(m)
    return predicate_from_name(p["name"])


def compile_gate(gate):
    """Attach the predicate callables (gate['_terms']) once, in place. Returns the same dict."""
    gate["_terms"] = [(float(p["coef"]), _predicate(p)) for p in gate.get("predictors", [])]
    return gate


def load(path):
    """Load and compile a gate JSON written by flurry_study.export_gate."""
    with open(path) as fh:
        return compile_gate(json.load(fh))


def eta(rec, gate):
    """The joint logit score (log-odds) for one slot record: intercept + sum of coefficients whose predicate
    holds. Higher = more precursor-like. No exponential needed -- the gate compares eta to an eta threshold."""
    if "_terms" not in gate:
        compile_gate(gate)
    e = float(gate.get("intercept", 0.0))
    for coef, fn in gate["_terms"]:
        if fn(rec):
            e += coef
    return e


def fires(rec, gate, tier="1"):
    """True when the slot's eta clears the exported top-(tier)% threshold. tier '1' = top 1% (tightest, the
    paying cell), '2' = top 2%. Unknown tier -> no fire (fail closed)."""
    thr = (gate.get("threshold") or {}).get(str(tier))
    if thr is None:
        return False
    return eta(rec, gate) >= float(thr)
