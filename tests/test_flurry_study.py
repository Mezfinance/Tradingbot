import json

from tradingbot import flurry_study as fl


def _buy(res, dsol):
    """One constant-product buy of dsol SOL: SOL enters, tokens leave. Returns (tokΔ<0, solΔ>0)."""
    tok, sol = res
    k = tok * sol
    new_sol = sol + dsol
    new_tok = k / new_sol
    res[0], res[1] = new_tok, new_sol
    return new_tok - tok, dsol


def _rec(continuation=True, pull=False):
    """A synthetic launch_record: a quiet warm-up, a 5-buy flurry at slot 100, then either a 5-buy continuation
    at slot 101 (edge) or flat (no edge); optionally an LP pull at slot 101."""
    res = [1e9, 100.0]
    rows, sec = [], 0.0

    def add(slot, n, dsol):
        nonlocal sec
        for _ in range(n):
            dtok, dsolv = _buy(res, dsol)
            rows.append([sec, slot, "w", round(dtok, 6), round(dsolv, 9), round(res[0], 6), round(res[1], 9)])
            sec += 0.05

    add(90, 1, 0.1); add(91, 1, 0.1); add(92, 1, 0.1)   # quiet
    add(100, 5, 2.0)                                     # the flurry slot
    if pull:
        tok, sol = res
        res[0], res[1] = tok * 0.05, sol * 0.05          # both vaults collapse = withdrawal
        rows.append([sec, 101, "op", round(res[0] - tok, 6), round(res[1] - sol, 9),
                     round(res[0], 6), round(res[1], 9)])
    elif continuation:
        add(101, 5, 2.0)                                 # the wash bot keeps buying -> price up next slot
    else:
        add(101, 1, 0.1)                                 # flurry burst-and-stops -> flat
    add(102, 1, 0.1); add(103, 1, 0.1); add(104, 1, 0.1); add(105, 1, 0.1)
    return {"txs": rows}


def test_classify_reads_the_two_vault_signs():
    assert fl.classify(-1, 1) == "buy"     # tokens out, SOL in
    assert fl.classify(1, -1) == "sell"
    assert fl.classify(-1, -1) == "pull"   # both out = withdrawal
    assert fl.classify(1, 1) == "add"
    assert fl.classify(0, 0) == "other"


def test_slot_flow_counts_buys_and_carries_price():
    flow = fl.slot_flow(_rec())
    slots = {s["slot"]: s for s in flow["slots"]}
    assert slots[100]["n_buys"] == 5 and slots[100]["buy_sol"] > 9      # 5 x ~2 SOL
    assert slots[101]["n_buys"] == 5
    assert slots[101]["price"] > slots[100]["price"] > 0               # price rose over the continuation
    assert flow["pull_slot"] is None


def test_slot_flow_flags_the_pull_slot():
    flow = fl.slot_flow(_rec(pull=True))
    assert flow["pull_slot"] == 101


def test_scalp_is_positive_on_continuation_flat_on_quiet_and_minus_100_on_a_pull():
    flow = fl.slot_flow(_rec())
    slots = flow["slots"]
    i100 = next(i for i, s in enumerate(slots) if s["slot"] == 100)
    i90 = next(i for i, s in enumerate(slots) if s["slot"] == 90)
    assert fl._scalp(slots, None, i100, 1, 0.01) > 0        # flurry + continuation beats 1% cost
    assert fl._scalp(slots, None, i90, 1, 0.01) < 0         # quiet slot -> flat minus cost
    pflow = fl.slot_flow(_rec(pull=True))
    pslots = pflow["slots"]
    pi = next(i for i, s in enumerate(pslots) if s["slot"] == 100)
    assert fl._scalp(pslots, pflow["pull_slot"], pi, 1, 0.01) == -1.0   # pull inside the hold = trapped


def _run(monkeypatch, rec, n=80):
    from tradingbot import pool_tracker as pt
    pools = {f"P{i}": {"create_ts": 1.0, "token_mint": "M"} for i in range(n)}
    monkeypatch.setattr(pt, "scorer_rows", lambda q, since: pools)
    monkeypatch.setattr(pt, "feint_pools", lambda q, since: {})

    class S:
        def execute(self, *a):
            pass

        def query(self, sql, params=()):
            if "watch_flurry" in sql:
                return [{"pool": p, "data": json.dumps(rec)} for p in pools]
            return []

    class J:
        def query(self, *a, **k):
            return []

    return fl.study(J(), S(), helius=None, now=10.0)


def test_study_finds_a_planted_slot_scale_flurry_edge(monkeypatch):
    out = _run(monkeypatch, _rec(continuation=True))
    assert out["best"] is not None
    assert out["best"]["mean"] > out["best"]["control"]
    assert "BEST CELL" in fl.render(out)


def test_study_reports_no_edge_when_flurries_burst_and_stop(monkeypatch):
    out = _run(monkeypatch, _rec(continuation=False))
    assert out["best"] is None
    assert "NO CELL" in fl.render(out)


def test_motif_entry_scalp_pays_on_the_discovered_lead(monkeypatch):
    # _rec_causal: a new wallet buys in slot k, the bot wash-pumps k+1 -> entering on the quiet->new-wallet
    # lead (buy front of k+1) and selling at the flurry should beat the random-slot control.
    out = _run(monkeypatch, _rec_causal())
    assert out["motif_best"] is not None
    assert "N" in out["motif_best"]["motif"]                 # the discovered quiet->new-wallet precursor
    assert out["motif_best"]["mean"] > out["motif_best"]["control"]
    r = fl.render(out)
    assert "MOTIF-ENTRY SCALP" in r and "BEST LEAD:" in r


def test_discovered_rule_scalp_auto_feeds_the_survivors_and_scalps_them(monkeypatch):
    # the scalp should use the ACTUAL discovered rules (not a hardcoded list): run discover's search, then
    # scalp each holdout-confirmed survivor out of sample. On the planted causal data a survivor should pay.
    out = _run(monkeypatch, _rec_causal())
    d = out["discovered"]
    assert d["n_survivors"] >= 1                              # discover's own rules were carried in
    assert d["best"] is not None                              # at least one predicts the move AND pays on price
    assert d["best"]["mean"] > d["best"]["control"]
    assert any(r.get("target_hit", 0) > 0 for r in d["results"])   # the rules fire on real target slots
    assert "DISCOVERED-RULE SCALP" in fl.render(out)


# --- characterization (lead-lag: what in slot N predicts a flurry in N+1) ---------------------------

def _add(res, rows, sec0, slot, n, dsol, wallet):
    sec = sec0
    for _ in range(n):
        dtok, dsolv = _buy(res, dsol)
        rows.append([sec, slot, wallet, round(dtok, 6), round(dsolv, 9), round(res[0], 6), round(res[1], 9)])
        sec += 0.05
    return sec


def _rec_causal(cycles=40):
    """A new outsider wallet buys in slot k, then the bot pumps k+1 with real SOL (a clear PRICE up-move),
    then a long quiet stretch. So a first-ever buyer in slot N reliably precedes both a flurry AND a price
    move in N+1 -- the price target and the flurry target both get a strong, low-base-rate precursor."""
    res, rows, sec, slot = [1e9, 100.0], [], 0.0, 10
    for c in range(cycles):
        sec = _add(res, rows, sec, slot, 1, 0.2, f"out{c}")     # a brand-new wallet arrives (the precursor)
        sec = _add(res, rows, sec, slot + 1, 5, 3.0, "bot")     # the bot pumps the next slot -> price up >1%
        for q in range(2, 6):                                   # a long quiet stretch -> no move, keeps the base rate low
            sec = _add(res, rows, sec, slot + q, 1, 0.02, "bot")
        slot += 6
    return {"txs": rows}


def _rec_random(slots=1500, seed=1):
    """Buys AND sells drawn independently per slot -> price is a random walk, so nothing in N predicts the next
    slot's move. The proper null for the PRICE target (a buys-only stream trends up, which is trivially
    'predictable' and would plant a false signal)."""
    import random as _r
    rng = _r.Random(seed)
    res, rows, sec = [1e9, 100.0], [], 0.0
    for s in range(10, 10 + slots):
        for _ in range(rng.randint(0, 6)):
            tok, sol = res
            if rng.random() < 0.5:                          # sell: tokens in, SOL out (price down)
                new_sol = max(sol - 0.1, 1.0)
                new_tok = (tok * sol) / new_sol
                dtok, dsolv = new_tok - tok, new_sol - sol
                res[0], res[1] = new_tok, new_sol
            else:                                           # buy: tokens out, SOL in (price up)
                dtok, dsolv = _buy(res, 0.1)
            rows.append([sec, s, f"w{rng.randint(0, 50)}", round(dtok, 6), round(dsolv, 9),
                         round(res[0], 6), round(res[1], 9)])
            sec += 0.05
    return {"txs": rows}


def _run_char(monkeypatch, rec, n=40):
    from tradingbot import pool_tracker as pt
    pools = {f"P{i}": {"create_ts": 1.0, "token_mint": "M"} for i in range(n)}
    monkeypatch.setattr(pt, "scorer_rows", lambda q, since: pools)
    monkeypatch.setattr(pt, "feint_pools", lambda q, since: {})

    class S:
        def execute(self, *a):
            pass

        def query(self, sql, params=()):
            return [{"pool": p, "data": json.dumps(rec)} for p in pools] if "watch_flurry" in sql else []

    class J:
        def query(self, *a, **k):
            return []

    return fl.characterize(J(), S(), helius=None, now=10.0)


def test_characterize_finds_the_event_that_precedes_a_next_slot_flurry(monkeypatch):
    out = _run_char(monkeypatch, _rec_causal())
    assert out["base_rate"] > 0
    ev = {r["event"]: r for r in out["events"]}
    nb = ev["new_buyers"]                                       # the planted positive precursor
    assert nb["corr"] > 0.2 and nb["auc"] > 0.6 and nb["lift"] > 1.0
    # a flurry follows when the dominant bot is NOT the one buying this slot -> strong negative corr (also valid)
    assert ev["dominant_bought"]["corr"] < -0.2
    assert "SIGNAL:" in fl.render_char(out)


def test_characterize_reports_no_signal_on_a_memoryless_stream(monkeypatch):
    out = _run_char(monkeypatch, _rec_random(), n=6)
    assert out["base_rate"] > 0                                 # flurries do occur
    assert all(abs(r["corr"]) < 0.1 for r in out["events"])     # but nothing in N predicts N+1
    assert "NO SIGNAL" in fl.render_char(out)


# --- discovery (auto rule bank + motifs, holdout-confirmed; no event named in advance) --------------

def _run_disc(monkeypatch, make_rec, n):
    """Each pool gets its own record (distinct create_ts), so the time-holdout pools are independent draws."""
    from tradingbot import pool_tracker as pt
    pools = {f"P{i}": {"create_ts": float(i + 1), "token_mint": "M"} for i in range(n)}
    recs = {f"P{i}": make_rec(i) for i in range(n)}
    monkeypatch.setattr(pt, "scorer_rows", lambda q, since: pools)
    monkeypatch.setattr(pt, "feint_pools", lambda q, since: {})

    class S:
        def execute(self, *a):
            pass

        def query(self, sql, params=()):
            return [{"pool": p, "data": json.dumps(recs[p])} for p in pools] if "watch_flurry" in sql else []

    class J:
        def query(self, *a, **k):
            return []

    return fl.discover(J(), S(), helius=None, now=1000.0)


def test_discover_names_the_precursor_and_it_survives_the_holdout(monkeypatch):
    out = _run_disc(monkeypatch, lambda i: _rec_causal(), n=8)
    assert out["tested"] > 0 and out["split"] == "time"
    assert out["n_survivors"] >= 1
    rules = {s["rule"] for s in out["survivors"]}
    # the data should name the new-wallet precursor without it being specified: a symbol, a boolean or a motif
    assert any("sym == N" in r or "new_present" in r or "new_buyers" in r or "N-1..N" in r for r in rules)
    assert all(s["lift_hold"] > 1 for s in out["survivors"] if s["z_hold"] > 0)
    assert "DISCOVERED" in fl.render_discover(out)


def test_discover_reports_nothing_survives_on_a_memoryless_stream(monkeypatch):
    out = _run_disc(monkeypatch, lambda i: _rec_random(slots=1500, seed=100 + i), n=6)
    assert out["tested"] > 0                                    # rules were generated and holdout-tested
    assert out["n_survivors"] == 0                              # none leads a next-slot flurry out of sample
    assert "NOTHING SURVIVES" in fl.render_discover(out)
