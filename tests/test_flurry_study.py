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


def test_tail_report_breaks_down_the_discovered_edge(monkeypatch):
    # the tail breakdown should re-discover survivors and report concentration: distinct winning pools, the
    # top-5/20 share of the positive return, and capped vs uncapped mean. On the planted data the edge is
    # spread across many pools (a broad tail), not one or two.
    rec = _rec_causal()
    pools = [f"P{i}" for i in range(80)]
    have = {p: rec for p in pools}
    metas = [{"pool": p, "token_mint": "M", "create_ts": float(i)} for i, p in enumerate(pools)]
    tgt = fl._tgt("price", 3, 2, 1.0)
    out = fl.tail_report(have, metas, tgt, 0.01, hold=2)
    assert out["rules"], "no survivors re-discovered"
    r = out["rules"][0]
    assert r["n"] > 0 and r["win_pools"] >= 1
    assert 0.0 <= r["top5_share"] <= 1.0 and 0.0 <= r["top20_share"] <= 1.0
    assert r["pools"] >= r["win_pools"]
    assert "TAIL-CONCENTRATION" in fl.render_tail(out)


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


def _rec_combo(cycles=60, seed=0):
    """Planted CONJUNCTION: price pumps only when a NEW wallet buys in slot N (marker A) AND the prior slot was
    busy (prev_buys high, marker B). A fires on P1(hit)+P2(miss), B on P1(hit)+P3(miss) -- each weak alone; only
    A AND B (= P1) is a clean precursor, so discover must AND them to isolate the edge."""
    res, rows, sec, slot = [1e9, 100.0], [], 0.0, 10
    for c in range(cycles):
        pat = c % 3
        if pat == 0:                                               # P1: busy prev -> NEW wallet -> PUMP (A&B -> hit)
            sec = _add(res, rows, sec, slot, 3, 0.05, "bot")
            sec = _add(res, rows, sec, slot + 1, 1, 0.05, f"newA{c}")
            sec = _add(res, rows, sec, slot + 2, 6, 1.0, "bot")    # the pump -> price up next slot
            sec = _add(res, rows, sec, slot + 3, 1, 0.02, "bot"); sec = _add(res, rows, sec, slot + 4, 1, 0.02, "bot")
        elif pat == 1:                                             # P2: quiet prev -> NEW wallet -> flat (A only)
            sec = _add(res, rows, sec, slot, 1, 0.02, "bot")
            sec = _add(res, rows, sec, slot + 1, 1, 0.05, f"newB{c}")
            sec = _add(res, rows, sec, slot + 2, 1, 0.02, "bot")
            sec = _add(res, rows, sec, slot + 3, 1, 0.02, "bot"); sec = _add(res, rows, sec, slot + 4, 1, 0.02, "bot")
        else:                                                     # P3: busy prev -> existing wallet -> flat (B only)
            sec = _add(res, rows, sec, slot, 3, 0.05, "bot")
            sec = _add(res, rows, sec, slot + 1, 1, 0.05, "bot")
            sec = _add(res, rows, sec, slot + 2, 1, 0.02, "bot")
            sec = _add(res, rows, sec, slot + 3, 1, 0.02, "bot"); sec = _add(res, rows, sec, slot + 4, 1, 0.02, "bot")
        slot += 6
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

def _run_disc(monkeypatch, make_rec, n, combos=False):
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

    return fl.discover(J(), S(), helius=None, now=1000.0, combos=combos)


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


def test_discover_combos_search_runs_and_tightens_the_guard(monkeypatch):
    # with --combos the conjunction search runs (beam over AND-ed filters), the family-wise bar WIDENS for the
    # bigger search, and any reported combo is >1 filter AND-ed and sharper out of sample than its parts.
    out = _run_disc(monkeypatch, lambda i: _rec_causal(), n=8, combos=True)
    assert out["combo_tested"] > 0                              # conjunctions were actually searched
    assert out["combo_zc"] >= out["zc"]                         # stricter guard than the single-rule pass
    assert isinstance(out["combos"], list)
    for c in out["combos"]:
        assert "  AND  " in c["rule"] and c["depth"] >= 2       # a combo is a conjunction, not a single rule
        assert c["gain_vs_part"] > 0                            # it improved on its best single filter out of sample
    assert "COMBINATIONS" in fl.render_discover(out)


def test_discover_combos_find_nothing_on_a_memoryless_stream(monkeypatch):
    # the critical overfitting guard: a bigger (combinatorial) search must NOT manufacture a false edge on
    # memoryless data -- the widened family-wise z threshold keeps every conjunction out.
    out = _run_disc(monkeypatch, lambda i: _rec_random(slots=1500, seed=200 + i), n=6, combos=True)
    assert out["combo_tested"] >= 0
    assert out["n_combos"] == 0                                 # no conjunction survives the stricter bar
    r = fl.render_discover(out)
    assert "COMBINATIONS" not in r or "does not" in r


def test_discover_combos_surfaces_a_planted_conjunction(monkeypatch):
    # the feature's reason to exist: when A-and-B predicts but A or B alone is weak, the conjunction search
    # must recover the stacked rule and show it is SHARPER than either single filter.
    out = _run_disc(monkeypatch, lambda i: _rec_combo(seed=i), n=10, combos=True)
    assert out["n_combos"] >= 1
    top = out["combos"][0]
    assert top["depth"] >= 2 and "  AND  " in top["rule"]       # it is a conjunction
    assert top["gain_vs_part"] > 0.5                            # clearly sharper out of sample than its best single filter
    assert any("new" in c["rule"] and "prev" in c["rule"] for c in out["combos"])  # recovered the A AND B structure
    assert "STACKED EDGE" in fl.render_discover(out)


def test_metas_from_pools_loads_json_and_jsonl(tmp_path):
    rows = [{"pool": "P1", "token_mint": "M1", "create_ts": 10.0},
            {"pool": "P2", "token_mint": "M2", "create_ts": 20.0},
            {"pool": "P1", "token_mint": "M1", "create_ts": 10.0},   # dup pool -> dropped
            {"pool": "P3", "create_ts": 30.0}]                        # no mint -> dropped
    j = tmp_path / "pools.json"
    j.write_text(json.dumps(rows))
    metas = fl.metas_from_pools(str(j))
    assert [m["pool"] for m in metas] == ["P1", "P2"]
    assert metas[0] == {"pool": "P1", "token_mint": "M1", "create_ts": 10.0}

    jl = tmp_path / "pools.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")
    metas2 = fl.metas_from_pools(str(jl))
    assert [m["pool"] for m in metas2] == ["P1", "P2"]


def test_study_uses_a_passed_metas_universe_without_touching_the_scorer(monkeypatch):
    # route #1: metas override must bypass pool_tracker.scorer_rows entirely.
    def _boom(*a, **k):
        raise AssertionError("scorer_rows must not be called when metas is passed")
    monkeypatch.setattr(fl.pool_tracker, "scorer_rows", _boom)
    rec = _rec(continuation=True)
    have = {"P1": rec}
    monkeypatch.setattr(fl, "fetch_flows", lambda store, helius, metas, span_s, mc, prog=None: (have, 0))
    metas = [{"pool": "P1", "token_mint": "M1", "create_ts": 1.0}]
    out = fl.study(None, None, hours=1.0, metas=metas, max_credits=0)
    assert out["n_flows"] == 1


def test_flows_from_file_loads_live_recs_and_bypasses_fetch(tmp_path, monkeypatch):
    # route #1 (live): box surf.py writes launch_record-shaped recs; study must run on them without Helius.
    rec = _rec(continuation=True)
    rows = [{"pool": "P1", "token_mint": "M1", "create_ts": 100.0, "rec": rec},
            {"pool": "P2", "token_mint": "M2", "create_ts": 200.0, "rec": rec}]
    p = tmp_path / "surf.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    metas, have = fl.flows_from_file(str(p))
    assert [m["pool"] for m in metas] == ["P1", "P2"]
    assert metas[0]["create_ts"] == 100.0 and set(have) == {"P1", "P2"}

    def _boom(*a, **k):
        raise AssertionError("fetch_flows must not run when have is passed")
    monkeypatch.setattr(fl, "fetch_flows", _boom)
    monkeypatch.setattr(fl.pool_tracker, "scorer_rows", _boom)
    out = fl.study(None, None, hours=1.0, metas=metas, have=have, max_credits=0)
    assert out["n_flows"] == 2


def test_entry_window_finds_a_delayed_ev_positive_entry(monkeypatch):
    # a signal whose up-move lands a few entry-slots LATER -> entry_window surfaces a later-offset EV+ cell.
    import random as _r
    def rec(ct, seed):
        rr=_r.Random(seed); sol,tok=100_000_000_000,1_000_000_000_000; txs=[]
        for slot in range(1,40):
            is_sell=(slot%6==0)
            buyers=[] if is_sell else ([rr.choice(["A","B","C"])] if rr.random()<0.6 else [])
            if slot>3 and ((slot-3)%6==0): sol=int(sol*1.04)
            if is_sell: txs.append([0,slot,None,1.0,-0.6,tok,sol])
            for w in buyers: txs.append([0,slot,w,-1.0,round(rr.uniform(.2,1.0),3),tok,sol])
            if not buyers and not is_sell: txs.append([0,slot,None,1.0,-1e-9,tok,sol])
        return {"create_ts":ct,"token_mint":f"M{seed}","txs":txs,"window_s":0,"complete":True,"fetched":len(txs)}
    metas=[{"pool":f"P{i}","token_mint":f"M{i}","create_ts":1000+i*10} for i in range(80)]
    have={f"P{i}":rec(1000+i*10,i) for i in range(80)}
    out=fl.entry_window(have,metas,fl._tgt("price",3,2,1.0),cost=0.01,window=10)
    assert out["n_hold"] >= 2 and out["split"] == "time"
    am=next(s for s in out["signals"] if s["signal"].startswith("abs_move"))
    # a later offset (k>=3) is EV+ (the move lands a few slots after the signal), and best_k found one
    assert am["best_k"] is not None
    assert any(am["per_k"][k].get("n",0)>=20 and am["per_k"][k]["mean"]>0 for k in range(3,8))
