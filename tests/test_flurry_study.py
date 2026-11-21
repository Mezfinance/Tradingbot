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


def test_discovered_combo_scalp_scalps_the_best_conjunctions(monkeypatch):
    # the scalp must also run on the best CONJUNCTIONS (not just singles): rebuild each discovered combo's
    # predicate by AND-ing its component rules, scalp it out of sample. On the planted A AND B price pump,
    # the stacked rule fires and PAYS beyond the control.
    from tradingbot import pool_tracker as pt
    n = 60
    pools = {f"P{i}": {"create_ts": float(i), "token_mint": "M"} for i in range(n)}
    recs = {f"P{i}": _rec_combo(seed=i) for i in range(n)}
    monkeypatch.setattr(pt, "scorer_rows", lambda q, since: pools)
    monkeypatch.setattr(pt, "feint_pools", lambda q, since: {})

    class S:
        def execute(self, *a):
            pass

        def query(self, sql, params=()):
            if "watch_flurry" in sql:
                return [{"pool": p, "data": json.dumps(recs[p])} for p in pools]
            return []

    class J:
        def query(self, *a, **k):
            return []

    out = fl.study(J(), S(), helius=None, now=10.0)
    d = out["discovered"]
    assert d["combo_results"], "no conjunction was scalped"
    assert any("  AND  " in r["rule"] for r in d["combo_results"])          # they are real conjunctions
    assert any("new" in r["rule"] and "prev" in r["rule"] for r in d["combo_results"])  # the planted A AND B
    cb = d["combo_best"]
    assert cb is not None and cb["mean"] > cb["control"]                    # the stacked lead pays out of sample
    assert "DISCOVERED-COMBO SCALP" in fl.render(out)


def test_abs_move_threshold_sweep_steps_the_entry_cut_up(monkeypatch):
    # sweep the abs_move entry threshold from the 0.0188 base upward, to read whether a BIGGER move nets a
    # higher mean. A higher threshold must fire on a SUBSET (monotonically fewer slots), the quantile column is
    # a share, and the table + best/none verdict must render.
    out = _run(monkeypatch, _rec_causal())
    sw = out["abs_sweep"]
    assert [r["thresh"] for r in sw["rows"]] == list(fl.ABSMOVE_GRID)   # the whole grid, base first
    fires = [r["fires"] for r in sw["rows"]]
    assert fires == sorted(fires, reverse=True)                        # a higher cut can only fire on fewer slots
    assert all(0.0 <= r["q"] <= 100.0 for r in sw["rows"])             # each row's quantile is a share
    # non-overlapping bands: every slot lands in exactly one, so the band fires sum to the base-threshold fires
    bands = sw["bands"]
    assert [(b["lo"], b["hi"]) for b in bands] == list(fl.ABSMOVE_BANDS)
    assert sum(b["fires"] for b in bands) == sw["rows"][0]["fires"]     # base >=0.0188 = union of the bands
    r = fl.render(out)
    assert "ABS_MOVE THRESHOLD SWEEP" in r and "(base q90)" in r
    assert "ABS_MOVE BANDS" in r
    assert ("BEST THRESHOLD" in r) or ("NO THRESHOLD beats" in r)
    assert ("BEST BAND" in r) or ("NO BAND beats" in r)


def test_moveval_isolates_the_chosen_side():
    down = {"f": {"move": -0.05, "abs_move": 0.05}}
    up = {"f": {"move": 0.05, "abs_move": 0.05}}
    assert fl._moveval(down, "down") == 0.05 and fl._moveval(up, "down") == 0.0     # down side sees only down-moves
    assert fl._moveval(up, "up") == 0.05 and fl._moveval(down, "up") == 0.0         # up side sees only up-moves
    assert fl._moveval(down, "abs") == 0.05 and fl._moveval(up, "abs") == 0.05      # abs sees both


def test_sell_side_down_move_sweep_captures_the_reversion(monkeypatch):
    # the sell-side (down-move) sweep fires AFTER a down-spike and scalps the bounce. On a planted
    # down-then-bounce, the down side's base-threshold hold-2 scalp must be positive and beat its control,
    # which the absolute sweep (mixing in the up-spike that reverts flat) need not be.
    out = _run(monkeypatch, _rec_reversion())
    dsw = out["down_sweep"]
    assert dsw["side"] == "down"
    base = dsw["rows"][0]
    assert base["fires"] > 0                                          # down-moves fire the sell side
    assert base["h2"]["n"] >= 50                                      # enough to judge
    assert base["h2"]["mean"] > dsw["control"]["h2"]["mean"]          # the reversion beats the quiet-slot control
    assert base["h2"]["mean"] > 0                                     # buy the trough, sell the bounce -> positive
    r = fl.render(out)
    assert "DOWN_MOVE THRESHOLD SWEEP" in r and "SELL SIDE" in r
    assert "DOWN_MOVE BANDS" in r


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


def _rec_reversion(cycles=40):
    """A big SELL drops the price in slot k (a down-spike), then the bot buys it back in k+1 (the bounce), then a
    quiet stretch. So a DOWN move in slot N reliably precedes an UP move in N+1 -- the reversion the sell-side
    sweep must capture: fire after the down-spike, buy the trough front of N+1, sell the bounce. The absolute
    sweep mixes this with the up-spike slot (which reverts flat), so only the down side should beat its control."""
    res, rows, sec, slot = [1e9, 100.0], [], 0.0, 10
    for c in range(cycles):
        for _ in range(3):                                      # a big sell -> SOL out, price DOWN
            tok, sol = res
            new_sol = max(sol - 1.0, 1.0)
            new_tok = (tok * sol) / new_sol
            dtok, dsolv = new_tok - tok, new_sol - sol
            res[0], res[1] = new_tok, new_sol
            rows.append([sec, slot, f"sell{c}", round(dtok, 6), round(dsolv, 9), round(res[0], 6), round(res[1], 9)])
            sec += 0.05
        sec = _add(res, rows, sec, slot + 1, 5, 3.0, "bot")     # the bot buys it back -> price UP (the bounce)
        for q in range(2, 6):                                   # quiet -> keeps the base rate low
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
    # the verdict must compare the strongest-edge conjunction (best by holdout LIFT, never the z-sorted top row)
    # against the best single rule, and must not contradict its own table (the bug: it picked combos[0] by z).
    r = fl.render_discover(out)
    best_combo = max(c["lift_hold"] for c in out["combos"])
    best_single = max(s["lift_hold"] for s in out["survivors"])
    if best_combo > best_single + 1e-9:
        assert "STACKED EDGE" in r                              # a conjunction out-lifts every single -> say so
    else:
        assert "does NOT beat" in r                             # tie or weaker -> single-dimensional, honestly
    assert "COMBINATIONS" in r


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


# --- economics-ranked discovery (discover_econ): select rules by scalp P&L, not lift ---------------

def _emit(specs):
    """Build a launch_record from per-slot specs [{buys, sells, fresh, price}, ...]. Price is set directly via
    the reserves (tok_after fixed at 1e9), so it is independent of the buy/sell magnitudes -- lets a test plant
    any price path with any slot classification. A fresh slot's buyers are brand-new wallets (sym N / F)."""
    rows, sec, wid = [], 0.0, [0]
    for slot, sp in enumerate(specs, start=10):
        sol_after = sp["price"] * 1e9
        for _ in range(sp.get("buys", 0)):
            payer = f"f{slot}_{wid[0]}" if sp.get("fresh") else "w0"
            wid[0] += 1
            rows.append([sec, slot, payer, -1.0, 2.0, 1e9, sol_after]); sec += 0.05
        for _ in range(sp.get("sells", 0)):
            rows.append([sec, slot, "w0", 1.0, -2.0, 1e9, sol_after]); sec += 0.05
        if not sp.get("buys") and not sp.get("sells"):
            rows.append([sec, slot, "w0", -1.0, 1e-9, 1e9, sol_after]); sec += 0.05  # a quiet price-carrying tick
    return {"txs": rows}


def _econ(builder, n=24):
    have = {f"P{i}": builder(i) for i in range(n)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(n)]
    return have, metas


def test_discover_econ_surfaces_a_median_rule_that_lift_misses():
    # Each flurry signal (n_buys=5) is followed by a steady +0.6% -- a positive, consistent SCALP return, but
    # BELOW the 1% target, so lift sees ~nothing. discover_econ (economics) must surface it; discover (lift) not.
    def broad(_i):
        specs = [{"buys": 1, "price": 100.0}] * 3                 # warmup
        for _ in range(40):
            specs += [{"buys": 5, "fresh": True, "price": 100.0}, # the flurry signal
                      {"buys": 1, "price": 100.6},                # +0.6% realization (< 1% target -> weak lift)
                      {"buys": 1, "price": 100.3},                # gentle decay (quiet control slots ~ -0.3%/0)
                      {"buys": 1, "price": 100.0}]
        return _emit(specs)
    have, metas = _econ(broad)
    tgt = fl._tgt("price", 3, 2, 1.0)
    econ = fl.discover_econ(have, metas, tgt, 0.0)
    assert econ["n_survivors"] >= 1
    assert any(s["label"] == "broad" and s["median"] > 0 and s["dir"] == "up" for s in econ["survivors"])
    assert "PAY" in fl.render_econ(econ)
    # the lift objective finds nothing on the SAME data (every move is < the 1% target) -> the whole point
    disc = fl.discover(None, None, metas=metas, have=have, target="price", move_pct=1.0, horizon=2, now=10.0)
    assert disc["n_survivors"] == 0


def test_discover_econ_surfaces_nothing_on_a_null_flat_market():
    # flurries with NO forward price move: every scalp is flat, so no rule can beat the control -- the
    # family-wise guard must not fabricate an edge.
    def null(_i):
        specs = [{"buys": 1, "price": 100.0}] * 3
        for _ in range(40):
            specs += [{"buys": 5, "fresh": True, "price": 100.0}, {"buys": 1, "price": 100.0},
                      {"buys": 1, "price": 100.0}, {"buys": 1, "price": 100.0}]
        return _emit(specs)
    have, metas = _econ(null)
    econ = fl.discover_econ(have, metas, fl._tgt("price", 3, 2, 1.0), 0.0)
    assert econ["n_survivors"] == 0
    assert "NO RULE" in fl.render_econ(econ)


def test_discover_econ_significance_separates_a_broad_edge_from_a_positive_skew_tail():
    # the two significance tests that split 'broad' (trust either side) from 'tail' (+300% skew, trust only on a
    # DOWN/reversion entry). A steady positive sample dominates the control on BOTH mean (Welch) and
    # distribution (Mann-Whitney) -> broad. A positive-skew sample (median < 0, mean > 0 from rare big wins)
    # clears the mean test but NOT the distribution test -> correctly labelled tail, not broad.
    steady = [0.006, 0.005, 0.007] * 50
    ctrl = [0.0, 0.001, -0.001] * 50
    assert fl._welch_z(steady, ctrl) > 3.0
    assert fl._mw_z(steady, ctrl)[1] > 3.0                        # dominates in distribution -> broad path

    import statistics as _st
    skew = [-0.01] * 90 + [3.0] * 10                              # median -1%, mean +29%
    assert _st.median(skew) < 0 < _st.fmean(skew)
    assert fl._welch_z(skew, [0.0] * 100) > 0                     # a mean edge (the tail)...
    assert fl._mw_z(skew, [0.0] * 100)[0] < 0.5                   # ...but does NOT dominate in distribution


# --- creation-moment scalp: the slot-1 prize ------------------------------------------------------

def _cre(prices, pull_pos=None):
    """A launch_record with one buy per slot at the given reserve-ratio prices (slot 0 = creation), and
    optionally an LP pull at slot position `pull_pos`."""
    rows, sec = [], 0.0
    for i, pr in enumerate(prices):
        rows.append([sec, 10 + i, f"w{i}", -1.0, 1.0, 1e9, pr * 1e9]); sec += 0.05
    if pull_pos is not None:
        rows.append([sec, 10 + pull_pos, "op", -1.0, -1.0, 1.0, 1.0])     # both vaults out = pull
    return {"txs": rows}


def test_first_buy_reactive_scalp_finds_a_prize_on_a_post_first_buy_ramp():
    # first buy in slot 0; a ramp 100 -> 110 -> 120 -> 125 after it. Backrunning (enter front of slot 1, hold
    # 2 = exit 2 slots later) captures the ramp, so a hold is positive on mean AND median.
    prices = [100.0, 110.0, 120.0, 125.0, 125.0, 125.0]
    have = {f"P{i}": _cre(prices) for i in range(60)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(60)]
    out = fl.creation_study(have, metas, 0.0)
    assert out["n_first_buy"] == 60 and out["pull_dist"]["none"] == 60   # every pool has a first buy, no pulls
    assert out["best"] is not None
    assert out["best"]["agg"]["mean"] > 0 and out["best"]["agg"]["median"] > 0
    assert {c["hold"] for c in out["cells"]} == {1, 2, 3}         # holds measured AFTER entry
    assert any(r["mean"] > 0 for r in out["ramp"])
    assert "PRIZE" in fl.render_creation(out)


def test_first_buy_reactive_scalp_no_prize_on_a_flat_open():
    have = {f"P{i}": _cre([100.0] * 6) for i in range(60)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(60)]
    out = fl.creation_study(have, metas, 0.0)
    assert out["best"] is None
    assert "NO CAPTURABLE MOVE" in fl.render_creation(out)


def test_first_buy_reactive_pull_histogram_is_relative_to_our_entry():
    # first buy in slot 0 (our entry = front of slot 1); a pull one slot after our entry traps the 2-slot hold.
    have = {f"P{i}": _cre([100.0, 100.0, 100.0, 100.0], pull_pos=2) for i in range(60)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(60)]
    out = fl.creation_study(have, metas, 0.0)
    assert out["pull_dist"]["1"] == 60 and out["pull_dist"]["none"] == 0   # pull lands 1 slot after our entry
    h2 = next(c for c in out["cells"] if c["hold"] == 2)["agg"]
    assert h2["rugged_pct"] > 0                                   # a pull within 2 slots of entry traps the stake
    assert "relative to OUR ENTRY" in fl.render_creation(out)


# --- slot-0 landing: how many slots to hold for EV+? ----------------------------------------------

def _cre2(prices, pull_pos=None):
    """Like _cre but the pull leaves no readable reserve (tok_after=0), so the price carries forward past it --
    lets a test keep a clean raw drift AFTER the pull slot while the net is still trapped by the rug."""
    rows, sec = [], 0.0
    for i, pr in enumerate(prices):
        rows.append([sec, 10 + i, f"w{i}", -1.0, 1.0, 1e9, pr * 1e9]); sec += 0.05
    if pull_pos is not None:
        rows.append([sec, 10 + pull_pos, "op", -1.0, -1.0, 0.0, 0.0])
    return {"txs": rows}


def test_slot0_ev_positive_within_the_safe_window():
    # a clean opening ramp, no pull: landing in slot 0 is EV+ from hold 1 (<= the 2-slot safe window) -> the prize.
    have = {f"P{i}": _cre([100.0, 110.0, 120.0, 125.0, 125.0, 125.0]) for i in range(60)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(60)]
    out = fl.slot0_study(have, metas, 0.0)
    assert out["break_even"] == 1 and out["break_even"] <= out["safe_hold"]
    txt = fl.render_slot0(out)
    assert "EV+ FROM HOLD 1" in txt and "WITHIN" in txt


def test_slot0_ev_positive_only_past_the_safe_window():
    # a dip then a rise: negative for holds 1-2, positive only at hold 3 -> EV+ but past the 2-slot safe window.
    have = {f"P{i}": _cre([100.0, 99.0, 99.0, 115.0, 120.0, 120.0]) for i in range(60)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(60)]
    out = fl.slot0_study(have, metas, 0.0)
    assert out["break_even"] == 3 and out["break_even"] > out["safe_hold"]
    assert "PAST the" in fl.render_slot0(out)


def test_slot0_drift_never_covers_cost():
    # flat opening, a 5% round-trip cost: no hold's raw drift clears cost, so none is EV+.
    have = {f"P{i}": _cre([100.0] * 8) for i in range(60)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(60)]
    out = fl.slot0_study(have, metas, 0.05)
    assert out["break_even"] is None and out["raw_break_even"] is None
    assert "DRIFT NEVER COVERS COST" in fl.render_slot0(out)


def test_slot0_not_ev_when_the_rug_eats_the_drift():
    # raw drift clears cost at hold 4, but every pool is pulled at slot 13 (hold 3), so the NET is never positive.
    have = {f"P{i}": _cre2([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0], pull_pos=3) for i in range(60)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(60)]
    out = fl.slot0_study(have, metas, 0.035)
    assert out["break_even"] is None and out["raw_break_even"] == 4
    assert "NOT EV+ AT ANY HOLD" in fl.render_slot0(out)


# --- buy-flurry banded by buy count, enter the slot after, hold 1-10 slots -------------------------

def _mb(per_slot):
    """A launch_record with `per_slot` = [(n_buys, price), ...]; slot i (number 10+i) has n_buys buy rows at price."""
    rows, sec = [], 0.0
    for idx, (nb, pr) in enumerate(per_slot):
        for _ in range(nb):
            rows.append([sec, 10 + idx, f"w{idx}", -1.0, 1.0, 1e9, pr * 1e9]); sec += 0.001
    return {"txs": rows}


def test_multibuy_bands_and_per_slot_hold_find_the_ideal_exit():
    # each pool: flat prefix (2 buys), a 20-buy flurry (band 15-25) at slot idx 8, then a ramp peaking at hold 5.
    flat = [(2, 100.0)] * 8
    ramp = [(20, 100.0), (2, 108.0), (2, 116.0), (2, 124.0), (2, 132.0), (2, 150.0),
            (2, 120.0), (2, 110.0), (2, 105.0), (2, 102.0), (2, 100.0)]
    have = {f"P{i}": _mb(flat + ramp) for i in range(60)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(60)]
    out = fl.multibuy_study(have, metas, 0.0)
    assert out["holds"] == list(range(1, 11))
    labels = [b["band"] for b in out["bands"]]
    assert labels == ["0-15", "15-25", "25-35", "35-45", "45-55", "55-65", "65-75", "75-85", "85-95", "95+"]
    row = next(b for b in out["bands"] if b["band"] == "15-25")
    assert row["best_hold"] == 5 and row["pays"]          # entering front of N+1, the ramp peaks 5 slots later
    assert out["best"] is not None and out["best"]["band"] == "15-25"
    assert abs(out["best"]["agg"]["median"] - 0.50) < 1e-6
    assert "ENTER SLOT AFTER" in fl.render_multibuy(out)


def test_multibuy_no_edge_when_flat():
    have = {f"P{i}": _mb([(2, 100.0)] * 20) for i in range(60)}
    metas = [{"pool": f"P{i}", "token_mint": "M", "create_ts": float(i)} for i in range(60)]
    out = fl.multibuy_study(have, metas, 0.01)        # a round-trip cost, so a flat hold is negative
    assert out["best"] is None
    assert "NO BAND x EXIT-HOLD PAYS" in fl.render_multibuy(out)


def test_pool_vq_decodes_the_i128_at_offset_245():
    raw = bytes(245) + (12345).to_bytes(16, "little", signed=True)
    assert fl._pool_vq(raw) == 12345
    assert fl._pool_vq(b"") == 0
    assert fl._pool_vq(bytes(100)) == 0            # too short to hold the field


def test_virtual_quote_basis_dampens_early_moves():
    # Raw vault ratio inflates early-life moves (virtual_quote dominates the tiny real quote vault); adding vq
    # gives the effective basis the live WS stream uses, so the ">=0.3" target fires on real moves, not artifacts.
    rec = {"create_ts": 0, "txs": [
        [0.0, 100, "w", -10.0, 1.0, 1000.0, 1.0],
        [0.1, 101, "w", -10.0, 1.0, 990.0, 2.0],
        [0.2, 102, "w", -10.0, 1.0, 980.0, 3.0],
    ]}

    def moves(flow):
        p = [s["price"] for s in flow["slots"]]
        return [p[i] / p[i - 1] - 1.0 for i in range(1, len(p))]

    saved = fl.USE_VQ
    try:
        fl.USE_VQ = False
        raw = moves(fl.slot_flow(rec))
        assert raw[0] > 1.0 and raw[1] > 0.4           # raw ratio: fake +100% / +50% moves

        fl.USE_VQ = True
        rec["vq"] = 30_000_000_000                     # 30 SOL virtual_quote (raw lamports)
        eff = moves(fl.slot_flow(rec))
        assert all(m < 0.1 for m in eff)               # effective basis: the real move is a few %

        rec.pop("vq")                                  # vq on but the rec carries none -> graceful raw fallback
        assert moves(fl.slot_flow(rec)) == raw
    finally:
        fl.USE_VQ = saved


def _cre_front(create_ts, n_ep=25, up=0.40, signal=True, seed=0):
    """A pool where (signal) a brand-new wallet buys the slot BEFORE a >= up spike, so sym==N front-runs it;
    (not signal) up-moves happen with no new-wallet precursor. Consecutive slot numbers, constant-product buys."""
    import random as _r
    rng = _r.Random("%s-%s" % (seed, create_ts))
    sol, tok = 100.0, 100000.0
    rows, sec, slot = [], 0.0, 0
    up_frac = (1.0 + up) ** 0.5 - 1.0        # constant-product: price x (1+frac)^2

    def buy(payer, frac):
        nonlocal sol, tok, sec, slot
        k = sol * tok
        nsol = sol + frac * sol
        ntok = k / nsol
        rows.append([sec, slot, payer, round(ntok - tok, 6), round(frac * sol, 9), round(ntok, 6), round(nsol, 9)])
        sol, tok, sec, slot = nsol, ntok, sec + 0.4, slot + 1

    for _ in range(5):
        buy("w0", 0.002)
    for e in range(n_ep):
        if signal:
            buy("new%d_%d" % (create_ts, e), 0.01)    # precursor: brand-new wallet, ~flat -> sym N
            buy("wb", up_frac)                          # the >= up spike (T), repeat wallet
        else:
            buy("wb", up_frac if rng.random() < 0.5 else 0.01)   # spike with no new-wallet before it
            buy("new%d_%d" % (create_ts, e), 0.01)               # new wallets AFTER the move, not before
        buy("w0", 0.002)
        buy("w0", 0.002)
    return {"create_ts": create_ts, "txs": rows}


def _flat_front(create_ts):
    """No >= 0.3 move at all: only tiny filler buys -> no events to front-run."""
    sol, tok = 100.0, 100000.0
    rows, sec, slot = [], 0.0, 0
    for _ in range(90):
        k = sol * tok
        nsol = sol + 0.002 * sol
        ntok = k / nsol
        rows.append([sec, slot, "w%d" % (slot % 4), round(ntok - tok, 6), round(0.002 * sol, 9),
                     round(ntok, 6), round(nsol, 9)])
        sol, tok, sec, slot = nsol, ntok, sec + 0.4, slot + 1
    return {"create_ts": create_ts, "txs": rows}


def test_frontrun_finds_the_precursor_and_it_pays():
    saved = fl.USE_VQ
    try:
        fl.USE_VQ = False
        metas = [{"pool": "P%d" % i, "token_mint": "M%d" % i, "create_ts": float(i)} for i in range(6)]
        have = {"P%d" % i: _cre_front(i) for i in range(6)}
        out = fl.frontrun(have, metas, move=0.30, cost=0.01)
        assert out["events"] > 50                                    # the planted spikes are detected
        assert out["ceiling"]["1-1"]["mean"] > 0.10                  # enter T-1 / exit T+1 captures the spike
        surv = [s["rule"] for L in out["leads"] for s in out["survivors"][L]]
        assert any("sym == N" in r for r in surv)                   # the new-wallet precursor is discovered
        assert out["net"], "a surviving indicator should reach the net"
        top = out["net"][0]
        # it fires on the precursor slot and the spike lands in the hold -> beats the random-slot control
        assert any(top["holds"][h]["n"] >= 50 and top["holds"][h]["mean"] > out["control"][h]["mean"]
                   for h in fl.FRONT_HOLDS)
        lg = out["logit"]                                            # the multivariate "regression with confidence"
        assert lg["predictors"] and lg["predictors"][0]["name"] == "intercept"
        nrow = next(p for p in lg["predictors"] if "sym == N" in p["name"])
        assert nrow["coef"] > 0                                      # precursor points the right way in the joint fit
        assert all(0.0 <= p["p"] <= 1.0 and fl.np.isfinite(p["z"]) for p in lg["predictors"])
        assert lg["auc_hold"] is not None and lg["auc_hold"] > 0.5   # the joint model predicts the up-move out of sample
        gate = out["logit_gate"]                                     # the high-precision gate test is well-formed
        assert gate["rows"] and "control" in gate
        assert all(r["n"] >= 0 and set(r["holds"]) == set(fl.FRONT_HOLDS) for r in gate["rows"])
        assert gate["rows"][-1]["n"] <= gate["rows"][0]["n"]         # a stricter top-k% fires on fewer slots
        assert fl.render_frontrun(out)
    finally:
        fl.USE_VQ = saved


def test_logit_fit_recovers_a_positive_coefficient_with_finite_se():
    rng = fl.np.random.default_rng(0)
    n = 2000
    x = (rng.random(n) < 0.5).astype(float)
    p = 1.0 / (1.0 + fl.np.exp(-(-1.5 + 2.5 * x)))                     # y strongly driven by x
    y = (rng.random(n) < p).astype(float)
    X = fl.np.column_stack([fl.np.ones(n), x])
    beta, se, _mu = fl._logit_fit(X, y)
    assert beta[1] > 1.5 and se[1] > 0 and fl.np.isfinite(se[1])       # recovers the planted slope
    z = beta[1] / se[1]
    pv = fl._wald_p(z)
    assert z > 3 and 0.0 <= pv <= 1.0 and pv < 0.01


def test_frontrun_flat_market_has_no_events_and_no_survivor():
    saved = fl.USE_VQ
    try:
        fl.USE_VQ = False
        metas = [{"pool": "F%d" % i, "token_mint": "M%d" % i, "create_ts": float(i)} for i in range(6)]
        have = {"F%d" % i: _flat_front(i) for i in range(6)}
        out = fl.frontrun(have, metas, move=0.30, cost=0.01)
        assert out["events"] == 0                                   # nothing to front-run
        assert not out["ceiling"]                                   # no perfect-foresight cell either
        assert all(not out["survivors"][L] for L in out["leads"])   # and no indicator fabricated
        assert not out["logit"].get("predictors")                   # nothing to fit the logistic on
        assert fl.render_frontrun(out)
    finally:
        fl.USE_VQ = saved
