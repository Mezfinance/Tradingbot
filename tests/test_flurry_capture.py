"""The two-feed box capture (train=serve front-run study): current slot from the shred decode (price-free),
prior from the Helius-WS confirmed decode capped at the captured frontier. No Helius history reconstruction.

Pins the honest live shape: the gate is fit on exactly what the box serves live -- a shred-fresh current slot N
(no price) and a confirmation-lagged prior. The frontier, captured per decision, decides N-1 vs N-k, so the study
can never hand the gate a confirmed slot the box did not yet have at the N->N+1 boundary (the look-ahead that made
the offline hybrid diverge)."""
import json

from tradingbot import flurry_study as fs


def _shred(slot, n_buys, buyers):      # price-free (shreds carry no reserves): 7 fields, no price
    return [slot, n_buys, 0.0, 0.0, 0, 0.0, buyers]


def _conf(slot, n_buys, buyers, price, n_sells=0):   # confirmed: 8 fields, WITH price
    return [slot, n_buys, 0.0, 0.0, n_sells, 0.0, buyers, price]


def test_flow_from_series_prices_and_net_sol():
    priced = fs._flow_from_series([[5, 2, 1.5, 1.0, 1, 0.4, ["a", "b"], 1.07]], priced=True)
    s = priced["slots"][0]
    assert s["slot"] == 5 and s["price"] == 1.07
    assert abs(s["net_sol"] - (1.5 - 0.4)) < 1e-9       # buy_sol - sell_sol
    assert s["buyers"] == ["a", "b"]
    # a price-free (shred) series gets the placeholder price so _slot_bank runs; the hybrid builder zeroes the
    # price-derived features of the current slot anyway.
    free = fs._flow_from_series([_shred(5, 2, ["a"])], priced=False)
    assert free["slots"][0]["price"] == 1.0
    # slots come back sorted
    out = fs._flow_from_series([_shred(9, 1, ["z"]), _shred(1, 1, ["a"])], priced=False)
    assert [s["slot"] for s in out["slots"]] == [1, 9]


def _cap(frontier=None):
    # current (shred) counts differ from confirmed counts for the same slot, so a prior feature taken from
    # confirmed is provable (and the current feature is provably from shred).
    shred = [_shred(k, k, [f"s{k}"]) for k in range(1, 6)]            # slot k: n_buys = k
    confirmed = [_conf(k, 10 + k, [f"c{k}"], 1.0 + 0.01 * k) for k in range(1, 6)]   # slot k: n_buys = 10 + k
    cap = {"shred": shred, "confirmed": confirmed}
    if frontier is not None:
        cap["frontier"] = frontier
    return cap


def test_current_slot_is_price_free_and_prior_comes_from_confirmed():
    tgt = fs._tgt("price", 3, 2, 20.0)
    bank = fs._hybrid_bank_from_capture(_cap(), 0.0, tgt)            # no frontier -> lag 1 (prior = N-1)
    assert len(bank) == 5
    last = bank[-1]                                                  # decision slot 5
    # the current slot has NO live price from shreds -> its own move features are zeroed, never a fire input
    assert last["f"]["abs_move"] == 0.0 and last["f"]["move"] == 0.0 and last["f"]["price_up"] == 0.0
    # the current slot's own counts come from the SHRED decode (what the box sees live)
    assert last["f"]["n_buys"] == 5.0
    # the prior (N-1) count features come from the CONFIRMED decode, not the shred decode
    assert last["f"]["prev_buys"] == 14.0                            # confirmed slot 4 = 10 + 4


def test_frontier_caps_the_prior_so_a_deeper_lag_uses_n_minus_k_not_n_minus_1():
    tgt = fs._tgt("price", 3, 2, 20.0)
    lag1 = fs._hybrid_bank_from_capture(_cap(frontier=[[5, 4]]), 0.0, tgt)   # confirmed through N-1
    lag3 = fs._hybrid_bank_from_capture(_cap(frontier=[[5, 2]]), 0.0, tgt)   # confirmed only through N-3
    # at decision slot 5 the prior is the freshest CONFIRMED slot the box actually had: slot 4 vs slot 2
    assert lag1[-1]["f"]["prev_buys"] == 14.0                        # confirmed slot 4
    assert lag3[-1]["f"]["prev_buys"] == 12.0                        # confirmed slot 2 -- the frontier cap, not N-1
    # the current slot is identical regardless of the lag (it is the shred decode of N)
    assert lag1[-1]["f"]["n_buys"] == lag3[-1]["f"]["n_buys"] == 5.0


def test_empty_series_yields_no_bank():
    tgt = fs._tgt("price", 3, 2, 20.0)
    assert fs._hybrid_bank_from_capture({"shred": [], "confirmed": []}, 0.0, tgt) == []


def test_flows_from_file_detects_a_capture_row(tmp_path):
    p = tmp_path / "flows.jsonl"
    rows = [
        {"pool": "CAP", "token_mint": "mintA", "create_ts": 100,
         "shred": [_shred(1, 1, ["a"])], "confirmed": [_conf(1, 2, ["b"], 1.0)],
         "frontier": [[2, 1]], "pull_slot": None},
        {"pool": "REC", "token_mint": "mintB", "create_ts": 200,
         "rec": {"create_ts": 200, "txs": []}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    metas, have = fs.flows_from_file(str(p))
    assert {m["pool"] for m in metas} == {"CAP", "REC"}
    assert "confirmed" in have["CAP"] and "shred" in have["CAP"]    # capture dict (so _frontrun takes the capture path)
    assert have["CAP"]["frontier"] == [[2, 1]]
    assert "txs" in have["REC"]                                     # launch_record rec (existing path)


def test_scalp_rec_prices_by_slot_number_not_decision_position():
    # the decision (shred) slot set is not positionally aligned with the confirmed price path, so the scalp must
    # locate the entry by the slot NUMBER, never by rec['idx'] (a decision-view position).
    slots = [{"slot": 5, "price": 1.0}, {"slot": 6, "price": 1.2}, {"slot": 7, "price": 1.1}]
    cpos = {s["slot"]: i for i, s in enumerate(slots)}
    rec = {"slot": 5, "idx": 99}                                     # idx is bogus on purpose -> must be ignored
    assert fs._scalp_rec(slots, cpos, None, rec, 1, 0.0) == fs._scalp(slots, None, 0, 1, 0.0)  # enters at slot 5
    # a decision slot absent from the confirmed price path (WS opened late) -> unfillable -> None, never IndexError
    assert fs._scalp_rec(slots, cpos, None, {"slot": 1, "idx": 0}, 1, 0.0) is None


def test_frontrun_capture_mismatched_feeds_reaches_the_net_without_indexerror():
    # the regression: shred carries MORE active slots than confirmed (the Helius WS opens per-pool late), so a
    # decision-view index used against the confirmed price path ran off the end (IndexError at line _scalp). Enough
    # pools to pass the min-support gate (>= 2*DISC_MIN_SUPPORT recs) so the net + precursor gate scalp actually run.
    def pool(pfx, bump_at):
        shred, confirmed = [], []
        price = 1.0
        for k in range(1, 13):
            price = price * 1.25 if k == bump_at else price * 1.001
            buyers = ["%sn%d" % (pfx, k)] if k % 2 == 0 else ["%sR" % pfx]   # even: new wallet (N); odd: repeat (B/b)
            shred.append(_shred(k, 1, buyers))
            if k >= 3:                                              # confirmed opens LATE: slots 1,2 are shred-only
                confirmed.append(_conf(k, 1, buyers, price))
        return {"shred": shred, "confirmed": confirmed, "frontier": [], "pull_slot": None}

    metas = [{"pool": "P%d" % i, "token_mint": "m%d" % i, "create_ts": float(i)} for i in range(8)]
    have = {"P%d" % i: pool("P%d" % i, 6 + 2 * (i % 2)) for i in range(8)}   # bump at an even (N) slot
    out = fs.frontrun(have, metas, move=0.2)                        # must NOT raise (the IndexError regression)
    assert out["pools"] == 8 and out["events"] >= 1
    assert isinstance(out["net"], list)
    assert "rows" in out["logit_gate_precursor"]                    # the precursor-gate scalp ran over mismatched feeds


def test_ws_only_uses_the_confirmed_series_as_the_single_feed():
    # WS-only: the Helius-WS ('confirmed') series is the sole complete+priced feed -- no shred, no hybrid. A pool
    # with an EMPTY shred series (which two-feed mode would drop) must still be used from its confirmed series.
    saved = fs.WS_ONLY
    try:
        fs.WS_ONLY = True

        def pool(pfx, bump):
            confirmed, price = [], 1.0
            for k in range(1, 13):
                price = price * 1.25 if k == bump else price * 1.001
                confirmed.append(_conf(k, (3 if k == bump - 1 else 1), ["%sc%d" % (pfx, k)], price))
            return {"shred": [], "confirmed": confirmed, "frontier": [], "pull_slot": None}   # shred EMPTY on purpose

        metas = [{"pool": "P%d" % i, "token_mint": "m%d" % i, "create_ts": float(i)} for i in range(8)]
        have = {"P%d" % i: pool("P%d" % i, 6 + 2 * (i % 2)) for i in range(8)}
        out = fs.frontrun(have, metas, move=0.2)
        assert out["ws_only"] and out["pools"] == 8 and out["events"] >= 1   # empty shred no longer drops the pool
        assert "[WS-ONLY feed]" in fs.render_frontrun(out)
    finally:
        fs.WS_ONLY = saved


def test_frontrun_tail_concentration_is_computed_when_requested():
    # --tail adds the broad-vs-spikes breakdown of the precursor gate: distinct winning pools, capped vs raw mean,
    # top-5/20 share. Off by default; on, every cell's stats are well-formed.
    saved = fs.WS_ONLY
    try:
        fs.WS_ONLY = True

        def pool(pfx, bump):
            confirmed, price = [], 1.0
            for k in range(1, 13):
                price = price * 1.25 if k == bump else price * 1.001
                buyers = ["%sn%d" % (pfx, k)] if k % 2 == 0 else ["%sR" % pfx]   # even: new wallet; odd: repeat
                confirmed.append(_conf(k, 1, buyers, price))
            return {"shred": [], "confirmed": confirmed, "frontier": [], "pull_slot": None}

        metas = [{"pool": "P%d" % i, "token_mint": "m%d" % i, "create_ts": float(i)} for i in range(8)]
        have = {"P%d" % i: pool("P%d" % i, 6 + 2 * (i % 2)) for i in range(8)}
        base = fs.frontrun(have, metas, move=0.2)
        assert "gate_tail" not in base                       # off by default
        out = fs.frontrun(have, metas, move=0.2, tail=True)
        assert isinstance(out.get("gate_tail"), dict)
        for (pct, h), d in out["gate_tail"].items():
            assert d["win_pools"] <= d["winners"]            # distinct winning pools can't exceed winners
            assert d["winners"] <= d["n"]
            assert 0.0 <= d["top5_share"] <= 1.0 + 1e-9 and 0.0 <= d["top20_share"] <= 1.0 + 1e-9
        assert (not out["gate_tail"]) or "TAIL CONCENTRATION" in fs.render_frontrun(out)
    finally:
        fs.WS_ONLY = saved


def test_frontrun_runs_on_a_two_feed_capture_without_helius():
    # two pools, each with a +20% up-move so there are events to be positioned before; the study should run its
    # literal front-run code on the capture (ceiling + feature bank) and return cleanly, no Helius.
    def pool(pfx, bump_at):
        shred, confirmed = [], []
        price = 1.0
        for k in range(1, 13):
            price = price * 1.22 if k == bump_at else price * 1.001
            shred.append(_shred(k, (3 if k == bump_at - 1 else 1), [f"{pfx}s{k}"]))
            confirmed.append(_conf(k, (3 if k == bump_at - 1 else 1), [f"{pfx}c{k}"], price))
        return {"shred": shred, "confirmed": confirmed, "frontier": [], "pull_slot": None}

    metas = [{"pool": "P1", "token_mint": "m1", "create_ts": 1000},
             {"pool": "P2", "token_mint": "m2", "create_ts": 2000}]
    have = {"P1": pool("P1", 6), "P2": pool("P2", 7)}
    out = fs.frontrun(have, metas, move=0.2)
    assert out["mode"] == "frontrun"
    assert out["pools"] == 2
    assert "ceiling" in out and out["events"] >= 1                  # the +20% bumps are real up-events


# --- reserve-collapse artifact filter (DROP_ARTIFACTS): remove near-drain spikes ENTIRELY -----------

def _rec(rows):                          # a launch_record: rows [sec, slot, payer, tokD, solD, tok_after, sol_after]
    return {"txs": rows, "vq": 0}


def test_drop_artifacts_reserve_branch_drops_a_token_reserve_collapse():
    slots = [{"slot": 1, "tok": 1000.0, "price": 0.01}, {"slot": 2, "tok": 1000.0, "price": 0.01},
             {"slot": 3, "tok": 50.0, "price": 2.0}, {"slot": 4, "tok": 1000.0, "price": 0.011}]
    kept = fs._drop_artifacts(slots)                         # tok 50 < 0.15 x median(1000) -> dropped
    assert [s["slot"] for s in kept] == [1, 2, 4]


def test_drop_artifacts_price_branch_drops_the_spike_image_and_leaves_flat_untouched():
    slots = [{"slot": 1, "price": 1.0}, {"slot": 2, "price": 1.02}, {"slot": 3, "price": 20.0},
             {"slot": 4, "price": 1.03}]
    kept = fs._drop_artifacts(slots)                         # 20 > median(~1.025)/0.15 ~ 6.8 -> dropped
    assert [s["slot"] for s in kept] == [1, 2, 4]
    flat = [{"slot": k, "price": 1.0} for k in range(5)]     # price-free (shred) series: all equal -> untouched
    assert fs._drop_artifacts(flat) == flat


def test_flow_from_series_drops_a_capture_price_spike_only_when_enabled():
    rows = [[k, 1, 0.1, 0.1, 0, 0.0, [f"w{k}"], 1.0 + 0.01 * k] for k in range(4)]   # prices ~1.0..1.03
    rows.append([4, 1, 5.0, 5.0, 0, 0.0, ["x"], 20.0])                                # a near-drain spike
    saved = fs.DROP_ARTIFACTS
    try:
        fs.DROP_ARTIFACTS = False
        assert any(abs(s["price"] - 20.0) < 1e-9 for s in fs._flow_from_series(rows, priced=True)["slots"])
        fs.DROP_ARTIFACTS = True
        on = fs._flow_from_series(rows, priced=True)["slots"]
        assert all(s["price"] < 2.0 for s in on) and len(on) == 4   # spike gone, genuine slots kept
        # a price-free capture series is never touched (all prices become the 1.0 placeholder)
        free = fs._flow_from_series([[k, 1, 0.1, 0.1, 0, 0.0, [f"w{k}"]] for k in range(4)], priced=False)
        assert len(free["slots"]) == 4
    finally:
        fs.DROP_ARTIFACTS = saved


def test_slot_flow_drops_a_reserve_collapse_and_does_not_carry_its_price_forward():
    rows = [[0, k, f"w{k}", -10.0, 1.0, 1000.0 - 10 * k, 10.0 + k] for k in range(1, 5)]   # 4 normal buys
    rows.append([0, 5, "x", -940.0, 90.0, 50.0, 101.0])                                    # near-drain: tok 50
    saved = fs.DROP_ARTIFACTS
    try:
        fs.DROP_ARTIFACTS = False
        off = fs.slot_flow(_rec(rows))
        assert any(s["slot"] == 5 and s["price"] > 1.0 for s in off["slots"])   # artifact kept, price explodes
        fs.DROP_ARTIFACTS = True
        on = fs.slot_flow(_rec(rows))
        assert on is not None and all(s["slot"] != 5 for s in on["slots"])      # dropped entirely...
        assert all(s["price"] < 1.0 for s in on["slots"])                       # ...and its price never carried fwd
    finally:
        fs.DROP_ARTIFACTS = saved


def test_filter_by_liq_keeps_only_pools_at_or_above_the_floor():
    # peek5 carries no liquidity field, so filter_by_liq joins the scorer's recorded liq_usd. C has no record -> dropped.
    liq = {"A": 9000.0, "B": 2000.0, "D": 50000.0}

    def query(sql, params):
        return [{"pool": p, "liq_usd": liq[p]} for p in params if p in liq]

    metas = [{"pool": p, "token_mint": "m" + p, "create_ts": 1.0} for p in ("A", "B", "C", "D")]
    have = {p: {"confirmed": [[1, 0, 0.0, 0.0, 0, 0.0, [], 1.0]]} for p in ("A", "B", "C", "D")}
    kept, have2, stats = fs.filter_by_liq(query, metas, have, 5000.0)
    assert [m["pool"] for m in kept] == ["A", "D"]          # >= 5000 only (B below, C unknown)
    assert set(have2) == {"A", "D"}                         # have filtered to match the kept metas
    assert stats == {"floor": 5000.0, "total": 4, "with_liq": 3, "from_scorer": 3, "from_fetch": 0,
                     "kept": 2, "dropped_below": 1, "dropped_unknown": 1}


def test_filter_by_liq_extra_liq_backfills_a_scorer_gap():
    # C has no scorer record; an extra_liq fetch result >= floor rescues it from the unknown-drop.
    def query(sql, params):
        return [{"pool": "A", "liq_usd": 9000.0}] if "A" in params else []

    metas = [{"pool": p, "token_mint": "m" + p, "create_ts": 1.0} for p in ("A", "C")]
    have = {p: {"confirmed": []} for p in ("A", "C")}
    kept, have2, stats = fs.filter_by_liq(query, metas, have, 5000.0, extra_liq={"C": 12000.0})
    assert [m["pool"] for m in kept] == ["A", "C"]          # C now clears the floor from the fetch
    assert stats["from_scorer"] == 1 and stats["from_fetch"] == 1 and stats["dropped_unknown"] == 0


class _FakeStore:
    def __init__(self):
        self.rows = {}

    def execute(self, sql, params=()):
        if sql.startswith("INSERT"):
            self.rows[params[0]] = params[1]

    def query(self, sql, params=()):
        if "watch_capture_liq" in sql:
            return [{"pool": p, "liq_usd": self.rows[p]} for p in params if p in self.rows]
        return []


class _FakeHelius:
    def __init__(self, sol):          # sol: pool -> SOL vault reserve at second ~4
        self.sol = sol
        self.calls = 0

    def history_from_start(self, pool, span_s, pages):
        self.calls += 1
        s = self.sol.get(pool)
        if s is None:
            return ([], True, True)
        # one early row: [sec, slot, payer, dtok, dsol, tok_after, sol_after]
        return ([{"blockTime": 100, "slot": 1, "meta": {}, "transaction": {}}], True, True)

    # launch_record is patched per-test to avoid parsing a real tx


def test_backfill_liq_reads_seed_liq_and_caches(monkeypatch):
    store = _FakeStore()
    hel = _FakeHelius({"P1": 40.0, "P2": 10.0})   # P1 ~ $6.6k at $83 x 2 x 40; P2 ~ $1.6k
    import tradingbot.data.helius as _h
    monkeypatch.setattr(_h, "launch_record",
                        lambda txs, pool, mint, window_s=30: {"txs": [[2, 1, "x", 0, 0, 0, hel.sol[pool]]]})
    out = fs.backfill_liq(hel, store, ["P1", "P2", "P3"], {"P1": "m1", "P2": "m2", "P3": "m3"}, 83.0)
    assert round(out["P1"]) == round(2 * 40.0 * 83.0) and round(out["P2"]) == round(2 * 10.0 * 83.0)
    assert "P3" not in out                          # no reserve read -> cached 0 -> left unknown
    assert hel.calls == 3
    out2 = fs.backfill_liq(hel, store, ["P1", "P2", "P3"], {}, 83.0)   # all cached now
    assert hel.calls == 3 and round(out2["P1"]) == round(2 * 40.0 * 83.0)


def test_filter_by_liq_fails_open_when_the_table_is_missing():
    def query(sql, params):
        raise Exception("no such table: watch_launch_live")

    metas = [{"pool": "A", "token_mint": "mA", "create_ts": 1.0}]
    kept, have2, stats = fs.filter_by_liq(query, metas, {"A": {"confirmed": []}}, 5000.0)
    assert kept == metas and have2 == {"A": {"confirmed": []}} and "error" in stats   # no filter applied
