"""Step 1 of the failed-inclusive refit test: helius.swap_direction + launch_record(include_failed=).

Validates that a FAILED pumpswap tx (no vault delta) gets its direction from the instruction, with the same
orientation flip as the live decoder, and that launch_record emits flagged failed rows the study can toggle."""
from tradingbot.data import helius

WSOL = helius.SOL_MINT
TOKEN = "Tok1111111111111111111111111111111111111111"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
POOL = "Pool11111111111111111111111111111111111111"
PAYER = "Payer1111111111111111111111111111111111111"

SELL = helius._SWAP_SELL
BEQI = bytes([198, 46, 21, 82, 180, 217, 232, 112])   # buy_exact_quote_in
BBO = bytes([102, 6, 61, 18, 1, 218, 235, 234])       # buy (buy_base_out)


def _ix(disc, accts, program=helius.PUMPSWAP_PROGRAM):
    return {"programId": program, "accounts": list(accts), "data": helius.b58encode(disc + bytes(16))}


def _tx(instrs, err=None, bt=0, slot=0, token_deltas=None):
    meta = {"err": err}
    if token_deltas:                                   # [(owner, mint, pre, post), ...]
        meta["preTokenBalances"] = [{"owner": o, "mint": m, "uiTokenAmount": {"uiAmount": pre}}
                                    for o, m, pre, _ in token_deltas]
        meta["postTokenBalances"] = [{"owner": o, "mint": m, "uiTokenAmount": {"uiAmount": post}}
                                     for o, m, _, post in token_deltas]
    return {"blockTime": bt, "slot": slot, "meta": meta,
            "transaction": {"message": {"accountKeys": [{"pubkey": PAYER}], "instructions": instrs}}}


def test_swap_direction_normal_pool():
    accts = [POOL, "u", "g", TOKEN, WSOL]              # base = TOKEN
    assert helius.swap_direction(_tx([_ix(SELL, accts)])) is False   # sell ix -> token SELL
    assert helius.swap_direction(_tx([_ix(BEQI, accts)])) is True    # buy ix  -> token BUY


def test_swap_direction_sol_base_pool_flips():
    accts = [POOL, "u", "g", WSOL, TOKEN]              # base = WSOL -> the flip
    assert helius.swap_direction(_tx([_ix(SELL, accts)])) is True    # sell ix -> token BUY
    assert helius.swap_direction(_tx([_ix(BBO, accts)])) is False    # buy ix  -> token SELL


def test_swap_direction_rejects_non_sol_and_non_pumpswap():
    assert helius.swap_direction(_tx([_ix(SELL, [POOL, "u", "g", TOKEN, USDC])])) is None   # no WSOL side
    assert helius.swap_direction(_tx([_ix(SELL, [POOL, "u", "g", TOKEN, WSOL], program="Other")])) is None


def test_launch_record_includes_failed_rows_with_direction():
    good = _tx([_ix(BEQI, [POOL, "u", "g", TOKEN, WSOL])], err=None, bt=1, slot=10,
               token_deltas=[(POOL, TOKEN, 100.0, 90.0), (POOL, WSOL, 5.0, 6.0)])   # token out, SOL in = buy
    fbuy = _tx([_ix(BEQI, [POOL, "u", "g", TOKEN, WSOL])], err={"InstructionError": [3, {"Custom": 6040}]}, bt=2, slot=11)
    fsell = _tx([_ix(SELL, [POOL, "u", "g", TOKEN, WSOL])], err={"InstructionError": [3, {"Custom": 1}]}, bt=3, slot=12)
    txs = [good, fbuy, fsell]

    # success-only (default): failed txs skipped, behaviour unchanged
    rec0 = helius.launch_record(txs, POOL, TOKEN, window_s=180, include_failed=False)
    assert rec0["failed"] == 2 and rec0.get("failed_tagged") is False and len(rec0["txs"]) == 1

    # failed-inclusive: all three rows, failed flagged at index 8, direction encoded by sign, price 0 (carry-forward)
    rec = helius.launch_record(txs, POOL, TOKEN, window_s=180, include_failed=True)
    assert rec["failed_tagged"] is True and len(rec["txs"]) == 3
    rows = rec["txs"]
    assert rows[0][8] is False and rows[0][3] < 0 and rows[0][4] > 0           # success buy: real delta
    assert rows[1][8] is True and rows[1][3] < 0 and rows[1][4] > 0 and rows[1][5] == 0.0   # failed buy
    assert rows[2][8] is True and rows[2][3] > 0 and rows[2][4] < 0            # failed sell
    assert rows[1][2] == PAYER                                                # payer captured (the would-be buyer)


# --- Step 2: slot_flow toggles the failed rows ---
from tradingbot import flurry_study as fs   # noqa: E402


def _row(sec, slot, payer, td, sd, toka, sola, failed):
    return [sec, slot, payer, td, sd, toka, sola, True, failed]   # vis=True at 7, _FAILED at 8


def test_slot_flow_includes_failed_only_when_asked():
    rec = {"create_ts": 0, "failed_tagged": True, "txs": [
        _row(1, 10, "A", -10, 1, 90, 6, False),
        _row(2, 11, "B", -10, 1, 80, 7, False),
        _row(2, 11, "C", -1e-9, 1e-9, 0, 0, True),    # a FAILED buy in the same slot (new wallet C)
        _row(3, 12, "D", 5, -0.5, 85, 6.5, False),
    ]}
    s0 = fs.slot_flow(rec, include_failed=False)
    s1 = fs.slot_flow(rec, include_failed=True)
    a = next(s for s in s0["slots"] if s["slot"] == 11)
    b = next(s for s in s1["slots"] if s["slot"] == 11)
    assert a["n_buys"] == 1 and a["buyers"] == ["B"]              # success-only: C (failed) dropped
    assert b["n_buys"] == 2 and b["buyers"] == ["B", "C"]         # failed-inclusive: C counted as a buyer
    assert a["price"] == b["price"]                               # the failed row never moves the price


# --- Step 3: frontrun runs both views and render_frontrun_failed_compare works ---
def _causal_rows_tagged(cycles=45):
    """_rec_causal's shape (new wallet -> bot pump -> quiet) with vis(7)=True, _FAILED(8)=False on every row."""
    res, rows, sec, slot = [1e9, 100.0], [], 0.0, 10
    for c in range(cycles):
        res[0] -= 1000.0; res[1] += 0.2
        rows.append([sec, slot, f"out{c}", -1000.0, 0.2, res[0], res[1], True, False]); sec += 0.05
        for _ in range(5):
            res[0] -= 5000.0; res[1] += 3.0
            rows.append([sec, slot + 1, "bot", -5000.0, 3.0, res[0], res[1], True, False]); sec += 0.05
        for q in range(2, 6):
            res[0] -= 10.0; res[1] += 0.02
            rows.append([sec, slot + q, "bot", -10.0, 0.02, res[0], res[1], True, False]); sec += 0.05
        slot += 6
    return rows


def test_frontrun_include_failed_runs_both_views_and_compares():
    rows = _causal_rows_tagged()
    for r in rows[::7]:                                   # inject FAILED buys from fresh "ghost" wallets
        rows.append([r[0] + 0.01, r[1], f"ghost{r[1]}", -1e-9, 1e-9, 0.0, 0.0, True, True])
    rec = {"txs": rows, "failed_tagged": True, "create_ts": 0}
    pools = [f"P{i}" for i in range(80)]
    have = {p: rec for p in pools}
    metas = [{"pool": p, "token_mint": "M", "create_ts": float(i)} for i, p in enumerate(pools)]
    o_s = fs.frontrun(have, metas, include_failed=False)
    o_f = fs.frontrun(have, metas, include_failed=True)
    assert o_s["include_failed"] is False and o_f["include_failed"] is True
    assert o_f["pools"] == o_s["pools"]                   # failed toggles only the decision features, not the pools
    r = fs.render_frontrun_failed_compare(o_s, o_f)
    assert "FAILED-INCLUSIVE REFIT TEST" in r and "VERDICT" in r


# --- merge: our capture features + Helius price -> --flows recs ---
def test_history_window_filters_by_blocktime_and_paginates():
    h = helius.Helius(api_key="k")
    pages = [
        {"data": [{"slot": 10, "blockTime": 95}, {"slot": 11, "blockTime": 110}], "paginationToken": "t1"},
        {"data": [{"slot": 12, "blockTime": 150}, {"slot": 13, "blockTime": 310}], "paginationToken": "t2"},
    ]
    calls = []

    def fake_rpc(method, params, credits):
        bt = params[1].get("filters", {}).get("blockTime", {})
        calls.append(bt)
        gte = bt.get("gte", 0)
        p = pages[len(calls) - 1] if len(calls) <= len(pages) else {"data": [], "paginationToken": None}
        return {"data": [t for t in p["data"] if t["blockTime"] >= gte],   # server honours gte (lower bound)
                "paginationToken": p.get("paginationToken")}

    h._rpc = fake_rpc
    txs, ok = h.history_window("P", from_ts=100, span_s=180, max_pages=6)   # window [100, 280]
    assert ok is True
    assert [t["slot"] for t in txs] == [11, 12]        # 95 (server-dropped, gte) and 310 (>280, trimmed) gone
    assert calls[0] == {"gte": 100}                    # gte filter sent
    assert len(calls) == 2                             # stopped once a tx passed the window end (310 > 280)


def _wtx(slot, toka, sola, mint="M", pool="P", err=None):
    """A Helius jsonParsed tx carrying the pool's end-of-slot reserves (what history_window returns)."""
    return {"slot": slot, "blockTime": 100 + slot, "meta": {"err": err, "postTokenBalances": [
        {"owner": pool, "mint": mint, "uiTokenAmount": {"uiAmount": toka}},
        {"owner": pool, "mint": fs.he.SOL_MINT, "uiTokenAmount": {"uiAmount": sola}}]}}


def test_merge_capture_flows_overlays_our_counts_on_helius_price():
    cap = [{"pool": "P", "token_mint": "M", "create_ts": 100.0, "slots": [
        [0.0, 10, 1, 0, ["a"]],
        [1.0, 11, 2, 1, ["b", "c"]],
        [2.0, 12, 0, 1, []],
    ]}]
    win_txs = [_wtx(10, 95.0, 6.0), _wtx(11, 90.0, 7.0), _wtx(12, 95.0, 6.5),
               _wtx(13, 80.0, 9.0)]            # a slot our feed MISSED -> price anchor, no our count

    class _H:
        def history_window(self, pool, from_ts, span, pages):
            return (win_txs, True)

    rows, credits = fs.merge_capture_flows(cap, _H(), span_s=180, max_credits=10000, min_cap_slots=0, min_cap_swaps=0)
    assert len(rows) == 1 and rows[0]["pool"] == "P" and rows[0]["create_ts"] == 100.0
    txs = rows[0]["rec"]["txs"]
    s11 = [t for t in txs if t[1] == 11]
    assert len(s11) == 4                                    # anchor + 2 buys + 1 sell
    anchor = next(t for t in s11 if t[3] == 0.0 and t[4] == 0.0)
    assert anchor[5] == 90.0 and anchor[6] == 7.0 and anchor[7] is True
    assert [t[2] for t in s11 if t[3] < 0] == ["b", "c"]   # our buyers kept, in order
    s13 = [t for t in txs if t[1] == 13]
    assert len(s13) == 1 and s13[0][3] == 0.0              # missed slot: anchor only, no our count
    flow = fs.slot_flow(rows[0]["rec"], vis="direct")      # direct view counts our (vis-True) rows
    s11f = next(s for s in flow["slots"] if s["slot"] == 11)
    assert s11f["n_buys"] == 2 and s11f["buyers"] == ["b", "c"] and s11f["n_sells"] == 1
    assert abs(s11f["price"] - 7.0 / 90.0) < 1e-9          # the Helius reserves, not our sentinel


def test_merge_drops_pump_graduations():
    cap = [
        {"pool": "G", "token_mint": "ABCpump", "create_ts": 1.0,
         "slots": [[0, 10, 1, 0, ["a"]], [1, 11, 1, 0, ["b"]], [2, 12, 1, 0, ["c"]]]},
        {"pool": "H", "token_mint": "ABCxyz", "create_ts": 1.0,
         "slots": [[0, 10, 1, 0, ["a"]], [1, 11, 1, 0, ["b"]], [2, 12, 1, 0, ["c"]]]},
    ]

    class _H:
        def history_window(self, pool, from_ts, span, pages):
            return ([_wtx(10, 95.0, 6.0, mint="ABCxyz", pool=pool),
                     _wtx(11, 90.0, 7.0, mint="ABCxyz", pool=pool),
                     _wtx(12, 85.0, 8.0, mint="ABCxyz", pool=pool)], True)

    rows, _ = fs.merge_capture_flows(cap, _H(), span_s=180, max_credits=10000, min_cap_slots=0, min_cap_swaps=0)
    assert [r["pool"] for r in rows] == ["H"]          # graduation G (pump mint) dropped before any fetch


def test_merge_dedups_duplicate_pool_captures_keeping_the_richest():
    # the capture can emit a pool across several windows; the merge must fetch each pool ONCE (richest window).
    cap = [
        {"pool": "P", "token_mint": "Axyz", "create_ts": 1.0,
         "slots": [[0, 10, 1, 0, ["a"]], [1, 11, 1, 0, ["b"]], [2, 12, 1, 0, ["c"]]]},            # thin copy
        {"pool": "P", "token_mint": "Axyz", "create_ts": 1.0,
         "slots": [[i * 0.1, 10 + i, 2, 1, ["x", "y"]] for i in range(8)]},                        # rich copy
    ]
    fetched = []

    class _H:
        def history_window(self, pool, from_ts, span, pages):
            fetched.append(pool)
            return ([_wtx(10 + i, 90.0 - i, 6.0 + i, mint="Axyz", pool=pool) for i in range(8)], True)

    rows, _ = fs.merge_capture_flows(cap, _H(), span_s=180, max_credits=10000, min_cap_slots=6, min_cap_swaps=12)
    assert fetched == ["P"]                       # fetched once, not twice
    assert len(rows) == 1                          # the rich copy survived the activity filter


def test_merge_activity_prefilter_skips_thin_pools_before_fetching():
    # a thin pool (3 slots, 3 swaps) is skipped BEFORE the Helius fetch; an active one is fetched + kept.
    thin_slots = [[0, 10, 1, 0, ["a"]], [1, 11, 1, 0, ["b"]], [2, 12, 1, 0, ["c"]]]
    active_slots = [[i * 0.1, 10 + i, 2, 1, ["x", "y"]] for i in range(8)]   # 8 slots, 24 swaps
    cap = [{"pool": "THIN", "token_mint": "Axyz", "create_ts": 1.0, "slots": thin_slots},
           {"pool": "ACT", "token_mint": "Bxyz", "create_ts": 1.0, "slots": active_slots}]
    fetched = []

    class _H:
        def history_window(self, pool, from_ts, span, pages):
            fetched.append(pool)
            return ([_wtx(10 + i, 90.0 - i, 6.0 + i, mint="Bxyz", pool=pool) for i in range(8)], True)

    rows, _ = fs.merge_capture_flows(cap, _H(), span_s=180, max_credits=10000, min_cap_slots=6, min_cap_swaps=12)
    assert fetched == ["ACT"]                    # THIN never cost a Helius call
    assert [r["pool"] for r in rows] == ["ACT"]
