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
def test_merge_capture_flows_overlays_our_counts_on_helius_price(monkeypatch):
    cap = [{"pool": "P", "token_mint": "M", "create_ts": 100.0, "slots": [
        [0.0, 10, 1, 0, ["a"]],
        [1.0, 11, 2, 1, ["b", "c"]],
        [2.0, 12, 0, 1, []],
    ]}]
    helius_rows = {"txs": [
        [0.0, 10, "x", -5, 1, 95.0, 6.0],
        [1.0, 11, "y", -5, 1, 90.0, 7.0],
        [2.0, 12, "z", 5, -1, 95.0, 6.5],
        [3.0, 13, "w", -5, 2, 80.0, 9.0],   # a slot our feed MISSED -> price anchor, no our count
    ]}
    monkeypatch.setattr(fs.he, "launch_record", lambda *a, **k: helius_rows)

    class _H:
        def history_from_start(self, pool, span, pages):
            return ([], True, True)

    rows, credits = fs.merge_capture_flows(cap, _H(), span_s=180, max_credits=10000)
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
