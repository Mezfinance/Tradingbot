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
