"""Helius transactionSubscribe -> per-slot FULL aggregate: the rebase-feed core, matched to slot_flow."""
from tradingbot import flurry_study as fl
from tradingbot import helius_feature_ws as hw
from tradingbot.data import helius as he

POOL, MINT = "POOL", "TOKENMINT"


def _tx(slot, payer, tok_pre, tok_post, sol_pre, sol_post, err=None):
    """A Helius tx shaped like a transactionSubscribe notification body ({slot, transaction, meta})."""
    bal = lambda owner, mint, amt: {"owner": owner, "mint": mint, "uiTokenAmount": {"uiAmount": amt}}
    return {"slot": slot, "blockTime": slot,
            "transaction": {"message": {"accountKeys": [payer, "prog", "x"]}},
            "meta": {"err": err,
                     "preTokenBalances": [bal(POOL, MINT, tok_pre), bal(POOL, he.SOL_MINT, sol_pre)],
                     "postTokenBalances": [bal(POOL, MINT, tok_post), bal(POOL, he.SOL_MINT, sol_post)]}}


def test_swap_from_tx_classifies_by_vault_deltas():
    buy = hw.swap_from_tx(_tx(10, "A", 1000, 980, 100, 102), POOL, MINT)   # tokens out, SOL in
    assert buy["slot"] == 10 and buy["payer"] == "A" and buy["tok_delta"] == -20 and buy["sol_delta"] == 2
    assert buy["tok_after"] == 980 and buy["sol_after"] == 102
    sell = hw.swap_from_tx(_tx(11, "B", 980, 990, 102, 101), POOL, MINT)   # tokens in, SOL out
    assert sell["tok_delta"] == 10 and sell["sol_delta"] == -1
    assert hw.swap_from_tx(_tx(12, "C", 990, 970, 101, 103, err={"x": 1}), POOL, MINT) is None   # reverted
    assert hw.swap_from_tx(_tx(13, "D", 970, 970, 103, 103), POOL, MINT) is None                 # no vault change


def test_aggregator_matches_slot_flow_full_wallet_aggregation():
    # a mix of buys and sells across four slots; two buys land in slot 12.
    txs = [
        _tx(10, "A", 1000, 980, 100, 102),       # buy
        _tx(11, "B", 980, 995, 102, 101),        # sell
        _tx(12, "C", 995, 985, 101, 103),        # buy
        _tx(12, "D", 985, 970, 103, 106),        # buy (same slot)
        _tx(13, "E", 970, 985, 106, 105),        # sell
    ]
    # ground truth: launch_record -> slot_flow full
    rec = he.launch_record(txs, POOL, MINT, window_s=1000)
    want = {s["slot"]: s for s in fl.slot_flow(rec, "full")["slots"]}

    got = []
    agg = hw.FullSlotAggregator(got.append)
    for tx in txs:
        sw = hw.swap_from_tx(tx, POOL, MINT)
        if sw:
            agg.push(sw)
    agg.flush()
    got = {s["slot"]: s for s in got}

    assert set(got) == set(want)
    for slot, g in got.items():
        w = want[slot]
        for k in ("n_buys", "n_sells", "buyers"):
            assert g[k] == w[k], (slot, k, g[k], w[k])
        for k in ("buy_sol", "max_buy", "sell_sol"):
            assert abs(g[k] - w[k]) < 1e-9, (slot, k)
    # slot 12 collected both buys
    assert got[12]["n_buys"] == 2 and got[12]["buyers"] == ["C", "D"]


def test_aggregator_feeds_a_hybrid_full_bank_in_slot_order():
    # the aggregate shape is exactly what frontrun_live.SlotBank.push (the full bank) consumes
    from tradingbot import frontrun_live as live
    bank = live.SlotBank()
    got = []
    agg = hw.FullSlotAggregator(lambda s: got.append(bank.push(s)))
    for tx in [_tx(10, "A", 1000, 980, 100, 102), _tx(11, "C", 980, 970, 102, 104)]:
        agg.push(hw.swap_from_tx(tx, POOL, MINT))
    agg.flush()
    assert len(got) == 2 and all("sym" in r and "m2" in r for r in got)


def test_inline_helpers_are_pinned_to_helius():
    # the box runs helius_feature_ws flat (no tradingbot package), so it inlines helius._vault/_keys/SOL_MINT;
    # pin them to the originals so they can never drift.
    assert hw.SOL_MINT == he.SOL_MINT
    tx = _tx(42, "PAYER", 1000, 980, 100, 102)
    for side in ("preTokenBalances", "postTokenBalances"):
        assert hw._vault(tx, POOL, MINT, side) == he._vault(tx, POOL, MINT, side)
        assert hw._vault(tx, POOL, he.SOL_MINT, side) == he._vault(tx, POOL, he.SOL_MINT, side)
    assert hw._keys(tx) == he._keys(tx)
