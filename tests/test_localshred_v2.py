"""pump_amm v2 swaps (buy_v2 / sell_v2 / buy_exact_quote_in_v2) in the box's direct decode.

The big routers' flow moved to v2 (docs/ROUTER_DECODE.md); a v1-only decode_tx dropped every direct v2 swap.
The fixtures at the bottom are REAL instructions from `Router Data/router-raw26.jsonl.gz` (successful txs,
direction confirmed by router_direction.label_raw against the payer's token-balance change)."""
import hashlib

from tradingbot import localshred_consumer as lc
from tradingbot import router_direction as rd
from tradingbot.data import helius


def _anchor(name):
    return hashlib.sha256(("global:" + name).encode()).digest()[:8]


def test_v2_discriminators_are_the_anchor_hashes():
    assert lc.BUY_V2 == _anchor("buy_v2")
    assert lc.SELL_V2 == _anchor("sell_v2")
    assert lc.BUY_EXACT_QUOTE_IN_V2 == _anchor("buy_exact_quote_in_v2")
    # and the v1 set is unchanged
    assert lc.SELL == _anchor("sell") and lc.BUY_BASE_OUT == _anchor("buy")
    assert lc.BUY_EXACT_QUOTE_IN == _anchor("buy_exact_quote_in")


def test_one_definition_shared_with_router_direction_and_helius():
    assert rd.BUY_V2 is lc.BUY_V2 and rd.SELL_V2 is lc.SELL_V2
    assert rd.BUY_EXACT_QUOTE_IN_V2 is lc.BUY_EXACT_QUOTE_IN_V2
    assert lc.SELL_DISCS == helius._SWAP_SELLS
    assert lc.DIRECT_BUY_DISCS == helius._SWAP_BUYS


def _one(disc, base, quote):
    keys = ["PAYER", "POOL", "USER", "GCONF", base, quote]
    swaps = lc.decode_tx(keys, [(lc.PROGRAM, [1, 2, 3, 4, 5], disc + bytes(16))])
    assert len(swaps) == 1
    return swaps[0]


def test_v2_token_base_pool_maps_straight():
    for disc in (lc.BUY_V2, lc.BUY_EXACT_QUOTE_IN_V2):
        sw = _one(disc, "TOKEN", lc.WSOL)
        assert sw.is_buy is True and sw.pool == "POOL" and sw.token == "TOKEN" and sw.payer == "PAYER"
    sw = _one(lc.SELL_V2, "TOKEN", lc.WSOL)
    assert sw.is_buy is False and sw.token == "TOKEN"


def test_v2_sol_base_pool_flips_like_v1():
    assert _one(lc.SELL_V2, lc.WSOL, "TOKEN").is_buy is True        # token BUY arrives as sell_v2
    assert _one(lc.BUY_V2, lc.WSOL, "TOKEN").is_buy is False
    assert _one(lc.BUY_EXACT_QUOTE_IN_V2, lc.WSOL, "TOKEN").is_buy is False


def test_v2_non_sol_pool_and_unknown_disc_still_rejected():
    usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", usdc]
    assert lc.classify_instruction(lc.PROGRAM, [1, 2, 3, 4, 5], lc.BUY_V2 + bytes(16), keys, "PAYER") is None
    keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    near = bytes([lc.BUY_V2[0] ^ 1]) + lc.BUY_V2[1:]
    assert lc.classify_instruction(lc.PROGRAM, [1, 2, 3, 4, 5], near + bytes(16), keys, "PAYER") is None


def test_decode_tx_mixes_v1_and_v2():
    keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    a = [1, 2, 3, 4, 5]
    ins = [(lc.PROGRAM, a, lc.BUY_BASE_OUT + bytes(16)), (lc.PROGRAM, a, lc.SELL_V2 + bytes(16)),
           (lc.PROGRAM, a, lc.BUY_EXACT_QUOTE_IN_V2 + bytes(16)), ("Other", a, lc.BUY_V2 + bytes(16))]
    assert [s.is_buy for s in lc.decode_tx(keys, ins)] == [True, False, True]


class _Rec:
    def __init__(self):
        self.records = []

    def on_slot(self, rec):
        self.records.append(dict(rec))
        return None


def test_consumer_counts_direct_v2_swaps_and_not_as_cpi_drops():
    c = lc.Consumer(gate=None, runner_factory=lambda: _Rec())
    keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    c.on_tx(10, keys, [(lc.PROGRAM, [1, 2, 3, 4, 5], lc.BUY_V2 + bytes(16))])
    c.on_tx(10, ["P2"] + keys[1:], [(lc.PROGRAM, [1, 2, 3, 4, 5], lc.SELL_V2 + bytes(16))])
    assert c.direct_swaps == 2 and c.cpi_txs == 0
    st = c.pools["POOL"]
    assert st.n_buys == 1 and st.n_sells == 1 and st.buyers == ["PAYER"]


# --- real v2 instructions (first 5 accounts + base58 data), from router-raw26 -------------------------------
REAL = [
    ("buy", ["7xtfb2uLcQzaTN1w8EK6PMkdTFiSazKAjWANrJwAUHNn", "FpGQdUdXMGT3RHJuyuSzDweRMEhoEzZ2Sfxs8dbyonUh",
             "ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw", "4qFwbK9Usb7h12BKzjf8edSEbX8oxLhpKJVHLRWnpump",
             "So11111111111111111111111111111111111111112"], "HnQ14hy6fxG8sHim7ky7p7VFT9Bwnejw5"),
    ("sell", ["9yYrfKGoV5QWc2PNotkdkzrVjGoDaDLMC61p44sFE3UR", "7qydQ4hMfauog4intNV71P9x2bgzGVsoQ3zb4qF1ANen",
              "ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw", "4m7aMhBsPEDQJux59WK4s9pCDubyNqamaFGGFtwBpump",
              "So11111111111111111111111111111111111111112"], "9Zq9ZwMbe2bmvPNnZEhZojyGFHMR52def"),
    ("buy", ["9yYrfKGoV5QWc2PNotkdkzrVjGoDaDLMC61p44sFE3UR", "Dnq77xMNwqPTj8NRhXvD827HNJNgQvZpqob3oMFus7tE",
             "ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw", "4m7aMhBsPEDQJux59WK4s9pCDubyNqamaFGGFtwBpump",
             "So11111111111111111111111111111111111111112"], "JkK5BJGr6ZGoRpM96qUVW9q6ZvHNtFjYK"),
]


def test_real_v2_instructions_decode_to_the_balance_confirmed_direction():
    for want, accts, d58 in REAL:
        keys = accts                                   # instruction account i -> account_keys[i]
        payer = accts[1]                               # user (acct 1) == fee payer on these txs
        sw = lc.classify_instruction(lc.PROGRAM, [0, 1, 2, 3, 4], rd._b58decode(d58), keys, payer)
        assert sw is not None and sw.pool == accts[0] and sw.token == accts[3]
        assert sw.is_buy == (want == "buy")


def test_helius_swap_direction_reads_v2_for_failed_txs():
    for want, accts, d58 in REAL:
        tx = {"meta": {"err": {"x": 1}}, "transaction": {"message": {
            "accountKeys": [{"pubkey": accts[1]}],
            "instructions": [{"programId": helius.PUMPSWAP_PROGRAM, "accounts": accts, "data": d58}]}}}
        assert helius.swap_direction(tx) is (want == "buy")


def test_v2_counter_and_info_line(capsys):
    c = lc.Consumer(gate=None, runner_factory=lambda: _Rec())
    c.v2_log_s = 0                                  # no timed print while counting
    keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    for d in (lc.BUY_V2, lc.SELL_V2, lc.BUY_BASE_OUT, lc.SELL):
        c.on_tx(10, keys, [(lc.PROGRAM, [1, 2, 3, 4, 5], d + bytes(16))])
    assert c.v2_swaps == 2 and c.direct_swaps == 4
    assert c.v2_line() == "v2 decoded: 2 of 4 direct swaps (50.0%)"
    assert capsys.readouterr().out == ""
    c.v2_log_s = 60.0
    c._v2_last -= 61                                # due: the next tx prints the line once
    c.on_tx(11, keys, [(lc.PROGRAM, [1, 2, 3, 4, 5], lc.BUY_V2 + bytes(16))])
    c.on_tx(11, keys, [(lc.PROGRAM, [1, 2, 3, 4, 5], lc.BUY_V2 + bytes(16))])
    assert capsys.readouterr().out == "v2 decoded: 3 of 5 direct swaps (60.0%)\n"


def test_v2_flag_on_swaps():
    keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    flags = [lc.decode_tx(keys, [(lc.PROGRAM, [1, 2, 3, 4, 5], d + bytes(16))])[0].v2
             for d in (lc.BUY_V2, lc.SELL_V2, lc.BUY_EXACT_QUOTE_IN_V2, lc.BUY_BASE_OUT, lc.SELL, lc.BUY_EXACT_QUOTE_IN)]
    assert flags == [True, True, True, False, False, False]
