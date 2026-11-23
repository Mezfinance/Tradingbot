from tradingbot import localshred_consumer as lc
from tradingbot import localshred_tap as tap


# Duck-typed stand-ins for a solders VersionedTransaction (message.account_keys / message.instructions,
# each compiled instruction's program_id_index / accounts (bytes of indices) / data (bytes)).
class _Ix:
    def __init__(self, program_id_index, accounts, data):
        self.program_id_index = program_id_index
        self.accounts = bytes(accounts)
        self.data = data


class _Msg:
    def __init__(self, account_keys, instructions):
        self.account_keys = account_keys
        self.instructions = instructions


class _Tx:
    def __init__(self, account_keys, instructions):
        self.message = _Msg(account_keys, instructions)


def test_tx_view_resolves_program_id_and_account_indices():
    keys = ["PAYER", lc.PROGRAM, "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    #                 ^prog@1   ^pool@2                    ^base@5  ^quote@6
    ix = _Ix(1, [2, 3, 4, 5, 6], lc.SELL + bytes(16))     # a pumpswap swap: pool@2, base_mint@5
    account_keys, instructions, fee_payer = tap.tx_view(_Tx(keys, [ix]))
    assert fee_payer == "PAYER"
    assert instructions == [(lc.PROGRAM, [2, 3, 4, 5, 6], lc.SELL + bytes(16))]


def test_tx_view_skips_lut_only_program_index():
    keys = ["PAYER", "POOL"]                               # only 2 static keys
    ix = _Ix(5, [0, 1], b"\x00" * 8)                       # program_id_index 5 is out of range (LUT)
    _, instructions, _ = tap.tx_view(_Tx(keys, [ix]))
    assert instructions == []                              # skipped, not an exception


def test_on_transaction_decodes_and_routes_into_the_consumer():
    keys = ["PAYER", lc.PROGRAM, "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    accts = [2, 3, 4, 5, 6]
    # buy_exact_quote_in -> a token BUY; a non-pumpswap ix is ignored by decode_tx.
    ixs = [_Ix(0, accts, b"\x00" * 24),                    # program_id_index 0 = "PAYER", not pumpswap
           _Ix(1, accts, lc.BUY_EXACT_QUOTE_IN + bytes(16))]
    rec = []

    class _Rec:
        def on_slot(self, r):
            rec.append(dict(r))
            return None

    c = lc.Consumer(gate={}, runner_factory=lambda: _Rec())
    tap.on_transaction(c, 42, _Tx(keys, ixs))
    tap.on_transaction(c, 43, _Tx(keys, ixs))             # advances the pool -> flush slot 42
    assert len(rec) == 1
    assert rec[0]["slot"] == 42 and rec[0]["n_buys"] == 1 and rec[0]["buyers"] == ["PAYER"]


# --- LUT resolution (the fuller decode): v0 message with address_table_lookups ---

class _Lk:
    def __init__(self, account_key, writable_indexes, readonly_indexes):
        self.account_key = account_key
        self.writable_indexes = writable_indexes
        self.readonly_indexes = readonly_indexes


class _MsgV0:
    def __init__(self, account_keys, instructions, lookups):
        self.account_keys = account_keys
        self.instructions = instructions
        self.address_table_lookups = lookups


class _TxV0:
    def __init__(self, account_keys, instructions, lookups):
        self.message = _MsgV0(account_keys, instructions, lookups)


def test_lut_indexed_swap_is_skipped_without_a_resolver():
    # pool/user/gconf/base/quote all live in a LUT (indices 2..6 are past the 2 static keys).
    keys = ["PAYER", lc.PROGRAM]
    lk = _Lk("TABLE", [0, 1], [2, 3, 4])
    ix = _Ix(1, [2, 3, 4, 5, 6], lc.SELL + bytes(16))
    account_keys, instructions, _ = tap.tx_view(_TxV0(keys, [ix], [lk]))   # resolve=None (default)
    assert account_keys == keys and instructions == []                     # LUT-blind: skipped, no crash


def test_lut_resolver_recovers_the_swap():
    keys = ["PAYER", lc.PROGRAM]
    lk = _Lk("TABLE", [0, 1], [2, 3, 4])                                   # writable: pool,user; readonly: gconf,base,quote
    table = ["POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    ix = _Ix(1, [2, 3, 4, 5, 6], lc.SELL + bytes(16))

    def resolve(t):
        return table if t == "TABLE" else None

    account_keys, instructions, fee_payer = tap.tx_view(_TxV0(keys, [ix], [lk]), resolve)
    assert account_keys == ["PAYER", lc.PROGRAM, "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    assert instructions == [(lc.PROGRAM, [2, 3, 4, 5, 6], lc.SELL + bytes(16))]
    swaps = lc.decode_tx(account_keys, instructions, fee_payer)
    assert len(swaps) == 1
    assert swaps[0].pool == "POOL" and swaps[0].token == "TOKEN" and swaps[0].payer == "PAYER"
    assert swaps[0].is_buy is False                                        # SELL of a token-base pool


def test_lut_order_is_static_then_all_writable_then_all_readonly():
    lk1 = _Lk("T1", [0], [1])
    lk2 = _Lk("T2", [0], [1])

    def resolve(t):
        return {"T1": ["W1", "R1"], "T2": ["W2", "R2"]}.get(t)

    keys, ok = tap.resolve_account_keys(_MsgV0(["S0"], [], [lk1, lk2]), resolve)
    assert ok is True
    assert keys == ["S0", "W1", "W2", "R1", "R2"]                          # Solana order: writable before readonly


def test_partial_lut_resolution_falls_back_to_static():
    lk1, lk2 = _Lk("T1", [0], []), _Lk("T2", [0], [])

    def resolve(t):
        return ["X"] if t == "T1" else None                               # T2 unresolved

    keys, ok = tap.resolve_account_keys(_MsgV0(["S0"], [], [lk1, lk2]), resolve)
    assert ok is False and keys == ["S0"]                                 # no partial resolve -> no index corruption


def test_lut_index_out_of_table_range_falls_back():
    lk = _Lk("T", [5], [])                                                # index 5 but the table has 1 entry
    keys, ok = tap.resolve_account_keys(_MsgV0(["S0"], [], [lk]), lambda t: ["only"])
    assert ok is False and keys == ["S0"]


def test_resolver_unused_on_a_legacy_message():
    keys = ["PAYER", lc.PROGRAM, "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    ix = _Ix(1, [2, 3, 4, 5, 6], lc.SELL + bytes(16))
    # a legacy _Tx has no address_table_lookups -> resolve is irrelevant, full static keys
    account_keys, instructions, _ = tap.tx_view(_Tx(keys, [ix]), resolve=lambda t: ["junk"])
    assert account_keys == keys and instructions == [(lc.PROGRAM, [2, 3, 4, 5, 6], lc.SELL + bytes(16))]


# --- box-path tests that use real solders objects (skip where solders isn't installed) ---

def _le(n):
    return int(n).to_bytes(8, "little")


def _entry_bytes(txs):
    # solana_entry::Entry = num_hashes:u64 + hash:[u8;32] + Vec<VersionedTransaction>
    return _le(0) + bytes(32) + _le(len(txs)) + b"".join(bytes(t) for t in txs)


def _vec_entries(entries):
    return _le(len(entries)) + b"".join(entries)


class _StreamMsg:
    def __init__(self, slot, entries):
        self.slot = slot
        self.entries = entries


def _empty_tx():
    from solders.keypair import Keypair
    from solders.message import Message
    from solders.hash import Hash
    from solders.transaction import VersionedTransaction
    kp = Keypair()
    return VersionedTransaction(Message.new_with_blockhash([], kp.pubkey(), Hash.default()), [kp])


def _pumpswap_sell_tx():
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.instruction import Instruction, AccountMeta
    from solders.message import Message
    from solders.hash import Hash
    from solders.transaction import VersionedTransaction
    payer = Keypair()
    pool, user, gconf, token = (Pubkey.new_unique() for _ in range(4))
    wsol = Pubkey.from_string(lc.WSOL)
    metas = [AccountMeta(pool, False, True), AccountMeta(user, False, True),
             AccountMeta(gconf, False, False), AccountMeta(token, False, False),   # base_mint @ acct 3
             AccountMeta(wsol, False, False)]                                       # quote_mint @ acct 4
    ix = Instruction(Pubkey.from_string(lc.PROGRAM), lc.SELL + bytes(16), metas)
    tx = VersionedTransaction(Message.new_with_blockhash([ix], payer.pubkey(), Hash.default()), [payer])
    return tx, str(pool), str(payer.pubkey())


def test_deserialize_entries_round_trips_a_real_vec_of_entries(pytestconfig):
    import pytest
    pytest.importorskip("solders")
    tx0, tx1 = _empty_tx(), _empty_tx()
    buf = _vec_entries([_entry_bytes([tx0]), _entry_bytes([tx1])])   # 2 entries, one tx each
    got = [bytes(t) for t in tap.deserialize_entries(buf)]
    assert got == [bytes(tx0), bytes(tx1)]
    # and a single entry carrying two txs
    buf2 = _vec_entries([_entry_bytes([tx0, tx1])])
    assert [bytes(t) for t in tap.deserialize_entries(buf2)] == [bytes(tx0), bytes(tx1)]


def test_consume_stream_decodes_a_real_pumpswap_sell_end_to_end():
    import pytest
    pytest.importorskip("solders")
    tx, pool_addr, payer_addr = _pumpswap_sell_tx()
    buf = _vec_entries([_entry_bytes([tx])])
    rec = []

    class _Rec:
        def on_slot(self, r):
            rec.append(dict(r))
            return None

    c = lc.Consumer(gate={}, runner_factory=lambda: _Rec())
    stream = [_StreamMsg(100, buf), _StreamMsg(101, buf)]   # second advances the pool -> flush slot 100
    tap.consume_stream(stream, c)
    assert len(rec) == 1
    r = rec[0]
    assert r["slot"] == 100 and r["n_sells"] == 1 and r["n_buys"] == 0 and r["buyers"] == []
