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
