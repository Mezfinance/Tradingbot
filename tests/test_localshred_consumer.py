from tradingbot import localshred_consumer as lc
from tradingbot import precursor_gate


# account_keys layout used by the builders: [payer, pool, user, gconf, mint3, mint4, ...]
def _tx(disc, base_mint, quote_mint, pool="POOL", payer="PAYER"):
    keys = [payer, pool, "USER", "GCONF", base_mint, quote_mint]
    accts = [1, 2, 3, 4, 5]                       # pool@idx1, base_mint@idx4, quote_mint@idx5
    data = disc + bytes(16)                       # discriminator + amount_in + min_out (values unused)
    return keys, [(lc.PROGRAM, accts, data)]


def _classify_one(disc, base, quote):
    keys, ins = _tx(disc, base, quote)
    swaps = lc.decode_tx(keys, ins)
    assert len(swaps) == 1
    return swaps[0]


def test_token_base_pool_maps_instruction_kind_straight():
    # token is base (base_mint != WSOL): buy ix -> token buy, sell ix -> token sell.
    buy = _classify_one(lc.BUY_EXACT_QUOTE_IN, "TOKEN", lc.WSOL)
    assert buy.is_buy is True and buy.pool == "POOL" and buy.token == "TOKEN" and buy.payer == "PAYER"
    sell = _classify_one(lc.SELL, "TOKEN", lc.WSOL)
    assert sell.is_buy is False and sell.token == "TOKEN"


def test_sol_base_pool_flips_so_a_token_buy_is_the_sell_instruction():
    # WSOL is the base (token_is_base False): the box flip -> a token BUY arrives as the `sell` ix.
    sell_ix = _classify_one(lc.SELL, lc.WSOL, "TOKEN")
    assert sell_ix.is_buy is True and sell_ix.token == "TOKEN"     # token BUY
    buy_ix = _classify_one(lc.BUY_EXACT_QUOTE_IN, lc.WSOL, "TOKEN")
    assert buy_ix.is_buy is False and buy_ix.token == "TOKEN"      # token SELL


def test_non_pumpswap_and_malformed_are_ignored():
    keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    assert lc.classify_instruction("OtherProgram", [1, 2, 3, 4, 5], lc.SELL + bytes(16), keys, "PAYER") is None
    assert lc.classify_instruction(lc.PROGRAM, [1, 2, 3, 4, 5], b"\x00\x01", keys, "PAYER") is None   # short data
    assert lc.classify_instruction(lc.PROGRAM, [1, 2, 3], lc.SELL + bytes(16), keys, "PAYER") is None  # too few accts
    assert lc.classify_instruction(lc.PROGRAM, [1, 2, 3, 4, 5], bytes(8) + bytes(16), keys, "PAYER") is None  # unknown disc


def test_decode_tx_keeps_only_top_level_pumpswap_swaps():
    keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
    accts = [1, 2, 3, 4, 5]
    ins = [
        ("OtherProgram", accts, b"\x00" * 24),                 # dropped
        (lc.PROGRAM, accts, lc.BUY_EXACT_QUOTE_IN + bytes(16)),  # a token buy
        (lc.PROGRAM, accts, lc.SELL + bytes(16)),               # a token sell
    ]
    swaps = lc.decode_tx(keys, ins)
    assert [s.is_buy for s in swaps] == [True, False]
    assert all(s.payer == "PAYER" for s in swaps)


class _RecRunner:
    """Captures the per-slot records fed to on_slot and fires on a chosen set of slots."""
    def __init__(self, fire_on=()):
        self.records = []
        self.fire_on = set(fire_on)

    def on_slot(self, rec):
        self.records.append(dict(rec))
        if rec["slot"] in self.fire_on:
            return {"signal_slot": rec["slot"], "entry_slot": rec["slot"] + 1,
                    "exit_slot": rec["slot"] + 2, "hold": 2, "tier": "1", "eta": -3.0,
                    "sym": "S", "m2": "BS"}
        return None


def test_per_slot_aggregate_and_flush_on_slot_advance():
    rec = _RecRunner()
    c = lc.Consumer(gate={}, runner_factory=lambda: rec)
    # slot 10: two buys (payers A, B) + one sell; slot 11 opens -> slot 10 flushes.
    c.on_swap(10, "P", "TOK", True, "A")
    c.on_swap(10, "P", "TOK", True, "B")
    c.on_swap(10, "P", "TOK", False, "X")
    assert rec.records == []                                   # nothing finalized until the pool advances
    c.on_swap(11, "P", "TOK", False, "Y")                      # advances -> flush slot 10
    assert len(rec.records) == 1
    r = rec.records[0]
    assert r["slot"] == 10 and r["n_buys"] == 2 and r["n_sells"] == 1 and r["buyers"] == ["A", "B"]
    assert r["price"] == 0.0 and r["buy_sol"] == 0.0           # gate is price/magnitude free
    c.flush_all()                                              # slot 11: 0 buys / 1 sell
    assert rec.records[1]["slot"] == 11 and rec.records[1]["n_buys"] == 0 and rec.records[1]["n_sells"] == 1


def test_each_pool_gets_its_own_runner_and_slot():
    made = []

    def factory():
        r = _RecRunner()
        made.append(r)
        return r

    c = lc.Consumer(gate={}, runner_factory=factory)
    c.on_swap(5, "P1", "T1", True, "A")
    c.on_swap(5, "P2", "T2", True, "B")
    assert len(made) == 2                                      # one Runner per pool
    c.flush_all()
    assert made[0].records[0]["buyers"] == ["A"]
    assert made[1].records[0]["buyers"] == ["B"]


def test_on_fire_gets_pool_token_and_slot():
    seen = []
    rec = _RecRunner(fire_on={10})
    c = lc.Consumer(gate={}, on_fire=seen.append, runner_factory=lambda: rec)
    c.on_swap(10, "POOLX", "TOKX", False, "A")                 # net sellers
    c.on_swap(11, "POOLX", "TOKX", False, "B")                 # advance -> flush slot 10 -> fires
    assert len(seen) == 1
    line = lc.fire_line(seen[0])
    assert line == {"pool": "POOLX", "slot": 10, "eta": -3.0, "sym": "S", "m2": "BS",
                    "tier": "1", "token_mint": "TOKX"}


def test_end_to_end_through_a_real_precursor_gate():
    # A minimal real gate: fire (eta >= 0.5) exactly when sym == 'S'. Proves decode -> aggregate ->
    # frontrun_live.Runner -> precursor_gate compose and emit a box-shaped fire.
    gate = precursor_gate.compile_gate({
        "intercept": 0.0, "hold": 2, "threshold": {"1": 0.5},
        "predictors": [{"coef": 1.0, "name": "sym == S  (net sellers)",
                        "predicate": {"kind": "symbol", "symv": "S"}}],
    })
    fires = []
    c = lc.Consumer(gate=gate, tier="1", on_fire=fires.append)
    # pool P, slot 10: one sell, zero buys -> sym 'S'; slot 11 (another sell) closes slot 10.
    keys, ins = _tx(lc.SELL, "TOKEN", lc.WSOL, pool="P")       # token_is_base True, sell ix -> token sell
    c.on_tx(10, keys, ins)
    c.on_tx(11, keys, ins)
    assert len(fires) == 1
    assert fires[0]["pool"] == "P" and fires[0]["sym"] == "S" and fires[0]["signal_slot"] == 10
    assert fires[0]["entry_slot"] == 11 and fires[0]["tier"] == "1"
