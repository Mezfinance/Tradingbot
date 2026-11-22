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


def test_buy_base_out_is_decoded_as_a_token_buy():
    # pump_amm `buy` (buy_base_out) is the dominant buy path; a TOKEN/WSOL pool -> a token BUY.
    buy = _classify_one(lc.BUY_BASE_OUT, "TOKEN", lc.WSOL)
    assert buy.is_buy is True and buy.token == "TOKEN"
    # and on a SOL-base pool it flips (token buy arrives as... still buy_base_out here, WSOL is base):
    sol_base = _classify_one(lc.BUY_BASE_OUT, lc.WSOL, "TOKEN")
    assert sol_base.is_buy is False and sol_base.token == "TOKEN"   # inst_buy True, token_is_base False -> sell


def test_non_sol_paired_pool_is_ignored():
    # a USDC/token pool (neither side WSOL) is not a SOL-paired pool the box tracks -> None.
    usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", usdc]
    assert lc.classify_instruction(lc.PROGRAM, [1, 2, 3, 4, 5], lc.BUY_BASE_OUT + bytes(16), keys, "PAYER") is None
    # two WSOL sides (degenerate) also rejected
    keys2 = ["PAYER", "POOL", "USER", "GCONF", lc.WSOL, lc.WSOL]
    assert lc.classify_instruction(lc.PROGRAM, [1, 2, 3, 4, 5], lc.SELL + bytes(16), keys2, "PAYER") is None


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


class _AlwaysFire:
    """A runner that fires on every slot -- isolates the Consumer's freshness gate from the real gate."""
    def on_slot(self, rec):
        return {"signal_slot": rec["slot"], "entry_slot": rec["slot"] + 1, "exit_slot": rec["slot"] + 2,
                "hold": 2, "tier": "1", "eta": -3.0, "sym": "S", "m2": "BS"}


def test_freshness_gate_fires_only_on_fresh_pools_after_warmup():
    fires = []
    c = lc.Consumer(gate={}, on_fire=fires.append, runner_factory=lambda: _AlwaysFire(),
                    fresh_window_slots=10)
    # OLD: present from the first slot (start_slot=100), so it's mid-life -> never eligible.
    c.on_swap(100, "OLD", "T", False, "a")
    c.on_swap(105, "OLD", "T", False, "a")   # flush OLD@100: first_slot-start = 0 < 10 (warm-up) -> suppressed
    c.on_swap(118, "OLD", "T", False, "a")   # flush OLD@105: still < 10 -> suppressed
    # NEW: first seen at 115, after the 10-slot warm-up (115-100=15) -> eligible for its first 10 slots.
    c.on_swap(115, "NEW", "U", False, "b")
    c.on_swap(118, "NEW", "U", False, "b")   # flush NEW@115: age 0 <= 10 -> FIRES
    c.on_swap(130, "NEW", "U", False, "b")   # flush NEW@118: age 3 <= 10 -> FIRES
    c.on_swap(145, "NEW", "U", False, "b")   # flush NEW@130: age 15 > 10 -> suppressed (outside fresh window)
    c.flush_all()                            # OLD@118 suppressed, NEW@145 age 30 suppressed
    assert [f["pool"] for f in fires] == ["NEW", "NEW"]
    assert sorted(f["signal_slot"] for f in fires) == [115, 118]


def test_freshness_gate_off_by_default_fires_on_everything():
    fires = []
    c = lc.Consumer(gate={}, on_fire=fires.append, runner_factory=lambda: _AlwaysFire())  # window 0 = off
    c.on_swap(100, "OLD", "T", False, "a")
    c.on_swap(105, "OLD", "T", False, "a")   # flush OLD@100 -> fires (no freshness gate)
    assert len(fires) == 1 and fires[0]["pool"] == "OLD"


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


# --- capture mode (fit the study on our own feed) ---
class _Clock:
    def __init__(self, t=1000.0):
        self.t = t
    def __call__(self):
        return self.t


def test_capture_sink_skips_warmup_and_emits_fresh_pools():
    out = []
    clk = _Clock(1000.0)
    cap = lc.CaptureSink(out.append, span_s=10, warm_s=5, min_slots=3, clock=clk, wall=lambda: 42.0)
    clk.t = 1002.0                                     # inside the warm-up (start 1000, warm until 1005)
    cap.on_capture("EARLY", "T", {"slot": 1, "n_buys": 1, "n_sells": 0, "buyers": ["a"]})
    assert "EARLY" not in cap.pools                    # skipped (maybe mid-life)
    clk.t = 1006.0                                      # after warm-up: a genuinely fresh pool
    for s in range(3):
        cap.on_capture("FRESH", "TOK", {"slot": 10 + s, "n_buys": 2, "n_sells": 1, "buyers": ["x", "y"]})
    assert len(cap.pools["FRESH"]["slots"]) == 3
    clk.t = 1022.0                                      # FRESH ages past span+5 -> emitted on the next capture
    cap.on_capture("OTHER", "T2", {"slot": 99, "n_buys": 1, "n_sells": 0, "buyers": ["z"]})
    assert len(out) == 1
    r = out[0]
    assert r["pool"] == "FRESH" and r["token_mint"] == "TOK" and r["create_ts"] == 42.0
    assert len(r["slots"]) == 3
    assert r["slots"][0][1] == 10 and r["slots"][0][2] == 2 and r["slots"][0][3] == 1 and r["slots"][0][4] == ["x", "y"]


def test_capture_sink_drops_pools_with_too_few_slots():
    out = []
    clk = _Clock(1000.0)
    cap = lc.CaptureSink(out.append, span_s=10, warm_s=0, min_slots=3, clock=clk, wall=lambda: 1.0)
    clk.t = 1001.0
    cap.on_capture("P", "T", {"slot": 1, "n_buys": 1, "n_sells": 0, "buyers": ["a"]})
    cap.close()
    assert out == []                                   # 1 slot < min_slots -> not emitted


def test_consumer_on_capture_fires_per_finalized_slot():
    recs = []
    c = lc.Consumer(gate={}, runner_factory=lambda: _RecRunner(),
                    on_capture=lambda pool, tok, r: recs.append((pool, tok, r["slot"], r["n_buys"], r["n_sells"])))
    c.on_swap(10, "P", "TOK", True, "a")
    c.on_swap(10, "P", "TOK", False, "s")
    c.on_swap(11, "P", "TOK", True, "b")               # advance -> flush slot 10 -> capture
    assert recs == [("P", "TOK", 10, 1, 1)]


# --- hand-made filter (pump.fun graduations are out of the study's population) ---
def test_is_handmade_flags_pump_mints():
    assert lc.is_handmade("ABCxyz") is True
    assert lc.is_handmade("SoMeMintpump") is False
    assert lc.is_handmade(None) is False and lc.is_handmade("") is False


def test_consumer_skips_pump_graduations_by_default():
    recs = []
    c = lc.Consumer(gate={}, runner_factory=lambda: _RecRunner(),
                    on_capture=lambda p, t, r: recs.append(p))
    c.on_swap(10, "GRAD", "ABCpump", True, "a")       # a graduation (pump mint) -> skipped
    c.on_swap(11, "GRAD", "ABCpump", True, "b")
    assert recs == [] and "GRAD" not in c.pools
    c.on_swap(10, "HM", "ABCxyz", True, "a")          # a hand-made pool (non-pump) -> processed
    c.on_swap(11, "HM", "ABCxyz", True, "b")
    assert recs == ["HM"]
