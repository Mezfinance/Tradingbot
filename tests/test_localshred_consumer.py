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


def test_arm_gate_fires_only_on_armed_pools():
    # The box's Helius sec-4 hand-made read fills `armed`; is_armed reads it. True fires; False (confirmed not
    # hand-made / pulled) and None (not read yet) suppress -- and the bank keeps advancing while unarmed, so a
    # pool armed LATE fires on its later slots.
    fires = []
    armed = {"YES": True, "NO": False}       # LATE absent -> is_armed returns None (undecided)
    c = lc.Consumer(gate={}, on_fire=fires.append, runner_factory=lambda: _AlwaysFire(),
                    is_armed=lambda p: armed.get(p))
    c.on_swap(10, "YES", "T", True, "a")
    c.on_swap(11, "YES", "T", True, "a")     # flush YES@10 -> armed -> FIRES
    c.on_swap(10, "NO", "U", True, "b")
    c.on_swap(11, "NO", "U", True, "b")      # flush NO@10 -> confirmed not armed -> suppressed
    c.on_swap(10, "LATE", "V", True, "c")
    c.on_swap(11, "LATE", "V", True, "c")    # flush LATE@10 -> undecided (None) -> suppressed
    armed["LATE"] = True                      # the sec-4 read lands
    c.on_swap(12, "LATE", "V", True, "c")    # flush LATE@11 -> now armed -> FIRES (history kept building)
    assert [f["pool"] for f in fires] == ["YES", "LATE"]
    assert sorted(f["signal_slot"] for f in fires) == [10, 11]


def test_arm_gate_off_by_default_fires_on_everything():
    fires = []
    c = lc.Consumer(gate={}, on_fire=fires.append, runner_factory=lambda: _AlwaysFire())  # is_armed None = off
    c.on_swap(10, "P", "T", True, "a")
    c.on_swap(11, "P", "T", True, "a")       # no arm gate -> fires
    assert len(fires) == 1 and fires[0]["pool"] == "P"


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


def test_fire_consumer_wires_the_gate_and_a_fires_writer(tmp_path):
    import json
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({
        "intercept": -10.0, "predictors": [{"coef": 20.0, "name": "sym == N"}],
        "threshold": {"1": 0.0}, "hold": 2, "vis": "direct"}))
    fires_path = tmp_path / "fires.jsonl"
    c = lc.fire_consumer(str(gate_path), str(fires_path), fresh_window_slots=0, tier="1")
    assert c.on_fire is not None and c.handmade_only is True and c.fresh_window_slots == 0
    assert "_terms" in c.gate                                   # gate loaded + compiled
    # the wired on_fire writer appends exactly the box front-fires.jsonl record
    c.on_fire({"pool": "P", "signal_slot": 7, "eta": 1.2, "sym": "N", "m2": ".N",
               "tier": "1", "token_mint": "Mxyz"})
    rec = json.loads(fires_path.read_text().strip())
    assert rec == {"pool": "P", "slot": 7, "eta": 1.2, "sym": "N", "m2": ".N",
                   "tier": "1", "token_mint": "Mxyz"}


def test_cpi_drop_counter_counts_invisible_routed_swaps():
    # direct_swaps = top-level pumpswap swaps we decode; cpi_txs = pumpswap invoked via CPI (program id in the
    # account list, but no top-level pumpswap instruction) -> a swap this feed drops. A gauge of the CPI gap.
    c = lc.Consumer(gate={}, runner_factory=lambda: _AlwaysFire())
    keys, ins = _tx(lc.SELL, "TOKEN", lc.WSOL, pool="P")          # a direct top-level pumpswap swap
    c.on_tx(10, keys, ins)
    assert c.direct_swaps == 1 and c.cpi_txs == 0
    # pumpswap is in the account list but the only top-level instruction is a router -> CPI-routed, dropped
    cpi_keys = ["PAYER", "ROUTER", lc.PROGRAM, "POOL", "TOKEN", lc.WSOL]
    cpi_ins = [("JupiterRouter", [0, 1, 2, 3], b"\x00" * 16)]
    assert c.on_tx(11, cpi_keys, cpi_ins) == []                  # nothing decoded
    assert c.cpi_txs == 1 and c.direct_swaps == 1                # one CPI counted, direct unchanged
    # a tx with neither pumpswap top-level nor pumpswap in the keys is not a drop
    c.on_tx(12, ["A", "B", "C"], [("Other", [0, 1], b"\x00" * 8)])
    assert c.cpi_txs == 1


def _cpi_router_tx(verdict_prog="ROUTER", pool="POOL", payer="PAYER"):
    # a routed tx: top-level instruction is the router's; pumpswap is only in the account list (CPI), and the
    # pool is one of the router instruction's accounts
    keys = [payer, verdict_prog, lc.PROGRAM, pool, "TOKENMINT", lc.WSOL]
    ins = [(verdict_prog, [3, 4, 5], b"\x01" * 8)]           # references the pool at account index 3
    return keys, ins


def test_routed_recovery_off_by_default():
    c = lc.Consumer(gate={}, runner_factory=lambda: _AlwaysFire())   # route_classify None
    keys, ins = _cpi_router_tx()
    c.add_pool("POOL", "TOKENMINT")
    c.on_tx(5, keys, ins)
    assert c.cpi_txs == 1 and "POOL" not in c.pools          # counted, but nothing recovered


def test_routed_recovery_true_direction_and_buyer():
    seen = {}
    c = lc.Consumer(gate={}, runner_factory=lambda: _AlwaysFire(),
                    route_classify=lambda prog, data, accts, keys, token: seen.get("v", "buy"))
    c.add_pool("POOL", "TOKENMINT")
    keys, ins = _cpi_router_tx()
    seen["v"] = "buy"
    c.on_tx(5, keys, ins)
    st = c.pools["POOL"]
    assert st.n_buys == 1 and st.buyers == ["PAYER"] and st.n_sells == 0
    seen["v"] = "sell"
    c.on_tx(5, *_cpi_router_tx(payer="W2"))                  # same slot, a routed sell
    assert st.n_sells == 1 and st.buyers == ["PAYER"]        # seller not added as a buyer


def test_routed_recovery_passes_the_pool_token_to_the_classifier():
    got = {}
    def _rc(prog, data, accts, keys, token):
        got["token"] = token
        return "buy"
    c = lc.Consumer(gate={}, runner_factory=lambda: _AlwaysFire(), route_classify=_rc)
    c.add_pool("POOL", "TOKENMINT")
    c.on_tx(5, *_cpi_router_tx())
    assert got["token"] == "TOKENMINT" and c.pools["POOL"].n_buys == 1   # token_order rules can use it


def test_routed_recovery_buyer_plus_counts_a_buy_and_drop_skips():
    v = {"x": "buyer_plus"}
    c = lc.Consumer(gate={}, runner_factory=lambda: _AlwaysFire(),
                    route_classify=lambda *a: v["x"])
    c.add_pool("POOL", "TOKENMINT")
    keys, ins = _cpi_router_tx()
    c.on_tx(5, keys, ins)
    assert c.pools["POOL"].n_buys == 1                       # buyer_plus -> a buy
    v["x"] = "drop"
    c.on_tx(5, *_cpi_router_tx(payer="W3"))
    assert c.pools["POOL"].n_buys == 1                       # drop -> nothing added


def test_routed_recovery_needs_a_tracked_pool():
    c = lc.Consumer(gate={}, runner_factory=lambda: _AlwaysFire(),
                    route_classify=lambda *a: "buy")         # pool NOT added
    keys, ins = _cpi_router_tx()
    c.on_tx(5, keys, ins)
    assert "POOL" not in c.pools and c.cpi_txs == 1          # unknown pool -> not recovered


def test_add_pool_bootstraps_a_routed_first_pool_and_fires():
    fires = []
    c = lc.Consumer(gate={}, on_fire=fires.append, runner_factory=lambda: _AlwaysFire(),
                    route_classify=lambda *a: "buy")
    c.add_pool("POOL", "TOKENMINT")                          # discovered before any swap
    keys, ins = _cpi_router_tx()
    c.on_tx(5, keys, ins)                                    # routed buy in slot 5 (pool bootstrapped)
    c.on_tx(6, *_cpi_router_tx(payer="W2"))                 # slot advances -> flush slot 5 -> _AlwaysFire fires
    assert len(fires) == 1 and fires[0]["pool"] == "POOL"


def test_router_breakdown_tallies_the_invoking_aggregator():
    # Phase 0: a dropped-CPI tx is attributed to the non-infra top-level program that carried it, so
    # router_top() sizes which aggregators a CPI-aware decoder must cover.
    c = lc.Consumer(gate={}, runner_factory=lambda: _AlwaysFire())
    JUP = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"
    CB = "ComputeBudget111111111111111111111111111111"
    keys = ["PAYER", CB, JUP, lc.PROGRAM, "POOL", "TOKEN", lc.WSOL]
    # top level: a compute-budget ix (infra, ignored) and a Jupiter route (the real router); pumpswap only via CPI
    ins = [(CB, [1], b"\x02\x00\x00\x00"), (JUP, [0, 2, 3, 4, 5, 6], b"\x00" * 16)]
    c.on_tx(10, keys, ins)
    assert c.cpi_txs == 1 and c.cpi_routers == {JUP: 1}            # infra + pumpswap excluded, Jupiter tallied
    c.on_tx(11, keys, ins)
    assert c.router_top()[0] == (JUP, 2)
