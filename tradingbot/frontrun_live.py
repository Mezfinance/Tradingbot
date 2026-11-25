"""Live front-run runner core: turn a per-slot ShredStream feed into a gate decision.

The study fits the precursor gate on CAUSAL _slot_bank features (flurry_study, CAUSAL_WALLET forced on) and
exports it (flurry_study.export_gate -> gate.json). This module rebuilds the same per-slot record ONLINE, one
active slot at a time, and scores the exported gate with tradingbot.precursor_gate -- so a slot fires live
exactly when it fired in the backtest (SlotBank is parity-tested against flurry_study._slot_bank). On a fire it
emits a FireSignal naming the entry and exit SLOT NUMBERS; the box owns execution (land a buy at the front of
N+1, sell at N+hold) and NET fill logging, because that lives in the ShredStream executor (absfast_tight.py),
not here. Pure python, no numpy.

The per-slot aggregate the box feeds (one active slot, in slot order, same as flurry_study.slot_flow emits):
    {"slot": int, "price": float>0 (effective vq reserve ratio at END of the slot, carry-forward if no trade
      left a readable reserve), "n_buys": int, "buy_sol": float, "max_buy": float, "n_sells": int,
      "sell_sol": float, "buyers": [wallet, ...] (buy-side payers, in order), "net_sol": optional}

Box loop sketch (pseudocode):
    gate = precursor_gate.load("gate.json")
    runner = frontrun_live.Runner(gate, tier="1")          # top 1% of the joint logit
    for slot_agg in shredstream_active_slots():            # as each slot N closes
        sig = runner.on_slot(slot_agg)
        if sig:                                            # land a buy at the FRONT of sig["entry_slot"]=N+1, hold
            executor.front_run(sig)                        # to sig["exit_slot"]=N+hold, log swap_return + net_return
"""
from collections import Counter

try:                                  # in-package (tradingbot.frontrun_live) and standalone on the box both work
    from . import precursor_gate
except ImportError:                   # noqa: E722 -- box copy sits next to precursor_gate.py
    import precursor_gate

DEFAULT_FLURRY_MIN = 3   # the 'F' symbol threshold flurry_study.frontrun uses (DEFAULT_FLURRY_MIN); only affects sym==F


class SlotBank:
    """Online, causal reconstruction of a flurry_study._slot_bank record. Push one active slot at a time, in slot
    order; each push returns the record for that slot. Mirrors _slot_bank's loop body with CAUSAL_WALLET on
    (wallet ranks from a running counter through slot N only -- no look-ahead), so live records == study records."""

    def __init__(self, flurry_min: int = DEFAULT_FLURRY_MIN):
        self.flurry_min = flurry_min
        self.seen: set = set()
        self.run_cnt: Counter = Counter()
        self.last_flurry = None
        self.first = None
        self.prev_sym = "."
        self.prev2_sym = "."
        self.prev_f = None
        self.prev_price = None

    def _build(self, s: dict, commit: bool) -> dict:
        slot = s["slot"]
        first = self.first if self.first is not None else slot
        buyers = s.get("buyers") or []
        new_b = [b for b in buyers if b not in self.seen]
        rc = self.run_cnt if commit else self.run_cnt.copy()
        rc.update(buyers)                                   # include slot N's own buyers (visible as it closes)
        top3 = {w for w, _ in rc.most_common(3)}
        top1 = rc.most_common(1)[0][0] if rc else None
        pprice = self.prev_price if self.prev_price is not None else s["price"]
        move = (s["price"] / pprice - 1.0) if pprice else 0.0
        since = float(min(slot - self.last_flurry, 50)) if self.last_flurry is not None else 50.0
        outsider = 1.0 if any(b not in top3 for b in buyers) else 0.0
        dominant = 1.0 if (top1 and top1 in buyers) else 0.0
        nbuys = float(s["n_buys"])
        buy_sol = float(s["buy_sol"])
        sell_sol = float(s["sell_sol"])
        n_sells = float(s["n_sells"])
        net_sol = float(s.get("net_sol", buy_sol - sell_sol))
        f = {
            "n_buys": nbuys, "buy_sol": buy_sol, "max_buy": float(s["max_buy"]),
            "net_sol": net_sol, "n_sells": n_sells, "sell_sol": sell_sol,
            "distinct_buyers": float(len(set(buyers))), "new_buyers": float(len(new_b)),
            "mean_buy": (buy_sol / nbuys if nbuys else 0.0), "abs_move": abs(move), "move": move,
            "slots_since_flurry": since, "age_slots": float(min(slot - first, 1000)),
            "outside_buy": outsider, "dominant_bought": dominant,
            "new_present": 1.0 if new_b else 0.0, "sell_present": 1.0 if n_sells > 0 else 0.0,
            "price_up": 1.0 if move > 0 else 0.0,
            "prev_buys": (self.prev_f["n_buys"] if self.prev_f else 0.0),
            "prev_buy_sol": (self.prev_f["buy_sol"] if self.prev_f else 0.0),
            "prev_new_buyers": (self.prev_f["new_buyers"] if self.prev_f else 0.0),
            "prev_net_sol": (self.prev_f["net_sol"] if self.prev_f else 0.0),
            "prev_abs_move": (self.prev_f["abs_move"] if self.prev_f else 0.0),
            "prev_move": (self.prev_f["move"] if self.prev_f else 0.0),
        }
        if nbuys >= self.flurry_min:
            sym = "F"
        elif n_sells > nbuys:
            sym = "S"
        elif new_b:
            sym = "N"
        elif outsider:
            sym = "O"
        elif dominant:
            sym = "B"
        elif nbuys > 0:
            sym = "b"
        else:
            sym = "."
        rec = {"f": f, "sym": sym, "m2": self.prev_sym + sym, "m3": self.prev2_sym + self.prev_sym + sym}
        if commit:
            if self.first is None:
                self.first = slot
            self.seen.update(buyers)                         # run_cnt was already advanced in place (rc is self.run_cnt)
            self.prev2_sym, self.prev_sym, self.prev_f = self.prev_sym, sym, f
            self.prev_price = s["price"]
            if nbuys >= self.flurry_min:
                self.last_flurry = slot
        return rec

    def push(self, s: dict) -> dict:
        """Commit one COMPLETE active slot (in slot order) and return its record -- advances the running state."""
        return self._build(s, True)

    def peek(self, s: dict) -> dict:
        """Record for a PARTIAL current slot WITHOUT advancing the state -- the streamed-tx scorer peeks the growing
        current slot after each tx (history = the prior committed slots), then push()es it once the slot closes."""
        return self._build(s, False)


class Runner:
    """Score the exported precursor gate on each closing slot and emit a FireSignal when it clears the top-`tier`%
    threshold. The signal names SLOT NUMBERS: enter at the front of N+1, exit at N+hold (hold from the gate)."""

    def __init__(self, gate: dict, tier: str = "1", flurry_min: int = DEFAULT_FLURRY_MIN):
        self.gate = gate if "_terms" in gate else precursor_gate.compile_gate(gate)
        self.tier = str(tier)
        self.hold = int(gate.get("hold", 2))
        self.bank = SlotBank(flurry_min)

    def on_slot(self, slot_agg: dict):
        """Process one active slot N. Returns a FireSignal dict when the gate fires, else None."""
        rec = self.bank.push(slot_agg)
        if not precursor_gate.fires(rec, self.gate, self.tier):
            return None
        n = slot_agg["slot"]
        return {"signal_slot": n, "entry_slot": n + 1, "exit_slot": n + self.hold,
                "hold": self.hold, "tier": self.tier, "eta": precursor_gate.eta(rec, self.gate),
                "sym": rec["sym"], "m2": rec["m2"]}


class StreamingRunner:
    """Score the gate on slot N's FIRST streamed transaction -- the earliest DURING-N decision, so a buy can be sent
    to land at the FRONT of N+1.

    A complete slot is only knowable after the NEXT slot starts (that is what "complete" means), so a complete-slot
    decision (Runner.on_slot) can never be sent before N+1 begins -- it tops out at front-of-N+2 (feed-lag 2-3 in
    practice). To land front-of-N+1 we must decide DURING N. This runner keeps the prior slots committed to the bank
    as COMPLETE (a tx in a new slot proves the old one closed) and scores the gate on the PARTIAL current slot at its
    FIRST tx -- [prior complete history] + [N's first tx]. It decides once per slot, on that first tx (later txs of
    the slot only accumulate, for when the slot is committed as history).

    This serves the MID_SLOT re-fit gate (flurry_study --frontrun --mid-slot): that gate is fit on exactly this
    record -- the first-tx partial + completed prior slots, predicting N+1 -- using the SAME SlotBank peek/push code,
    so the gate fires on the same partials live as in the backtest (train == serve). A gate fit the OLD way (on the
    COMPLETE slot, flurry_study without --mid-slot) must be served by Runner, not this runner: scoring a partial with
    a complete-slot gate is train/serve skew (the current slot's symbol is provisional at its first tx).

    Feed raw swaps (helius_feature_ws.swap_from_tx shape: slot, payer, tok_delta, sol_delta, tok_after, sol_after),
    NOT FullSlotAggregator output -- this runner owns its own bank and commits slots itself from the tx stream, so
    do not also feed its bank the aggregator's complete slots (double-commit). The box sets the real entry slot from
    its own slot clock (the next CHAIN slot), since `signal_slot` here is the partial slot's number."""

    def __init__(self, gate: dict, tier: str = "1", flurry_min: int = DEFAULT_FLURRY_MIN):
        self.gate = gate if "_terms" in gate else precursor_gate.compile_gate(gate)
        self.tier = str(tier)
        self.hold = int(gate.get("hold", 2))
        self.bank = SlotBank(flurry_min)
        self.cur: dict | None = None        # the current slot's running partial aggregate

    @staticmethod
    def _blank(slot: int) -> dict:
        return {"slot": slot, "n_buys": 0, "buy_sol": 0.0, "max_buy": 0.0, "n_sells": 0,
                "sell_sol": 0.0, "buyers": [], "price": None}

    def _add(self, swap: dict) -> None:
        s = self.cur
        tokd, sold = swap["tok_delta"], swap["sol_delta"]
        if tokd < 0 and sold > 0:                           # buy: tokens out, SOL in
            s["n_buys"] += 1
            s["buy_sol"] += sold
            s["max_buy"] = max(s["max_buy"], sold)
            s["buyers"].append(swap["payer"])
        elif tokd > 0 and sold < 0:                         # sell: tokens in, SOL out
            s["n_sells"] += 1
            s["sell_sol"] += -sold
        if swap.get("tok_after"):                           # a pull/seed moves no wallet state but updates the price
            s["price"] = swap["sol_after"] / swap["tok_after"]

    def _priced(self, agg: dict) -> dict:
        if agg["price"] is None:                            # no readable reserve yet: carry the last committed price
            a = dict(agg)                                   # (as FullSlotAggregator._flush does) so move reads 0
            a["price"] = self.bank.prev_price if self.bank.prev_price is not None else 0.0
            return a
        return agg

    def on_tx(self, swap: dict):
        """One streamed swap. Commits the prior slot to history when a new slot's first tx arrives, then DECIDES on
        that first tx (the earliest during-N point): scores the gate on [prior complete history] + [N's first tx]
        and returns a FireSignal if it trips. Later txs of the same slot only accumulate (no decision), so the gate
        fires at most once per slot, at its start."""
        slot = swap["slot"]
        first = False
        if self.cur is not None and slot != self.cur["slot"]:
            self.bank.push(self._priced(self.cur))          # the prior slot is complete now -> commit to history
            self.cur = None
        if self.cur is None:
            self.cur = self._blank(slot)
            first = True                                    # this is slot N's first tx -> the decision point
        self._add(swap)
        if not first:
            return None                                     # decide only on the first tx (front of N+1 from slot start)
        rec = self.bank.peek(self._priced(self.cur))        # gate on prior-complete history + N's first-tx partial
        if not precursor_gate.fires(rec, self.gate, self.tier):
            return None
        return {"signal_slot": slot, "entry_slot": slot + 1, "exit_slot": slot + self.hold,
                "hold": self.hold, "tier": self.tier, "eta": precursor_gate.eta(rec, self.gate),
                "sym": rec["sym"], "m2": rec["m2"], "partial": True}


class HybridRunner:
    """The box's live HYBRID architecture, EXACTLY mirroring flurry_study._hybrid_bank (parity-pinned by
    tests/test_hybrid_live.py). Two feeds, one gate:

      * the SHRED stream scores the CURRENT slot -- fed to on_direct_slot(slot_agg): the direct bank gives the
        slot's own symbol + wallet features (its own abs_move/move/price_up are zeroed, since shreds carry no
        reserves, so there is no live current-slot price);
      * the Helius WS stream rebases the HISTORY -- every COMPLETE slot it delivers is fed to on_full_slot(slot_agg):
        the full bank gives that slot's real (Helius) symbol + features, which become the motif history (N-1, N-2)
        and the lagged prev_* for scoring later slots.

    So scoring a shred slot N = direct(N) current features + full(N-1, N-2) rebased history, exactly as the study
    (which forces CAUSAL_WALLET on -- the online banks here are causal by construction, so they match).
    TIMING CONTRACT: before on_direct_slot(N), the box must have delivered via on_full_slot every FULL slot < N
    that exists (Helius at `processed` is ~1 slot behind, so by the time the shred closes N the WS has N-1). If the
    WS has not yet delivered a prior slot, the history for that position reads as a dead '.' -- the box should wait
    the few ms for the WS to catch up to N-1 so the live record matches the study's."""

    def __init__(self, gate: dict, tier: str = "1", flurry_min: int = DEFAULT_FLURRY_MIN):
        self.gate = gate if "_terms" in gate else precursor_gate.compile_gate(gate)
        self.tier = str(tier)
        self.hold = int(gate.get("hold", 2))
        self.direct = SlotBank(flurry_min)                  # SHRED feed -> current slot
        self.full = SlotBank(flurry_min)                    # Helius WS feed -> rebased history
        self.full_sym: dict = {}
        self.full_f: dict = {}
        self.full_order: list = []
        self.last_rec: dict = {}

    def on_full_slot(self, slot_agg: dict) -> None:
        """A COMPLETE slot from the Helius WS: advance the full bank and store its symbol + features (the rebase)."""
        slot = slot_agg["slot"]
        r = self.full.push(slot_agg)
        self.full_sym[slot] = r["sym"]
        self.full_f[slot] = r["f"]
        self.full_order.append(slot)

    def hybrid_rec(self, slot: int, direct_rec: dict) -> dict:
        """Combine the shred current slot with the Helius-rebased history -- flurry_study._hybrid_bank, online."""
        f = dict(direct_rec["f"])
        f["abs_move"] = 0.0; f["move"] = 0.0; f["price_up"] = 0.0   # no live current-slot price from shreds
        prior = [s for s in self.full_order if s < slot]
        psl = prior[-1] if prior else None
        p2sl = prior[-2] if len(prior) >= 2 else None
        pf = self.full_f.get(psl) or {}
        f["prev_buys"] = pf.get("n_buys", 0.0); f["prev_buy_sol"] = pf.get("buy_sol", 0.0)
        f["prev_new_buyers"] = pf.get("new_buyers", 0.0); f["prev_net_sol"] = pf.get("net_sol", 0.0)
        f["prev_abs_move"] = pf.get("abs_move", 0.0); f["prev_move"] = pf.get("move", 0.0)
        sym = direct_rec["sym"]
        return {"f": f, "sym": sym, "m2": self.full_sym.get(psl, ".") + sym,
                "m3": self.full_sym.get(p2sl, ".") + self.full_sym.get(psl, ".") + sym}

    def on_direct_slot(self, slot_agg: dict):
        """A closed slot from the SHRED stream: score direct(N) current + full(N-1,N-2) rebased history. Returns a
        FireSignal when the gate fires, else None. The scored record is on self.last_rec for the parity verifier."""
        slot = slot_agg["slot"]
        rec = self.hybrid_rec(slot, self.direct.push(slot_agg))
        self.last_rec = rec
        if not precursor_gate.fires(rec, self.gate, self.tier):
            return None
        return {"signal_slot": slot, "entry_slot": slot + 1, "exit_slot": slot + self.hold,
                "hold": self.hold, "tier": self.tier, "eta": precursor_gate.eta(rec, self.gate),
                "sym": rec["sym"], "m2": rec["m2"]}


class HybridConsumer:
    """Per-pool orchestration of HybridRunners for the box: route the two live feeds to the right pool's runner.
    The box drives it with two entry points:
      * on_helius_slot(pool, slot_agg) -- when the Helius transactionSubscribe FullSlotAggregator emits a complete
        slot (the rebase). Deliver these as they arrive.
      * on_shred_slot(pool, slot_agg) -- when the SHRED stream closes a slot (the direct current-slot aggregate).
        Scores it and, on a fire, stamps the pool and forwards the FireSignal. The box should DELAY this call ~1
        slot (~500 ms) so the Helius WS has rebased the prior slots first (HybridRunner's timing contract).
    on_pool(pool) fires the first time a pool is seen, so the box opens that pool's transactionSubscribe.
    Fresh / hand-made filtering stays on the shred Consumer upstream -- only its finalized slots for fresh
    hand-made pools should be routed here."""

    def __init__(self, gate: dict, tier: str = "1", flurry_min: int = DEFAULT_FLURRY_MIN,
                 on_fire=None, on_pool=None, fresh_window_slots: int = 0):
        self.gate = gate if "_terms" in gate else precursor_gate.compile_gate(gate)
        self.tier = str(tier)
        self.flurry_min = flurry_min
        self.on_fire = on_fire
        self.on_pool = on_pool
        # Fire only on FRESH pools (as the study's universe is): the pool was first seen after a warm-up of
        # fresh_window_slots (confidence we caught it from its start) and the fire is within that window of first
        # sight. 0 = off (tests / replay). Mirrors localshred_consumer.Consumer._fresh_ok on the SHRED first-sight.
        # No per-pool dedup: the gate fires on EVERY qualifying slot (a setup seen twice fires twice), so the live
        # rate is read raw against the study's ~1000 fires / 12 h on ~200 winners (top-1% logit) -- a fresh washbot
        # genuinely setting up twice IS two signals.
        self.fresh_window_slots = int(fresh_window_slots)
        self._start_slot = None
        self._first: dict = {}
        self.runners: dict = {}

    def _runner(self, pool: str) -> "HybridRunner":
        r = self.runners.get(pool)
        if r is None:
            r = HybridRunner(self.gate, self.tier, self.flurry_min)
            self.runners[pool] = r
            if self.on_pool:
                self.on_pool(pool)                           # box opens transactionSubscribe(accountInclude=[pool])
        return r

    def _fresh_ok(self, pool: str, slot: int) -> bool:
        w = self.fresh_window_slots
        if w <= 0:
            return True
        first = self._first.get(pool)
        if self._start_slot is None or first is None:
            return False
        if (first - self._start_slot) < w:                   # pool first seen during warm-up -> maybe mid-life
            return False
        return (slot - first) <= w

    def on_helius_slot(self, pool: str, slot_agg: dict) -> None:
        self._runner(pool).on_full_slot(slot_agg)

    def on_shred_slot(self, pool: str, slot_agg: dict):
        slot = slot_agg["slot"]
        if self._start_slot is None:
            self._start_slot = slot
        self._first.setdefault(pool, slot)                   # first sight of the pool on the SHRED feed
        sig = self._runner(pool).on_direct_slot(slot_agg)    # advance the bank even if suppressed (keep tracking)
        if sig is None or not self._fresh_ok(pool, slot):
            return None
        sig["pool"] = pool
        if self.on_fire:
            self.on_fire(sig)
        return sig

    def drop(self, pool: str) -> None:
        self.runners.pop(pool, None)                         # pool aged out of the fresh window


class WsOnlyConsumer:
    """Per-pool orchestration of WS-ONLY Runners for the box -- the one-feed architecture the --ws-only study fits.

    There is a SINGLE feed: a Helius transactionSubscribe (commitment=processed, maxSupportedTransactionVersion>=1)
    per fresh pool. The processed WS is same-slot + complete (proven: ws_latency 100% same-slot, 0 drops), so that
    one feed is BOTH the current-slot decision and the price path -- no shred stream, no hybrid rebase, no frontier.
    The box aggregates each pool's notifications to complete per-slot aggregates with
    helius_feature_ws.FullSlotAggregator (same shape flurry_study.slot_flow emits) and routes them here; this scores
    the gate on the closing slot N and, on a fire for a FRESH pool, emits the FireSignal (enter front of N+1, exit
    N+hold). Mirrors HybridConsumer's fresh-pool gate and pool routing, minus the second feed and its timing
    contract -- so it is strictly simpler and has no WS-vs-shred ordering to get wrong.

    Box loop sketch (pseudocode; the WS socket + the Jito send + NET fill logging are box-only):
        import helius_feature_ws as hw
        gate = precursor_gate.load("gate_wsonly.json")
        con  = WsOnlyConsumer(gate, tier="0.5", on_fire=on_fire, on_pool=open_sub, fresh_window_slots=450)
        aggs = {}
        def open_sub(pool):                                  # con calls this the first time it sees a pool
            ws_subscribe(pool)                               # transactionSubscribe(accountInclude=[pool], processed, maxTxVersion>=1)
        def on_notification(pool, token_mint, result):       # one WS tx notification
            swap = hw.swap_from_tx(hw.normalize_notification(result), pool, token_mint)
            if swap: aggs.setdefault(pool, hw.FullSlotAggregator(lambda s, _p=pool: con.on_slot(_p, s))).push(swap)
        def on_fire(sig):                                    # land a buy at the FRONT of sig["entry_slot"]=N+1 via a
            executor.front_run(sig)                          # Jito bundle, sell at N+hold, log swap_return + net_return
            #                                                # (swap % = the move even a $0.01 probe would've made)
    The box opens each pool's subscription at launch detection (the LP-mint sighting the scorer already uses), so a
    pool's first WS slot ~= its creation and the fresh-window gate is meaningful. No per-pool fire dedup (every
    qualifying slot fires), matching HybridConsumer and the study's raw fire rate."""

    def __init__(self, gate: dict, tier: str = "1", flurry_min: int = DEFAULT_FLURRY_MIN,
                 on_fire=None, on_pool=None, fresh_window_slots: int = 0):
        self.gate = gate if "_terms" in gate else precursor_gate.compile_gate(gate)
        self.tier = str(tier)
        self.flurry_min = flurry_min
        self.on_fire = on_fire
        self.on_pool = on_pool
        self.fresh_window_slots = int(fresh_window_slots)    # see HybridConsumer: new-pools-only warm-up gate
        self._start_slot = None
        self._first: dict = {}
        self.runners: dict = {}

    def _runner(self, pool: str) -> "Runner":
        r = self.runners.get(pool)
        if r is None:
            r = Runner(self.gate, self.tier, self.flurry_min)
            self.runners[pool] = r
            if self.on_pool:
                self.on_pool(pool)                           # box opens transactionSubscribe(accountInclude=[pool])
        return r

    def _fresh_ok(self, pool: str, slot: int) -> bool:
        w = self.fresh_window_slots
        if w <= 0:
            return True
        first = self._first.get(pool)
        if self._start_slot is None or first is None:
            return False
        if (first - self._start_slot) < w:                   # pool first seen during warm-up -> maybe mid-life
            return False
        return (slot - first) <= w

    def on_slot(self, pool: str, slot_agg: dict):
        """One complete per-slot aggregate for `pool` (from FullSlotAggregator). Advances the pool's bank (always,
        to keep the online history correct) and returns/forwards a FireSignal when the gate fires on a fresh pool."""
        slot = slot_agg["slot"]
        if self._start_slot is None:
            self._start_slot = slot
        self._first.setdefault(pool, slot)                   # first sight of the pool on the WS feed
        sig = self._runner(pool).on_slot(slot_agg)           # advance the bank even if suppressed (keep tracking)
        if sig is None or not self._fresh_ok(pool, slot):
            return None
        sig["pool"] = pool
        if self.on_fire:
            self.on_fire(sig)
        return sig

    def drop(self, pool: str) -> None:
        self.runners.pop(pool, None)                         # pool aged out of the fresh window


def swap_return(entry_fill: float, exit_fill: float) -> float:
    """SWAP-only return: exit_fill / entry_fill - 1, the price move with NO cost term. Log it beside net_return on
    every fire so a tiny ($0.01) probe still shows the % the trade WOULD have made at a size where the fixed
    per-tx fees are negligible -- the signal's raw edge, read apart from the fee drag that dominates (and buries)
    a cent-sized trade's net. Same idea as canary_rugs.swap_return, computed from the two fill prices."""
    if not entry_fill:
        return 0.0
    return exit_fill / entry_fill - 1.0


def net_return(entry_fill: float, exit_fill: float, cost: float = 0.01) -> float:
    """NET return of a filled front-run: swap_return - cost (round-trip cost, default 1%). The box passes its REAL
    fill prices (entry at the front of N+1, exit at N+hold) so the live number is fills, not the study's
    mid-to-mid -- the only thing that settles whether the up-side tail survives contact. At $0.01 the cost term
    swamps this, so read swap_return(entry, exit) alongside it to see the move the trade actually captured."""
    if not entry_fill:
        return 0.0
    return swap_return(entry_fill, exit_fill) - cost
