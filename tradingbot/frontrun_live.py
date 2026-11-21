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
        if sig:                                            # land a buy at the FRONT of sig["entry_slot"]=N+1,
            executor.front_run(sig)                        # hold to sig["exit_slot"]=N+hold, log NET fills
"""
from collections import Counter

from . import precursor_gate

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

    def push(self, s: dict) -> dict:
        slot = s["slot"]
        if self.first is None:
            self.first = slot
        buyers = s.get("buyers") or []
        new_b = [b for b in buyers if b not in self.seen]
        self.seen.update(buyers)
        self.run_cnt.update(buyers)                         # include slot N's own buyers (visible as it closes)
        top3 = {w for w, _ in self.run_cnt.most_common(3)}
        top1 = self.run_cnt.most_common(1)[0][0] if self.run_cnt else None
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
            "slots_since_flurry": since, "age_slots": float(min(slot - self.first, 1000)),
            "outside_buy": outsider, "dominant_bought": dominant,
            "new_present": 1.0 if new_b else 0.0, "sell_present": 1.0 if n_sells > 0 else 0.0,
            "price_up": 1.0 if move > 0 else 0.0,
            "prev_buys": (self.prev_f["n_buys"] if self.prev_f else 0.0),
            "prev_buy_sol": (self.prev_f["buy_sol"] if self.prev_f else 0.0),
            "prev_new_buyers": (self.prev_f["new_buyers"] if self.prev_f else 0.0),
            "prev_net_sol": (self.prev_f["net_sol"] if self.prev_f else 0.0),
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
        self.prev2_sym, self.prev_sym, self.prev_f = self.prev_sym, sym, f
        self.prev_price = s["price"]
        if nbuys >= self.flurry_min:
            self.last_flurry = slot
        return rec


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


def net_return(entry_fill: float, exit_fill: float, cost: float = 0.01) -> float:
    """NET return of a filled front-run: exit_fill / entry_fill - 1 - cost (round-trip cost, default 1%). The box
    passes its REAL fill prices (entry at the front of N+1, exit at N+hold) so the live number is fills, not the
    study's mid-to-mid -- the only thing that settles whether the up-side tail survives contact."""
    if not entry_fill:
        return 0.0
    return exit_fill / entry_fill - 1.0 - cost
