"""Symbol-flip confusion (2026-11-23): turn buyer recall into the gate-level number.

The completeness tool said the box's feed carries 95.5% of buyer wallets, but that is NOT the gate's metric.
The hybrid gate is 100% symbol/motif: it keys on the per-slot SYMBOL (F/S/N/O/B/b/.) and a slot flips if even
one `N` (first-ever wallet) or `O` (outsider) buyer is among the ones the box missed. The gate fires on exactly
that tail, so a small buyer-miss rate makes a large FIRE-level divergence -- the 45% hybrid-verify reproduction.

This measures the flip directly, at the slot level, over a capture, on Fly (research Helius key only, read only):

  * The box's live architecture is flurry_study._hybrid_bank: the CURRENT slot's symbol comes from the box's
    shred decode, the history (motif, prev_*) is rebased to the Helius FULL view. So the only thing that can
    differ between the pre-LUT box and the gate's own view is the CURRENT slot's symbol.
  * Two current-slot visibilities, history rebased to full in both (same Helius txs, so apples to apples):
      direct  = top-level pumpswap INCLUDING LUT = the view the gate was fit on (vis "direct") = what
                run_hybrid_lut now serves with LUT on.
      box     = top-level AND pool is a static key = box_visible = the LUT-BLIND view the box served pre-LUT.
  * For every slot the gate's own (direct) view fires, does the pre-LUT box fire too? That reproduction rate is
    the gate-level number the hybrid-verify measures (38/85). The misses are decomposed: the box lost the whole
    slot (`absent`), or saw it but assigned a different symbol (`flip`), with the share where the correct symbol
    was `N`/`O` (the organic precursor the box misread as washbot-dominant). Since run_hybrid_lut serves the
    `direct` view, this quantifies exactly how much LUT lifts reproduction, and what residual (routed, ~0.2%)
    LUT cannot reach.

Fly-only (imports flurry_study -> numpy). The box comparator (shred_completeness) stays numpy-free.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from typing import Callable, Optional

from . import flurry_study as fs
from . import precursor_gate as pg
from .data import helius as he
from .shred_completeness import MAX_PAGES, _pct, freshness, load_capture  # noqa: F401 (freshness/load_capture re-used by the CLI)

ORGANIC = ("N", "O")   # the organic precursor symbols a missed buyer flips away from


def bank_by_slot(txs: list[dict], pool: str, mint: str, box: bool) -> dict:
    """{slot: _slot_bank-style rec} for the box's live architecture (flurry_study._hybrid_bank): the current
    slot's symbol/features from the chosen visibility, the motif history rebased to the FULL (Helius) view.
    box=False -> current slot from `direct` (LUT-included, the gate's fit view); box=True -> from box_visible
    (LUT-blind). Keyed by the current slot id (the hybrid bank is aligned to slot_flow(rec,'direct')'s slots)."""
    rec = he.launch_record(txs, pool, mint, window_s=1e12, with_vis=True, vis_box_exact=box)
    direct_flow = fs.slot_flow(rec, "direct")       # the current-slot series the hybrid bank iterates
    if not direct_flow:
        return {}
    bank = fs._hybrid_bank(rec, 0.0, fs._tgt())     # [] if full/direct flow unusable
    slots = [s["slot"] for s in direct_flow["slots"]]
    return {slots[i]: bank[i] for i in range(min(len(slots), len(bank)))}


def confuse(box_bank: dict, dir_bank: dict, gate: dict, tier: str = "1") -> dict:
    """Compare one pool's box-visible vs direct current-slot symbols and gate fires. `dir_bank` is the gate's
    own view (ground truth for the fire decision); `box_bank` is the pre-LUT box. Returns per-pool counts."""
    out = {k: 0 for k in ("direct_fires", "box_fires", "reproduced", "false_fires",
                          "missed", "miss_absent", "miss_flip", "miss_flip_organic",
                          "elig", "elig_agree")}
    confusion: Counter = Counter()      # (correct direct sym -> what the box saw) on MISSED direct-fire slots
    false_sym: Counter = Counter()      # the box's symbol on a false fire (box fires, direct does not)
    for slot, rec_d in dir_bank.items():
        fire_d = pg.fires(rec_d, gate, tier)
        rec_b = box_bank.get(slot)
        fire_b = pg.fires(rec_b, gate, tier) if rec_b else False
        sym_d = rec_d.get("sym")
        sym_b = rec_b.get("sym") if rec_b else "∅"      # box never saw the slot
        out["direct_fires"] += fire_d
        out["box_fires"] += fire_b
        if fire_d or fire_b:
            out["elig"] += 1
            if sym_b == sym_d:
                out["elig_agree"] += 1
        if fire_d and fire_b:
            out["reproduced"] += 1
        elif fire_b and not fire_d:
            out["false_fires"] += 1
            false_sym[sym_b] += 1
        elif fire_d and not fire_b:                          # a gate-view fire the box missed
            out["missed"] += 1
            confusion[(sym_d, sym_b)] += 1
            if rec_b is None:
                out["miss_absent"] += 1
            else:
                out["miss_flip"] += 1
                if sym_d in ORGANIC:
                    out["miss_flip_organic"] += 1
    return {"counts": out, "confusion": confusion, "false_sym": false_sym}


def study(cap_rows: list[dict], helius, gate: dict, tier: str = "1", max_credits: int = 40_000,
          span_s: float = 180.0, max_pools: int = 300, now: Optional[float] = None,
          progress: Optional[Callable[[int, int, int], None]] = None) -> dict:
    """Over a sample of captured pools, measure the box-visible-vs-direct symbol flip on gate-fire slots.
    Fetches each pool's trading window from Helius (research key), stops before a fetch could pass max_credits."""
    fresh = freshness(cap_rows, now)
    pools = [cr for cr in cap_rows if cr.get("pool") and cr.get("token_mint")
             and not str(cr.get("token_mint")).endswith("pump")         # hand-made only (drop pump.fun graduations)
             and len(cr.get("slots") or []) >= 1]
    pools.sort(key=lambda cr: hashlib.sha1(cr["pool"].encode()).hexdigest())   # stable: a cap leaves an even sample
    agg = Counter()
    confusion: Counter = Counter()
    false_sym: Counter = Counter()
    credits, compared, no_bank = 0, 0, 0
    per_page = MAX_PAGES * 10
    for i, cr in enumerate(pools):
        if compared >= max_pools or helius is None or credits + per_page > max_credits:
            break
        ct = cr.get("create_ts") or 0
        txs, _ok = helius.history_window(cr["pool"], ct - 30, span_s + 60, MAX_PAGES)
        credits += per_page
        dir_bank = bank_by_slot(txs, cr["pool"], cr["token_mint"], box=False)
        box_bank = bank_by_slot(txs, cr["pool"], cr["token_mint"], box=True)
        if not dir_bank:
            no_bank += 1
        else:
            compared += 1
            r = confuse(box_bank, dir_bank, gate, tier)
            agg.update(r["counts"])
            confusion.update(r["confusion"])
            false_sym.update(r["false_sym"])
        if progress:
            progress(i + 1, len(pools), credits)
    df = agg["direct_fires"]
    summary = {
        "pools_in_capture": fresh["pools"], "hand_made_pools": len(pools),
        "compared": compared, "no_bank": no_bank, "credits": credits, "tier": tier,
        "gate_vis": gate.get("vis"), "gate_move": gate.get("move"),
        "direct_fires": df, "box_fires": agg["box_fires"], "reproduced": agg["reproduced"],
        "missed": agg["missed"], "false_fires": agg["false_fires"],
        "reproduction_pct": _pct(agg["reproduced"], df),
        "false_fire_rate_pct": _pct(agg["false_fires"], agg["box_fires"]),
        "miss_absent": agg["miss_absent"], "miss_flip": agg["miss_flip"],
        "miss_flip_organic": agg["miss_flip_organic"],
        "miss_organic_pct": _pct(agg["miss_flip_organic"], agg["missed"]),
        "elig": agg["elig"], "symbol_agree_pct": _pct(agg["elig_agree"], agg["elig"]),
    }
    top_conf = [{"correct": a, "box_saw": b, "n": n} for (a, b), n in confusion.most_common(15)]
    top_false = [{"box_sym": s, "n": n} for s, n in false_sym.most_common(8)]
    return {"freshness": fresh, "summary": summary, "confusion": top_conf, "false_sym": top_false}


def render(out: dict) -> str:
    s = out["summary"]
    L = []
    L.append("SYMBOL-FLIP CONFUSION -- does the pre-LUT box fire where the gate's own (direct/LUT-on) view fires?")
    L.append("  (current-slot symbol: box_visible vs direct; history rebased to FULL in both = the live hybrid)")
    L.append("")
    L.append(f"  gate         : vis={s['gate_vis']} move={s['gate_move']} tier={s['tier']}  "
             f"(direct = what run_hybrid_lut serves with LUT on)")
    L.append(f"  compared     : {s['compared']} pools  ({s['no_bank']} skipped: no usable bank)  "
             f"{s['credits']} research credits")
    L.append("")
    L.append("FIRE REPRODUCTION  (of the gate-view fires, how many the pre-LUT box also fires)")
    L.append(f"  direct-view fires : {s['direct_fires']}")
    L.append(f"  box reproduced    : {s['reproduced']}   ({s['reproduction_pct']}%)   "
             f"<- predicts the hybrid-verify rate; LUT lifts this toward ~100% (direct = LUT-on)")
    L.append(f"  box missed        : {s['missed']}   (lost the slot: {s['miss_absent']}  |  "
             f"symbol flipped: {s['miss_flip']}, of which {s['miss_flip_organic']} were N/O = "
             f"{s['miss_organic_pct']}% of misses)")
    L.append(f"  box FALSE fires   : {s['false_fires']}   ({s['false_fire_rate_pct']}% of box fires)   "
             f"<- box fired where the correct view does NOT")
    L.append(f"  symbol agreement  : {s['symbol_agree_pct']}%  on {s['elig']} fire-eligible slots")
    L.append("")
    L.append("WHAT THE BOX SAW INSTEAD, on missed gate-view fires  (correct symbol -> box's symbol)")
    for c in out["confusion"]:
        tag = "  <- organic precursor lost" if c["correct"] in ORGANIC else ""
        box = "(absent)" if c["box_saw"] == "∅" else c["box_saw"]
        L.append(f"  {c['correct']:>2} -> {box:<8} {c['n']:>5}{tag}")
    if out["false_sym"]:
        L.append("")
        L.append("BOX FALSE-FIRE symbols  (what the box saw when it fired and the correct view did not)")
        for c in out["false_sym"]:
            L.append(f"  {c['box_sym']:>2}  {c['n']:>5}")
    return "\n".join(L)
