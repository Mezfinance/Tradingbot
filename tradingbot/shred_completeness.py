"""Shred-feed completeness comparator (2026-11-23).

Diffs the box's LocalShred ``--capture`` decode (per-slot n_buys / n_sells / buyers of each fresh hand-made
pool) against Helius ground truth (vault-delta ``slot_flow``), per pool-slot, to answer the owner's question
directly: "see exactly what we do and don't have". Every chain trade the capture MISSED is classified:

  - absent : a top-level pumpswap swap whose pool account is a STATIC key -- the box CAN decode it today, yet
             the capture did not record it. This is genuine shred-feed loss (a shred that never arrived, or a
             decode the box dropped for another reason). It is NOT fixed by any decoder; only a more complete
             feed fixes it. The 2026-11-23 box-exact verdict said the m2 residual lives here.
  - lut    : a top-level pumpswap swap whose pool account is LUT-indexed -- the box drops it today (drop_lut
             ~3.9%), but the fuller LUT-resolved decode (localshred_tap.resolve_account_keys, built 2026-11-23)
             recovers it. The gap between what we have and what LUT resolution reaches.
  - routed : an aggregator-routed (CPI) pumpswap swap. A compiled message carries no inner instruction, so no
             top-level decode ever sees it; only a router decoder recovers it. The 2026-11-23 step-1 verdict
             (routers = 0.2% of flow on our pools) says NOT to build one, so this is the irreducible remainder.

Three ceilings make the lever explicit (all as a share of every chain trade in the box's watch window):
  capture / all      = HAVE%  (headline: of every chain trade, the share the box's feed actually recorded)
  box_visible / all  = what today's LUT-blind decode can reach  (its gap to HAVE% = absent = feed loss)
  direct / all       = what the fuller LUT-resolved decode can reach  (direct - box_visible = LUT-recoverable)
                       (all - direct = routed, unrecoverable without a router decoder we are not building)

Buyer-set recall (per pool, |cap_buyers & chain_buyers| / |chain_buyers|) is the gate-relevant read: the
deployed precursor gate is 100% symbol/motif and keys on the per-slot BUYER SET (a first-ever wallet = ``N``,
which SUPPRESSES fires), so a slot whose buyers the capture missed is misread as a different symbol live --
exactly the 45% hybrid-verify divergence. Recall near 1.0 means the feed sees the wallets the gate needs.

Runs on the box (it has ``--capture`` and the research Helius key in ``~/.shredenv``). Research key only,
never the live key. Read only; no trading effect.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, Optional

from .data import helius as he

# Row layout + classify, INLINED from flurry_study so the comparator imports no numpy -- it must run on the
# box, whose venv (run_hybrid.py's) is deliberately numpy-free, and importing flurry_study pulls numpy. These
# mirror flurry_study exactly (pinned by tests/test_shred_completeness.py::test_row_layout_matches_flurry_study):
# a launch_record row is [sec, slot, payer, tokΔ, solΔ, tokA, solA, vis?].
_SLOT, _PAYER, _TOKD, _SOLD, _VIS = 1, 2, 3, 4, 7
MAX_PAGES = 6


def _classify(tok_change: float, sol_change: float) -> str:
    if tok_change < 0 and sol_change > 0:
        return "buy"        # tokens leave the pool, SOL enters
    if tok_change > 0 and sol_change < 0:
        return "sell"       # tokens enter, SOL leaves
    return "other"          # pull / add / other -- not a buyer-or-seller trade


def load_capture(path: str) -> list[dict]:
    """Read a ``--capture`` JSONL and collapse duplicate captures of the same pool to the richest (most slots),
    the same de-dup ``merge_capture_flows`` does (the capture can emit a pool across several windows)."""
    best: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cr = json.loads(line)
            except ValueError:
                continue
            p = cr.get("pool")
            if p and (p not in best or len(cr.get("slots") or []) > len(best[p].get("slots") or [])):
                best[p] = cr
    return list(best.values())


def freshness(cap_rows: list[dict], now: Optional[float] = None) -> dict:
    """Describe how current the capture is, so an unknown-age ``ourfeed.jsonl`` is read honestly: pool count,
    the create_ts (first-seen wall clock) range, and the age of the newest pool."""
    now = time.time() if now is None else now
    cts = [cr.get("create_ts") or 0 for cr in cap_rows if cr.get("create_ts")]
    if not cts:
        return {"pools": len(cap_rows), "create_ts_min": None, "create_ts_max": None,
                "newest_age_s": None, "span_hours": None}
    lo, hi = min(cts), max(cts)
    return {"pools": len(cap_rows), "create_ts_min": lo, "create_ts_max": hi,
            "newest_age_s": round(now - hi, 1), "span_hours": round((hi - lo) / 3600.0, 2),
            "newest_iso": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(hi)),
            "oldest_iso": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(lo))}


def _slot_counts(rec_vis: dict, mode: str) -> dict:
    """{slot: {'n_buys', 'n_sells', 'buyers': set}} straight from the ``launch_record`` rows -- same
    classification as ``slot_flow`` (buy = tokens out / SOL in, buyer = fee payer on a buy) but with no price
    path and no minimum-slots gate, since the comparator needs per-slot counts only. ``mode='full'`` counts
    every trade; ``mode='direct'`` only rows whose 8th element (``with_vis``: top-level pumpswap) is truthy."""
    out: dict = {}
    for r in rec_vis.get("txs") or []:
        if mode == "direct" and not (len(r) > _VIS and bool(r[_VIS])):
            continue
        kind = _classify(r[_TOKD] or 0.0, r[_SOLD] or 0.0)
        if kind not in ("buy", "sell"):
            continue
        sl = int(r[_SLOT])
        d = out.setdefault(sl, {"n_buys": 0, "n_sells": 0, "buyers": set()})
        if kind == "buy":
            d["n_buys"] += 1
            if r[_PAYER]:
                d["buyers"].add(r[_PAYER])
        else:
            d["n_sells"] += 1
    return out


_ZERO = {"n_buys": 0, "n_sells": 0, "buyers": set()}


def compare_pool(cap_rec: dict, txs: list[dict], route_classify=None) -> Optional[dict]:
    """One pool's capture vs Helius ground truth, over the box's watch window (the capture's slot range). Returns
    per-category trade counts + buyer-set recall, or None when there is nothing to compare (no mint, no capture
    slots, or Helius saw no trades in the window). Chain truth at three visibility levels from one tx set:
    full = every trade; direct = top-level pumpswap (incl. LUT); box = top-level AND pool is a static key."""
    pool, mint = cap_rec.get("pool"), cap_rec.get("token_mint")
    cap_slots = cap_rec.get("slots") or []
    if not pool or not mint or not cap_slots or not txs:
        return None
    rec_vis = he.launch_record(txs, pool, mint, window_s=1e12, with_vis=True)                     # vis=_top_level
    rec_box = he.launch_record(txs, pool, mint, window_s=1e12, with_vis=True, vis_box_exact=True)  # vis=_box_visible
    all_c = _slot_counts(rec_vis, "full")      # every chain trade (vault-delta direction)
    dir_c = _slot_counts(rec_vis, "direct")    # top-level pumpswap, LUT included
    box_c = _slot_counts(rec_box, "direct")    # top-level AND pool static = what the box decodes today
    if not all_c:
        return None
    # capture keyed by absolute slot (merged across any duplicate rows)
    cap: dict = {}
    for s in cap_slots:
        sl = int(s[1])
        d = cap.setdefault(sl, {"n_buys": 0, "n_sells": 0, "buyers": set()})
        d["n_buys"] += int(s[2])
        d["n_sells"] += int(s[3])
        d["buyers"] |= set(s[4] or [])
    if not cap:
        return None
    lo, hi = min(cap), max(cap)                # the box's watch window: don't count chain slots it never saw
    slots = sorted(s for s in (set(all_c) | set(cap)) if lo <= s <= hi)
    t = {k: 0 for k in ("cap_buys", "all_buys", "box_buys", "dir_buys",
                         "cap_sells", "all_sells", "box_sells", "dir_sells")}
    cap_b, chain_b, box_b = set(), set(), set()
    for sl in slots:
        a, d, b, c = all_c.get(sl, _ZERO), dir_c.get(sl, _ZERO), box_c.get(sl, _ZERO), cap.get(sl, _ZERO)
        t["all_buys"] += a["n_buys"]; t["dir_buys"] += d["n_buys"]; t["box_buys"] += b["n_buys"]
        t["all_sells"] += a["n_sells"]; t["dir_sells"] += d["n_sells"]; t["box_sells"] += b["n_sells"]
        t["cap_buys"] += min(c["n_buys"], a["n_buys"])       # clamp: capture can't legitimately exceed chain
        t["cap_sells"] += min(c["n_sells"], a["n_sells"])
        cap_b |= c["buyers"]; chain_b |= a["buyers"]; box_b |= b["buyers"]
    all_t = t["all_buys"] + t["all_sells"]
    if all_t == 0:
        return None
    dir_t = t["dir_buys"] + t["dir_sells"]
    box_t = t["box_buys"] + t["box_sells"]
    cap_t = t["cap_buys"] + t["cap_sells"]
    rr = route_recovery(txs, pool, mint, lo, hi, route_classify) if route_classify else {}
    return {
        **rr,
        "pool": pool, "slots_compared": len(slots),
        "all": all_t, "direct": dir_t, "box_visible": box_t, "capture": cap_t,
        "routed": all_t - dir_t,                 # aggregator CPI (no top-level decode sees it)
        "lut": dir_t - box_t,                    # top-level but LUT-pooled (fuller decode recovers)
        "absent": max(0, box_t - cap_t),         # box-visible yet not captured = feed loss
        **t,
        "chain_buyers": len(chain_b), "box_buyers": len(box_b),
        "cap_hit_chain": len(cap_b & chain_b), "cap_hit_box": len(cap_b & box_b),
    }


_INFRA = frozenset({
    "ComputeBudget111111111111111111111111111111", "11111111111111111111111111111111",
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL", "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",
    "Memo1UhkJRfHyvLMcVucJwxXeuD728EqVDDwQDxFMNo"})          # == localshred_consumer.INFRA_PROGRAMS


def _inner_direction(tx: dict, pool: str, mint: str) -> Optional[str]:
    """TRUE token direction of a routed swap on `pool`, from its inner pumpswap instruction (v1 or v2 disc +
    the box orientation flip). None if the tx has no clean inner swap on this pool."""
    for grp in (tx.get("meta") or {}).get("innerInstructions") or []:
        for ix in grp.get("instructions") or []:
            a = ix.get("accounts") or []
            if ix.get("programId") != he.PUMPSWAP_PROGRAM or len(a) < 5 or a[0] != pool:
                continue
            try:
                disc = he._b58decode(ix.get("data") or "")[:8]
            except ValueError:
                continue
            inst_buy = True if disc in he._SWAP_BUYS else False if disc in he._SWAP_SELLS else None
            if inst_buy is None or (a[3] == he.SOL_MINT) == (a[4] == he.SOL_MINT):
                continue
            token_is_base = a[3] != he.SOL_MINT
            return "buy" if inst_buy == token_is_base else "sell"
    return None


def route_recovery(txs: list[dict], pool: str, mint: str, lo: int, hi: int, route_classify) -> dict:
    """How the box's router rules (route_rules / localshred_consumer._recover_routed) would read each SUCCESSFUL
    routed swap on `pool` inside the capture's slot window [lo, hi]. Mirrors the consumer: the first top-level
    non-infra, non-pumpswap instruction whose accounts include the pool gets route_classify(..., token). Counts:
    routed_seen (clean inner swap on the pool, no top-level pumpswap), rule_right / rule_wrong (rule fired,
    vs the inner-swap truth), rule_tail (buyer_plus: counted a buy; tail_right = it really was a buy),
    rule_drop (denylist / pin-target / abstain)."""
    c = {k: 0 for k in ("routed_seen", "rule_right", "rule_wrong", "rule_tail", "tail_right", "rule_drop")}
    for tx in txs:
        if (tx.get("meta") or {}).get("err") is not None or not (lo <= int(tx.get("slot") or -1) <= hi):
            continue
        msg = (tx.get("transaction") or {}).get("message") or {}
        top = msg.get("instructions") or []
        if any(ix.get("programId") == he.PUMPSWAP_PROGRAM for ix in top):
            continue                                              # direct swap: the capture's job, not a router's
        truth = _inner_direction(tx, pool, mint)
        if truth is None:
            continue
        c["routed_seen"] += 1
        verdict = "drop"
        for ix in top:
            p, a = ix.get("programId"), ix.get("accounts") or []
            if p in _INFRA or p == he.PUMPSWAP_PROGRAM or "parsed" in ix or pool not in a:
                continue
            try:
                data = he._b58decode(ix.get("data") or "")
            except ValueError:
                data = b""
            verdict = route_classify(p, data, list(range(len(a))), list(a), mint)
            break
        if verdict in ("buy", "sell"):
            c["rule_right" if verdict == truth else "rule_wrong"] += 1
        elif verdict == "buyer_plus":
            c["rule_tail"] += 1
            c["tail_right"] += truth == "buy"
        else:
            c["rule_drop"] += 1
    return c


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def study(cap_rows: list[dict], helius, max_credits: int = 40_000, span_s: float = 180.0,
          max_pools: int = 300, now: Optional[float] = None,
          progress: Optional[Callable[[int, int, int], None]] = None, route_classify=None) -> dict:
    """Compare a sample of captured pools against Helius. Fetches each pool's trading window
    (``history_window`` aligned to the capture's first-seen ``create_ts``, like ``merge_capture_flows``),
    stops before a fetch could pass ``max_credits``, and aggregates. Research Helius key only."""
    fresh = freshness(cap_rows, now)
    pools = [cr for cr in cap_rows if cr.get("pool") and cr.get("token_mint")
             and not str(cr.get("token_mint")).endswith("pump")        # hand-made only (drop pump.fun graduations)
             and len(cr.get("slots") or []) >= 1]
    pools.sort(key=lambda cr: hashlib.sha1(cr["pool"].encode()).hexdigest())   # stable: a cap leaves an even sample
    agg = {k: 0 for k in ("all", "direct", "box_visible", "capture", "routed", "lut", "absent",
                           "chain_buyers", "box_buyers", "cap_hit_chain", "cap_hit_box",
                           "routed_seen", "rule_right", "rule_wrong", "rule_tail", "tail_right", "rule_drop")}
    rows, credits, compared, no_chain = [], 0, 0, 0
    per_page = MAX_PAGES * 10
    for i, cr in enumerate(pools):
        if compared >= max_pools:
            break
        if helius is None or credits + per_page > max_credits:
            break
        ct = cr.get("create_ts") or 0
        txs, _ok = helius.history_window(cr["pool"], ct - 30, span_s + 60, MAX_PAGES)
        credits += per_page
        res = compare_pool(cr, txs, route_classify)
        if res is None:
            no_chain += 1
            if progress:
                progress(i + 1, len(pools), credits)
            continue
        compared += 1
        for k in agg:
            agg[k] += res.get(k, 0)
        rows.append({
            "pool": res["pool"][:8] + "...", "slots": res["slots_compared"],
            "have_pct": _pct(res["capture"], res["all"]), "box_pct": _pct(res["box_visible"], res["all"]),
            "dir_pct": _pct(res["direct"], res["all"]),
            "routed": res["routed"], "lut": res["lut"], "absent": res["absent"], "all": res["all"],
            "buyer_recall_chain": _pct(res["cap_hit_chain"], res["chain_buyers"]),
            "buyer_recall_box": _pct(res["cap_hit_box"], res["box_buyers"]),
        })
        if progress:
            progress(i + 1, len(pools), credits)
    all_t = agg["all"]
    missing = all_t - agg["capture"]
    summary = {
        "pools_in_capture": fresh["pools"], "hand_made_pools": len(pools),
        "compared": compared, "no_chain_in_window": no_chain, "credits": credits,
        "chain_trades": all_t, "captured_trades": agg["capture"],
        "have_pct": _pct(agg["capture"], all_t),
        "box_ceiling_pct": _pct(agg["box_visible"], all_t),       # today's LUT-blind decode ceiling
        "lut_ceiling_pct": _pct(agg["direct"], all_t),            # fuller LUT-resolved decode ceiling
        "missing_trades": missing,
        "missing_absent": agg["absent"], "missing_lut": agg["lut"], "missing_routed": agg["routed"],
        "missing_absent_pct": _pct(agg["absent"], missing), "missing_lut_pct": _pct(agg["lut"], missing),
        "missing_routed_pct": _pct(agg["routed"], missing),
        "buyer_recall_chain_pct": _pct(agg["cap_hit_chain"], agg["chain_buyers"]),
        "buyer_recall_box_pct": _pct(agg["cap_hit_box"], agg["box_buyers"]),
    }
    if route_classify is not None:                    # router rules: what they would add on the same pools/slots
        summary.update({
            "routed_seen": agg["routed_seen"], "rule_right": agg["rule_right"], "rule_wrong": agg["rule_wrong"],
            "rule_tail": agg["rule_tail"], "tail_right": agg["tail_right"], "rule_drop": agg["rule_drop"],
            "rule_precision_pct": _pct(agg["rule_right"], agg["rule_right"] + agg["rule_wrong"]),
            "rule_recall_of_routed_pct": _pct(agg["rule_right"], agg["routed_seen"]),
            # headline: real (successful) swaps the box would decode = captured direct + rule-decoded routed
            "decoded_with_rules_pct": _pct(agg["capture"] + agg["rule_right"], all_t),
        })
    return {"freshness": fresh, "summary": summary, "pools": rows}


def render(out: dict) -> str:
    f, s = out["freshness"], out["summary"]
    L = []
    L.append("SHRED-FEED COMPLETENESS -- the box's LocalShred capture vs Helius ground truth")
    L.append("")
    L.append("CAPTURE FRESHNESS")
    if f.get("create_ts_max"):
        L.append(f"  pools in file : {f['pools']}   trading window: {f.get('oldest_iso')} .. {f.get('newest_iso')}"
                 f"  ({f.get('span_hours')} h)")
        age = f.get("newest_age_s") or 0
        L.append(f"  newest pool   : {age/3600.0:.1f} h old"
                 + ("   (stale -- re-capture for a current read)" if age > 6 * 3600 else "   (current)"))
    else:
        L.append(f"  pools in file : {f['pools']}   (no create_ts -- cannot date the capture)")
    L.append("")
    L.append("COVERAGE  (sampled pools, over the box's watch window per pool)")
    L.append(f"  compared      : {s['compared']} pools   ({s['no_chain_in_window']} skipped: no chain trades in "
             f"the window)   {s['credits']} research credits")
    L.append(f"  chain trades  : {s['chain_trades']}   captured: {s['captured_trades']}")
    L.append(f"  HAVE          : {s['have_pct']}%   <- of every chain trade, the share the feed recorded")
    L.append(f"  box ceiling   : {s['box_ceiling_pct']}%   (today's LUT-blind decode; gap to HAVE = feed loss)")
    L.append(f"  + LUT resolve : {s['lut_ceiling_pct']}%   (the fuller decode's ceiling)")
    L.append("")
    L.append(f"WHAT WE MISS  ({s['missing_trades']} trades)")
    L.append(f"  absent (feed loss)      : {s['missing_absent']}  ({s['missing_absent_pct']}%)  "
             f"-- box-visible yet not captured; NO decoder fixes this, only a more complete feed")
    L.append(f"  LUT-recoverable         : {s['missing_lut']}  ({s['missing_lut_pct']}%)  "
             f"-- direct swaps the fuller LUT-resolved decode recovers")
    L.append(f"  routed (aggregator CPI) : {s['missing_routed']}  ({s['missing_routed_pct']}%)  "
             f"-- invisible to any top-level decode; router decoder NOT being built (0.2% of flow)")
    if "decoded_with_rules_pct" in s:
        L.append("")
        L.append("ROUTER RULES  (what route_rules would add, same pools and slots; successful routed swaps only)")
        L.append(f"  routed swaps seen     : {s['routed_seen']}")
        L.append(f"  rule-decoded right    : {s['rule_right']}  wrong: {s['rule_wrong']}  "
                 f"(precision {s['rule_precision_pct']}%, {s['rule_recall_of_routed_pct']}% of routed)")
        L.append(f"  tail (counted a buy)  : {s['rule_tail']}  ({s['tail_right']} really buys)   dropped: {s['rule_drop']}")
        L.append(f"  REAL SWAPS DECODED    : {s['have_pct']}% captured now -> {s['decoded_with_rules_pct']}% with the rules")
    L.append("")
    L.append("BUYER-SET RECALL  (the gate keys on the per-slot buyer set; a missed buyer misreads the symbol)")
    L.append(f"  vs every chain buyer : {s['buyer_recall_chain_pct']}%")
    L.append(f"  vs box-visible buyer : {s['buyer_recall_box_pct']}%")
    L.append("")
    L.append(f"{'pool':11} {'slots':>5} {'have%':>6} {'box%':>6} {'dir%':>6} {'routed':>6} {'lut':>5} "
             f"{'absent':>6} {'all':>5} {'buyrec_chain%':>13} {'buyrec_box%':>11}")
    for r in out["pools"][:60]:
        L.append(f"{r['pool']:11} {r['slots']:>5} {r['have_pct']:>6} {r['box_pct']:>6} {r['dir_pct']:>6} "
                 f"{r['routed']:>6} {r['lut']:>5} {r['absent']:>6} {r['all']:>5} "
                 f"{r['buyer_recall_chain']:>13} {r['buyer_recall_box']:>11}")
    return "\n".join(L)
