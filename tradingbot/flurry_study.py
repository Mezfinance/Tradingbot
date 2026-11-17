"""Slot-level buy-flurry continuation study (11-17, read only; Helius research key only with --fetch).

The question behind the ShredStream reactive-backrun idea: when a buy flurry lands in one Solana slot, does the
NEXT 1-2 slots show enough price continuation to beat a round-trip cost + tip? That is the edge a reactive scalp
needs -- detect the flurry in slot N (via ShredStream, sub-slot), buy at the front of N+1, sell at N+2/N+3,
riding the wash bot's continued buying. And a <=2-slot exposure should dodge the operator's reactive LP pull
(canary-slots: 100% of pulls land >=2 slots after our buy, median ~18s), which is why this is the one framing
where "money stays safe" is data-backed.

For a sample of pools we read the first `span_s` of trading (Helius `history_from_start`, oldest-first) and, via
`helius.launch_record`, get one row per tx: [sec, slot, payer, pool_token_change, pool_sol_change, tokens_after,
sol_after]. From those we rebuild a per-slot series: price = reserve ratio sol_after/tokens_after carried forward
between trades, per-slot buy count and buy SOL (a buy = tokens leave the pool, SOL enters; a pull = both leave),
and the first LP-pull slot. Then for each flurry definition (a slot whose buy signal clears a threshold) we measure
the net scalp return at holds of 1/2/3 slots -- buy at price_end(s) (front of s+1), sell at price_end(s+h), minus
`cost_pct`, and -100% if a pull lands in (s, s+h] -- against a random non-flurry active-slot control, with the
pull-in-window rate.

Observational (untouched pools), so hypothesis-generating per the methodology invariant -- but a 1-2 slot scalp
exits BEFORE the reactive pull window, so it is far less contaminated than the 10-60s ignition studies that didn't
reproduce (our buy can't be reacted-to-and-pulled inside 2 slots; it only adds a little buy pressure to the flow
we ride, a modest haircut). A positive result is validated live with the existing 1-slot probe infra before any
ShredStream build. CLI `flurry-study [--hours N] [--span-s 180] [--cost-pct 1.0] [--fetch] [--max-credits N]
[--rps 2] [--probed-only] [--json]`.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics

from tradingbot import pool_tracker
from tradingbot.data import helius as he

SCHEMA = "CREATE TABLE IF NOT EXISTS watch_flurry(pool TEXT PRIMARY KEY, data TEXT, ts REAL)"
MAX_PAGES = 6          # history_from_start pages (10 credits / 100 txs)
MINUTE6_S = 360

# flurry definitions: (label, field, threshold).  A slot is a flurry slot if field >= threshold.
# field 'n_buys' = buy txs in the slot; 'buy_sol' = SOL bought into the pool that slot; 'net_sol' = buy - sell SOL.
FLURRY_DEFS = [
    ("buys>=3", "n_buys", 3), ("buys>=5", "n_buys", 5), ("buys>=8", "n_buys", 8),
    ("buy>=1sol", "buy_sol", 1.0), ("buy>=3sol", "buy_sol", 3.0), ("buy>=5sol", "buy_sol", 5.0),
    ("net>=1sol", "net_sol", 1.0), ("net>=3sol", "net_sol", 3.0),
]
HOLDS = (1, 2, 3)      # sell this many slots after the flurry slot
DEFAULT_SPAN_S = 180.0
DEFAULT_COST_PCT = 1.0  # round-trip pool fee (~0.6%) + a modest Jito tip, as a % of notional
CONTROL_PER_POOL = 60   # random non-flurry active slots sampled per pool for the baseline


# row indices from helius.launch_record txs
_SLOT, _TOKD, _SOLD, _TOKA, _SOLA = 1, 3, 4, 5, 6


def classify(tok_change: float, sol_change: float) -> str:
    """A pumpswap tx by the sign of the pool's token and SOL vault changes."""
    if tok_change < 0 and sol_change > 0:
        return "buy"        # tokens leave the pool, SOL enters
    if tok_change > 0 and sol_change < 0:
        return "sell"       # tokens enter, SOL leaves
    if tok_change < 0 and sol_change < 0:
        return "pull"       # both leave = an LP withdrawal (the rug)
    if tok_change > 0 and sol_change > 0:
        return "add"        # both enter = a deposit
    return "other"


def slot_flow(rec: dict) -> dict | None:
    """Rebuild the per-slot series from a launch_record dict. Returns None if there is no usable price path.
    `slots` is a sorted list of per-slot dicts {slot, n_buys, buy_sol, n_sells, sell_sol, price}; `price` is the
    reserve ratio at the END of that slot. `pull_slot` is the first slot with an LP withdrawal (or None)."""
    txs = rec.get("txs") or []
    by_slot: dict[int, dict] = {}
    pull_slot = None
    for r in txs:
        slot = r[_SLOT]
        if slot is None:
            continue
        tok_after, sol_after = r[_TOKA], r[_SOLA]
        kind = classify(r[_TOKD] or 0.0, r[_SOLD] or 0.0)
        s = by_slot.setdefault(slot, {"slot": slot, "n_buys": 0, "buy_sol": 0.0, "n_sells": 0, "sell_sol": 0.0,
                                      "price": None})
        if kind == "buy":
            s["n_buys"] += 1
            s["buy_sol"] += float(r[_SOLD] or 0.0)        # SOL added to the pool
        elif kind == "sell":
            s["n_sells"] += 1
            s["sell_sol"] += -float(r[_SOLD] or 0.0)
        elif kind == "pull" and pull_slot is None:
            pull_slot = slot
        if tok_after and tok_after > 0:                   # price after this tx = end-of-slot price so far
            s["price"] = float(sol_after) / float(tok_after)
    slots = [by_slot[k] for k in sorted(by_slot)]
    # carry a price forward over slots whose last tx did not leave a readable reserve
    last = None
    for s in slots:
        if s["price"] is None:
            s["price"] = last
        last = s["price"]
    slots = [s for s in slots if s["price"] and s["price"] > 0]
    if len(slots) < 3:
        return None
    for s in slots:
        s["net_sol"] = s["buy_sol"] - s["sell_sol"]
    return {"slots": slots, "pull_slot": pull_slot}


def _price_at(slots: list[dict], slot: int) -> float | None:
    """Carry-forward price at the end of `slot`: the last active slot at or before it (flat between trades)."""
    price = None
    for s in slots:
        if s["slot"] <= slot:
            price = s["price"]
        else:
            break
    return price


def _scalp(slots: list[dict], pull_slot, s_idx: int, hold: int, cost: float) -> float | None:
    """Return of buying at the end of flurry slot `slots[s_idx]` (front of the next slot) and selling `hold`
    slots later: exit/entry - 1 - cost, or -1.0 if a pull lands in (s, s+hold]. None if no exit price."""
    s0 = slots[s_idx]["slot"]
    entry = slots[s_idx]["price"]
    if not entry:
        return None
    if pull_slot is not None and s0 < pull_slot <= s0 + hold:
        return -1.0                                       # trapped: the LP was pulled inside our hold
    exit_price = _price_at(slots, s0 + hold)
    if not exit_price:
        return None
    return exit_price / entry - 1.0 - cost


def _agg(rets: list[float]) -> dict:
    rets = [r for r in rets if r is not None]
    if not rets:
        return {"n": 0}
    return {"n": len(rets), "mean": statistics.fmean(rets), "median": statistics.median(rets),
            "win_pct": 100.0 * sum(1 for r in rets if r > 0) / len(rets),
            "rugged_pct": 100.0 * sum(1 for r in rets if r <= -0.999) / len(rets)}


def fetch_flows(store, helius, metas: list[dict], span_s: float, max_credits: int, progress=None) -> tuple[dict, int]:
    """Cache the first `span_s` of each pool's trading in watch.db `watch_flurry` (research key, stable hash order
    so a credit cap leaves an even sample). Returns {pool: launch_record dict} and the research credits spent."""
    store.execute(SCHEMA)
    have: dict = {}
    for r in store.query("SELECT pool, data FROM watch_flurry", ()):
        try:
            have[r["pool"]] = json.loads(r["data"])
        except Exception:  # noqa: BLE001
            pass
    if helius is None:
        return have, 0
    order = sorted(metas, key=lambda m: hashlib.sha1(("fl:" + m["pool"]).encode()).hexdigest())
    credits = 0
    for i, m in enumerate(order):
        pool = m["pool"]
        if pool in have and (have[pool].get("txs") or have[pool].get("error")):
            continue
        if credits + MAX_PAGES * 10 > max_credits:
            break
        txs, _more, _err = helius.history_from_start(pool, span_s, MAX_PAGES)
        credits += MAX_PAGES * 10
        rec = he.launch_record(txs, pool, m["token_mint"], window_s=span_s)
        store.execute("INSERT OR REPLACE INTO watch_flurry(pool, data, ts) VALUES(?,?,?)",
                      (pool, json.dumps(rec), m.get("create_ts") or 0))
        have[pool] = rec
        if progress:
            progress(i + 1, len(order), credits)
    return have, credits


def study(journal, store, cfg=None, helius=None, hours: float = 720.0, span_s: float = DEFAULT_SPAN_S,
          cost_pct: float = DEFAULT_COST_PCT, probed_only: bool = False, max_credits: int = 20_000,
          progress=None, now: float | None = None, seed: int = 0) -> dict:
    """Assemble the pool sample, fetch/read each pool's slot flow, and aggregate the scalp returns per flurry
    definition x hold vs a random-slot control. Population = the scorer's finished liquid pools (or only pools we
    probed, with --probed-only). Read only; Helius only with `helius` set."""
    import time
    now = now if now is not None else time.time()
    since = now - hours * 3600
    scorer = pool_tracker.scorer_rows(store.query, since)
    pools = set(scorer)
    if probed_only:
        pools &= set(pool_tracker.feint_pools(journal.query, since))
    metas = []
    for pool in pools:
        s = scorer.get(pool) or {}
        create, mint = s.get("create_ts"), s.get("token_mint")
        if create and mint:
            metas.append({"pool": pool, "token_mint": mint, "create_ts": create})
    have, credits = fetch_flows(store, helius, metas, span_s, max_credits, progress)
    cost = cost_pct / 100.0
    rng = random.Random(seed)
    # per (def label, hold): collected scalp returns; control[hold]: scalp returns from random non-flurry slots
    flurry: dict = {d[0]: {h: [] for h in HOLDS} for d in FLURRY_DEFS}
    control: dict = {h: [] for h in HOLDS}
    n_flows = flurries_total = 0
    buys_per_slot = []
    for m in metas:
        rec = have.get(m["pool"])
        if not rec:
            continue
        flow = slot_flow(rec)
        if not flow:
            continue
        n_flows += 1
        slots, pull_slot = flow["slots"], flow["pull_slot"]
        buys_per_slot.extend(s["n_buys"] for s in slots)
        # flurry slots per definition
        for label, field, thr in FLURRY_DEFS:
            for i, s in enumerate(slots):
                if s[field] >= thr:
                    flurries_total += 1 if label == FLURRY_DEFS[0][0] else 0
                    for h in HOLDS:
                        r = _scalp(slots, pull_slot, i, h, cost)
                        if r is not None:
                            flurry[label][h].append(r)
        # control: random active slots that are NOT a flurry under the loosest definition (buys>=3)
        quiet = [i for i, s in enumerate(slots) if s["n_buys"] < 3]
        rng.shuffle(quiet)
        for i in quiet[:CONTROL_PER_POOL]:
            for h in HOLDS:
                r = _scalp(slots, pull_slot, i, h, cost)
                if r is not None:
                    control[h].append(r)
    results = []
    for label, field, thr in FLURRY_DEFS:
        row = {"flurry": label}
        for h in HOLDS:
            row[f"h{h}"] = _agg(flurry[label][h])
        results.append(row)
    ctrl = {f"h{h}": _agg(control[h]) for h in HOLDS}
    # best cell: the flurry x hold with the highest mean that clears the control and has a sane n
    best = None
    for row in results:
        for h in HOLDS:
            a = row[f"h{h}"]
            c = ctrl[f"h{h}"]
            if a.get("n", 0) >= 50 and a.get("mean", -9) > 0 and a.get("mean", -9) > c.get("mean", 0):
                edge = a["mean"] - c.get("mean", 0)
                if best is None or edge > best["edge"]:
                    best = {"flurry": row["flurry"], "hold": h, "mean": a["mean"], "control": c.get("mean"),
                            "edge": edge, "n": a["n"], "win_pct": a["win_pct"], "rugged_pct": a["rugged_pct"]}
    return {
        "hours": hours, "span_s": span_s, "cost_pct": cost_pct, "probed_only": probed_only, "credits": credits,
        "n_pools": len(metas), "n_flows": n_flows, "flurries_total": flurries_total,
        "median_buys_per_slot": statistics.median(buys_per_slot) if buys_per_slot else 0,
        "results": results, "control": ctrl, "best": best,
    }


# --- render ----------------------------------------------------------------------------------------

def _c(a: dict) -> str:
    if not a or not a.get("n"):
        return f"{'-':>26}"
    return f"{a['mean']*100:+6.2f}% n{a['n']:<5d} w{a['win_pct']:3.0f} r{a['rugged_pct']:3.0f}"


def render(out: dict) -> str:
    L = []
    L.append("BUY-FLURRY SLOT-CONTINUATION STUDY  (reactive-backrun scalp: buy front of N+1, sell N+h)")
    L.append(f"  pools {out['n_pools']} ({out['n_flows']} with a slot flow){', probed only' if out['probed_only'] else ''}; "
             f"span {out['span_s']:.0f}s; cost {out['cost_pct']:.2f}%; {out['credits']} research credits")
    L.append(f"  median buys/slot {out['median_buys_per_slot']:.0f}; flurries (buys>=3) {out['flurries_total']}")
    L.append("  each cell = net scalp mean / n / win% / rugged% (rugged = pulled inside the hold)")
    L.append("")
    head = f"  {'flurry':<11}" + "".join(f"{'hold '+str(h):>28}" for h in HOLDS)
    L.append(head)
    for row in out["results"]:
        L.append(f"  {row['flurry']:<11}" + "".join(f"  {_c(row[f'h{h}'])}" for h in HOLDS))
    L.append(f"  {'control':<11}" + "".join(f"  {_c(out['control'][f'h{h}'])}" for h in HOLDS))
    L.append("")
    b = out.get("best")
    if b:
        L.append(f"  BEST CELL: {b['flurry']} hold {b['hold']} slot(s) -> net {b['mean']*100:+.2f}% "
                 f"(control {b['control']*100:+.2f}%, edge {b['edge']*100:+.2f} pts) over n={b['n']}, "
                 f"win {b['win_pct']:.0f}%, rugged {b['rugged_pct']:.0f}%")
        L.append("  -> a real slot-scale flurry edge over cost. Validate live with the 1-slot probe before any ShredStream build.")
    else:
        L.append("  NO CELL clears cost + the control with a sane n: flurries burst-and-stop at slot scale, so a")
        L.append("  reactive 1-2 slot scalp has no edge. ShredStream would be chasing ~0 -- do not build it.")
    return "\n".join(L)
