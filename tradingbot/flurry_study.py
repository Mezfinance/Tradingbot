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
from collections import Counter

import numpy as np

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

# motif-entry scalp: enter on the DISCOVERED lead (a precursor slot), not on the flurry itself.
# discover() named the quiet->new-wallet transition (.N / ..N / sym == N) as what precedes an N+1 flurry --
# the lead single-feature characterize() missed. So position BEFORE the flurry: buy at the end of the lead
# slot (front of N+1, where the predicted flurry lands) and sell at N+1/N+2. Each fn runs over a _slot_bank rec.
MOTIF_ENTRIES = [
    ("sym==N new wallet",  lambda r: r["sym"] == "N"),
    (".N quiet->new",      lambda r: r["m2"] == ".N"),
    ("..N 2quiet->new",    lambda r: r["m3"] == "..N"),
    (".NN new->new",       lambda r: r["m3"] == ".NN"),
    ("sym==F persistence", lambda r: r["sym"] == "F"),
    (".F quiet->flurry",   lambda r: r["m2"] == ".F"),
]
MOTIF_HOLDS = (1, 2)    # the lead predicts a flurry in N+1, so sell 1-2 slots after the lead slot


# row indices from helius.launch_record txs: [sec, slot, payer, tokΔ, solΔ, tok_after, sol_after]
_SLOT, _PAYER, _TOKD, _SOLD, _TOKA, _SOLA = 1, 2, 3, 4, 5, 6


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
                                      "price": None, "buyers": [], "max_buy": 0.0})
        if kind == "buy":
            s["n_buys"] += 1
            dsol = float(r[_SOLD] or 0.0)                 # SOL added to the pool
            s["buy_sol"] += dsol
            s["max_buy"] = max(s["max_buy"], dsol)
            s["buyers"].append(r[_PAYER])
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


RET_CAP = 3.0   # +300%, as every other pumpswap study: bounds reserve-ratio spikes (a near-full buy can leave
#               # ~0 tokens for one tx, exploding sol_after/tok_after) so a single artifact can't swamp the mean


def _scalp(slots: list[dict], pull_slot, s_idx: int, hold: int, cost: float) -> float | None:
    """Return of buying at the end of flurry slot `slots[s_idx]` (front of the next slot) and selling `hold`
    slots later: exit/entry - 1 - cost, capped to [-1.0, +RET_CAP], or -1.0 if a pull lands in (s, s+hold].
    None if no exit price."""
    s0 = slots[s_idx]["slot"]
    entry = slots[s_idx]["price"]
    if not entry:
        return None
    if pull_slot is not None and s0 < pull_slot <= s0 + hold:
        return -1.0                                       # trapped: the LP was pulled inside our hold
    exit_price = _price_at(slots, s0 + hold)
    if not exit_price:
        return None
    return max(-1.0, min(exit_price / entry - 1.0 - cost, RET_CAP))


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
    motif: dict = {label: {h: [] for h in MOTIF_HOLDS} for label, _fn in MOTIF_ENTRIES}
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
        # motif-entry scalp: enter on a discovered lead slot (buy front of N+1), sell 1-2 slots later. bank is
        # aligned to `slots` by index, so a matched rec i is the same slot _scalp enters on.
        bank = _slot_bank(flow, DEFAULT_FLURRY_MIN, 0.0)
        for label, fn in MOTIF_ENTRIES:
            for i, rec_s in enumerate(bank):
                if fn(rec_s):
                    for h in MOTIF_HOLDS:
                        r = _scalp(slots, pull_slot, i, h, cost)
                        if r is not None:
                            motif[label][h].append(r)
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
    # motif-entry scalp: aggregate each discovered-lead entry vs the same control, pick the best lead
    motif_results = []
    for label, _fn in MOTIF_ENTRIES:
        row = {"motif": label}
        for h in MOTIF_HOLDS:
            row[f"h{h}"] = _agg(motif[label][h])
        motif_results.append(row)
    motif_best = None
    for row in motif_results:
        for h in MOTIF_HOLDS:
            a = row[f"h{h}"]
            c = ctrl[f"h{h}"]
            if a.get("n", 0) >= 50 and a.get("mean", -9) > 0 and a.get("mean", -9) > c.get("mean", 0):
                edge = a["mean"] - c.get("mean", 0)
                if motif_best is None or edge > motif_best["edge"]:
                    motif_best = {"motif": row["motif"], "hold": h, "mean": a["mean"], "control": c.get("mean"),
                                  "edge": edge, "n": a["n"], "win_pct": a["win_pct"], "rugged_pct": a["rugged_pct"]}
    return {
        "hours": hours, "span_s": span_s, "cost_pct": cost_pct, "probed_only": probed_only, "credits": credits,
        "n_pools": len(metas), "n_flows": n_flows, "flurries_total": flurries_total,
        "median_buys_per_slot": statistics.median(buys_per_slot) if buys_per_slot else 0,
        "results": results, "control": ctrl, "best": best,
        "motif_results": motif_results, "motif_best": motif_best,
        "discovered": discovered_scalp(have, metas, DEFAULT_FLURRY_MIN, cost, MOTIF_HOLDS, seed),
    }


# --- lead-lag: what in slot N correlates with a buy flurry in slot N+1 -----------------------------
# The horse (not the scalp): no presupposed cause -- every candidate slot-N event is scored for its
# correlation / AUC / lift with "the very next slot is a buy flurry". The events rank themselves.

DEFAULT_FLURRY_MIN = 3    # a slot is a flurry if it has >= this many buys (sweepable with --flurry-min)
MAXLAG = 10               # autocorrelation of buys/slot out to this many slots (periodicity / a clock)

# event feature at slot N -> does slot N+1 have a flurry.  (name, human description)
EVENTS = [
    ("buys", "buy count in N (does a burst beget a burst)"),
    ("buy_sol", "SOL bought in N"),
    ("max_buy", "biggest single buy in N"),
    ("net_sol", "net buy-minus-sell SOL in N"),
    ("sells", "sell count in N (a dip the bot defends)"),
    ("distinct_buyers", "distinct buyer wallets in N"),
    ("new_buyers", "first-ever buyers in N (a new wallet arrives)"),
    ("outside_buy", "a non-top-3 wallet bought in N (an outsider shows up)"),
    ("dominant_bought", "the pool's #1 buyer bought in N (its own loop)"),
    ("is_flurry_N", "N itself was a flurry (persistence)"),
    ("price_move", "price change over N"),
    ("price_up", "price rose in N"),
    ("slots_since_flurry", "slots since the previous flurry (a clock)"),
    ("age_slots", "pool age in slots at N"),
]
# static conditions, not slot-N events you can detect and react to: reported, but never counted as a
# tradeable precursor (age just says flurries cluster early; slots_since_flurry is a clock).
STATIC_EVENTS = {"age_slots", "slots_since_flurry"}


def _rank_auc(feat: np.ndarray, label: np.ndarray) -> float:
    """Tie-aware Mann-Whitney AUC = P(feature higher when N+1 is a flurry). 0.5 = no lead-lag signal."""
    n1 = float(label.sum())
    n0 = float(len(label) - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(feat, kind="mergesort")
    sf = feat[order]
    ranks_sorted = np.arange(1, len(feat) + 1, dtype=float)
    start = 0
    for i in range(1, len(feat) + 1):               # average ranks within tie groups
        if i == len(feat) or sf[i] != sf[start]:
            ranks_sorted[start:i] = (start + i + 1) / 2.0
            start = i
    ranks = np.empty(len(feat))
    ranks[order] = ranks_sorted
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _pool_rows(flow: dict, flurry_min: int) -> list[tuple[dict, float]]:
    """Per active slot N: the event features and the target (slot N+1 is a flurry)."""
    slots = flow["slots"]
    by_slot = {s["slot"]: s for s in slots}
    cnt: Counter = Counter()
    for s in slots:
        cnt.update(s.get("buyers") or [])
    top3 = {w for w, _ in cnt.most_common(3)}
    top1 = cnt.most_common(1)[0][0] if cnt else None
    seen: set = set()
    last_flurry = None
    first_slot = slots[0]["slot"]
    out = []
    for idx, s in enumerate(slots):
        slot = s["slot"]
        buyers = s.get("buyers") or []
        new_b = {b for b in buyers if b not in seen}
        seen.update(buyers)
        prev_price = slots[idx - 1]["price"] if idx > 0 else s["price"]
        move = (s["price"] / prev_price - 1.0) if prev_price else 0.0
        since = float(min(slot - last_flurry, 50)) if last_flurry is not None else 50.0
        feat = {
            "buys": float(s["n_buys"]), "buy_sol": float(s["buy_sol"]), "max_buy": float(s["max_buy"]),
            "net_sol": float(s["net_sol"]), "sells": float(s["n_sells"]),
            "distinct_buyers": float(len(set(buyers))), "new_buyers": float(len(new_b)),
            "outside_buy": 1.0 if any(b not in top3 for b in buyers) else 0.0,
            "dominant_bought": 1.0 if (top1 and top1 in buyers) else 0.0,
            "is_flurry_N": 1.0 if s["n_buys"] >= flurry_min else 0.0,
            "price_move": move, "price_up": 1.0 if move > 0 else 0.0,
            "slots_since_flurry": since, "age_slots": float(min(slot - first_slot, 1000)),
        }
        nxt = by_slot.get(slot + 1)
        target = 1.0 if (nxt and nxt["n_buys"] >= flurry_min) else 0.0
        out.append((feat, target))
        if s["n_buys"] >= flurry_min:
            last_flurry = slot
    return out


def _dense_buys(flow: dict) -> list[float]:
    """Buy count per slot over [first, last], zero-filled, for autocorrelation."""
    slots = flow["slots"]
    lo, hi = slots[0]["slot"], slots[-1]["slot"]
    series = [0.0] * (hi - lo + 1)
    for s in slots:
        series[s["slot"] - lo] = float(s["n_buys"])
    return series


def _autocorr(series: list[float], maxlag: int) -> list[float]:
    x = np.asarray(series, float)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0 or len(x) < 2:
        return [0.0] * maxlag
    return [float((x[:-k] * x[k:]).sum() / denom) if 0 < k < len(x) else 0.0 for k in range(1, maxlag + 1)]


def characterize(journal, store, cfg=None, helius=None, hours: float = 720.0, span_s: float = DEFAULT_SPAN_S,
                 flurry_min: int = DEFAULT_FLURRY_MIN, probed_only: bool = False, max_credits: int = 20_000,
                 progress=None, now: float | None = None) -> dict:
    """For every candidate slot-N event, its correlation / AUC / lift with a buy flurry in slot N+1, plus the
    autocorrelation of buys/slot (a clock). Answers 'what predicts a next-slot flurry' with no presupposed cause."""
    import time
    now = now if now is not None else time.time()
    since = now - hours * 3600
    scorer = pool_tracker.scorer_rows(store.query, since)
    pools = set(scorer)
    if probed_only:
        pools &= set(pool_tracker.feint_pools(journal.query, since))
    metas = [{"pool": p, "token_mint": (scorer[p] or {}).get("token_mint"),
              "create_ts": (scorer[p] or {}).get("create_ts")}
             for p in pools if (scorer[p] or {}).get("token_mint") and (scorer[p] or {}).get("create_ts")]
    have, credits = fetch_flows(store, helius, metas, span_s, max_credits, progress)
    rows: list[tuple[dict, float]] = []
    autos: list[tuple[list[float], int]] = []
    n_flows = 0
    for m in metas:
        rec = have.get(m["pool"])
        if not rec:
            continue
        flow = slot_flow(rec)
        if not flow:
            continue
        n_flows += 1
        rows.extend(_pool_rows(flow, flurry_min))
        series = _dense_buys(flow)
        autos.append((_autocorr(series, MAXLAG), len(series)))
    out = {"hours": hours, "span_s": span_s, "flurry_min": flurry_min, "probed_only": probed_only,
           "credits": credits, "n_pools": len(metas), "n_flows": n_flows, "n_slots": len(rows)}
    if not rows:
        out["events"] = []
        out["base_rate"] = 0.0
        out["autocorr"] = [0.0] * MAXLAG
        return out
    label = np.array([t for _f, t in rows])
    base = float(label.mean())
    results = []
    for name, desc in EVENTS:
        feat = np.array([f[name] for f, _t in rows], float)
        if feat.std() == 0:
            continue
        auc = _rank_auc(feat, label)
        corr = float(np.corrcoef(feat, label)[0, 1])
        thr = float(np.median(feat))
        hi = feat > thr
        if hi.sum() == 0 or hi.sum() == len(feat):
            hi = feat > 0
        rate_hi = float(label[hi].mean()) if hi.sum() else 0.0
        rate_lo = float(label[~hi].mean()) if (~hi).sum() else 0.0
        results.append({"event": name, "desc": desc, "corr": corr, "auc": auc,
                        "rate_present": rate_hi, "rate_absent": rate_lo,
                        "lift": (rate_hi / base) if base else 0.0, "n_present": int(hi.sum())})
    results.sort(key=lambda r: abs(r["corr"]), reverse=True)
    tw = float(sum(w for _a, w in autos)) or 1.0
    autocorr = [sum(a[k] * w for a, w in autos) / tw for k in range(MAXLAG)]
    out.update({"base_rate": base, "events": results, "autocorr": autocorr})
    return out


def render_char(out: dict) -> str:
    L = []
    L.append("WHAT IN SLOT N CORRELATES WITH A BUY FLURRY IN SLOT N+1  (lead-lag, no presupposed cause)")
    L.append(f"  pools {out['n_pools']} ({out['n_flows']} with a flow){', probed only' if out['probed_only'] else ''}; "
             f"span {out['span_s']:.0f}s; flurry = >={out['flurry_min']} buys/slot; {out['credits']} research credits")
    L.append(f"  {out['n_slots']} active slots; base rate of a flurry in the NEXT slot = {out['base_rate']*100:.1f}%")
    if not out["events"]:
        L.append("  no usable slots.")
        return "\n".join(L)
    L.append("")
    L.append(f"  {'event (slot N)':<20}{'corr':>8}{'AUC':>7}{'P(flurry N+1)':>15}{'lift':>7}{'n':>8}   description")
    for r in out["events"]:
        L.append(f"  {r['event']:<20}{r['corr']:+8.3f}{r['auc']:7.3f}"
                 f"{r['rate_present']*100:13.1f}% {r['lift']:6.2f}x{r['n_present']:8d}   {r['desc']}")
    L.append("")
    ac = out["autocorr"]
    L.append("  buys/slot autocorrelation by lag (a clock shows as a peak at the period):")
    L.append("    " + "  ".join(f"L{k+1}:{ac[k]:+.2f}" for k in range(len(ac))))
    L.append("")
    lag1 = ac[0] if ac else 0.0
    # actionable = a real slot-N EVENT (not a static condition) that discriminates the next-slot flurry.
    # Pearson corr is suppressed by the ~1% base rate, so gate on AUC + lift, not corr.
    actionable = [r for r in out["events"]
                  if r["event"] not in STATIC_EVENTS and (r["auc"] - 0.5) >= 0.1 and r["lift"] >= 1.5]
    best_p = max((r["rate_present"] for r in actionable), default=0.0)
    static_hit = next((r for r in out["events"] if r["event"] in STATIC_EVENTS and abs(r["corr"]) >= 0.1), None)
    if actionable and best_p >= 0.25:
        names = ", ".join(f"{r['event']} ({r['auc']:.2f} AUC, {r['lift']:.1f}x)" for r in actionable[:4])
        L.append(f"  SIGNAL: an actionable slot-N precursor makes the next-slot flurry likely -- {names} "
                 f"(P {best_p*100:.0f}%, base {out['base_rate']*100:.1f}%).")
        L.append("  -> fit a 1-slot-ahead predictor, position BEFORE the flurry (buy in N, ride N+1), validate live")
        L.append("     with the 1-slot probe; read --scalp for the price edge first.")
    elif actionable:
        names = ", ".join(f"{r['event']} ({r['auc']:.2f} AUC, {r['lift']:.1f}x)" for r in actionable[:4])
        L.append(f"  WEAK CONDITIONAL PRECURSOR ONLY: a busier slot N leads the next-slot flurry -- {names} -- but")
        L.append(f"  P(flurry next) rises only to {best_p*100:.1f}% (base {out['base_rate']*100:.1f}%)"
                 + (f", the top raw correlation is {static_hit['event']} (a static condition, not a precursor)," if static_hit else ",")
                 + f" and buys mean-revert at lag 1 ({lag1:+.2f}).")
        L.append("  -> too weak to position on alone. The go/no-go is whether a flurry slot PAYS a 1-2 slot scalp (--scalp)")
        L.append("     and whether --discover finds a combination; otherwise ShredStream has little to key on.")
    else:
        top = out["events"][0]
        L.append(f"  NO SIGNAL: nothing in slot N predicts a flurry in N+1 (top: {top['event']} corr {top['corr']:+.3f}, "
                 f"AUC {top['auc']:.3f}); lag-1 autocorr {lag1:+.2f}. Flurries are ~memoryless one slot ahead, so no")
        L.append("  detector can position ahead of them -- the ShredStream backrun has nothing to key on. Drop it.")
    return "\n".join(L)


# --- discover: let the data name the event (no hand-picked list) -----------------------------------
# Unsupervised complement to characterize(): mechanically generate a large bank of candidate rules per
# slot (thresholds across every numeric, wallet-archetype booleans, and 1-3 slot symbol motifs), then
# keep only the ones whose lead on "N+1 is a flurry" SURVIVES a time-holdout -- discover the pools in
# the earlier half, confirm on the later half -- with a family-wise z guard for the number of rules
# tested, so a flattering-but-spurious rule is reported as noise rather than a signal. The survivor's
# rule string IS the discovered event: readable, so you can name it.

NUMERIC_FIELDS = [
    "n_buys", "buy_sol", "max_buy", "net_sol", "n_sells", "sell_sol", "distinct_buyers", "new_buyers",
    "mean_buy", "abs_move", "slots_since_flurry", "age_slots",
    "prev_buys", "prev_buy_sol", "prev_new_buyers", "prev_net_sol",
]
BOOL_FIELDS = ["outside_buy", "dominant_bought", "new_present", "sell_present", "price_up"]
THRESH_Q = (50, 75, 90)   # quantiles (of the train slice) each numeric is cut at
DISC_MIN_SUPPORT = 30     # a rule must have >= this many present AND absent slots in BOTH halves
DISC_TOP = 20             # survivors reported
DISC_SCALP_TOP = 12       # holdout-confirmed survivor rules carried into the discovered-rule scalp
SYM_DESC = {"F": "N was itself a flurry", "S": "net sellers in N", "N": "a brand-new wallet bought in N",
            "O": "an outsider (non-top-3) bought in N", "B": "the dominant wallet looped in N",
            "b": "light buying in N", ".": "a dead slot"}


def _slot_bank(flow: dict, flurry_min: int, create_ts: float) -> list[dict]:
    """Per active slot N: a generic numeric feature dict, a discrete symbol, 2- and 3-slot motifs ending at N,
    the target (N+1 is a flurry) and the pool's create_ts (for the time split). No hand-picked events."""
    slots = flow["slots"]
    by_slot = {s["slot"]: s for s in slots}
    cnt: Counter = Counter()
    for s in slots:
        cnt.update(s.get("buyers") or [])
    top3 = {w for w, _ in cnt.most_common(3)}
    top1 = cnt.most_common(1)[0][0] if cnt else None
    seen: set = set()
    last_flurry = None
    first = slots[0]["slot"]
    recs = []
    prev_sym, prev2_sym, prev = ".", ".", None
    for idx, s in enumerate(slots):
        slot = s["slot"]
        buyers = s.get("buyers") or []
        new_b = [b for b in buyers if b not in seen]
        seen.update(buyers)
        pprice = slots[idx - 1]["price"] if idx > 0 else s["price"]
        move = (s["price"] / pprice - 1.0) if pprice else 0.0
        since = float(min(slot - last_flurry, 50)) if last_flurry is not None else 50.0
        outsider = 1.0 if any(b not in top3 for b in buyers) else 0.0
        dominant = 1.0 if (top1 and top1 in buyers) else 0.0
        nbuys = float(s["n_buys"])
        f = {
            "n_buys": nbuys, "buy_sol": float(s["buy_sol"]), "max_buy": float(s["max_buy"]),
            "net_sol": float(s["net_sol"]), "n_sells": float(s["n_sells"]), "sell_sol": float(s["sell_sol"]),
            "distinct_buyers": float(len(set(buyers))), "new_buyers": float(len(new_b)),
            "mean_buy": (float(s["buy_sol"]) / nbuys if nbuys else 0.0), "abs_move": abs(move),
            "slots_since_flurry": since, "age_slots": float(min(slot - first, 1000)),
            "outside_buy": outsider, "dominant_bought": dominant,
            "new_present": 1.0 if new_b else 0.0, "sell_present": 1.0 if s["n_sells"] > 0 else 0.0,
            "price_up": 1.0 if move > 0 else 0.0,
            "prev_buys": (prev["n_buys"] if prev else 0.0), "prev_buy_sol": (prev["buy_sol"] if prev else 0.0),
            "prev_new_buyers": (prev["new_buyers"] if prev else 0.0),
            "prev_net_sol": (prev["net_sol"] if prev else 0.0),
        }
        if s["n_buys"] >= flurry_min:
            sym = "F"
        elif s["n_sells"] > s["n_buys"]:
            sym = "S"
        elif new_b:
            sym = "N"
        elif outsider:
            sym = "O"
        elif dominant:
            sym = "B"
        elif s["n_buys"] > 0:
            sym = "b"
        else:
            sym = "."
        nxt = by_slot.get(slot + 1)
        target = 1.0 if (nxt and nxt["n_buys"] >= flurry_min) else 0.0
        recs.append({"f": f, "sym": sym, "m2": prev_sym + sym, "m3": prev2_sym + prev_sym + sym,
                     "target": target, "create_ts": create_ts})
        prev2_sym, prev_sym, prev = prev_sym, sym, f
        if s["n_buys"] >= flurry_min:
            last_flurry = slot
    return recs


def _build_specs(train: list[dict]) -> list[dict]:
    """Mechanically generate the candidate-rule bank from the TRAIN slice only (no look-ahead): a threshold at
    each quantile of every numeric, each wallet-archetype boolean, each slot symbol, and every 2-/3-slot motif
    with enough train support. Each spec carries a fn(rec)->bool and a readable rule string."""
    specs: list[dict] = []
    for fld in NUMERIC_FIELDS:
        vals = np.array([r["f"][fld] for r in train], float)
        if vals.std() == 0:
            continue
        for q in THRESH_Q:
            thr = float(np.quantile(vals, q / 100.0))
            if thr <= float(vals.min()):
                continue                                  # a threshold at the floor selects everything
            specs.append({"rule": f"{fld} >= {thr:.3g} (q{q})", "kind": "numeric",
                          "fn": (lambda r, _f=fld, _t=thr: r["f"][_f] >= _t)})
    for fld in BOOL_FIELDS:
        specs.append({"rule": f"{fld}", "kind": "bool", "fn": (lambda r, _f=fld: r["f"][_f] >= 1.0)})
    syms = {r["sym"] for r in train}
    for x in sorted(syms):
        specs.append({"rule": f"sym == {x}  ({SYM_DESC.get(x, x)})", "kind": "symbol",
                      "fn": (lambda r, _x=x: r["sym"] == _x)})
    for key, label in (("m2", "N-1..N"), ("m3", "N-2..N")):
        counts = Counter(r[key] for r in train)
        for mot, c in counts.items():
            if c >= DISC_MIN_SUPPORT and len(train) - c >= DISC_MIN_SUPPORT:
                specs.append({"rule": f"{label} = {mot}", "kind": "motif",
                              "fn": (lambda r, _k=key, _m=mot: r[_k] == _m)})
    return specs


def _eval_spec(spec: dict, recs: list[dict], tgt: np.ndarray, base: float) -> dict | None:
    """Present-slot flurry rate, lift and a z-score vs base for one rule on one slice. None if support is thin."""
    pres = np.fromiter((1 if spec["fn"](r) else 0 for r in recs), dtype=bool, count=len(recs))
    n = int(pres.sum())
    if n < DISC_MIN_SUPPORT or len(recs) - n < DISC_MIN_SUPPORT:
        return None
    rate = float(tgt[pres].mean())
    se = (base * (1.0 - base) / n) ** 0.5
    z = (rate - base) / se if se > 0 else 0.0
    return {"n": n, "rate": rate, "lift": (rate / base if base else 0.0), "z": z}


def discover(journal, store, cfg=None, helius=None, hours: float = 720.0, span_s: float = DEFAULT_SPAN_S,
             flurry_min: int = DEFAULT_FLURRY_MIN, probed_only: bool = False, max_credits: int = 20_000,
             progress=None, now: float | None = None) -> dict:
    """Data-driven discovery of the slot-N event that precedes an N+1 flurry: no event named in advance, a large
    auto-generated rule bank + sequence motifs, kept only where the lead survives a time-holdout with a
    family-wise z guard. Complements characterize(); same Helius-cached slot flows, read only."""
    import math
    import time
    now = now if now is not None else time.time()
    since = now - hours * 3600
    scorer = pool_tracker.scorer_rows(store.query, since)
    pools = set(scorer)
    if probed_only:
        pools &= set(pool_tracker.feint_pools(journal.query, since))
    metas = [{"pool": p, "token_mint": (scorer[p] or {}).get("token_mint"),
              "create_ts": (scorer[p] or {}).get("create_ts")}
             for p in pools if (scorer[p] or {}).get("token_mint") and (scorer[p] or {}).get("create_ts")]
    have, credits = fetch_flows(store, helius, metas, span_s, max_credits, progress)
    recs: list[dict] = []
    n_flows = 0
    for m in metas:
        rec = have.get(m["pool"])
        if not rec:
            continue
        flow = slot_flow(rec)
        if not flow:
            continue
        n_flows += 1
        recs.extend(_slot_bank(flow, flurry_min, m["create_ts"] or 0.0))
    out = {"hours": hours, "span_s": span_s, "flurry_min": flurry_min, "probed_only": probed_only,
           "credits": credits, "n_pools": len(metas), "n_flows": n_flows, "n_slots": len(recs),
           "survivors": [], "tested": 0, "base_train": 0.0, "base_hold": 0.0, "split": "none", "zc": 0.0}
    if len(recs) < 4 * DISC_MIN_SUPPORT:
        return out
    # time split by pool create_ts so the holdout pools are disjoint in time from the discovery pools
    tss = sorted({r["create_ts"] for r in recs})
    cut = tss[len(tss) // 2] if len(tss) > 1 else None
    train = [r for r in recs if cut is not None and r["create_ts"] < cut]
    hold = [r for r in recs if cut is not None and r["create_ts"] >= cut]
    if len(train) < 2 * DISC_MIN_SUPPORT or len(hold) < 2 * DISC_MIN_SUPPORT:
        rng = random.Random(0)                            # fall back to a random slot split (pools too clustered in time)
        idx = list(range(len(recs)))
        rng.shuffle(idx)
        k = len(idx) // 2
        train = [recs[i] for i in idx[:k]]
        hold = [recs[i] for i in idx[k:]]
        out["split"] = "random"
    else:
        out["split"] = "time"
    tgt_tr = np.array([r["target"] for r in train], float)
    tgt_ho = np.array([r["target"] for r in hold], float)
    base_tr, base_ho = float(tgt_tr.mean()), float(tgt_ho.mean())
    out["base_train"], out["base_hold"] = base_tr, base_ho
    specs = _build_specs(train)
    evaluated = []
    for spec in specs:
        tr = _eval_spec(spec, train, tgt_tr, base_tr)
        if tr is None or tr["lift"] == 1.0:
            continue
        ho = _eval_spec(spec, hold, tgt_ho, base_ho)
        if ho is None:
            continue
        evaluated.append((spec, tr, ho))
    k = max(len(evaluated), 2)
    zc = max(3.0, math.sqrt(2.0 * math.log(k)))           # family-wise guard: ~max of k standard normals
    out["tested"] = len(evaluated)
    out["zc"] = zc
    survivors = []
    for spec, tr, ho in evaluated:
        same_dir = (tr["lift"] > 1) == (ho["lift"] > 1)   # the lead must point the same way in both halves
        if same_dir and abs(ho["z"]) >= zc:
            survivors.append({"rule": spec["rule"], "kind": spec["kind"],
                              "lift_train": tr["lift"], "lift_hold": ho["lift"], "rate_hold": ho["rate"],
                              "z_hold": ho["z"], "n_hold": ho["n"]})
    survivors.sort(key=lambda d: abs(d["z_hold"]), reverse=True)
    out["survivors"] = survivors[:DISC_TOP]
    out["n_survivors"] = len(survivors)
    return out


def render_discover(out: dict) -> str:
    L = []
    L.append("DISCOVERED EVENTS: what in slot N leads an N+1 flurry  (data-driven, holdout-confirmed, no event named)")
    L.append(f"  pools {out['n_pools']} ({out['n_flows']} with a flow){', probed only' if out['probed_only'] else ''}; "
             f"span {out['span_s']:.0f}s; flurry = >={out['flurry_min']} buys/slot; {out['credits']} research credits")
    L.append(f"  {out['n_slots']} active slots; {out['split']} split; "
             f"base rate of a next-slot flurry train {out['base_train']*100:.1f}% / holdout {out['base_hold']*100:.1f}%")
    if not out["n_slots"] or out["tested"] == 0:
        L.append("  too few slots to discover anything.")
        return "\n".join(L)
    L.append(f"  {out['tested']} candidate rules auto-generated and holdout-tested; "
             f"family-wise z threshold {out['zc']:.2f} (guards against spurious winners)")
    L.append("")
    if out["survivors"]:
        L.append(f"  {'discovered rule (slot N)':<34}{'train lift':>11}{'hold lift':>11}{'P(flurry)':>11}{'z':>7}{'n':>7}")
        for r in out["survivors"]:
            L.append(f"  {r['rule']:<34}{r['lift_train']:10.2f}x{r['lift_hold']:10.2f}x"
                     f"{r['rate_hold']*100:10.1f}%{r['z_hold']:+7.1f}{r['n_hold']:7d}")
        L.append("")
        top = out["survivors"][0]
        direction = "more" if top["lift_hold"] > 1 else "less"
        L.append(f"  DISCOVERED ({out['n_survivors']} rule(s) survive the holdout): the strongest is "
                 f"'{top['rule']}' -> a next-slot flurry is {top['lift_hold']:.1f}x {'likelier' if top['lift_hold']>1 else 'rarer'} "
                 f"when it holds ({direction} than base).")
        L.append("  -> these are candidate precursors the data named itself, confirmed out of sample. Next: fold them into")
        L.append("     a 1-slot-ahead flurry predictor, cross-check against characterize(), and validate live with the 1-slot probe.")
    else:
        L.append(f"  NOTHING SURVIVES THE HOLDOUT: of {out['tested']} auto-generated rules, none leads a next-slot flurry")
        L.append("  on the held-out pools beyond chance. Any in-sample pattern was overfit -- consistent with NO SIGNAL:")
        L.append("  flurries are ~memoryless one slot ahead, so there is no event to key a ShredStream backrun on. Drop it.")
    return "\n".join(L)


# --- discovered-rule scalp: scalp the holdout-confirmed survivors, out of sample ------------------
# Closes the loop between discover() and the scalp: discover names the rules that LEAD a next-slot flurry
# out of sample; this measures whether ENTERING on each of those exact rules (buy front of N+1, sell N+h)
# actually PAYS on price -- on the same holdout pools the rule was confirmed on, so it is out of sample too.
# A rule can lead a flurry yet not move price enough to beat cost; that gap is the whole question.

def discovered_scalp(have: dict, metas: list[dict], flurry_min: int, cost: float,
                     holds=MOTIF_HOLDS, seed: int = 0) -> dict:
    import math
    pools = []
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if not flow:
            continue
        pools.append({"flow": flow, "bank": _slot_bank(flow, flurry_min, m.get("create_ts") or 0.0),
                      "ct": m.get("create_ts") or 0.0})
    out = {"n_pools": len(pools), "n_survivors": 0, "results": [], "control": {}, "best": None,
           "split": "none", "zc": 0.0}
    if len(pools) < 4:
        return out
    tss = sorted({p["ct"] for p in pools})
    cut = tss[len(tss) // 2] if len(tss) > 1 else None
    train = [p for p in pools if cut is not None and p["ct"] < cut]
    hold = [p for p in pools if cut is not None and p["ct"] >= cut]
    if len(train) < 2 or len(hold) < 2:                   # pools clustered in time -> random pool split
        rng0 = random.Random(0)
        idx = list(range(len(pools)))
        rng0.shuffle(idx)
        k = len(idx) // 2
        train, hold, out["split"] = [pools[i] for i in idx[:k]], [pools[i] for i in idx[k:]], "random"
    else:
        out["split"] = "time"
    train_recs = [r for p in train for r in p["bank"]]
    hold_recs = [r for p in hold for r in p["bank"]]
    if len(train_recs) < 2 * DISC_MIN_SUPPORT or len(hold_recs) < 2 * DISC_MIN_SUPPORT:
        return out
    tgt_tr = np.array([r["target"] for r in train_recs], float)
    tgt_ho = np.array([r["target"] for r in hold_recs], float)
    base_tr, base_ho = float(tgt_tr.mean()), float(tgt_ho.mean())
    evald = []
    for spec in _build_specs(train_recs):                 # discover the rules on the train pools only
        tr = _eval_spec(spec, train_recs, tgt_tr, base_tr)
        if tr is None or tr["lift"] == 1.0:
            continue
        ho = _eval_spec(spec, hold_recs, tgt_ho, base_ho)
        if ho is None:
            continue
        evald.append((spec, tr, ho))
    zc = max(3.0, math.sqrt(2.0 * math.log(max(len(evald), 2))))
    out["zc"] = zc
    survivors = [(spec, ho) for spec, tr, ho in evald
                 if ((tr["lift"] > 1) == (ho["lift"] > 1)) and ho["lift"] > 1 and ho["z"] >= zc]
    survivors.sort(key=lambda x: x[1]["z"], reverse=True)
    survivors = survivors[:DISC_SCALP_TOP]
    out["n_survivors"] = len(survivors)
    rng = random.Random(seed)
    results = []
    for spec, ho in survivors:                            # scalp each survivor on the HOLDOUT pools (out of sample)
        agg = {h: [] for h in holds}
        for p in hold:
            slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
            for i, rec_s in enumerate(p["bank"]):
                if spec["fn"](rec_s):
                    for h in holds:
                        r = _scalp(slots, pull, i, h, cost)
                        if r is not None:
                            agg[h].append(r)
        row = {"rule": spec["rule"], "lift_hold": ho["lift"], "z_hold": ho["z"]}
        for h in holds:
            row[f"h{h}"] = _agg(agg[h])
        results.append(row)
    control = {h: [] for h in holds}
    for p in hold:
        slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
        quiet = [i for i, s in enumerate(slots) if s["n_buys"] < 3]
        rng.shuffle(quiet)
        for i in quiet[:CONTROL_PER_POOL]:
            for h in holds:
                r = _scalp(slots, pull, i, h, cost)
                if r is not None:
                    control[h].append(r)
    ctrl = {f"h{h}": _agg(control[h]) for h in holds}
    best = None
    for row in results:
        for h in holds:
            a, c = row[f"h{h}"], ctrl[f"h{h}"]
            if a.get("n", 0) >= 50 and a.get("mean", -9) > 0 and a.get("mean", -9) > c.get("mean", 0):
                edge = a["mean"] - c.get("mean", 0)
                if best is None or edge > best["edge"]:
                    best = {"rule": row["rule"], "hold": h, "mean": a["mean"], "control": c.get("mean"),
                            "edge": edge, "n": a["n"], "win_pct": a["win_pct"], "median": a["median"]}
    out.update({"results": results, "control": ctrl, "best": best, "n_hold": len(hold)})
    return out


# --- render (scalp) --------------------------------------------------------------------------------

def _c(a: dict) -> str:
    # mean is capped at +/-RET_CAP so skewed memecoin returns don't drown the average; median + win% are the
    # robust reads (a scalp edge needs a mean clearly > cost AND the median/win% to agree, not one outlier).
    if not a or not a.get("n"):
        return f"{'-':>26}"
    s = f"{a['mean']*100:+.0f}% md{a['median']*100:+.0f}% w{a['win_pct']:.0f} n{a['n']}"
    return f"{s:>26}"


def render(out: dict) -> str:
    L = []
    L.append("BUY-FLURRY SLOT-CONTINUATION STUDY  (reactive-backrun scalp: buy front of N+1, sell N+h)")
    L.append(f"  pools {out['n_pools']} ({out['n_flows']} with a slot flow){', probed only' if out['probed_only'] else ''}; "
             f"span {out['span_s']:.0f}s; cost {out['cost_pct']:.2f}%; {out['credits']} research credits")
    L.append(f"  median buys/slot {out['median_buys_per_slot']:.0f}; flurries (buys>=3) {out['flurries_total']}")
    L.append(f"  each cell = net scalp mean (capped +-{RET_CAP*100:.0f}%) / median / win% / n")
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
    if "motif_results" in out:
        L.append("")
        L.append("  MOTIF-ENTRY SCALP  (position BEFORE the flurry on the discovered lead: buy front of N+1, sell N+h)")
        L.append(f"  {'lead slot N':<20}" + "".join(f"{'hold '+str(h):>28}" for h in MOTIF_HOLDS))
        for row in out["motif_results"]:
            L.append(f"  {row['motif']:<20}" + "".join(f"  {_c(row[f'h{h}'])}" for h in MOTIF_HOLDS))
        L.append(f"  {'control':<20}" + "".join(f"  {_c(out['control'][f'h{h}'])}" for h in MOTIF_HOLDS))
        mb = out.get("motif_best")
        if mb:
            L.append(f"  BEST LEAD: {mb['motif']} hold {mb['hold']} -> net {mb['mean']*100:+.2f}% "
                     f"(control {mb['control']*100:+.2f}%, edge {mb['edge']*100:+.2f} pts) over n={mb['n']}, "
                     f"win {mb['win_pct']:.0f}%, rugged {mb['rugged_pct']:.0f}%")
            L.append("  -> the discovered precursor pays as an entry. Validate live with the 1-slot probe before ShredStream.")
        else:
            L.append("  NO LEAD clears cost + the control: the discovered precursor leads a flurry but not enough PRICE")
            L.append("  to scalp. The lead-lag is real yet not tradeable at slot scale on untouched pools.")
    d = out.get("discovered")
    if d is not None:
        L.append("")
        L.append("  DISCOVERED-RULE SCALP  (auto-fed from discover's survivors; scalped OUT OF SAMPLE on the holdout pools)")
        L.append(f"  {d['n_survivors']} holdout-confirmed rule(s), {d['split']} split, {d.get('n_hold', 0)} holdout pools; "
                 f"each scalps buy front of N+1, sell N+h")
        if d["results"]:
            L.append(f"  {'discovered rule (enter on it)':<34}" + "".join(f"{'hold '+str(h):>28}" for h in MOTIF_HOLDS))
            for row in d["results"]:
                L.append(f"  {row['rule'][:34]:<34}" + "".join(f"  {_c(row[f'h{h}'])}" for h in MOTIF_HOLDS))
            L.append(f"  {'control':<34}" + "".join(f"  {_c(d['control'][f'h{h}'])}" for h in MOTIF_HOLDS))
        db = d.get("best")
        if db:
            L.append(f"  BEST DISCOVERED RULE: '{db['rule']}' hold {db['hold']} -> net {db['mean']*100:+.1f}% "
                     f"(median {db['median']*100:+.1f}%, control {db['control']*100:+.1f}%, edge {db['edge']*100:+.1f} pts) "
                     f"n={db['n']}, win {db['win_pct']:.0f}%")
            L.append("  -> a discovered lead that also PAYS on price out of sample. Validate live with the 1-slot probe.")
        else:
            L.append("  NO DISCOVERED RULE pays on price out of sample: the leads predict a next-slot flurry but the price")
            L.append("  move doesn't beat cost. Lead-lag real, scalp not -> ShredStream has nothing profitable to key on.")
    return "\n".join(L)
