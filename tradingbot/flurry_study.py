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
MOTIF_HOLDS = (1, 2)    # the lead predicts the move in N+1, so sell 1-2 slots after the lead slot

DEFAULT_FLURRY_MIN = 3    # a slot is a flurry if it has >= this many buys (sweepable with --flurry-min)
MAXLAG = 10               # autocorrelation of buys/slot out to this many slots (periodicity / a clock)

# --- the predicted target ---------------------------------------------------------------------------
# The whole point of the 11-18 reframe: predict what MAKES MONEY (a price move), not what merely clusters
# (a buy-count flurry). A flurry of a few tiny wash buys is a reliable thing to predict (discover found it
# at z+25) but it moves price a fraction of a % over 1-2 slots, so its scalp dies at cost. "price" asks the
# right question -- does price rise enough next slot(s) to beat a round trip -- so a surviving rule predicts
# money. "flurry" is kept for comparison (--target flurry).
DEFAULT_TARGET = "price"    # "price" (an up-move) | "flurry" (next slot has >= flurry_min buys)
DEFAULT_MOVE_PCT = 1.0      # price target: a rise of this many % counts as a hit (default = the round-trip cost)
DEFAULT_HORIZON = 2         # price target: the up-move may land within the next this-many slots (the scalp window)


def _tgt(kind=DEFAULT_TARGET, flurry_min=DEFAULT_FLURRY_MIN, horizon=DEFAULT_HORIZON, move=DEFAULT_MOVE_PCT) -> dict:
    return {"kind": kind, "flurry_min": flurry_min, "horizon": horizon, "move": move}


def _target_hit(slots: list[dict], by_slot: dict, idx: int, tgt: dict) -> float:
    """1.0 if slot N (=slots[idx]) is followed by the predicted thing, else 0.0. kind 'flurry' = slot N+1 has
    >= flurry_min buys; kind 'price' = price rises >= move% within the next `horizon` slots (end-of-slot
    granularity, carried forward). The price target is what a scalp needs, so a rule that predicts it predicts
    money. `_price_at` resolves at call time, so it may be defined below this helper."""
    s = slots[idx]
    if tgt["kind"] == "flurry":
        nxt = by_slot.get(s["slot"] + 1)
        return 1.0 if (nxt and nxt["n_buys"] >= tgt["flurry_min"]) else 0.0
    base = s["price"]
    if not base:
        return 0.0
    thr = tgt["move"] / 100.0
    for k in range(1, tgt["horizon"] + 1):
        p = _price_at(slots, s["slot"] + k)
        if p and p / base - 1.0 >= thr:
            return 1.0
    return 0.0


def target_label(tgt: dict) -> str:
    if tgt["kind"] == "flurry":
        return f"a buy flurry (>={tgt['flurry_min']} buys) in slot N+1"
    return f"a >={tgt['move']:.1f}% price up-move in the next {tgt['horizon']} slot(s)"


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


def _best_cell(rows: list[dict], ctrl: dict, holds, key: str) -> dict | None:
    """Pick the strongest cell that beats the control on mean AND median AND win% at n>=50. The triple guard
    is the point: with the +300% cap a single reserve-ratio outlier (a +300% artifact at a ~1% win rate) can
    lift the MEAN above the control, so mean alone declared false edges ('BEST LEAD ... win 1% -> pays'). A
    real price edge also beats the control's median and wins at least as often; an outlier beats neither."""
    best = None
    for row in rows:
        for h in holds:
            a, c = row[f"h{h}"], ctrl.get(f"h{h}", {})
            if a.get("n", 0) < 50 or a.get("mean", -9) <= 0:
                continue
            if a["mean"] <= c.get("mean", 0) or a.get("median", -9) < c.get("median", -9) \
                    or a.get("win_pct", 0) < c.get("win_pct", 0):
                continue
            edge = a["mean"] - c.get("mean", 0)
            if best is None or edge > best["edge"]:
                best = {key: row[key], "hold": h, "mean": a["mean"], "control": c.get("mean"), "edge": edge,
                        "n": a["n"], "win_pct": a["win_pct"], "median": a.get("median"),
                        "rugged_pct": a.get("rugged_pct", 0.0)}
    return best


def metas_from_pools(path: str) -> list[dict]:
    """Load a pool universe captured live (box absmove.py MODE=capture): a JSON list or JSONL of
    {pool, token_mint, create_ts}. This is how route #1 runs the study's literal code on exactly the WSOL-base
    pumpswap pools ShredStream surfaced live, instead of pool_tracker.scorer_rows."""
    metas, seen = [], set()
    with open(path) as fh:
        text = fh.read().strip()
    rows = []
    if text.startswith("["):
        rows = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    for r in rows:
        pool, mint = r.get("pool"), r.get("token_mint")
        if not pool or not mint or pool in seen:
            continue
        seen.add(pool)
        metas.append({"pool": pool, "token_mint": mint, "create_ts": r.get("create_ts") or 0})
    return metas


def flows_from_file(path: str) -> tuple[list[dict], dict]:
    """Load launch_record recs reconstructed live by the box surf.py (MODE=surf): a JSON list or JSONL of
    {pool, token_mint, create_ts, rec}, where rec is a helius.launch_record-shaped dict {create_ts, txs:[...]}.
    Returns (metas, have) so study/discover/characterize run their LITERAL code on live-captured slot flows,
    bypassing Helius. create_ts = the pool's first-seen wall-clock, so discover's time split is a true
    prospective live walk-forward (early-session pools train, later-session pools test). Route #1 (live)."""
    with open(path) as fh:
        text = fh.read().strip()
    rows = json.loads(text) if text.startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    metas, have, seen = [], {}, set()
    for r in rows:
        pool, mint, rec = r.get("pool"), r.get("token_mint"), r.get("rec")
        if not pool or not mint or not rec or pool in seen:
            continue
        seen.add(pool)
        ct = r.get("create_ts") or rec.get("create_ts") or 0
        metas.append({"pool": pool, "token_mint": mint, "create_ts": ct})
        have[pool] = rec
    return metas, have


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
          progress=None, now: float | None = None, seed: int = 0, target: str = DEFAULT_TARGET,
          move_pct: float = DEFAULT_MOVE_PCT, horizon: int = DEFAULT_HORIZON, metas=None, have=None) -> dict:
    """Assemble the pool sample, fetch/read each pool's slot flow, and aggregate the scalp returns per flurry
    definition x hold vs a random-slot control, plus the motif-entry and discovered-rule scalps (both pointed at
    `target`). Population = the scorer's finished liquid pools (or only --probed-only), or a live-captured pool
    list passed as `metas` (route #1). Read only."""
    import time
    tgt = _tgt(target, DEFAULT_FLURRY_MIN, horizon, move_pct)
    now = now if now is not None else time.time()
    since = now - hours * 3600
    if metas is None:
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
    if have is None:
        have, credits = fetch_flows(store, helius, metas, span_s, max_credits, progress)
    else:
        credits = 0
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
        bank = _slot_bank(flow, 0.0, tgt)
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
    # best cell: the flurry x hold that beats the control on mean, median AND win% at a sane n (_best_cell)
    best = _best_cell(results, ctrl, HOLDS, "flurry")
    # motif-entry scalp: aggregate each discovered-lead entry vs the same control, pick the best lead
    motif_results = []
    for label, _fn in MOTIF_ENTRIES:
        row = {"motif": label}
        for h in MOTIF_HOLDS:
            row[f"h{h}"] = _agg(motif[label][h])
        motif_results.append(row)
    motif_best = _best_cell(motif_results, ctrl, MOTIF_HOLDS, "motif")
    return {
        "hours": hours, "span_s": span_s, "cost_pct": cost_pct, "probed_only": probed_only, "credits": credits,
        "n_pools": len(metas), "n_flows": n_flows, "flurries_total": flurries_total, "target": tgt,
        "median_buys_per_slot": statistics.median(buys_per_slot) if buys_per_slot else 0,
        "results": results, "control": ctrl, "best": best,
        "motif_results": motif_results, "motif_best": motif_best,
        "discovered": discovered_scalp(have, metas, tgt, cost, MOTIF_HOLDS, seed),
    }


# --- lead-lag: what in slot N correlates with a buy flurry in slot N+1 -----------------------------
# The horse (not the scalp): no presupposed cause -- every candidate slot-N event is scored for its
# correlation / AUC / lift with "the very next slot is a buy flurry". The events rank themselves.

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


def _pool_rows(flow: dict, tgt: dict) -> list[tuple[dict, float]]:
    """Per active slot N: the event features and the target (what we predict follows, per `tgt`)."""
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
            "is_flurry_N": 1.0 if s["n_buys"] >= tgt["flurry_min"] else 0.0,
            "price_move": move, "price_up": 1.0 if move > 0 else 0.0,
            "slots_since_flurry": since, "age_slots": float(min(slot - first_slot, 1000)),
        }
        target = _target_hit(slots, by_slot, idx, tgt)
        out.append((feat, target))
        if s["n_buys"] >= tgt["flurry_min"]:
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
                 progress=None, now: float | None = None, target: str = DEFAULT_TARGET,
                 move_pct: float = DEFAULT_MOVE_PCT, horizon: int = DEFAULT_HORIZON, metas=None, have=None) -> dict:
    """For every candidate slot-N event, its correlation / AUC / lift with the predicted thing in the next
    slot(s) (a price up-move by default; a buy flurry with target='flurry'), plus the buys/slot autocorrelation.
    Answers 'what predicts a next-slot MOVE' with no presupposed cause. `metas` overrides the scorer universe
    with a live-captured pool list (route #1)."""
    import time
    tgt = _tgt(target, flurry_min, horizon, move_pct)
    now = now if now is not None else time.time()
    since = now - hours * 3600
    if metas is None:
        scorer = pool_tracker.scorer_rows(store.query, since)
        pools = set(scorer)
        if probed_only:
            pools &= set(pool_tracker.feint_pools(journal.query, since))
        metas = [{"pool": p, "token_mint": (scorer[p] or {}).get("token_mint"),
                  "create_ts": (scorer[p] or {}).get("create_ts")}
                 for p in pools if (scorer[p] or {}).get("token_mint") and (scorer[p] or {}).get("create_ts")]
    if have is None:
        have, credits = fetch_flows(store, helius, metas, span_s, max_credits, progress)
    else:
        credits = 0
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
        rows.extend(_pool_rows(flow, tgt))
        series = _dense_buys(flow)
        autos.append((_autocorr(series, MAXLAG), len(series)))
    out = {"hours": hours, "span_s": span_s, "flurry_min": flurry_min, "probed_only": probed_only,
           "credits": credits, "n_pools": len(metas), "n_flows": n_flows, "n_slots": len(rows), "target": tgt}
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
    tgt = out.get("target") or _tgt()
    tl = target_label(tgt)
    L = []
    L.append(f"WHAT IN SLOT N PREDICTS {tl.upper()}  (lead-lag, no presupposed cause)")
    L.append(f"  pools {out['n_pools']} ({out['n_flows']} with a flow){', probed only' if out['probed_only'] else ''}; "
             f"span {out['span_s']:.0f}s; target = {tl}; {out['credits']} research credits")
    L.append(f"  {out['n_slots']} active slots; base rate of the target = {out['base_rate']*100:.1f}%")
    if not out["events"]:
        L.append("  no usable slots.")
        return "\n".join(L)
    L.append("")
    L.append(f"  {'event (slot N)':<20}{'corr':>8}{'AUC':>7}{'P(target)':>15}{'lift':>7}{'n':>8}   description")
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
    price = tgt["kind"] == "price"
    if actionable and best_p >= 0.25:
        names = ", ".join(f"{r['event']} ({r['auc']:.2f} AUC, {r['lift']:.1f}x)" for r in actionable[:4])
        L.append(f"  SIGNAL: an actionable slot-N precursor makes {tl} likely -- {names} "
                 f"(P {best_p*100:.0f}%, base {out['base_rate']*100:.1f}%).")
        L.append("  -> fit a 1-slot-ahead predictor, position in N and ride N+1, validate live with the 1-slot probe;")
        L.append("     confirm it PAYS with the motif / discovered-rule scalp (--scalp).")
    elif actionable:
        names = ", ".join(f"{r['event']} ({r['auc']:.2f} AUC, {r['lift']:.1f}x)" for r in actionable[:4])
        L.append(f"  WEAK CONDITIONAL PRECURSOR ONLY: a busier slot N leads {tl} -- {names} -- but")
        L.append(f"  P(target) rises only to {best_p*100:.1f}% (base {out['base_rate']*100:.1f}%)"
                 + (f", the top raw correlation is {static_hit['event']} (a static condition, not a precursor)," if static_hit else ",")
                 + f" and buys mean-revert at lag 1 ({lag1:+.2f}).")
        L.append("  -> too weak to position on alone. The go/no-go is whether it PAYS a 1-2 slot scalp (--scalp) and")
        L.append("     whether --discover finds a combination; otherwise ShredStream has little to key on.")
    else:
        top = out["events"][0]
        tail = ("price is ~a random walk one slot ahead" if price else "flurries are ~memoryless one slot ahead")
        L.append(f"  NO SIGNAL: nothing in slot N predicts {tl} (top: {top['event']} corr {top['corr']:+.3f}, "
                 f"AUC {top['auc']:.3f}); lag-1 autocorr {lag1:+.2f}. {tail}, so no")
        L.append("  detector can position ahead of it -- the ShredStream backrun has nothing to key on. Drop it.")
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
COMBO_MAX_DEPTH = 4       # AND together up to this many single filters (conjunction depth: pairs, triples, quads)
COMBO_BEAM = 24           # beam search: keep this many best partial conjunctions at each depth (wide so high-lift
                          # small-n pairs are not pruned by big-n/high-z ones before a deeper stack off them is tested;
                          # the family-wise z guard auto-widens with the search, so a wider beam cannot overfit)
COMBO_TOP = 15            # conjunction survivors reported
SYM_DESC = {"F": "N was itself a flurry", "S": "net sellers in N", "N": "a brand-new wallet bought in N",
            "O": "an outsider (non-top-3) bought in N", "B": "the dominant wallet looped in N",
            "b": "light buying in N", ".": "a dead slot"}


def _slot_bank(flow: dict, create_ts: float, tgt: dict) -> list[dict]:
    """Per active slot N: a generic numeric feature dict, a discrete symbol, 2- and 3-slot motifs ending at N,
    the target (what `tgt` predicts follows) and the pool's create_ts (for the time split). No hand-picked
    events. The symbol "F" still uses the buy-count flurry threshold; only the TARGET follows `tgt`."""
    flurry_min = tgt["flurry_min"]
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
        target = _target_hit(slots, by_slot, idx, tgt)
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
             progress=None, now: float | None = None, target: str = DEFAULT_TARGET,
             move_pct: float = DEFAULT_MOVE_PCT, horizon: int = DEFAULT_HORIZON, combos: bool = False,
             metas=None, have=None) -> dict:
    """Data-driven discovery of the slot-N event that precedes the predicted thing (a price up-move by default;
    a buy flurry with target='flurry'): no event named in advance, a large auto-generated rule bank + sequence
    motifs, kept only where the lead survives a time-holdout with a family-wise z guard. `metas` overrides the
    scorer universe with a live-captured pool list (route #1). Read only."""
    import math
    import time
    tgt = _tgt(target, flurry_min, horizon, move_pct)
    now = now if now is not None else time.time()
    since = now - hours * 3600
    if metas is None:
        scorer = pool_tracker.scorer_rows(store.query, since)
        pools = set(scorer)
        if probed_only:
            pools &= set(pool_tracker.feint_pools(journal.query, since))
        metas = [{"pool": p, "token_mint": (scorer[p] or {}).get("token_mint"),
                  "create_ts": (scorer[p] or {}).get("create_ts")}
                 for p in pools if (scorer[p] or {}).get("token_mint") and (scorer[p] or {}).get("create_ts")]
    if have is None:
        have, credits = fetch_flows(store, helius, metas, span_s, max_credits, progress)
    else:
        credits = 0
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
        recs.extend(_slot_bank(flow, m["create_ts"] or 0.0, tgt))
    out = {"hours": hours, "span_s": span_s, "flurry_min": flurry_min, "probed_only": probed_only, "target": tgt,
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
    if combos and len(evaluated) >= 2:
        out.update(_combo_layer(evaluated, train, hold, tgt_tr, tgt_ho, base_tr, base_ho))
    else:
        out.update({"combos": [], "n_combos": 0, "combo_zc": 0.0, "combo_tested": 0})
    return out


def _combo_layer(evaluated, train, hold, tgt_tr, tgt_ho, base_tr, base_ho) -> dict:
    """Beam search over CONJUNCTIONS of the single-rule filters (AND them together). A combination is kept only
    if it (a) sharpens the TRAIN signal over its parent at each step, (b) clears a family-wise z guard WIDENED
    for the number of conjunctions tested (so the much larger search space can't manufacture a false positive),
    and (c) is MORE significant out of sample than ANY of its own single filters -- otherwise the combination
    adds nothing a single rule didn't. Selection is on TRAIN only; the holdout is read once per candidate for
    the final judgement. Candidate filters = the single rules that already have support in both halves."""
    import math
    cands = []
    for spec, tr, ho in evaluated:
        mtr = np.fromiter((1 if spec["fn"](r) else 0 for r in train), dtype=bool, count=len(train))
        mho = np.fromiter((1 if spec["fn"](r) else 0 for r in hold), dtype=bool, count=len(hold))
        cands.append({"spec": spec, "mtr": mtr, "mho": mho, "tr": tr, "ho": ho})

    def ev(mask, tgt, base):
        n = int(mask.sum())
        if n < DISC_MIN_SUPPORT or len(mask) - n < DISC_MIN_SUPPORT:
            return None
        rate = float(tgt[mask].mean())
        se = (base * (1.0 - base) / n) ** 0.5
        return {"n": n, "rate": rate, "lift": (rate / base if base else 0.0),
                "z": ((rate - base) / se if se > 0 else 0.0)}

    seeds = sorted(range(len(cands)), key=lambda i: abs(cands[i]["tr"]["z"]), reverse=True)[:COMBO_BEAM]
    beam = [{"idxs": (i,), "mtr": cands[i]["mtr"], "mho": cands[i]["mho"], "tr": cands[i]["tr"]} for i in seeds]
    seen = {frozenset(e["idxs"]) for e in beam}
    tested = []                                               # (idxs, train_stats, hold_stats) for every conjunction evaluated
    k_conj = 0
    for _ in range(2, COMBO_MAX_DEPTH + 1):
        level = []
        for e in beam:
            for j in range(len(cands)):
                if j in e["idxs"]:
                    continue
                key = frozenset(e["idxs"] + (j,))
                if key in seen:
                    continue
                seen.add(key)
                cmtr = e["mtr"] & cands[j]["mtr"]
                tr = ev(cmtr, tgt_tr, base_tr)
                if tr is None:
                    continue
                k_conj += 1
                if abs(tr["z"]) <= abs(e["tr"]["z"]):          # the added filter must sharpen the TRAIN signal
                    continue
                cmho = e["mho"] & cands[j]["mho"]
                ho = ev(cmho, tgt_ho, base_ho)
                idxs = tuple(sorted(e["idxs"] + (j,)))
                level.append({"idxs": idxs, "mtr": cmtr, "mho": cmho, "tr": tr})
                if ho is not None:
                    tested.append((idxs, tr, ho))
        if not level:
            break
        beam = sorted(level, key=lambda x: abs(x["tr"]["z"]), reverse=True)[:COMBO_BEAM]
    k_total = max(len(cands) + k_conj, 2)
    zc = max(3.0, math.sqrt(2.0 * math.log(k_total)))         # stricter than the single-rule guard (bigger search)
    survs = []
    for idxs, tr, ho in tested:
        same_dir = (tr["lift"] > 1) == (ho["lift"] > 1)
        part_z = max((abs(cands[i]["ho"]["z"]) for i in idxs if cands[i]["ho"]), default=0.0)
        part_lift = max((abs(cands[i]["ho"]["lift"] - 1.0) for i in idxs if cands[i]["ho"]), default=0.0)
        if same_dir and abs(ho["z"]) >= zc and abs(ho["z"]) > part_z and abs(ho["lift"] - 1.0) > part_lift:
            survs.append({"rule": "  AND  ".join(cands[i]["spec"]["rule"] for i in idxs), "depth": len(idxs),
                          "lift_train": tr["lift"], "lift_hold": ho["lift"], "rate_hold": ho["rate"],
                          "z_hold": ho["z"], "n_hold": ho["n"], "gain_vs_part": abs(ho["lift"] - 1.0) - part_lift})
    survs.sort(key=lambda d: abs(d["z_hold"]), reverse=True)
    return {"combos": survs[:COMBO_TOP], "n_combos": len(survs), "combo_zc": zc, "combo_tested": k_conj}


def render_discover(out: dict) -> str:
    tgt = out.get("target") or _tgt()
    tl = target_label(tgt)
    L = []
    L.append(f"DISCOVERED EVENTS: what in slot N leads {tl}  (data-driven, holdout-confirmed, no event named)")
    L.append(f"  pools {out['n_pools']} ({out['n_flows']} with a flow){', probed only' if out['probed_only'] else ''}; "
             f"span {out['span_s']:.0f}s; target = {tl}; {out['credits']} research credits")
    L.append(f"  {out['n_slots']} active slots; {out['split']} split; "
             f"base rate of the target train {out['base_train']*100:.1f}% / holdout {out['base_hold']*100:.1f}%")
    if not out["n_slots"] or out["tested"] == 0:
        L.append("  too few slots to discover anything.")
        return "\n".join(L)
    L.append(f"  {out['tested']} candidate rules auto-generated and holdout-tested; "
             f"family-wise z threshold {out['zc']:.2f} (guards against spurious winners)")
    L.append("")
    if out["survivors"]:
        L.append(f"  {'discovered rule (slot N)':<34}{'train lift':>11}{'hold lift':>11}{'P(target)':>11}{'z':>7}{'n':>7}")
        for r in out["survivors"]:
            L.append(f"  {r['rule']:<34}{r['lift_train']:10.2f}x{r['lift_hold']:10.2f}x"
                     f"{r['rate_hold']*100:10.1f}%{r['z_hold']:+7.1f}{r['n_hold']:7d}")
        L.append("")
        top = out["survivors"][0]
        direction = "more" if top["lift_hold"] > 1 else "less"
        L.append(f"  DISCOVERED ({out['n_survivors']} rule(s) survive the holdout): the strongest is "
                 f"'{top['rule']}' -> the target is {top['lift_hold']:.1f}x {'likelier' if top['lift_hold']>1 else 'rarer'} "
                 f"when it holds ({direction} than base).")
        L.append("  -> candidate precursors the data named itself, confirmed out of sample. Next: confirm they PAY with the")
        L.append("     discovered-rule scalp (--scalp), cross-check characterize(), then validate live with the 1-slot probe.")
    else:
        tail = ("price is ~a random walk one slot ahead" if tgt["kind"] == "price" else "flurries are ~memoryless")
        L.append(f"  NOTHING SURVIVES THE HOLDOUT: of {out['tested']} auto-generated rules, none leads {tl}")
        L.append("  on the held-out pools beyond chance. Any in-sample pattern was overfit -- consistent with NO SIGNAL:")
        L.append(f"  {tail}, so there is no event to key a ShredStream backrun on. Drop it.")
    if out.get("combo_tested"):
        L.append("")
        L.append(f"  COMBINATIONS (single filters AND-ed together; {out['combo_tested']} conjunctions beam-searched; "
                 f"stricter family-wise z threshold {out['combo_zc']:.2f}):")
        if out["combos"]:
            L.append(f"  {'discovered conjunction (slot N)':<62}{'hold lift':>10}{'P(tgt)':>8}{'z':>7}{'n':>6}")
            for r in out["combos"]:
                rule = r["rule"] if len(r["rule"]) <= 60 else r["rule"][:57] + "..."
                L.append(f"  {rule:<62}{r['lift_hold']:9.2f}x{r['rate_hold']*100:7.1f}%{r['z_hold']:+7.1f}{r['n_hold']:6d}")
            best_s = max(out["survivors"], key=lambda s: s["lift_hold"], default=None)
            best_single = best_s["lift_hold"] if best_s else 1.0
            b = max(out["combos"], key=lambda c: c["lift_hold"])
            if b["lift_hold"] > best_single:
                L.append(f"  -> STACKED EDGE: '{b['rule']}' ({b['depth']} filters) leads {tl} {b['lift_hold']:.1f}x out of sample,")
                L.append(f"     BEATING the best single rule ({best_single:.1f}x). Confirm it PAYS (--scalp), then the live 1-slot probe.")
            else:
                L.append(f"  -> the best conjunction ({b['lift_hold']:.1f}x) is sharper than its own filters but does NOT beat the best")
                L.append(f"     single rule ('{best_s['rule'] if best_s else '-'}' {best_single:.1f}x): stacking surfaces no cleaner edge --")
                L.append("     the signal is single-dimensional, a single rule is the ceiling. Put effort into execution/sizing, not more filters.")
        else:
            L.append("  no conjunction beat BOTH the stricter z bar AND its own single filters -> stacking filters does not")
            L.append("  surface a cleaner edge here; the signal is single-dimensional (or absent). A single rule is as good as it gets.")
    return "\n".join(L)


# --- discovered-rule scalp: scalp the holdout-confirmed survivors, out of sample ------------------
# Closes the loop between discover() and the scalp: discover names the rules that LEAD a next-slot flurry
# out of sample; this measures whether ENTERING on each of those exact rules (buy front of N+1, sell N+h)
# actually PAYS on price -- on the same holdout pools the rule was confirmed on, so it is out of sample too.
# A rule can lead a flurry yet not move price enough to beat cost; that gap is the whole question.

def discovered_scalp(have: dict, metas: list[dict], tgt: dict, cost: float,
                     holds=MOTIF_HOLDS, seed: int = 0) -> dict:
    import math
    pools = []
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if not flow:
            continue
        pools.append({"flow": flow, "bank": _slot_bank(flow, m.get("create_ts") or 0.0, tgt),
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
        hits = []                                         # per fire: did the predicted target follow (proves the rule fires on real target slots)
        for p in hold:
            slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
            for i, rec_s in enumerate(p["bank"]):
                if spec["fn"](rec_s):
                    hits.append(rec_s["target"])
                    for h in holds:
                        r = _scalp(slots, pull, i, h, cost)
                        if r is not None:
                            agg[h].append(r)
        row = {"rule": spec["rule"], "lift_hold": ho["lift"], "z_hold": ho["z"], "fires": len(hits),
               "target_hit": (statistics.fmean(hits) if hits else 0.0)}
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
    best = _best_cell(results, ctrl, holds, "rule")
    out.update({"results": results, "control": ctrl, "best": best, "n_hold": len(hold)})
    return out


def _raw_scalp(slots: list[dict], pull_slot, s_idx: int, hold: int, cost: float):
    """(raw, capped) return of the same scalp as _scalp, so the tail report can see whether the capped
    mean rests on a few uncapped reserve-ratio spikes. raw has no +RET_CAP ceiling (pulls are still -1.0)."""
    s0 = slots[s_idx]["slot"]
    entry = slots[s_idx]["price"]
    if not entry:
        return None
    if pull_slot is not None and s0 < pull_slot <= s0 + hold:
        return (-1.0, -1.0)
    exit_price = _price_at(slots, s0 + hold)
    if not exit_price:
        return None
    raw = exit_price / entry - 1.0 - cost
    return (max(-1.0, raw), max(-1.0, min(raw, RET_CAP)))


def tail_report(have: dict, metas: list[dict], tgt: dict, cost: float,
                hold: int = 2, top_rules: int = 6, seed: int = 0) -> dict:
    """Is the discovered edge a broad tail or a few pools? Re-discovers the holdout-confirmed survivors
    (same train/holdout split and z guard as discovered_scalp), then for each, on the OUT-OF-SAMPLE
    holdout pools at one hold, breaks the per-trade returns down by concentration: winners, capped vs
    uncapped mean, how much of the summed positive return sits in the top few winners, distinct pools,
    and the biggest single trades. A real positive-skew edge has many winning pools; a fragile one is
    a handful of capped +300% spikes. Read only."""
    import math
    pools = []
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if not flow:
            continue
        pools.append({"pool": m["pool"], "flow": flow,
                      "bank": _slot_bank(flow, m.get("create_ts") or 0.0, tgt),
                      "ct": m.get("create_ts") or 0.0})
    out = {"n_pools": len(pools), "hold": hold, "target": tgt, "cost": cost, "rules": [], "control": {}}
    if len(pools) < 4:
        return out
    tss = sorted({p["ct"] for p in pools})
    cut = tss[len(tss) // 2] if len(tss) > 1 else None
    train = [p for p in pools if cut is not None and p["ct"] < cut]
    hold_pools = [p for p in pools if cut is not None and p["ct"] >= cut]
    if len(train) < 2 or len(hold_pools) < 2:
        rng0 = random.Random(0)
        idx = list(range(len(pools)))
        rng0.shuffle(idx)
        k = len(idx) // 2
        train, hold_pools, out["split"] = [pools[i] for i in idx[:k]], [pools[i] for i in idx[k:]], "random"
    else:
        out["split"] = "time"
    out["n_hold"] = len(hold_pools)
    train_recs = [r for p in train for r in p["bank"]]
    hold_recs = [r for p in hold_pools for r in p["bank"]]
    if len(train_recs) < 2 * DISC_MIN_SUPPORT or len(hold_recs) < 2 * DISC_MIN_SUPPORT:
        return out
    tgt_tr = np.array([r["target"] for r in train_recs], float)
    tgt_ho = np.array([r["target"] for r in hold_recs], float)
    base_tr, base_ho = float(tgt_tr.mean()), float(tgt_ho.mean())
    evald = []
    for spec in _build_specs(train_recs):
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
    survivors = survivors[:top_rules]
    out["n_survivors"] = len(survivors)

    def _breakdown(trades: list[tuple]) -> dict:
        # trades = [(pool, raw, capped)]
        if not trades:
            return {"n": 0}
        capped = [t[2] for t in trades]
        raw = [t[1] for t in trades]
        wins = sorted(((t[2], t[0]) for t in trades if t[2] > 0), reverse=True)
        pos_sum = sum(c for c, _ in wins) or 1.0
        top5 = sum(c for c, _ in wins[:5]) / pos_sum
        top20 = sum(c for c, _ in wins[:20]) / pos_sum
        return {"n": len(trades), "win": len(wins),
                "capped_mean": statistics.fmean(capped), "raw_mean": statistics.fmean(raw),
                "median": statistics.median(capped),
                "top5_share": top5, "top20_share": top20,
                "pools": len({t[0] for t in trades}), "win_pools": len({p for _, p in wins}),
                "max": max(raw), "top_winners": [(p, c) for c, p in wins[:10]]}

    for spec, ho in survivors:
        trades = []
        for p in hold_pools:
            slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
            for i, rec_s in enumerate(p["bank"]):
                if spec["fn"](rec_s):
                    r = _raw_scalp(slots, pull, i, hold, cost)
                    if r is not None:
                        trades.append((p["pool"], r[0], r[1]))
        b = _breakdown(trades)
        b["rule"] = spec["rule"]
        b["z_hold"] = ho["z"]
        out["rules"].append(b)
    rng = random.Random(seed)
    ctrl_trades = []
    for p in hold_pools:
        slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
        quiet = [i for i, s in enumerate(slots) if s["n_buys"] < 3]
        rng.shuffle(quiet)
        for i in quiet[:CONTROL_PER_POOL]:
            r = _raw_scalp(slots, pull, i, hold, cost)
            if r is not None:
                ctrl_trades.append((p["pool"], r[0], r[1]))
    out["control"] = _breakdown(ctrl_trades)
    return out


def render_tail(out: dict) -> str:
    tgt = out.get("target") or _tgt()
    L = []
    L.append("TAIL-CONCENTRATION BREAKDOWN  (is the discovered edge a broad tail or a few pools?)")
    L.append(f"  {out['n_pools']} cached pools; {out.get('split', '?')} split; {out.get('n_hold', 0)} holdout pools; "
             f"hold {out['hold']} slot(s); cost {out['cost']*100:.1f}%; target = {target_label(tgt)}")
    if not out.get("rules"):
        L.append("  not enough cached flow to re-discover survivors.")
        return "\n".join(L)
    c = out.get("control") or {}
    cm = c.get("capped_mean", 0.0)
    L.append(f"  control (random quiet slots): capped-mean {cm*100:+.2f}%, raw-mean {c.get('raw_mean',0)*100:+.2f}%, "
             f"win {c.get('win',0)}/{c.get('n',0)}")
    L.append("")
    L.append(f"  {'discovered rule':<26}{'n':>6}{'win':>6}{'cap-mn':>8}{'edge':>7}{'raw-mn':>8}"
             f"{'top5':>6}{'top20':>7}{'pools':>6}{'win-pl':>7}{'max':>8}")
    for r in out["rules"]:
        if not r.get("n"):
            continue
        edge = (r["capped_mean"] - cm) * 100
        L.append(f"  {r['rule'][:26]:<26}{r['n']:>6}{r['win']:>6}{r['capped_mean']*100:>7.1f}%{edge:>+7.1f}"
                 f"{r['raw_mean']*100:>7.1f}%{r['top5_share']*100:>5.0f}%{r['top20_share']*100:>6.0f}%"
                 f"{r['pools']:>6}{r['win_pools']:>7}{r['max']*100:>+7.0f}%")
    L.append("")
    L.append("  edge = capped-mean - control capped-mean;  raw-mn = uncapped mean (if >> cap-mn, a few spikes carry it)")
    L.append("  top5/top20 = share of the summed POSITIVE return from the top 5/20 winning trades (high = fragile)")
    L.append("  pools = distinct pools that fired;  win-pl = distinct pools with a winning trade (high = broad tail)")
    best = out["rules"][0] if out["rules"] else None
    if best and best.get("n"):
        L.append("")
        L.append(f"  TOP WINNERS for '{best['rule']}': "
                 + ", ".join(f"{p[:6]} {c*100:+.0f}%" for p, c in best.get("top_winners", [])))
    return "\n".join(L)


# --- entry-window: patience study (the owner's reframe, 11-18) -------------------------------------
# The 1-2 slot HOLD stays mandatory (it dodges the reactive LP pull -- canary-slots: pulls land >=2 slots
# after our buy). But nothing forces us to ENTER at N+1. Given a slot-N signal, we can wait and take the
# safe 1-2 slot ride at any entry slot in N+1..N+WINDOW. Two things this buys that immediate entry can't:
# (1) while we wait we hold no capital, so a pull during the wait costs nothing -- we just never enter;
# (2) we can wait for the up-move to actually start (the washbot's next buy-leg) and ride it, instead of
# guessing it lands next slot. So: is there a signal at N that makes a 1-2 slot buy EV+ somewhere in the
# next WINDOW entry slots, and can we pick that slot causally? Out of sample on the time-holdout.

# candidate slot-N signals, surf-readable only (shreds give buys+payers+counts, not sell SIZE, so no
# sell_sol/buy-size thresholds here -- those are ~0 on reconstructed flows). abs_move uses the DATA's own
# q90 (built per run in entry_window), not a hardcoded cut from the offline distribution. Add/remove freely.
ENTRY_SIGNALS = [
    ("sell_present",      lambda r: r["f"]["sell_present"] >= 1),      # a washbot sell leg (the 11-18 lead)
    ("new_present",       lambda r: r["f"]["new_present"] >= 1),       # a brand-new wallet bought
    ("outside_buy",       lambda r: r["f"]["outside_buy"] >= 1),       # a non-top-3 wallet bought
    ("dominant_bought",   lambda r: r["f"]["dominant_bought"] >= 1),   # the #1 wallet looped
    ("n_buys>=3",         lambda r: r["f"]["n_buys"] >= 3),            # a buy flurry
    ("n_sells>=1",        lambda r: r["f"]["n_sells"] >= 1),           # any sell leg (count, readable on surf)
]


def _ew_scalp_best(slots, pull, idx, holds, cost):
    """Best 1-2 slot scalp at entry index idx (the safe hold stays raw-slot <=2 for pull safety)."""
    best = None
    for h in holds:
        r = _scalp(slots, pull, idx, h, cost)
        if r is not None:
            best = r if best is None else max(best, r)
    return best


def entry_window(have, metas, tgt, cost, window: int = 10, holds=(1, 2), seed: int = 0) -> dict:
    """For each slot-N signal, sweep the ENTRY offset 1..window (in active slots) and measure the 1-2 slot
    scalp at each entry (pool must be ALIVE at entry, else skip -- a pull during the wait is a no-op, not a
    loss). Reports: per-offset EV, the best-of-window ceiling (perfect timing, look-ahead), and a causal
    'wait for the up-tick then ride' rule, all vs a random-active-slot control, on the time-holdout."""
    pools = []
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if not flow:
            continue
        pools.append({"ct": m.get("create_ts") or 0.0, "slots": flow["slots"], "pull": flow["pull_slot"],
                      "bank": _slot_bank(flow, m.get("create_ts") or 0.0, tgt)})
    out = {"n_pools": len(pools), "window": window, "holds": list(holds), "cost": cost, "target": tgt,
           "signals": [], "control": {}, "split": "none", "n_hold": 0}
    if len(pools) < 4:
        return out
    tss = sorted({p["ct"] for p in pools})
    cut = tss[len(tss) // 2] if len(tss) > 1 else None
    hold_pools = [p for p in pools if cut is not None and p["ct"] >= cut]
    if len(hold_pools) < 2:
        hold_pools, out["split"] = pools, "all"
    else:
        out["split"] = "time"
    out["n_hold"] = len(hold_pools)

    def run(signal_fn):
        per_k = {k: [] for k in range(1, window + 1)}
        ceil, causal = [], []
        for p in hold_pools:
            slots, pull, bank, n = p["slots"], p["pull"], p["bank"], len(p["slots"])
            for i, rec in enumerate(bank):
                if not signal_fn(rec):
                    continue
                best_win = None
                done_causal = False
                for k in range(1, window + 1):
                    j = i + k
                    if j >= n:
                        break
                    sj = slots[j]["slot"]
                    if pull is not None and pull <= sj:        # pool pulled at/before this entry -> we never enter
                        break
                    sc = _ew_scalp_best(slots, pull, j, holds, cost)
                    if sc is None:
                        continue
                    per_k[k].append(sc)
                    best_win = sc if best_win is None else max(best_win, sc)
                    if not done_causal and bank[j]["f"]["price_up"] >= 1:   # wait for the up-tick, then ride
                        causal.append(sc)
                        done_causal = True
                if best_win is not None:
                    ceil.append(best_win)
        return per_k, ceil, causal

    cpk, cceil, ccausal = run(lambda r: (r["f"]["n_buys"] > 0 or r["f"]["n_sells"] > 0))   # control = any active slot
    out["control"] = {"per_k": {k: _agg(v) for k, v in cpk.items()}, "ceiling": _agg(cceil), "causal": _agg(ccausal)}
    # abs_move signal uses the DATA's own q90 (a fixed cut from the offline distribution rarely fires on surf)
    moves = [r["f"]["abs_move"] for p in hold_pools for r in p["bank"] if r["f"]["abs_move"] > 0]
    sigs = list(ENTRY_SIGNALS)
    if len(moves) >= 20:
        q90 = sorted(moves)[int(0.90 * (len(moves) - 1))]
        out["abs_move_q90"] = q90
        sigs.append((f"abs_move>=q90({q90*100:.2f}%)", lambda r, _t=q90: r["f"]["abs_move"] >= _t))
    for name, fn in sigs:
        pk, ceil, causal = run(fn)
        best_k = None
        for k in range(1, window + 1):
            a = _agg(pk[k])
            if a.get("n", 0) >= 30 and a.get("mean", -9) > 0 and a["mean"] > out["control"]["per_k"][k].get("mean", 0):
                if best_k is None or a["mean"] > best_k["mean"]:
                    best_k = {"k": k, **a}
        out["signals"].append({"signal": name, "per_k": {k: _agg(v) for k, v in pk.items()},
                               "ceiling": _agg(ceil), "causal": _agg(causal), "best_k": best_k})
    return out


def _ew_causal(pools, fn, window, holds, cost) -> list:
    """The causal 'wait for the first up-tick in the window, then take the 1-2 slot ride' returns for a signal,
    across pools. One value per signal firing (or none if no live up-tick in the window). Pool must be alive
    at entry (a pull during the wait = no entry)."""
    vals = []
    for p in pools:
        slots, pull, bank, n = p["slots"], p["pull"], p["bank"], len(p["slots"])
        for i, rec in enumerate(bank):
            if not fn(rec):
                continue
            for k in range(1, window + 1):
                j = i + k
                if j >= n:
                    break
                if pull is not None and pull <= slots[j]["slot"]:
                    break
                if bank[j]["f"]["price_up"] >= 1:          # first up-tick in the window
                    sc = _ew_scalp_best(slots, pull, j, holds, cost)
                    if sc is not None:
                        vals.append(sc)
                    break
    return vals


def entry_window_discover(have, metas, tgt, cost, window: int = 10, holds=(1, 2), seed: int = 0) -> dict:
    """Discover version of the patience study: no signal named in advance. Auto-generate the candidate rule
    bank on the TRAIN half (_build_specs), then for each candidate run the causal entry-window rule (wait for
    the first up-tick in N+1..N+W, take the 1-2 slot ride) on the HOLDOUT, and keep only the signals whose
    holdout EV beats the random-active-slot control past the family-wise z guard (same direction on train),
    so a flattering-but-spurious signal is reported as noise. The causal rule is a single deterministic value
    per signal (no entry-slot cherry-pick), so the only multiple comparison is across rules -> the zc guard."""
    import math
    pools = []
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if not flow:
            continue
        pools.append({"ct": m.get("create_ts") or 0.0, "slots": flow["slots"], "pull": flow["pull_slot"],
                      "bank": _slot_bank(flow, m.get("create_ts") or 0.0, tgt)})
    out = {"n_pools": len(pools), "window": window, "holds": list(holds), "cost": cost, "target": tgt,
           "survivors": [], "n_survivors": 0, "tested": 0, "zc": 0.0, "split": "none", "n_hold": 0,
           "control_mean": 0.0, "control_n": 0}
    if len(pools) < 4:
        return out
    tss = sorted({p["ct"] for p in pools})
    cut = tss[len(tss) // 2] if len(tss) > 1 else None
    train = [p for p in pools if cut is not None and p["ct"] < cut]
    hold = [p for p in pools if cut is not None and p["ct"] >= cut]
    if len(train) < 2 or len(hold) < 2:
        rng0 = random.Random(0)
        idx = list(range(len(pools))); rng0.shuffle(idx); k = len(idx) // 2
        train, hold, out["split"] = [pools[i] for i in idx[:k]], [pools[i] for i in idx[k:]], "random"
    else:
        out["split"] = "time"
    out["n_hold"] = len(hold)

    def _any(r):
        return r["f"]["n_buys"] > 0 or r["f"]["n_sells"] > 0
    ctrl = _ew_causal(hold, _any, window, holds, cost)
    if len(ctrl) < DISC_MIN_SUPPORT:
        return out
    cm = statistics.fmean(ctrl)
    cv = statistics.pvariance(ctrl) if len(ctrl) > 1 else 0.0
    out["control_mean"], out["control_n"] = cm, len(ctrl)

    train_recs = [r for p in train for r in p["bank"]]
    evald = []
    for spec in _build_specs(train_recs):
        tr = _ew_causal(train, spec["fn"], window, holds, cost)
        if len(tr) < DISC_MIN_SUPPORT:
            continue
        ho = _ew_causal(hold, spec["fn"], window, holds, cost)
        if len(ho) < DISC_MIN_SUPPORT:
            continue
        evald.append((spec, tr, ho))
    out["tested"] = len(evald)
    zc = max(3.0, math.sqrt(2.0 * math.log(max(len(evald), 2))))
    out["zc"] = zc
    survivors = []
    for spec, tr, ho in evald:
        trm, hom = statistics.fmean(tr), statistics.fmean(ho)
        hv = statistics.pvariance(ho) if len(ho) > 1 else 0.0
        se = (hv / len(ho) + cv / len(ctrl)) ** 0.5
        z = (hom - cm) / se if se > 0 else 0.0
        if (trm > 0) == (hom > 0) and hom > cm and z >= zc:
            survivors.append({"rule": spec["rule"], "train_mean": trm, "hold_mean": hom, "z": z,
                              "n_hold": len(ho), "win": 100.0 * sum(1 for x in ho if x > 0) / len(ho),
                              "edge": hom - cm})
    survivors.sort(key=lambda s: s["z"], reverse=True)
    out["survivors"] = survivors[:DISC_TOP]
    out["n_survivors"] = len(survivors)
    return out


def render_entry_window_discover(out: dict) -> str:
    L = []
    L.append("ENTRY-WINDOW DISCOVERY  (no signal named in advance; causal up-tick-ride on the holdout, z-guarded)")
    L.append(f"  {out['n_pools']} pools; {out['split']} split; {out['n_hold']} holdout pools; window {out['window']} "
             f"active slots; hold {out['holds']} raw slots; cost {out['cost']*100:.1f}%")
    L.append(f"  control (up-tick ride after a random active slot): mean {out['control_mean']*100:+.2f}% over n={out['control_n']}")
    L.append(f"  {out['tested']} rules auto-generated + holdout-tested; family-wise z threshold {out['zc']:.2f}")
    L.append("")
    if out["survivors"]:
        L.append(f"  {'discovered signal (slot N)':<34}{'train':>8}{'holdout':>8}{'edge':>7}{'win':>6}{'z':>7}{'n':>7}")
        for s in out["survivors"]:
            L.append(f"  {s['rule'][:34]:<34}{s['train_mean']*100:>+7.1f}{s['hold_mean']*100:>+7.1f}%"
                     f"{s['edge']*100:>+6.1f}{s['win']:>5.0f}%{s['z']:>+7.1f}{s['n_hold']:>7d}")
        L.append("")
        top = out["survivors"][0]
        L.append(f"  DISCOVERED ({out['n_survivors']} signal(s) survive): the strongest is '{top['rule']}' -> waiting for the")
        L.append(f"  first up-tick after it and taking the 1-2 slot ride nets {top['hold_mean']*100:+.1f}% out of sample "
                 f"(control {out['control_mean']*100:+.1f}%, +{top['edge']*100:.1f} pts).")
        L.append("  -> a patience rule the data named itself, out of sample, past the multiple-comparisons guard.")
        L.append("     Next: cross-check on a second capture, then validate live with the 1-slot probe.")
    else:
        L.append("  NOTHING SURVIVES: no auto-generated slot-N signal makes the delayed 1-2 slot ride EV+ beyond the")
        L.append("  control past the z guard. Patience doesn't rescue the scalp at slot scale on these pools.")
    return "\n".join(L)


def render_entry_window(out: dict) -> str:
    tgt = out.get("target") or _tgt()
    L = []
    L.append("ENTRY-WINDOW (patience) STUDY  (1-2 slot hold stays; sweep WHICH entry slot in N+1..N+W to take it)")
    L.append(f"  {out['n_pools']} pools; {out['split']} split; {out['n_hold']} holdout pools; window {out['window']} "
             f"active slots; hold {out['holds']} raw slots; cost {out['cost']*100:.1f}%")
    L.append("  a pull during the wait = no entry (no loss); only a pull inside the 1-2 slot hold = -100%.")
    L.append("")
    ks = list(range(1, out["window"] + 1))
    L.append(f"  {'signal':<16}" + "".join(f"{('k'+str(k)):>7}" for k in ks) + f"{'ceil':>8}{'causal':>8}{'bestK':>14}")
    def cell(a):
        return f"{a['mean']*100:>+6.1f}" if a and a.get("n", 0) >= 20 else f"{'-':>6} "
    cr = out["control"]
    L.append(f"  {'control(any)':<16}" + "".join(f"{cell(cr['per_k'][k])}" for k in ks)
             + f"{cell(cr['ceiling']):>8}{cell(cr['causal']):>8}{'':>14}")
    for s in out["signals"]:
        bk = s["best_k"]
        bkt = (f"k{bk['k']} {bk['mean']*100:+.1f}% n{bk['n']}" if bk else "none")
        L.append(f"  {s['signal']:<16}" + "".join(f"{cell(s['per_k'][k])}" for k in ks)
                 + f"{cell(s['ceiling']):>8}{cell(s['causal']):>8}{bkt:>14}")
    L.append("")
    L.append("  cells = mean 1-2 slot scalp % entering k active slots after the signal; ceil = best-of-window (look-ahead,")
    L.append("  upper bound); causal = enter at the first up-tick slot in the window then ride (realistic).")
    winners = [s for s in out["signals"] if s["best_k"] or (s["causal"].get("n", 0) >= 30 and s["causal"].get("mean", -9) > 0
               and s["causal"]["mean"] > cr["causal"].get("mean", 0))]
    if winners:
        L.append("")
        for s in winners:
            c = s["causal"]
            tail = (f"causal(up-tick ride) {c['mean']*100:+.1f}% n{c['n']} vs control {cr['causal'].get('mean',0)*100:+.1f}%"
                    if c.get("n", 0) >= 30 else "")
            bk = s["best_k"]
            fx = (f"best fixed wait k{bk['k']} {bk['mean']*100:+.1f}% (control {cr['per_k'][bk['k']].get('mean',0)*100:+.1f}%)" if bk else "")
            L.append(f"  SIGNAL '{s['signal']}': {fx}{'; ' if fx and tail else ''}{tail}")
        L.append("  -> a 1-2 slot buy made EV+ by WAITING for the right entry slot. Validate live with the 1-slot probe.")
    else:
        L.append("  NO SIGNAL makes the 1-2 slot buy EV+ over the window beyond the control: patience doesn't rescue it")
        L.append("  at slot scale on these pools.")
    return "\n".join(L)


# --- render (scalp) --------------------------------------------------------------------------------

def _c(a: dict) -> str:
    # mean is capped at +/-RET_CAP so skewed memecoin returns don't drown the average; median + win% are the
    # robust reads (a scalp edge needs a mean clearly > cost AND the median/win% to agree, not one outlier).
    if not a or not a.get("n"):
        return f"{'-':>26}"
    s = f"{a['mean']*100:+.0f}% md{a['median']*100:+.0f}% w{a['win_pct']:.0f} n{a['n']}"
    return f"{s:>26}"


def render(out: dict) -> str:
    tgt = out.get("target") or _tgt()
    L = []
    L.append("BUY-FLURRY SLOT-CONTINUATION STUDY  (reactive-backrun scalp: buy front of N+1, sell N+h)")
    L.append(f"  pools {out['n_pools']} ({out['n_flows']} with a slot flow){', probed only' if out['probed_only'] else ''}; "
             f"span {out['span_s']:.0f}s; cost {out['cost_pct']:.2f}%; {out['credits']} research credits")
    L.append(f"  median buys/slot {out['median_buys_per_slot']:.0f}; flurries (buys>=3) {out['flurries_total']}; "
             f"discover/motif target = {target_label(tgt)}")
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
            L.append("  NO LEAD clears cost + the control on mean/median/win%: the precursor leads the target but entering")
            L.append("  on it doesn't pay a 1-2 slot scalp. Not tradeable at slot scale on untouched pools.")
    d = out.get("discovered")
    if d is not None:
        L.append("")
        L.append("  DISCOVERED-RULE SCALP  (auto-fed from discover's survivors; scalped OUT OF SAMPLE on the holdout pools)")
        L.append(f"  {d['n_survivors']} holdout-confirmed rule(s), {d['split']} split, {d.get('n_hold', 0)} holdout pools; "
                 f"target = {target_label(tgt)}")
        if d["results"]:
            L.append(f"  {'discovered rule':<28}{'target%':>8}{'fires':>7}" + "".join(f"{'hold '+str(h):>28}" for h in MOTIF_HOLDS))
            for row in d["results"]:
                L.append(f"  {row['rule'][:28]:<28}{row.get('target_hit', 0.0)*100:7.0f}%{row.get('fires', 0):7d}"
                         + "".join(f"  {_c(row[f'h{h}'])}" for h in MOTIF_HOLDS))
            L.append(f"  {'control':<28}{'':>8}{'':>7}" + "".join(f"  {_c(d['control'][f'h{h}'])}" for h in MOTIF_HOLDS))
            L.append("  (target% = share of fires the predicted move actually followed -- proves the rule fires on real target slots)")
        db = d.get("best")
        if db:
            L.append(f"  BEST DISCOVERED RULE: '{db['rule']}' hold {db['hold']} -> net {db['mean']*100:+.1f}% "
                     f"(median {db['median']*100:+.1f}%, control {db['control']*100:+.1f}%, edge {db['edge']*100:+.1f} pts) "
                     f"n={db['n']}, win {db['win_pct']:.0f}%")
            L.append("  -> a discovered lead that also PAYS out of sample. Validate live with the 1-slot probe before ShredStream.")
        elif tgt["kind"] == "price":
            L.append("  NO DISCOVERED RULE pays out of sample: even pointed straight at PRICE, no slot-N rule predicts a")
            L.append("  tradeable 1-2 slot up-move beyond chance -> price is ~a random walk at slot scale, drop ShredStream.")
        else:
            L.append("  NO DISCOVERED RULE pays on price: the leads predict a next-slot flurry (target%) but it doesn't move")
            L.append("  price enough. Re-run with --target price to point discovery at the move itself.")
    return "\n".join(L)
