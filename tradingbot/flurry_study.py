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

# When True, slot_flow adds the pool's virtual_quote (raw lamports -> SOL) to the SOL reserve so the price is the
# live EFFECTIVE (vq-aware) basis, not the raw vault ratio. Raw ratio inflates early-life moves (vq dominates the
# quote side when little real SOL is in yet), manufacturing fake big abs-moves. Off by default so every existing
# mode's numbers stay reproducible; the frontrun mode turns it on. Needs rec["vq"], populated by fetch_flows on a
# --fetch run (and backfilled onto the existing cache). Toggle on the CLI with --vq.
USE_VQ = False
# When True, the wallet-rank features (top1/top3 -> sym B/O, dominant_bought, outside_buy) are computed ONLINE
# (only wallets seen through slot N), not over the whole flow. The global version is look-ahead: it knows who the
# pool's dominant wallet turns out to be, so a gate using sym==B / dominant_bought fires on different slots live
# than in the backtest. Off by default so every other mode stays byte-reproducible; frontrun forces it on so the
# exported gate is live-valid (precursor_gate scores the same slots the study scored).
CAUSAL_WALLET = False

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
# abs_move entry-threshold sweep: the base 0.0188 is abs_move q90 on the Fly scorer universe; step it up to
# see whether requiring a BIGGER slot move -- deeper into the tail -- nets a higher mean scalp.
ABSMOVE_GRID = (0.0188, 0.025, 0.035, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30)
# non-overlapping BANDS from the same edges (last open-ended): the cumulative >=t sweep keeps the whole tail in
# every row, so a rising mean can be the top slice carrying it -- bands isolate WHICH move size actually pays.
ABSMOVE_BANDS = tuple(zip(ABSMOVE_GRID, ABSMOVE_GRID[1:] + (float("inf"),)))

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
_VIS = 7   # optional 8th row element (helius.launch_record with_vis): True = top-level pumpswap swap the box can
#          # decode from shreds; absent/False = an aggregator-routed CPI, direction invisible to the box.
_FAILED = 8  # optional 9th row element (helius.launch_record include_failed): True = a REVERTED swap. The live
#            # shred feed counts these (no status in shreds); the training view (vault deltas) never saw them.


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


def _pool_vq(raw: bytes) -> int:
    """pumpswap Pool.virtual_quote_reserves (i128 @245), raw base units (lamports for WSOL). 0 if unreadable.
    Mirrors execution.pumpswap_direct.pool_extra without importing the executor (and its chain deps)."""
    return int.from_bytes(raw[245:261], "little", signed=True) if raw and len(raw) >= 261 else 0


def slot_flow(rec: dict, vis: str = "full", include_failed: bool = False) -> dict | None:
    """Rebuild the per-slot series from a launch_record dict. Returns None if there is no usable price path.
    `slots` is a sorted list of per-slot dicts {slot, n_buys, buy_sol, n_sells, sell_sol, price}; `price` is the
    reserve ratio at the END of that slot. `pull_slot` is the first slot with an LP withdrawal (or None).

    `vis` restricts which trades contribute to the per-slot BUYER/SELL aggregates, to reproduce what the live box
    can actually see from ShredStream (the price path always uses every trade -- it is the real pool state):
      "full"       -- every trade (the study baseline; byte-identical to before).
      "direct"     -- only top-level pumpswap swaps (row[_VIS] True): the box's literal decoded view.
      "buyer_plus" -- direct swaps as-is, plus each aggregator-routed trade counted as a buyer with its fee payer
                      but no magnitude (the box sees the payer in the shred but not the CPI's direction/size),
                      approximating recovery of the organic new-wallet signal.
    Needs with_vis-tagged rows for the restricted modes; untagged rows read as not-direct."""
    txs = rec.get("txs") or []
    by_slot: dict[int, dict] = {}
    pull_slot = None
    vq_sol = (rec.get("vq") or 0) / 1e9 if USE_VQ else 0.0   # virtual_quote (raw lamports) -> SOL: effective basis
    for r in txs:
        slot = r[_SLOT]
        if slot is None:
            continue
        if (len(r) > _FAILED and bool(r[_FAILED])) and not include_failed:
            continue                                       # a reverted swap: in the live feed but not training
        tok_after, sol_after = r[_TOKA], r[_SOLA]
        kind = classify(r[_TOKD] or 0.0, r[_SOLD] or 0.0)
        s = by_slot.setdefault(slot, {"slot": slot, "n_buys": 0, "buy_sol": 0.0, "n_sells": 0, "sell_sol": 0.0,
                                      "price": None, "buyers": [], "max_buy": 0.0})
        visible = vis == "full" or (len(r) > _VIS and bool(r[_VIS]))
        if visible:
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
        elif vis == "buyer_plus" and kind in ("buy", "sell"):
            s["n_buys"] += 1                                  # aggregator trade: payer visible, direction/size not
            s["buyers"].append(r[_PAYER])
        if tok_after and tok_after > 0:                   # price after this tx = end-of-slot price so far
            s["price"] = (float(sol_after) + vq_sol) / float(tok_after)
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


def _has_vis(rec: dict | None) -> bool:
    """A cached flow is visibility-tagged iff its rows carry the 8th (top-level pumpswap) element."""
    rows = (rec or {}).get("txs") or []
    return bool(rows) and len(rows[0]) > _VIS


def _has_failed(rec: dict | None) -> bool:
    """A cached flow is failed-inclusive iff it was built with include_failed (so it carries the reverted swaps
    the live shred feed sees). A failed-inclusive rec also carries the vis tag (include_failed implies it)."""
    return bool((rec or {}).get("failed_tagged"))


def fetch_flows(store, helius, metas: list[dict], span_s: float, max_credits: int, progress=None,
                with_vis: bool = False, include_failed: bool = False) -> tuple[dict, int]:
    """Cache the first `span_s` of each pool's trading in watch.db `watch_flurry` (research key, stable hash order
    so a credit cap leaves an even sample). Returns {pool: launch_record dict} and the research credits spent.
    `with_vis` tags each row with its box visibility (top-level pumpswap vs aggregator CPI) and REFETCHES any cached
    row that lacks the tag -- the one piece the cached launch_record does not store (it needs the raw tx logs).
    `include_failed` likewise refetches rows lacking the reverted swaps (the failed-inclusive refit test): it caches
    a SUPERSET (success + failed-flagged rows) that slot_flow(include_failed=False/True) reads either way, so the
    success-only and failed-inclusive views come from one fetch on identical pools. include_failed implies with_vis."""
    store.execute(SCHEMA)
    have: dict = {}
    for r in store.query("SELECT pool, data FROM watch_flurry", ()):
        try:
            have[r["pool"]] = json.loads(r["data"])
        except Exception:  # noqa: BLE001
            pass
    if helius is None:
        return have, 0

    def _order_key(m):
        c = have.get(m["pool"])
        # when tagging, re-tag already-cached flows FIRST (cheap, focuses the budget on the flow set) before
        # spending it discovering brand-new pools; otherwise the hash order spreads it across every candidate.
        retag = c and c.get("txs") and ((with_vis and not _has_vis(c)) or (include_failed and not _has_failed(c)))
        return (0 if retag else 1, hashlib.sha1(("fl:" + m["pool"]).encode()).hexdigest())
    order = sorted(metas, key=_order_key)
    credits = 0
    for i, m in enumerate(order):
        pool = m["pool"]
        cached = have.get(pool)
        done = cached and (cached.get("txs") or cached.get("error"))
        # a cached row satisfies the request iff it carries every tag asked for (error rows can't be tagged, leave them)
        tagged = done and (cached.get("error") or
                           ((not with_vis or _has_vis(cached)) and (not include_failed or _has_failed(cached))))
        if done and tagged:
            continue
        if credits + MAX_PAGES * 10 > max_credits:
            break
        txs, _more, _err = helius.history_from_start(pool, span_s, MAX_PAGES)
        credits += MAX_PAGES * 10
        rec = he.launch_record(txs, pool, m["token_mint"], window_s=span_s, with_vis=with_vis,
                               include_failed=include_failed)
        if cached and cached.get("vq") is not None and "vq" not in rec:
            rec["vq"] = cached["vq"]                                  # keep the vq basis across a vis refetch
        store.execute("INSERT OR REPLACE INTO watch_flurry(pool, data, ts) VALUES(?,?,?)",
                      (pool, json.dumps(rec), m.get("create_ts") or 0))
        have[pool] = rec
        if progress:
            progress(i + 1, len(order), credits)
    # vq backfill: add each usable pool's virtual_quote (for the effective-reserve basis) in one batched read per
    # 100 pools. Covers both pools fetched just now and any older cache rows that predate the vq field.
    need = [m["pool"] for m in order
            if (have.get(m["pool"]) or {}).get("txs") and "vq" not in have[m["pool"]]]
    ct = {m["pool"]: (m.get("create_ts") or 0) for m in order}
    for j in range(0, len(need), 100):
        if credits + 1 > max_credits:
            break
        chunk = need[j:j + 100]
        acc = helius.accounts(chunk)
        credits += 1
        if not acc:
            continue
        for pool in chunk:
            data = acc.get(pool)
            have[pool]["vq"] = _pool_vq(data[1]) if data else 0
            store.execute("INSERT OR REPLACE INTO watch_flurry(pool, data, ts) VALUES(?,?,?)",
                          (pool, json.dumps(have[pool]), ct.get(pool, 0)))
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
        "abs_sweep": abs_move_sweep(have, metas, tgt, cost, MOTIF_HOLDS, seed),
        "down_sweep": abs_move_sweep(have, metas, tgt, cost, MOTIF_HOLDS, seed, side="down"),
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
    causal = CAUSAL_WALLET
    if not causal:                                          # global (look-ahead) wallet ranks
        cnt: Counter = Counter()
        for s in slots:
            cnt.update(s.get("buyers") or [])
        gtop3 = {w for w, _ in cnt.most_common(3)}
        gtop1 = cnt.most_common(1)[0][0] if cnt else None
    run_cnt: Counter = Counter()                            # online ranks: only wallets seen through slot N
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
        if causal:
            run_cnt.update(buyers)                          # include slot N's own buyers (visible as it closes)
            top3 = {w for w, _ in run_cnt.most_common(3)}
            top1 = run_cnt.most_common(1)[0][0] if run_cnt else None
        else:
            top3, top1 = gtop3, gtop1
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
            "mean_buy": (float(s["buy_sol"]) / nbuys if nbuys else 0.0), "abs_move": abs(move), "move": move,
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
            specs.append({"rule": f"{fld} >= {thr:.3g} (q{q})", "kind": "numeric", "field": fld, "thr": thr,
                          "fn": (lambda r, _f=fld, _t=thr: r["f"][_f] >= _t)})
    for fld in BOOL_FIELDS:
        specs.append({"rule": f"{fld}", "kind": "bool", "field": fld,
                      "fn": (lambda r, _f=fld: r["f"][_f] >= 1.0)})
    syms = {r["sym"] for r in train}
    for x in sorted(syms):
        specs.append({"rule": f"sym == {x}  ({SYM_DESC.get(x, x)})", "kind": "symbol", "symv": x,
                      "fn": (lambda r, _x=x: r["sym"] == _x)})
    for key, label in (("m2", "N-1..N"), ("m3", "N-2..N")):
        counts = Counter(r[key] for r in train)
        for mot, c in counts.items():
            if c >= DISC_MIN_SUPPORT and len(train) - c >= DISC_MIN_SUPPORT:
                specs.append({"rule": f"{label} = {mot}", "kind": "motif", "mkey": key, "mot": mot,
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
    # also scalp the best CONJUNCTIONS, out of sample on the same holdout pools. A combo's predicate is
    # rebuilt by AND-ing its component single rules (looked up by rule string), so nothing non-serialisable
    # is stored and the combo scalp uses the exact same fires -> scalp machinery as the singles.
    combo_results = []
    combo_best = None
    if len(evald) >= 2:
        clay = _combo_layer(evald, train_recs, hold_recs, tgt_tr, tgt_ho, base_tr, base_ho)
        spec_by_rule = {spec["rule"]: spec for spec, _, _ in evald}
        for c in clay["combos"][:DISC_SCALP_TOP]:
            fns = [spec_by_rule[p]["fn"] for p in c["rule"].split("  AND  ") if p in spec_by_rule]
            if len(fns) != c["depth"]:
                continue
            agg = {h: [] for h in holds}
            hits = []
            for p in hold:
                slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
                for i, rec_s in enumerate(p["bank"]):
                    if all(f(rec_s) for f in fns):
                        hits.append(rec_s["target"])
                        for h in holds:
                            r = _scalp(slots, pull, i, h, cost)
                            if r is not None:
                                agg[h].append(r)
            row = {"rule": c["rule"], "depth": c["depth"], "lift_hold": c["lift_hold"], "z_hold": c["z_hold"],
                   "fires": len(hits), "target_hit": (statistics.fmean(hits) if hits else 0.0)}
            for h in holds:
                row[f"h{h}"] = _agg(agg[h])
            combo_results.append(row)
        combo_best = _best_cell(combo_results, ctrl, holds, "rule")
    out.update({"results": results, "control": ctrl, "best": best, "n_hold": len(hold),
                "combo_results": combo_results, "combo_best": combo_best, "n_combos": len(combo_results)})
    return out


def _moveval(rec: dict, side: str) -> float:
    """Signed-move magnitude on the chosen side. side='abs' -> |move| (both directions); 'down' -> the down-move
    magnitude (0 if the slot moved up); 'up' -> the up-move magnitude. The sell side ('down') is the reversion
    test: fire after a down-spike, buy the trough front of N+1, sell the bounce N+h -- the one direction where
    the mid-to-mid study is a floor not a ceiling (our own impact still costs, but the market move between the
    fills works WITH us: buy low, sell the recovery), unlike buying an up-spike top that reverts against us."""
    mv = rec["f"].get("move", 0.0)
    if side == "down":
        return -mv if mv < 0 else 0.0
    if side == "up":
        return mv if mv > 0 else 0.0
    return rec["f"]["abs_move"]


def abs_move_sweep(have: dict, metas: list[dict], tgt: dict, cost: float,
                   holds=MOTIF_HOLDS, seed: int = 0, grid=ABSMOVE_GRID, side: str = "abs") -> dict:
    """Sweep the move ENTRY threshold: fire the scalp (buy front of N+1, sell N+h) on every slot whose move on
    `side` >= t, for an ascending grid of t. side='abs' is the absolute move (both directions); side='down' is
    the SELL-SIDE reversion test (fire after a down-spike, bet on the bounce). The base (0.0188 = abs_move q90)
    is the reference; the sweep answers whether requiring a BIGGER move -- deeper into the tail -- nets a higher
    mean. A fixed threshold is a pre-specified hypothesis (not discovered from the data), so there is nothing to
    overfit: this runs on EVERY pool (more n), no holdout split. `q` per row = the quantile the threshold sits
    at (share of active slots below it, on this side)."""
    pools = []
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if flow:
            pools.append({"flow": flow, "bank": _slot_bank(flow, m.get("create_ts") or 0.0, tgt)})
    out = {"n_pools": len(pools), "rows": [], "control": {}, "best": None, "side": side}
    if not pools:
        return out
    allmoves = sorted(_moveval(r, side) for p in pools for r in p["bank"] if _moveval(r, side) > 0)

    def _q(t: float) -> float:                            # share of active (moving) slots below t = its quantile
        return 100.0 * sum(1 for m in allmoves if m < t) / len(allmoves) if allmoves else 0.0

    rows = []
    for t in grid:
        agg = {h: [] for h in holds}
        fires = 0
        for p in pools:
            slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
            for i, rec in enumerate(p["bank"]):
                if _moveval(rec, side) >= t:
                    fires += 1
                    for h in holds:
                        r = _scalp(slots, pull, i, h, cost)
                        if r is not None:
                            agg[h].append(r)
        row = {"thresh": t, "q": _q(t), "fires": fires}
        for h in holds:
            row[f"h{h}"] = _agg(agg[h])
        rows.append(row)
    rng = random.Random(seed)
    control = {h: [] for h in holds}
    for p in pools:
        slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
        quiet = [i for i, s in enumerate(slots) if s["n_buys"] < 3]
        rng.shuffle(quiet)
        for i in quiet[:CONTROL_PER_POOL]:
            for h in holds:
                r = _scalp(slots, pull, i, h, cost)
                if r is not None:
                    control[h].append(r)
    ctrl = {f"h{h}": _agg(control[h]) for h in holds}
    bands = []                                            # non-overlapping ranges: which move SIZE pays, not >=t
    for lo, hi in ABSMOVE_BANDS:
        agg = {h: [] for h in holds}
        fires = 0
        for p in pools:
            slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
            for i, rec in enumerate(p["bank"]):
                mv = _moveval(rec, side)
                if lo <= mv < hi:
                    fires += 1
                    for h in holds:
                        r = _scalp(slots, pull, i, h, cost)
                        if r is not None:
                            agg[h].append(r)
        label = f"{lo*100:.2f}-{'inf' if hi == float('inf') else f'{hi*100:.2f}'}"
        row = {"band": label, "lo": lo, "hi": hi, "fires": fires}
        for h in holds:
            row[f"h{h}"] = _agg(agg[h])
        bands.append(row)
    out.update({"rows": rows, "control": ctrl, "best": _best_cell(rows, ctrl, holds, "thresh"),
                "bands": bands, "band_best": _best_cell(bands, ctrl, holds, "band")})
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


# --- economics-ranked discovery: select rules by the scalp P&L they'd have made, not by lift -------
# discover() ranks the candidate rule bank by LIFT (binary separation on the price target), then scalps the
# survivors. A rule whose lift is unremarkable -- most of its fires are flat, so it doesn't "predict" the
# >=move_pct hit well -- but whose SCALP RETURN distribution is strong (a high median, or a fat fills-favoured
# tail) never reaches the scalp pass. discover_econ folds the economics INTO the selection: each rule is kept
# only where it beats the random-slot control significantly on the MEAN (Welch z) OR on the distribution/median
# (Mann-Whitney z), out of sample on a create_ts holdout, under a family-wise z guard. Each survivor is
# labelled broad (a median/distribution edge -- trust either side) vs tail (a mean/+300%-skew edge only --
# trust it on a DOWN/reversion entry where the fills work WITH us, distrust it on an UP/momentum entry where
# the spike reverts against us), and carries its move direction + tail concentration so you apply that trust.
ECON_MIN_FIRES = 50   # per-rule fires (train and holdout) for a candidate: a sanity floor + stable significance


def _welch_z(a, b) -> float:
    """z of mean(a) - mean(b) under unequal variances (Welch). The +300% cap / -100% rugs make the return
    distribution heavy-tailed, so this is deliberately conservative for a tail-carried mean."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    se = (aa.var(ddof=1) / len(aa) + bb.var(ddof=1) / len(bb)) ** 0.5
    return float((aa.mean() - bb.mean()) / se) if se > 0 else 0.0


def _mw_z(a, b):
    """Mann-Whitney (auc, z) that sample a stochastically dominates b -- a distribution/median edge robust to
    the tail. Reuses _rank_auc on the concatenated samples (auc = P(a's return > b's)); z from the normal
    approximation of U. This is the median-side significance test that a tail-inflated mean can't fake."""
    n1, n0 = len(a), len(b)
    if n1 < 2 or n0 < 2:
        return 0.5, 0.0
    feat = np.asarray(list(a) + list(b), float)
    lab = np.asarray([1] * n1 + [0] * n0, float)
    auc = _rank_auc(feat, lab)
    se = ((n1 + n0 + 1.0) / (12.0 * n1 * n0)) ** 0.5
    return auc, (float((auc - 0.5) / se) if se > 0 else 0.0)


def discover_econ(have: dict, metas: list[dict], tgt: dict, cost: float,
                  holds=MOTIF_HOLDS, seed: int = 0, stake: float = 50.0) -> dict:
    """Economics-ranked discovery (see the block comment). Same candidate rule bank as discover() plus signed
    move thresholds (move>=+q up entries, move<=-q down entries, so momentum AND reversion rules compete), but
    each rule is SELECTED by its scalp P&L -- significant vs the control on mean OR median, out of sample,
    family-wise guarded -- not by lift. Read only; a surfaced rule is a hypothesis, the live 1-slot run is the
    arbiter, and the mid-to-mid scalp overstates an UP tail (fills invert) while it is a FLOOR for a DOWN tail
    (fills favour the bounce), which is why each survivor carries a broad/tail label and a move direction."""
    import math
    pools = []
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if not flow:
            continue
        pools.append({"flow": flow, "bank": _slot_bank(flow, m.get("create_ts") or 0.0, tgt),
                      "ct": m.get("create_ts") or 0.0})
    out = {"n_pools": len(pools), "n_survivors": 0, "survivors": [], "control": {}, "split": "none",
           "zc": 0.0, "stake": stake, "n_specs": 0, "holds": list(holds)}
    if len(pools) < 4:
        return out
    tss = sorted({p["ct"] for p in pools})
    cut = tss[len(tss) // 2] if len(tss) > 1 else None
    train = [p for p in pools if cut is not None and p["ct"] < cut]
    hold = [p for p in pools if cut is not None and p["ct"] >= cut]
    if len(train) < 2 or len(hold) < 2:                       # pools clustered in time -> random pool split
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
    specs = _build_specs(train_recs)                          # the standard bank (discovered on train only)...
    mv = np.array([r["f"]["move"] for r in train_recs], float)
    if mv.std() > 0:                                          # ...plus signed move entries so BOTH sides compete
        for q in (75, 90, 95):
            thr = float(np.quantile(mv, q / 100.0))
            if thr > 0:
                specs.append({"rule": f"move >= {thr*100:.2f}% (up q{q})", "kind": "move",
                              "fn": (lambda r, _t=thr: r["f"]["move"] >= _t)})
        for q in (25, 10, 5):
            thr = float(np.quantile(mv, q / 100.0))
            if thr < 0:
                specs.append({"rule": f"move <= {thr*100:.2f}% (down q{q})", "kind": "move",
                              "fn": (lambda r, _t=thr: r["f"]["move"] <= _t)})

    def _fires(pool_set, fn):                                 # scalp returns per hold for the slots fn fires on
        per = {h: [] for h in holds}
        moves = []
        for p in pool_set:
            slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
            for i, rs in enumerate(p["bank"]):
                if fn(rs):
                    moves.append(rs["f"].get("move", 0.0))
                    for h in holds:
                        r = _scalp(slots, pull, i, h, cost)
                        if r is not None:
                            per[h].append(r)
        return per, moves

    def _control(pool_set, rng):                              # random quiet active slots = the baseline
        per = {h: [] for h in holds}
        for p in pool_set:
            slots, pull = p["flow"]["slots"], p["flow"]["pull_slot"]
            quiet = [i for i, s in enumerate(slots) if s["n_buys"] < 3]
            rng.shuffle(quiet)
            for i in quiet[:CONTROL_PER_POOL]:
                for h in holds:
                    r = _scalp(slots, pull, i, h, cost)
                    if r is not None:
                        per[h].append(r)
        return per

    ctrl_tr = _control(train, random.Random(seed + 1))
    ctrl_ho = _control(hold, random.Random(seed))
    out["control"] = {f"h{h}": _agg(ctrl_ho[h]) for h in holds}
    evaluated = []
    for spec in specs:
        tr_per, _ = _fires(train, spec["fn"])
        if max((len(tr_per[h]) for h in holds), default=0) < ECON_MIN_FIRES:
            continue                                          # thin on train -> not a candidate (and not in K)
        ho_per, moves = _fires(hold, spec["fn"])
        evaluated.append((spec, tr_per, ho_per, moves))
    k = max(len(evaluated) * max(len(holds), 1), 2)
    zc = max(3.0, math.sqrt(2.0 * math.log(k)))               # family-wise guard over every (rule x hold) tested
    out["zc"], out["n_specs"] = zc, len(evaluated)
    survivors = []
    for spec, tr_per, ho_per, moves in evaluated:
        best = None
        for h in holds:
            rr, cc, trr, tcc = ho_per[h], ctrl_ho[h], tr_per[h], ctrl_tr[h]
            if len(rr) < ECON_MIN_FIRES or len(cc) < 10 or len(trr) < ECON_MIN_FIRES or len(tcc) < 10:
                continue
            ra, ca = _agg(rr), _agg(cc)
            wz, (_, mwz) = _welch_z(rr, cc), _mw_z(rr, cc)
            twz, (_, tmwz) = _welch_z(trr, tcc), _mw_z(trr, tcc)
            mean_ok = wz >= zc and twz > 0 and ra["mean"] > ca["mean"]          # significant on the mean...
            dist_ok = mwz >= zc and tmwz > 0 and ra["median"] >= ca["median"]   # ...or the distribution/median
            if not (mean_ok or dist_ok):
                continue
            score = ra["median"] if dist_ok else ra["mean"]   # rank broad rules on median, tail rules on mean
            cand = {"hold": h, "rr": list(rr), "mean": ra["mean"], "median": ra["median"],
                    "win_pct": ra["win_pct"], "rugged_pct": ra["rugged_pct"], "n": ra["n"], "welch_z": wz,
                    "mw_z": mwz, "ctrl_mean": ca["mean"], "ctrl_median": ca["median"], "ctrl_win": ca["win_pct"],
                    "broad": dist_ok, "score": score}
            if best is None or cand["score"] > best["score"]:
                best = cand
        if best is None:
            continue
        wins = sorted((r for r in best["rr"] if r > 0), reverse=True)
        pos = sum(wins) or 1.0
        dirn = statistics.fmean(moves) if moves else 0.0
        survivors.append({
            "rule": spec["rule"], "kind": spec["kind"], "hold": best["hold"], "n": best["n"],
            "mean": best["mean"], "median": best["median"], "win_pct": best["win_pct"],
            "rugged_pct": best["rugged_pct"], "welch_z": best["welch_z"], "mw_z": best["mw_z"],
            "ctrl_mean": best["ctrl_mean"], "ctrl_median": best["ctrl_median"], "ctrl_win": best["ctrl_win"],
            "edge": best["mean"] - best["ctrl_mean"], "net_per_fire": best["mean"] * stake,
            "label": ("broad" if best["broad"] else "tail"), "dir": ("down" if dirn < 0 else "up"),
            "move_mean": dirn, "top5_share": sum(wins[:5]) / pos})
    # broad rules first (ranked by median, trustworthy either side), then tail rules (ranked by mean): the
    # reader trusts a DOWN tail (reversion bounce, fills favour us) and distrusts an UP tail (spike artifact).
    survivors.sort(key=lambda s: (0 if s["label"] == "broad" else 1,
                                  -(s["median"] if s["label"] == "broad" else s["mean"])))
    out["survivors"] = survivors[:DISC_TOP]
    out["n_survivors"] = len(out["survivors"])
    return out


def render_econ(out: dict) -> str:
    L = []
    L.append("ECONOMICS-RANKED DISCOVERY  (rules SELECTED by scalp P&L -- mean OR median -- not by lift; holdout-confirmed)")
    L.append(f"  pools {out['n_pools']}; {out['split']} split; {out.get('n_specs', 0)} candidate rules; "
             f"zc {out['zc']:.2f} (family-wise guard); stake ${out['stake']:.0f}; holds {tuple(out.get('holds', ()))}")
    cc = out.get("control") or {}
    if cc:
        L.append("  control (random quiet slots): " + "  ".join(f"h{h} {_c(cc.get(f'h{h}', {}))}" for h in out.get("holds", [])))
    L.append("")
    if not out["survivors"]:
        L.append("  NO RULE beats the control significantly on mean OR median out of sample: no slot-N rule pays a")
        L.append("  1-2 slot scalp beyond chance even ranked by economics -> consistent with price ~ a random walk at")
        L.append("  slot scale; drop ShredStream.")
        return "\n".join(L)
    L.append(f"  {out['n_survivors']} rule(s) PAY out of sample.  label: broad = median/distribution edge (trust either side);")
    L.append("  tail = mean/+300%-skew only -- trust a TAIL rule on a DOWN entry (fills favour the bounce), distrust it on an UP entry (the spike reverts).")
    L.append("")
    L.append(f"  {'discovered rule':<34}{'lbl':>6}{'dir':>5}{'hold':>5}{'fires':>7}{'median%':>9}{'mean%':>8}{'win%':>6}{'$/fire':>8}{'tail5':>7}{'MWz':>6}{'Wz':>6}")
    for s in out["survivors"]:
        L.append(f"  {s['rule'][:34]:<34}{s['label']:>6}{s['dir']:>5}{s['hold']:>5}{s['n']:>7}"
                 f"{s['median']*100:>8.1f}%{s['mean']*100:>7.1f}%{s['win_pct']:>5.0f}%"
                 f"{s['net_per_fire']:>+8.2f}{s['top5_share']*100:>6.0f}%{s['mw_z']:>6.1f}{s['welch_z']:>6.1f}")
    broad = [s for s in out["survivors"] if s["label"] == "broad"]
    downtail = [s for s in out["survivors"] if s["label"] == "tail" and s["dir"] == "down"]
    L.append("")
    if broad:
        b = broad[0]
        L.append(f"  BEST BROAD RULE: '{b['rule']}' ({b['dir']}) hold {b['hold']} -> median {b['median']*100:+.1f}%, "
                 f"mean {b['mean']*100:+.1f}%, win {b['win_pct']:.0f}%, ${b['net_per_fire']:+.2f}/fire at ${out['stake']:.0f}")
        L.append("  -> a broad (not tail-carried) edge that pays on the TYPICAL fire. The strongest candidate to validate live.")
    elif downtail:
        b = downtail[0]
        L.append(f"  BEST DOWN-TAIL RULE: '{b['rule']}' hold {b['hold']} -> mean {b['mean']*100:+.1f}% (median {b['median']*100:+.1f}%, "
                 f"win {b['win_pct']:.0f}%, top-5 {b['top5_share']*100:.0f}% of wins); ${b['net_per_fire']:+.2f}/fire at ${out['stake']:.0f}")
        L.append("  -> a positive-skew REVERSION edge: the mid-to-mid study is a FLOOR here (fills favour the bounce). Validate live on NET.")
    else:
        L.append("  Only UP-tail rules surfaced: a mean carried by +300% spikes that revert against the fill live. Distrust -- not a live candidate.")
    L.append("  Still mid-to-mid on untouched pools -> a hypothesis, not a go/no-go. The live 1-slot run is the arbiter.")
    return "\n".join(L)


# --- first-buy reactive scalp: see the pool's first buy (ShredStream), backrun it, hold a few slots ----
# The reactive-backrun exits in <=2 slots, which dodges the operator's LP pull (canary-slots: 100% land >=2
# slots after our mid-life buy). That settles the RISK side; the open question is RETURN -- a move we can
# reliably capture. This measures the trade the owner wants: ShredStream shows the pool's FIRST BUY in slot N,
# we backrun it -- enter front of slot N+1 ("straight away", the earliest a reactive buy can land) and hold k
# slots. It reports the return per hold-after-entry, where the first LP pull lands relative to OUR entry (the
# risk check for THIS trade -- the mid-life >=2-slot dodge may NOT hold at the creation moment), and the raw
# move the first buy leads to. Read only, free off the cached flows.
FIRST_BUY_HOLDS = (1, 2, 3)        # slots held AFTER our entry (entry = front of the slot after the first buy)


def creation_study(have: dict, metas: list[dict], cost: float, holds=FIRST_BUY_HOLDS) -> dict:
    """The reactive first-buy trade: ShredStream shows the pool's FIRST BUY (slot N); we backrun it -- enter
    front of slot N+1 ("straight away", the earliest a reactive buy can land) and hold k slots. Reports the
    return per hold-after-entry, where the first LP pull lands relative to OUR entry (the risk check for THIS
    trade), and the raw move the first buy leads to. Read only -- measured on untouched pools; the live arbiter
    is a real reactive entry with NET fills."""
    pools = []
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if flow and len(flow["slots"]) >= 2:
            pools.append(flow)
    out = {"n_pools": len(pools), "holds": list(holds), "cost": cost, "cells": [], "pull_dist": {},
           "ramp": [], "best": None, "n_first_buy": 0, "n_no_buy": 0}
    if not pools:
        return out

    def _fb(flow):                                            # index of the first slot that contains a buy
        for i, s in enumerate(flow["slots"]):
            if s.get("n_buys", 0) > 0:
                return i
        return None

    for k in holds:                                          # enter front of (first-buy+1), sell k slots later:
        rets = []                                            # _scalp(fb, k+1) exits at slot(fb)+k+1 = k slots after entry
        for flow in pools:
            fb = _fb(flow)
            if fb is None:
                continue
            r = _scalp(flow["slots"], flow["pull_slot"], fb, k + 1, cost)
            if r is not None:
                rets.append(r)
        out["cells"].append({"label": f"hold {k}", "hold": k, "agg": _agg(rets)})

    buckets = {"0": 0, "1": 0, "2": 0, "3-5": 0, "6+": 0, "none": 0}   # first pull relative to OUR entry
    for flow in pools:
        fb = _fb(flow)
        if fb is None:
            out["n_no_buy"] += 1
            continue
        out["n_first_buy"] += 1
        pull = flow["pull_slot"]
        if pull is None:
            buckets["none"] += 1
            continue
        pos = pull - (flow["slots"][fb]["slot"] + 1)          # slots from our entry (front of fb+1) to the pull
        key = ("0" if pos <= 0 else "1" if pos == 1 else "2" if pos == 2 else "3-5" if pos <= 5 else "6+")
        buckets[key] += 1
    out["pull_dist"] = buckets

    for k in holds:                                          # raw move: price k slots after the first-buy slot
        moves = []
        for flow in pools:
            fb = _fb(flow)
            if fb is None:
                continue
            p0 = flow["slots"][fb]["price"]
            pk = _price_at(flow["slots"], flow["slots"][fb]["slot"] + k)
            if p0 and pk:
                moves.append(pk / p0 - 1.0)
        if moves:
            out["ramp"].append({"to": k, "n": len(moves),
                                "mean": statistics.fmean(moves), "median": statistics.median(moves)})

    best = None                                               # the prize: a hold positive on mean AND median at n>=50
    for c in out["cells"]:
        a = c["agg"]
        if a.get("n", 0) < 50 or a.get("mean", -9) <= 0 or a.get("median", -9) <= 0:
            continue
        if best is None or a["median"] > best["agg"]["median"]:
            best = c
    out["best"] = best
    return out


def render_creation(out: dict) -> str:
    L = []
    L.append("FIRST-BUY REACTIVE SCALP  (ShredStream: see the pool's FIRST BUY in slot N, enter front of N+1, hold k slots)")
    L.append(f"  pools {out['n_pools']} ({out['n_first_buy']} with a first buy, {out['n_no_buy']} none); cost {out['cost']*100:.2f}%")
    L.append("")
    L.append("  THE TRADE (cell = mean / median / win% / n; hold k = slots held AFTER our entry):")
    for c in out["cells"]:
        L.append(f"    {c['label']:<9}  {_c(c['agg'])}")
    ramp = out.get("ramp") or []
    if ramp:
        L.append("")
        L.append("  RAW MOVE AFTER THE FIRST BUY (price k slots after the first-buy slot vs that slot's price):")
        for r in ramp:
            L.append(f"    +{r['to']} slot: mean {r['mean']*100:+.1f}%  median {r['median']*100:+.1f}%  (n {r['n']})")
    pd = out.get("pull_dist") or {}
    if pd:
        tot = sum(pd.values()) or 1
        rugged = tot - pd.get("none", 0)
        within2 = pd.get("0", 0) + pd.get("1", 0) + pd.get("2", 0)
        L.append("")
        L.append("  WHERE THE FIRST LP PULL LANDS relative to OUR ENTRY (front of first-buy + 1) -- the risk check:")
        L.append("    " + "  ".join(f"{k}: {pd[k]} ({100.0*pd[k]/tot:.0f}%)" for k in ("0", "1", "2", "3-5", "6+", "none")))
        L.append(f"    -> {100.0*within2/tot:.0f}% pull within 2 slots of our entry ({100.0*rugged/tot:.0f}% pull at all);"
                 f" a 2-slot hold is only safe if the within-2 share is ~0.")
    b = out.get("best")
    L.append("")
    if b:
        a = b["agg"]
        L.append(f"  PRIZE: {b['label']} after the first buy -> mean {a['mean']*100:+.1f}%, median {a['median']*100:+.1f}%, "
                 f"win {a['win_pct']:.0f}%, rugged {a.get('rugged_pct',0):.0f}% (n {a['n']})")
        L.append("  -> a move positive on the TYPICAL fire. Confirm the pull histogram: if pulls land within 2 slots")
        L.append("     of our entry, the 2-slot dodge does NOT hold for this trade and the stake can be trapped.")
    else:
        L.append("  NO CAPTURABLE MOVE: no hold-after-entry is positive on mean AND median at a sane n -- backrunning")
        L.append("  the first buy doesn't hand a reliable move on untouched pools.")
    L.append("")
    L.append("  Mid-to-mid on untouched pools -> a hypothesis. The live arbiter is a real reactive entry (NET fills).")
    return "\n".join(L)


# --- slot-0 landing: if we land in the creation slot, how long must we hold to be EV+? ----------------
# The most aggressive entry: land in slot 0 (the pool's first active slot, the creation/first-trade slot --
# via ShredStream + a bundle into creation). From there we buy at the end of slot 0 and sweep the hold length.
# Two forces pull against each other: the price drifts up over the opening slots (the wash ramp), which needs
# a few slots to clear the round-trip cost; but every extra slot held is more exposure to the LP pull (-100%).
# So we report, per hold: the NET scalp (with the rug), the RAW drift (no rug, = can drift alone cover cost?),
# and the cumulative share rugged by that hold -- then name the SMALLEST hold that is EV+ (mean AND median > 0),
# if any, and flag whether reaching it means holding past the ~2-slot safe window (i.e. re-taking rug risk).
SLOT0_HOLDS = tuple(range(1, 13)) + (15, 20, 30)   # slots held after the slot-0 entry (~400ms each)
SAFE_HOLD = 2                                       # the canary-slots dodge window: <=2 slots = no mid-life pull


def slot0_study(have: dict, metas: list[dict], cost: float, holds=SLOT0_HOLDS) -> dict:
    """Land in slot 0 (the pool's first active slot) and sweep the hold length: how many slots must we hold for
    the trade to be EV+, if at all? Per hold: NET scalp (with the -100% rug), RAW drift (no rug), and the
    cumulative share rugged by that hold. Names the smallest EV+ hold and whether it sits past the safe window.
    Read only -- mid-to-mid on untouched pools; the live arbiter is a real NET entry landed in the creation slot."""
    pools = []
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if flow and len(flow["slots"]) >= 2:
            pools.append(flow)
    out = {"n_pools": len(pools), "holds": list(holds), "cost": cost, "rows": [],
           "break_even": None, "raw_break_even": None, "safe_hold": SAFE_HOLD}
    if not pools:
        return out

    for h in holds:
        nets, raws, pulled = [], [], 0
        for flow in pools:
            s0 = flow["slots"][0]["slot"]
            p0 = flow["slots"][0]["price"]
            if not p0:
                continue
            r = _scalp(flow["slots"], flow["pull_slot"], 0, h, cost)
            if r is not None:
                nets.append(r)
            pk = _price_at(flow["slots"], s0 + h)            # raw drift: no rug, no cost (drift alone)
            if pk:
                raws.append(pk / p0 - 1.0)
            pull = flow["pull_slot"]
            if pull is not None and s0 < pull <= s0 + h:     # rugged by this hold (cumulative)
                pulled += 1
        agg = _agg(nets)
        raw = {"n": len(raws), "mean": statistics.fmean(raws) if raws else 0.0,
               "median": statistics.median(raws) if raws else 0.0}
        agg_n = agg.get("n", 0) or 1
        out["rows"].append({"hold": h, "agg": agg, "raw": raw,
                            "pulled_by_pct": 100.0 * pulled / agg_n})

    for row in out["rows"]:                                   # smallest EV+ hold: net mean AND median > 0, n>=50
        a = row["agg"]
        if a.get("n", 0) >= 50 and a.get("mean", -9) > 0 and a.get("median", -9) > 0:
            out["break_even"] = row["hold"]
            break
    for row in out["rows"]:                                   # smallest hold where drift alone clears cost (no rug)
        if row["raw"]["n"] >= 50 and row["raw"]["mean"] > cost:
            out["raw_break_even"] = row["hold"]
            break
    return out


def render_slot0(out: dict) -> str:
    L = []
    L.append("SLOT-0 LANDING: HOLD-TO-EV+  (land in the creation slot, sweep the hold length)")
    L.append(f"  pools {out['n_pools']}; cost {out['cost']*100:.2f}%; ~400ms/slot; safe window = {out['safe_hold']} slots (canary-slots dodge)")
    L.append("")
    L.append(f"  {'hold':>5}  {'NET (mean/md/win/n)':<28}  {'RAW drift (mean/md)':<22}  rugged-by")
    for row in out["rows"]:
        r = row["raw"]
        raw = f"{r['mean']*100:+.1f}% md{r['median']*100:+.1f}%"
        L.append(f"  {row['hold']:>5}  {_c(row['agg'])}  {raw:<22}  {row['pulled_by_pct']:.0f}%")
    L.append("")
    be, rbe, safe = out.get("break_even"), out.get("raw_break_even"), out["safe_hold"]
    if rbe is None:
        L.append("  DRIFT NEVER COVERS COST: at no hold does the raw opening drift (ignoring the rug) beat the round-trip")
        L.append(f"  cost of {out['cost']*100:.2f}% -- there is no move to capture from a slot-0 landing, so no hold is EV+.")
    elif be is None:
        L.append(f"  NOT EV+ AT ANY HOLD: raw drift first clears cost at hold {rbe}, but the rug (-100%) drag keeps the")
        L.append("  NET mean/median negative at every hold -- the pull eats the drift faster than it accrues.")
    else:
        row = next(x for x in out["rows"] if x["hold"] == be)
        a = row["agg"]
        L.append(f"  EV+ FROM HOLD {be}: net mean {a['mean']*100:+.1f}%, median {a['median']*100:+.1f}%, win {a['win_pct']:.0f}%, "
                 f"rugged-by {row['pulled_by_pct']:.0f}% (n {a['n']})")
        if be > safe:
            L.append(f"  -> but hold {be} is PAST the {safe}-slot safe window: reaching EV+ re-takes rug risk (the {row['pulled_by_pct']:.0f}%")
            L.append("     rugged-by is the price). NOT a risk-free scalp -- it trades the dodge for the drift.")
        else:
            L.append(f"  -> and hold {be} is WITHIN the {safe}-slot safe window: EV+ AND the pull is still dodged. This is the prize.")
    L.append("")
    L.append("  Mid-to-mid on untouched pools -> a hypothesis. The live arbiter is a real NET entry landed in slot 0.")
    return "\n".join(L)


# --- buy-flurry banded by buy count, enter the slot after, sweep the hold 1-10 slots -------------------
# The owner's reframe: entering CHANGES the dynamics, so the ideal exit is a few SLOTS after entry (their read:
# 4-10 slots), not a 5-10 SECOND window. So band each flurry slot by its buy count and, entering front of N+1,
# measure the net scalp at every hold 1..10 slots -- the hold where the mean peaks is the ideal exit for that
# band. A hold past ~2 slots re-exposes to the reactive LP pull, so rugged% climbs with the hold (read the grid
# as the move-vs-rug tradeoff per band). Read only, mid-to-mid on untouched pools; the live arbiter is a real
# NET reactive entry. Free off the cached flows.
MULTIBUY_BANDS = [(0, 15), (15, 25), (25, 35), (35, 45), (45, 55),
                  (55, 65), (65, 75), (75, 85), (85, 95), (95, 10 ** 9)]
MULTIBUY_HOLDS = tuple(range(1, 11))     # slots held after entering front of N+1
MULTIBUY_MIN_N = 45                      # sample floor for calling a band x hold a candidate exit


def _mb_label(lo: int, hi: int) -> str:
    return f"{lo}+" if hi >= 10 ** 9 else f"{lo}-{hi}"


def multibuy_study(have: dict, metas: list[dict], cost: float,
                   bands=MULTIBUY_BANDS, holds=MULTIBUY_HOLDS, seed: int = 0) -> dict:
    """Band each flurry slot by its buy count, enter front of N+1, sweep the hold 1..10 slots, and report the
    net scalp (mean/median/win/rugged/n) per band x hold -- so the hold where the mean peaks is the ideal exit
    for that band. Read only, mid-to-mid on untouched pools."""
    labels = [_mb_label(lo, hi) for lo, hi in bands]
    coll = {lab: {h: [] for h in holds} for lab in labels}
    control = {h: [] for h in holds}
    rng = random.Random(seed)
    n_flows = 0
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None
        if not flow:
            continue
        n_flows += 1
        slots, pull = flow["slots"], flow["pull_slot"]
        for i, s in enumerate(slots):
            nb = s.get("n_buys", 0)
            lab = next((l for (lo, hi), l in zip(bands, labels) if lo <= nb < hi), None)
            if lab is None:
                continue
            for h in holds:
                r = _scalp(slots, pull, i, h, cost)
                if r is not None:
                    coll[lab][h].append(r)
        quiet = [i for i, s in enumerate(slots) if s.get("n_buys", 0) < 3]   # reference: random light slots
        rng.shuffle(quiet)
        for i in quiet[:CONTROL_PER_POOL]:
            for h in holds:
                r = _scalp(slots, pull, i, h, cost)
                if r is not None:
                    control[h].append(r)
    out = {"n_flows": n_flows, "cost": cost, "holds": list(holds), "min_n": MULTIBUY_MIN_N,
           "control": {h: _agg(control[h]) for h in holds}, "bands": [], "best": None}
    best = None
    for lab in labels:
        cells = {h: _agg(coll[lab][h]) for h in holds}
        # ideal exit = the hold with the highest mean at a sane n; `pays` if it also clears 0 on mean AND median
        exits = [h for h in holds if cells[h].get("n", 0) >= MULTIBUY_MIN_N]
        be = max(exits, key=lambda h: cells[h]["mean"]) if exits else None
        pays = bool(be is not None and cells[be]["mean"] > 0 and cells[be]["median"] > 0
                    and cells[be]["mean"] > out["control"][be].get("mean", -9))
        out["bands"].append({"band": lab, "cells": cells, "best_hold": be, "pays": pays,
                             "n": cells[holds[0]].get("n", 0)})
        if pays and (best is None or cells[be]["median"] > best["agg"]["median"]):
            best = {"band": lab, "hold": be, "agg": cells[be]}
    out["best"] = best
    return out


def render_multibuy(out: dict) -> str:
    holds = out["holds"]
    L = []
    L.append("BUY-FLURRY BY BUY COUNT -> ENTER SLOT AFTER -> HOLD 1-10 SLOTS  (ideal exit = where the mean peaks)")
    L.append(f"  pools {out['n_flows']}; cost {out['cost']*100:.2f}%; ~0.4s/slot so hold 10 ~= 4s; min n {out['min_n']}")
    L.append("")
    head = f"  {'buys/slot':<9}  " + "".join(f"{('h'+str(h)):>5}" for h in holds) + f"  {'ideal':>6} {'n':>6}"
    L.append(head + "   (cells = net mean %)")
    for row in out["bands"]:
        cells = row["cells"]
        line = f"  {row['band']:<9}  "
        for h in holds:
            a = cells[h]
            line += f"{a['mean']*100:>+5.0f}" if a.get("n") else f"{'-':>5}"
        be = row["best_hold"]
        mark = (f"h{be}{'*' if row['pays'] else ''}") if be is not None else "-"
        line += f"  {mark:>6} {row['n']:>6}"
        L.append(line)
    c = out["control"]
    cline = f"  {'control':<9}  " + "".join(f"{c[h]['mean']*100:>+5.0f}" if c[h].get('n') else f"{'-':>5}" for h in holds)
    L.append(cline)
    L.append("")
    # at the ideal-exit hold, show the robust reads (median/win/rugged) per band so a mean peak isn't an outlier
    L.append("  AT THE IDEAL-EXIT HOLD (median / win% / rugged% / n):")
    for row in out["bands"]:
        be = row["best_hold"]
        if be is None:
            L.append(f"    {row['band']:<9}  (n below {out['min_n']} at every hold)")
            continue
        a = row["cells"][be]
        L.append(f"    {row['band']:<9}  h{be}: mean {a['mean']*100:+.1f}%  md {a['median']*100:+.1f}%  "
                 f"win {a['win_pct']:.0f}%  rug {a.get('rugged_pct',0):.0f}%  n {a['n']}")
    b = out.get("best")
    L.append("")
    if b:
        a = b["agg"]
        L.append(f"  BEST: {b['band']} buys/slot, exit at hold {b['hold']} -> mean {a['mean']*100:+.1f}%, "
                 f"median {a['median']*100:+.1f}%, win {a['win_pct']:.0f}%, rugged {a.get('rugged_pct',0):.0f}% (n {a['n']})")
        L.append("  -> a band x exit-hold that pays on mean AND median and beats the control. Validate live (NET).")
    else:
        L.append("  NO BAND x EXIT-HOLD PAYS: no buy-count band has a hold 1-10 positive on mean AND median that")
        L.append("  beats the control -- entering after a flurry and holding a few slots has no mid-to-mid edge here.")
    L.append("")
    L.append("  Mid-to-mid on untouched pools -> a hypothesis. A hold > ~2 slots re-exposes to the reactive pull")
    L.append("  (rugged% climbs with the hold); the live arbiter is a real reactive NET entry.")
    return "\n".join(L)


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
        cres = d.get("combo_results") or []
        if cres:
            L.append("")
            L.append(f"  DISCOVERED-COMBO SCALP  ({len(cres)} holdout-confirmed conjunction(s); same holdout pools, out of sample)")
            L.append(f"  {'discovered conjunction':<40}{'target%':>8}{'fires':>7}" + "".join(f"{'hold '+str(h):>28}" for h in MOTIF_HOLDS))
            for row in cres:
                L.append(f"  {row['rule'][:40]:<40}{row.get('target_hit', 0.0)*100:7.0f}%{row.get('fires', 0):7d}"
                         + "".join(f"  {_c(row[f'h{h}'])}" for h in MOTIF_HOLDS))
            L.append(f"  {'control':<40}{'':>8}{'':>7}" + "".join(f"  {_c(d['control'][f'h{h}'])}" for h in MOTIF_HOLDS))
            cb = d.get("combo_best")
            if cb:
                L.append(f"  BEST DISCOVERED COMBO: '{cb['rule']}' hold {cb['hold']} -> net {cb['mean']*100:+.1f}% "
                         f"(median {cb['median']*100:+.1f}%, control {cb['control']*100:+.1f}%, edge {cb['edge']*100:+.1f} pts) "
                         f"n={cb['n']}, win {cb['win_pct']:.0f}%")
                L.append("  -> a STACKED lead that also PAYS. Compare its edge to the best single above; validate live before ShredStream.")
            else:
                L.append("  NO DISCOVERED COMBO pays out of sample beyond its control: stacking leads but doesn't pay more than a single.")
    def _sweep_block(sw, mv, extra):
        if not (sw and sw.get("rows")):
            return
        L.append("")
        L.append(f"  {mv.upper()} THRESHOLD SWEEP  (enter on every slot with {mv} >= t, buy front of N+1, sell N+h; "
                 f"{sw['n_pools']} pools, every slot -- fixed thresholds, no holdout){extra}")
        L.append(f"  {mv+'>=':<13}{'~q':>5}{'fires':>8}" + "".join(f"{'hold '+str(h):>28}" for h in MOTIF_HOLDS))
        for row in sw["rows"]:
            tag = "  (base q90)" if abs(row["thresh"] - 0.0188) < 1e-9 else ""
            L.append(f"  {row['thresh']*100:>9.2f}%  {row['q']:>4.0f}{row['fires']:>8d}"
                     + "".join(f"  {_c(row[f'h{h}'])}" for h in MOTIF_HOLDS) + tag)
        L.append(f"  {'control':<13}{'':>5}{'':>8}" + "".join(f"  {_c(sw['control'][f'h{h}'])}" for h in MOTIF_HOLDS))
        sb = sw.get("best")
        if sb:
            L.append(f"  BEST THRESHOLD: {mv} >= {sb['thresh']*100:.2f}% hold {sb['hold']} -> net {sb['mean']*100:+.1f}% "
                     f"(median {sb['median']*100:+.1f}%, control {sb['control']*100:+.1f}%, edge {sb['edge']*100:+.1f} pts) "
                     f"n={sb['n']}, win {sb['win_pct']:.0f}%")
        else:
            L.append(f"  NO THRESHOLD beats the control on mean/median/win at n>=50: a bigger {mv} does not net a cleaner mean.")
        bands = sw.get("bands") or []
        if bands:
            L.append("")
            L.append(f"  {mv.upper()} BANDS  (non-overlapping lo<={mv}<hi -- isolates WHICH move size pays, vs the cumulative >=t above)")
            L.append(f"  {mv+' band %':<18}{'fires':>8}" + "".join(f"{'hold '+str(h):>28}" for h in MOTIF_HOLDS))
            for row in bands:
                L.append(f"  {row['band']:<18}{row['fires']:>8d}" + "".join(f"  {_c(row[f'h{h}'])}" for h in MOTIF_HOLDS))
            L.append(f"  {'control':<18}{'':>8}" + "".join(f"  {_c(sw['control'][f'h{h}'])}" for h in MOTIF_HOLDS))
            bb = sw.get("band_best")
            if bb:
                L.append(f"  BEST BAND: {mv} {bb['band']}% hold {bb['hold']} -> net {bb['mean']*100:+.1f}% "
                         f"(median {bb['median']*100:+.1f}%, control {bb['control']*100:+.1f}%, edge {bb['edge']*100:+.1f} pts) "
                         f"n={bb['n']}, win {bb['win_pct']:.0f}%")
            else:
                L.append(f"  NO BAND beats the control on mean/median/win at n>=50: no single move-size slice carries a clean edge.")

    _sweep_block(out.get("abs_sweep"), "abs_move", "")
    _sweep_block(out.get("down_sweep"), "down_move",
                 "  [SELL SIDE: fire after a DOWN-spike, buy the trough, bet the bounce -- the reversion test]")
    return "\n".join(L)


# --- front-run study (11-21): position BEFORE a >= FRONT_MOVE up-move, so the spike is the gain, not the entry.
# The up-side fill artifact (11-19: react = buy the spike top, sell the reversion trough) is sidestepped by being
# IN before the spike. The only hard constraint is the trade window (enter <=k slots before T, exit 1-2 after);
# the indicator LEAD (how far before T it fires) is swept and REPORTED, not fixed. Runs on the vq (effective)
# basis -- set USE_VQ True (the CLI --frontrun does) or the >= 0.3 target fires on raw-ratio basis artifacts.
FRONT_MOVE = 0.30            # the single-slot up-move (effective basis) we want to be positioned before
FRONT_LEADS = (1, 2, 3, 5, 8)   # indicator LEAD windows (raw slots ahead), swept -- each rule's profile is reported
FRONT_K = (1, 2, 3)         # ceiling: enter this many slots BEFORE the up-move slot T
FRONT_J = (1, 2)            # ceiling: exit this many slots AFTER T
FRONT_ENTRY_LEAD = 3        # the deployed enter-on-signal window: an up-move within this many slots ahead
FRONT_HOLDS = (2, 3, 4, 5)  # enter-on-signal holds (raw slots) covering the lead + a 1-2 slot exit
FRONT_CONTROL = 60          # random active slots per pool for the enter-on-signal control
FRONT_NET_TOP = 6           # surviving indicators carried into the actionable-net scalp


def _slot_moves(slots):
    return [0.0] + [(slots[i]["price"] / slots[i - 1]["price"] - 1.0) if slots[i - 1]["price"] else 0.0
                    for i in range(1, len(slots))]


def _up_event_slots(slots, move):
    """Slot NUMBERS of single-slot up-moves >= `move` -- the T's we want to front-run."""
    mv = _slot_moves(slots)
    return [slots[i]["slot"] for i in range(1, len(slots)) if mv[i] >= move]


def _front_ceiling(flow, move, cost):
    """Perfect-foresight: enter T-k, exit T+j (T = an up-move slot). Per (k,j), a list of returns.
    -1.0 if an LP pull lands in (T-k, T+j]. The ceiling: is there money even with perfect timing."""
    slots, pull = flow["slots"], flow["pull_slot"]
    out = {}
    for tn in _up_event_slots(slots, move):
        for k in FRONT_K:
            entry = _price_at(slots, tn - k)
            if not entry:
                continue
            for j in FRONT_J:
                if pull is not None and tn - k < pull <= tn + j:
                    r = -1.0
                else:
                    ex = _price_at(slots, tn + j)
                    if not ex:
                        continue
                    r = max(-1.0, min(ex / entry - 1.0 - cost, RET_CAP))
                out.setdefault((k, j), []).append(r)
    return out


def _front_label(sn, events, lead):
    """1.0 if the nearest up-move slot after `sn` lands within `lead` raw slots (events sorted ascending)."""
    for e in events:
        if e <= sn:
            continue
        return 1.0 if e - sn <= lead else 0.0
    return 0.0


def _ll(y, mu):
    mu = np.clip(mu, 1e-9, 1.0 - 1e-9)
    return float(np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)))


def _wald_p(z):
    import math
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def _logit_fit(X, y, ridge=1e-4, iters=60):
    """IRLS logistic with a tiny L2 ridge (intercept unpenalised) so a separated or collinear predictor still
    yields a finite covariance. Returns (beta, se, fitted mu); se from the ridged observed information."""
    _n, p = X.shape
    beta = np.zeros(p)
    pen = ridge * np.eye(p)
    pen[0, 0] = 0.0
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = np.clip(mu * (1.0 - mu), 1e-6, None)
        grad = X.T @ (y - mu) - pen @ beta
        H = X.T @ (X * w[:, None]) + pen
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
    w = np.clip(mu * (1.0 - mu), 1e-6, None)
    H = X.T @ (X * w[:, None]) + pen
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return beta, se, mu


def _front_logit(tr, ho, specfn, rules, lead):
    """Multivariate logistic on the surviving indicators: joint coefficients + Wald z/p for the up-move within
    `lead` slots, controlling for the correlation between indicators (a univariate lead that looks strong can
    add nothing once another is in the model). Fit on TRAIN; generalization read as the holdout AUC. This is the
    'regression with confidence' -- discover gives per-rule univariate z, the logistic gives the joint fit."""
    seen, uniq = set(), []
    for r in rules:                                        # dedupe, keep z-order, only rules we can evaluate
        if r in specfn and r not in seen:
            seen.add(r)
            uniq.append(r)
    if not uniq or not tr:
        return {"predictors": [], "n": 0}
    fns = [specfn[r] for r in uniq]

    def design(recs):
        X = np.ones((len(recs), len(uniq) + 1), float)
        for j, fn in enumerate(fns):
            X[:, j + 1] = np.fromiter((1.0 if fn(r) else 0.0 for r in recs), float, len(recs))
        return X

    Xtr = design(tr)
    ytr = np.array([r["target_by_lead"][lead] for r in tr], float)
    keep = [j for j in range(1, Xtr.shape[1]) if 0 < Xtr[:, j].sum() < len(tr)]   # drop constants (unidentified)
    cols = [0] + keep
    kept = [uniq[j - 1] for j in keep]
    Xtr = Xtr[:, cols]
    beta, se, mutr = _logit_fit(Xtr, ytr)
    z = np.where(se > 0, beta / se, 0.0)
    npred = max(len(kept), 1)
    preds = []
    for i, name in enumerate(["intercept"] + kept):
        zi = float(z[i])
        pi = _wald_p(zi)
        preds.append({"name": name, "coef": float(beta[i]), "se": float(se[i]), "z": zi, "p": pi,
                      "sig": bool(abs(zi) >= 1.96), "sig_bonf": bool(i > 0 and pi < 0.05 / npred)})
    auc = None
    if ho:
        Xho = design(ho)[:, cols]
        yho = np.array([r["target_by_lead"][lead] for r in ho], float)
        if 0 < yho.sum() < len(yho):
            auc = _rank_auc(Xho @ beta, yho)
    p0 = float(ytr.mean())
    ll0 = _ll(ytr, np.full(len(ytr), p0)) if 0 < p0 < 1 else 0.0
    r2 = (1.0 - _ll(ytr, mutr) / ll0) if ll0 < 0 else 0.0
    return {"lead": lead, "predictors": preds, "n": len(tr), "events": int(ytr.sum()),
            "auc_hold": auc, "pseudo_r2": r2, "n_predictors": len(kept)}


FRONT_GATE_PCTS = (80, 90, 95, 98, 99)   # fire the joint logit only on its top (100-pct)% of scores
FRONT_MOVE_FIELDS = {"abs_move", "move", "price_up"}   # the price move IN slot N -> backrun, not a precursor


def _is_move_rule(rule):
    """True if a discovered rule keys on the move IN slot N (abs_move/move/price_up) -- that is BACKRUNNING a move
    already landed, not front-running the next one. Motif/symbol rules (wallet state) are precursors -> False."""
    field = rule.split(" >=")[0].strip() if " >=" in rule else rule.strip()
    return field in FRONT_MOVE_FIELDS


def _front_logit_gate(ho, pool_flows, specfn, lg, cost, holds=FRONT_HOLDS):
    """The decisive precision test: score every HOLDOUT slot with the fitted joint logit (out of sample -- it was
    fit on train), then enter-on-signal only when that score is in its top k%. A single rule fires too broadly
    (16k fires for ~700 moves -> -1% median); the joint gate at a high threshold fires on far fewer, far cleaner
    slots. If it still has no positive-median cell, precision is not the lever and the base-rate wall holds."""
    preds = (lg or {}).get("predictors") or []
    if not preds or not ho:
        return {"rows": []}
    b0 = preds[0]["coef"]                                   # intercept
    terms = [(p["coef"], specfn[p["name"]]) for p in preds[1:] if p["name"] in specfn]
    by_pool, scores = {}, []
    for r in ho:
        eta = b0 + sum(c for c, fn in terms if fn(r))
        r["_eta"] = eta
        scores.append(eta)
        by_pool.setdefault(r["pool"], []).append(r)
    scores = np.array(scores, float)
    rows = []
    for pct in FRONT_GATE_PCTS:
        thr = float(np.quantile(scores, pct / 100.0))
        rr = {h: [] for h in holds}
        n_fire = 0
        for pool, recs in by_pool.items():
            flow = pool_flows.get(pool)
            if not flow:
                continue
            slots, pull = flow["slots"], flow["pull_slot"]
            for rec in recs:
                if rec["_eta"] < thr:
                    continue
                n_fire += 1
                for h in holds:
                    v = _scalp(slots, pull, rec["idx"], h, cost)
                    if v is not None:
                        rr[h].append(v)
        rows.append({"pct": pct, "n": n_fire, "thr": thr, "holds": {h: _agg(v) for h, v in rr.items()}})
    # holdout-only random control so the comparison is out of sample too
    rng = random.Random(0)
    ctrl = {h: [] for h in holds}
    for pool in by_pool:
        flow = pool_flows.get(pool)
        if not flow or len(flow["slots"]) < 3:
            continue
        slots, pull = flow["slots"], flow["pull_slot"]
        for idx in rng.sample(range(len(slots) - 1), min(FRONT_CONTROL, len(slots) - 1)):
            for h in holds:
                v = _scalp(slots, pull, idx, h, cost)
                if v is not None:
                    ctrl[h].append(v)
    return {"rows": rows, "control": {h: _agg(v) for h, v in ctrl.items()}}


def export_gate(out, hold: int = 2):
    """Serialise the fitted precursor front-run gate to a small dict the live box runner loads through
    tradingbot.precursor_gate: the joint-logistic intercept + non-move predictor coefficients (abs_move/move/
    price_up already excluded by the precursor fit) and the eta threshold at each top-k%. The runner scores a
    live _slot_bank record, fires when eta >= threshold[k], buys front of N+1, holds `hold`, sells N+hold.
    Returns None when no precursor predictor survived (nothing to deploy).

    The gate carries the `vis` it was FIT on. The live box sees top-level pumpswap only ('direct'), so a gate
    fit on 'full' scores the box's direct-only features with the wrong calibration and fires on the wrong slots
    (the 2026-11-21 frontmeasure divergence). Fit AND verify on the box's own view -- run --frontrun with
    --gate-vis direct, and frontrun-verify reads this field to replay through slot_flow(vis=)."""
    lg = out.get("logit_precursor") or {}
    gate = out.get("logit_gate_precursor") or {}
    preds = lg.get("predictors") or []
    rows = gate.get("rows") or []
    if len(preds) < 2 or not rows:
        return None
    intercept = float(preds[0]["coef"])
    meta = out.get("specmeta") or {}
    predictors = [{"name": p["name"], "coef": float(p["coef"]), "predicate": meta.get(p["name"])}
                  for p in preds[1:]]
    threshold = {str(100 - r["pct"]): float(r["thr"]) for r in rows if "thr" in r}
    return {
        "kind": "frontrun_precursor",
        "vis": out.get("vis", "full"),            # the ShredStream visibility the gate was fit on (box view = direct)
        "move": float(out.get("move", FRONT_MOVE)),
        "lead": int(out.get("net_lead", FRONT_ENTRY_LEAD)),
        "entry": "front of N+1 (immediately after the signal slot)",
        "hold": int(hold),
        "intercept": intercept,
        "predictors": predictors,
        "threshold": threshold,
        "built": {"pools": int(out.get("pools", 0)), "events": int(out.get("events", 0)),
                  "auc_hold": lg.get("auc_hold"), "move_pct": round(float(out.get("move", 0)) * 100, 2),
                  "n_predictors": lg.get("n_predictors")},
    }


def frontrun(have, metas, move=FRONT_MOVE, cost=DEFAULT_COST_PCT / 100.0, leads=FRONT_LEADS,
             seed=0, flurry_min=DEFAULT_FLURRY_MIN, vis="full"):
    """Can we predict a >= `move` up-move early enough to enter <=k slots before it and exit 1-2 after?
    (1) ceiling: enter T-k / exit T+j with perfect foresight; (2) indicator discovery, LEAD swept and reported
    (z-guarded time holdout, discover's discipline); (3) actionable enter-on-signal net vs a random-slot control.
    Needs USE_VQ (effective reserves), else the target is a raw-ratio artifact. Observational -> hypothesis only.
    Forces CAUSAL_WALLET: the wallet-rank features are online (no look-ahead), so the fitted/exported gate fires
    on the same slots live as in the backtest -- otherwise sym==B / dominant_bought would use hindsight."""
    import math
    _saved_causal = CAUSAL_WALLET
    try:
        return _frontrun(have, metas, move, cost, leads, seed, flurry_min, vis)
    finally:
        globals()["CAUSAL_WALLET"] = _saved_causal


def _frontrun(have, metas, move, cost, leads, seed, flurry_min, vis="full"):
    import math
    globals()["CAUSAL_WALLET"] = True                        # live-valid features for the gate (see frontrun docstring)
    base_tgt = _tgt("price", flurry_min, 1, move * 100.0)     # for features/symbols only; target overwritten
    ct = {m["pool"]: (m.get("create_ts") or 0) for m in metas}
    ceiling, all_recs, pool_flows = {}, [], {}
    n_events = n_slots = n_pools = 0
    for m in metas:
        rec = have.get(m["pool"])
        flow = slot_flow(rec) if rec else None               # FULL: the real price path, ceiling, target, net
        if not flow:
            continue
        feat = slot_flow(rec, vis) if vis != "full" else flow  # restricted: what the box can SEE to decide
        if not feat or len(feat["slots"]) != len(flow["slots"]):
            feat = flow                                      # price-driven slot set is identical; guard anyway
        n_pools += 1
        slots = flow["slots"]
        n_slots += len(slots)
        pool_flows[m["pool"]] = flow                         # scalp/net/gate price off FULL ground truth
        for key, rs in _front_ceiling(flow, move, cost).items():
            ceiling.setdefault(key, []).extend(rs)
        events = _up_event_slots(slots, move)                # the real up-moves we aim to be positioned before
        n_events += len(events)
        recs = _slot_bank(feat, ct.get(m["pool"], 0), base_tgt)   # features from the restricted view
        for idx, r in enumerate(recs):
            r["target_by_lead"] = {L: _front_label(feat["slots"][idx]["slot"], events, L) for L in leads}
            r["pool"], r["idx"] = m["pool"], idx
        all_recs.extend(recs)

    out = {"mode": "frontrun", "move": move, "cost": cost, "leads": list(leads), "use_vq": USE_VQ, "vis": vis,
           "pools": n_pools, "slots": n_slots, "events": n_events,
           "event_rate": (n_events / n_slots if n_slots else 0.0),
           "ceiling": {f"{k}-{j}": _agg(v) for (k, j), v in sorted(ceiling.items())},
           "survivors": {L: [] for L in leads}, "profile": {}, "net": [], "control": {}, "zc": 3.0,
           "net_lead": FRONT_ENTRY_LEAD}
    if len(all_recs) < DISC_MIN_SUPPORT * 2:
        return out

    # time-split by create_ts (discover's discipline); fall back to a random split if a half is too thin.
    cts = sorted(r["create_ts"] for r in all_recs)
    split = cts[len(cts) // 2]
    tr = [r for r in all_recs if r["create_ts"] <= split]
    ho = [r for r in all_recs if r["create_ts"] > split]
    rng = random.Random(seed)
    if len(tr) < DISC_MIN_SUPPORT * 2 or len(ho) < DISC_MIN_SUPPORT * 2:
        order = list(range(len(all_recs)))
        rng.shuffle(order)
        cut = len(order) // 2
        trs = set(order[:cut])
        tr = [all_recs[i] for i in order[:cut]]
        ho = [all_recs[i] for i in order[cut:]]

    specs = _build_specs(tr)
    specfn = {s["rule"]: s["fn"] for s in specs}
    out["specmeta"] = {s["rule"]: {k: s.get(k) for k in ("kind", "field", "thr", "symv", "mkey", "mot")}
                       for s in specs}                    # exact predicate params for export_gate (string is lossy)
    zc = max(3.0, math.sqrt(2.0 * math.log(max(len(specs) * len(leads), 2))))
    out["zc"] = zc
    profile, survivors = {}, {L: [] for L in leads}
    for L in leads:
        ttr = np.array([r["target_by_lead"][L] for r in tr], float)
        tho = np.array([r["target_by_lead"][L] for r in ho], float)
        btr = float(ttr.mean()) if len(ttr) else 0.0
        bho = float(tho.mean()) if len(tho) else 0.0
        surv = []
        for spec in specs:
            etr = _eval_spec(spec, tr, ttr, btr)
            eho = _eval_spec(spec, ho, tho, bho)
            if not etr or not eho:
                continue
            cell = {"lift": eho["lift"], "z": eho["z"], "p": eho["rate"], "n": eho["n"], "base": bho}
            profile.setdefault(spec["rule"], {})[L] = cell
            if (etr["lift"] - 1.0) * (eho["lift"] - 1.0) > 0 and eho["lift"] > 1.0 and abs(eho["z"]) >= zc:
                surv.append({"rule": spec["rule"], **cell})
        surv.sort(key=lambda s: -s["z"])
        survivors[L] = surv[:DISC_TOP]
    out["survivors"] = survivors
    out["profile"] = profile

    # actionable net: enter on the signal (front of idx+1), hold H, vs a random active-slot control.
    recs_by_pool = {}
    for r in all_recs:
        recs_by_pool.setdefault(r["pool"], []).append(r)
    ctrl = {h: [] for h in FRONT_HOLDS}
    for flow in pool_flows.values():
        slots, pull = flow["slots"], flow["pull_slot"]
        if len(slots) < 3:
            continue
        for idx in rng.sample(range(len(slots) - 1), min(FRONT_CONTROL, len(slots) - 1)):
            for h in FRONT_HOLDS:
                r = _scalp(slots, pull, idx, h, cost)
                if r is not None:
                    ctrl[h].append(r)
    out["control"] = {h: _agg(v) for h, v in ctrl.items()}
    lead_net = FRONT_ENTRY_LEAD if survivors.get(FRONT_ENTRY_LEAD) else next((L for L in leads if survivors.get(L)), leads[0])
    out["net_lead"] = lead_net
    net = []
    for p in survivors.get(lead_net, [])[:FRONT_NET_TOP]:
        fn = specfn.get(p["rule"])
        if not fn:
            continue
        rr = {h: [] for h in FRONT_HOLDS}
        for pool, flow in pool_flows.items():
            slots, pull = flow["slots"], flow["pull_slot"]
            for rec in recs_by_pool.get(pool, []):
                if not fn(rec):
                    continue
                for h in FRONT_HOLDS:
                    r = _scalp(slots, pull, rec["idx"], h, cost)
                    if r is not None:
                        rr[h].append(r)
        net.append({"rule": p["rule"], "z": p["z"], "lift": p["lift"],
                    "holds": {h: _agg(v) for h, v in rr.items()}})
    out["net"] = net
    out["logit"] = _front_logit(tr, ho, specfn, [s["rule"] for s in survivors.get(lead_net, [])], lead_net)
    out["logit_gate"] = _front_logit_gate(ho, pool_flows, specfn, out["logit"], cost)
    # precursor-only gate: drop the move-itself rules (abs_move/move/price_up = backrun) and refit on the wallet
    # setup, so the gate fires BEFORE the move = a genuine front-run test, not a continuation/backrun one.
    pre_rules = [s["rule"] for s in survivors.get(lead_net, []) if not _is_move_rule(s["rule"])]
    out["logit_precursor"] = _front_logit(tr, ho, specfn, pre_rules, lead_net)
    out["logit_gate_precursor"] = _front_logit_gate(ho, pool_flows, specfn, out["logit_precursor"], cost)
    return out


def render_frontrun(out):
    L = ["FRONT-RUN STUDY  (position BEFORE a >= %.0f%% up-move; enter <=%d before T, exit 1-2 after)"
         % (out["move"] * 100, FRONT_K[-1])]
    L.append("  pools %d  active slots %d  up-move events %d (%.2f%% of slots)  cost %.2f%%  vq basis %s"
             % (out["pools"], out["slots"], out["events"], out["event_rate"] * 100, out["cost"] * 100,
                "ON" if out.get("use_vq") else "OFF (raw ratio -- numbers are basis artifacts!)"))
    _vis = out.get("vis", "full")
    L.append("  gate FEATURES fit on the '%s' view%s (target/ceiling always FULL flow)"
             % (_vis, "  <-- the box's own ShredStream view: deployable gate" if _vis == "direct"
                else "  (historical full-flow study; the box sees direct-only)" if _vis == "full" else ""))

    def cell(a):
        if not a or not a.get("n"):
            return "     . "
        return "n%-4d %+6.1f/%+6.1f %3.0f%% rug%2.0f" % (a["n"], a["mean"] * 100, a["median"] * 100,
                                                          a["win_pct"], a.get("rugged_pct", 0))

    L.append("")
    L.append("  CEILING -- perfect foresight (enter T-k / exit T+j): is there money if we time it exactly")
    L.append("  k=before j=after   n mean/median win rug%")
    best = None
    for key in sorted(out["ceiling"]):
        a = out["ceiling"][key]
        L.append("  T-%s  exit T+%s      %s" % (key[0], key[-1], cell(a)))
        if a.get("n", 0) >= 50 and a.get("mean", -9) > 0 and (best is None or a["mean"] > best[1]):
            best = (key, a["mean"], a)
    L.append("  => CEILING %s" % ("+EV at T-%s/exit+%s (%.1f%% mean) -- a real front-run exists IF an indicator finds T"
                                   % (best[0][0], best[0][-1], best[1] * 100) if best
                                   else "NO +EV cell even with perfect foresight -> the move itself doesn't pay; drop"))

    leads = out["leads"]
    prof = out["profile"]
    L.append("")
    L.append("  INDICATOR LEAD PROFILE -- holdout z of each surviving rule at each lead (z guard %.2f); lead = raw slots the up-move lands within"
             % out["zc"])
    L.append("  %-40s%s" % ("rule", "".join("%8s" % ("L%d" % Ld) for Ld in leads)))
    seen = set()
    rows = []
    for Ld in leads:
        for s in out["survivors"].get(Ld, []):
            if s["rule"] in seen:
                continue
            seen.add(s["rule"])
            zs = prof.get(s["rule"], {})
            best_z = max((abs(zs[Ld]["z"]) for Ld in leads if Ld in zs), default=0.0)
            rows.append((best_z, s["rule"], zs))
    rows.sort(key=lambda x: -x[0])
    for _bz, rule, zs in rows[:DISC_TOP]:
        L.append("  %-40s%s" % (rule[:40], "".join(
            ("%+8.1f" % zs[Ld]["z"]) if Ld in zs else "%8s" % "." for Ld in leads)))
    if not rows:
        L.append("  (nothing survives the holdout at any lead -> no indicator predicts the up-move early enough)")

    L.append("")
    L.append("  ENTER-ON-SIGNAL NET (lead %d) -- fire the rule, buy front of N+1, hold H slots, minus cost; vs random-slot control"
             % out["net_lead"])
    L.append("  %-40s%s" % ("", "".join("%28s" % ("hold %d" % h) for h in FRONT_HOLDS)))
    L.append("  %-40s%s" % ("control (random active slot)",
                            "".join("  " + cell(out["control"].get(h, {})) for h in FRONT_HOLDS)))
    paid = None
    for nrow in out["net"]:
        L.append("  %-40s%s" % (nrow["rule"][:40], "".join("  " + cell(nrow["holds"].get(h, {})) for h in FRONT_HOLDS)))
        for h in FRONT_HOLDS:
            a, c = nrow["holds"].get(h, {}), out["control"].get(h, {})
            if a.get("n", 0) >= 50 and a.get("mean", -9) > 0 and a.get("median", -9) > 0 \
                    and a["mean"] > c.get("mean", 0) and a["win_pct"] >= c.get("win_pct", 0):
                edge = a["mean"] - c.get("mean", 0)
                if paid is None or edge > paid[0]:
                    paid = (edge, nrow["rule"], h, a)
    if not out["net"]:
        L.append("  (no surviving indicator to trade)")
    L.append("  => NET %s" % ("'%s' hold %d PAYS: net %+.1f%% (median %+.1f%%, win %.0f%%) vs control %+.1f%% -- validate LIVE (1-slot probe) before sizing"
                              % (paid[1][:40], paid[2], paid[3]["mean"] * 100, paid[3]["median"] * 100,
                                 paid[3]["win_pct"], out["control"].get(paid[2], {}).get("mean", 0) * 100) if paid
                              else "no surviving indicator's enter-on-signal beats the control on mean AND median -- the lead doesn't pay at the entry"))

    lg = out.get("logit") or {}
    L.append("")
    if lg.get("predictors"):
        L.append("  MULTIVARIATE LOGISTIC (lead %d) -- joint coefficients + Wald z/p; does each indicator add signal CONTROLLING for the others?"
                 % lg.get("lead", out["net_lead"]))
        L.append("  fit on %d train slots (%d up-moves); holdout AUC %s; pseudo-R2 %.3f"
                 % (lg["n"], lg["events"],
                    ("%.3f" % lg["auc_hold"]) if lg.get("auc_hold") is not None else "n/a", lg.get("pseudo_r2", 0.0)))
        L.append("  %-40s%9s%8s%8s%9s  %s" % ("predictor", "coef", "se", "z", "p", "keep"))
        for pr in lg["predictors"]:
            flag = "BONF" if pr.get("sig_bonf") else ("*" if pr["sig"] else "")
            L.append("  %-40s%+9.2f%8.2f%+8.1f%9.3f  %s"
                     % (pr["name"][:40], pr["coef"], pr["se"], pr["z"], pr["p"], flag))
        joint = [pr["name"] for pr in lg["predictors"] if pr.get("sig_bonf")]
        L.append("  => LOGIT %s" % ("independent signal (Bonferroni): %s -- the joint predictor to fold into the live front-run rule"
                                    % ", ".join(s[:24] for s in joint) if joint
                                    else "no indicator stays significant under Bonferroni once the others are controlled -- the univariate leads are largely one signal"))
    else:
        L.append("  MULTIVARIATE LOGISTIC: no surviving indicators at the deployed lead to fit")

    TAIL_MARGIN = 0.03  # mean must beat control by >=3 pts to count as a concentrated tail

    def gate_block(gate, title, front_run, backrun=None):
        # returns (best_tail_edge or None) so the precursor block can compare itself to the backrun gate
        if not (gate or {}).get("rows"):
            L.append("")
            L.append("  %s: no fitted joint logit to gate on" % title)
            return None
        L.append("")
        L.append("  %s (out of sample on holdout pools) -- fire only when the JOINT logit" % title)
        kind = "excludes abs_move/move/price_up -> fires on the wallet SETUP before the move = a true front-run" \
            if front_run else "includes abs_move = continuation/backrun contaminant"
        L.append("  score is in its top k%% (%s). Buy front of N+1, hold H, minus cost." % kind)
        L.append("  %-24s%s" % ("", "".join("%28s" % ("hold %d" % h) for h in FRONT_HOLDS)))
        gc = gate.get("control", {})
        L.append("  %-24s%s" % ("control (holdout random)",
                                "".join("  " + cell(gc.get(h, {})) for h in FRONT_HOLDS)))
        broad = None  # triple guard (mean AND median AND win): a BROAD edge
        tail = None   # mean clearly beats control, median at the cost floor: a positive-skew TAIL
        for row in gate["rows"]:
            lbl = "top %d%% (n%d)" % (100 - row["pct"], row["n"])
            L.append("  %-24s%s" % (lbl, "".join("  " + cell(row["holds"].get(h, {})) for h in FRONT_HOLDS)))
            for h in FRONT_HOLDS:
                a, c = row["holds"].get(h, {}), gc.get(h, {})
                if a.get("n", 0) < 30:
                    continue
                edge = a.get("mean", -9) - c.get("mean", 0)
                if a.get("mean", -9) > 0 and a.get("median", -9) > 0 \
                        and a["mean"] > c.get("mean", 0) and a.get("win_pct", 0) >= c.get("win_pct", 0):
                    if broad is None or edge > broad[0]:
                        broad = (edge, row["pct"], h, a)
                if a.get("mean", -9) > 0 and edge >= TAIL_MARGIN:
                    if tail is None or edge > tail[0]:
                        tail = (edge, row["pct"], h, a)
        tag = "GATE" if not front_run else "FRONT-RUN GATE"
        side = "an UP-side tail (mid-to-mid overstates: live fills buy the spike, sell the reversion)" if front_run \
            else "an UP-side tail (live fills work against it)"
        if broad:
            L.append("  => %s top %d%% / hold %d PAYS out of sample: net %+.1f%% (median %+.1f%%, win %.0f%%, n%d) -- a BROAD edge, validate LIVE (1-slot NET probe) before any ShredStream build"
                     % (tag, 100 - broad[1], broad[2], broad[3]["mean"] * 100, broad[3]["median"] * 100,
                        broad[3]["win_pct"], broad[3]["n"]))
        elif tail:
            e, pct, h, a = tail
            L.append("  => %s top %d%% / hold %d TAIL edge: net %+.1f%% vs control %+.1f%% (median %+.1f%%, win %.0f%%, n%d) -- a positive-skew tail, not broad; the median is pinned at the cost floor because the up-move is rare, so the tail IS the edge (as in this class)."
                     % (tag, 100 - pct, h, a["mean"] * 100, gc.get(h, {}).get("mean", 0) * 100,
                        a["median"] * 100, a["win_pct"], a["n"]))
            if front_run:
                if backrun is not None and e > backrun:
                    L.append("     precursor tail (+%.1f%%) OUT-MEANS the all-predictors/backrun gate (+%.1f%%) -- removing abs_move did NOT collapse the edge, so it is the wallet SETUP, not abs_move continuation."
                             % (e * 100, backrun * 100))
                L.append("     It is %s, so the mean is NOT bankable as shown -> settle with a LIVE 1-slot NET probe, do not drop on the median." % side)
            else:
                L.append("     It is %s -> the precursor-only gate below is the cleaner read; settle live." % side)
        elif front_run:
            L.append("  => %s no cell beats control on mean either -- the wallet setup does not lead the move at the entry; drop ShredStream" % tag)
        else:
            L.append("  => %s no cell beats control on mean -- precision is not the lever on the backrun gate; the precursor gate below is the real test" % tag)
        return tail[0] if tail else None

    backrun_edge = gate_block(out.get("logit_gate"), "LOGIT-GATE ENTER-ON-SIGNAL (all predictors)", False)
    gate_block(out.get("logit_gate_precursor"), "FRONT-RUN GATE (precursors only)", True, backrun=backrun_edge)

    L.append("")
    L.append("  Mid-to-mid on untouched pools (hypothesis only, per the invariant); the live arbiter is a real reactive NET entry.")
    return "\n".join(L)


# -- visibility experiment: does the box's actual ShredStream view carry the front-run edge? --------------------
# The study fit the gate on FULL flows (every trade, direction from pool vault deltas). The live box decodes only
# TOP-LEVEL pumpswap swaps from shreds (~40% of trades -- mostly the washbot); ~60% are aggregator-routed CPIs whose
# direction is invisible. So the box fires on a different, washbot-only slot composition than the study, and the
# feed verifier found every box fire diverges. This refits the SAME gate on three visibilities -- full (ceiling),
# direct (what the box literally decodes), buyer_plus (direct + aggregator buyers, approximating the organic signal)
# -- with the TARGET always the real full-flow up-move, and compares the precursor gate's holdout edge. Whichever of
# direct / buyer_plus keeps the edge is deployable from shreds; if only full has it, ShredStream cannot match the
# study and the approach is dropped. Mid-to-mid, untouched pools -> hypothesis; a live NET probe is the arbiter.

VIS_MODES = ("full", "direct", "buyer_plus")


def _gate_best(gate: dict, hold: int = 2, min_n: int = 30):
    """Best precursor-gate cell (highest hold-`hold` mean at n >= min_n) across the top-k%% rows, with the control.
    Returns (best_cell_or_None, control_mean). A tail edge is judged on the mean, per the owner's standing rule."""
    rows = (gate or {}).get("rows") or []
    ctrl = ((gate or {}).get("control") or {}).get(hold) or {}
    best = None
    for r in rows:
        a = (r.get("holds") or {}).get(hold) or {}
        if a.get("n", 0) >= min_n and a.get("mean") is not None:
            if best is None or a["mean"] > best["mean"]:
                best = {"pct": r["pct"], "n": a["n"], "mean": a["mean"],
                        "median": a.get("median"), "win": a.get("win_pct")}
    return best, (ctrl.get("mean") if ctrl else None)


def vis_experiment(have, metas, move=FRONT_MOVE, cost=DEFAULT_COST_PCT / 100.0,
                   flurry_min=DEFAULT_FLURRY_MIN, hold=2):
    """Run the front-run fit on each box-reproducible visibility and compare the precursor gate's holdout edge.
    Runs on the VISIBILITY-TAGGED flow pools only -- an untagged pool reads as not-direct, which would silently
    empty its direct view and over-count its buyer_plus view, so including it corrupts the restricted columns.
    Excluding it keeps all three columns on the same pool set (apples-to-apples); fetch more to grow the sample."""
    usable = [m for m in metas if slot_flow(have.get(m["pool"]) or {}) is not None]
    tagged = [m for m in usable if _has_vis(have.get(m["pool"]))]
    res = {}
    for mode in VIS_MODES:
        out = frontrun(have, tagged, move=move, cost=cost, flurry_min=flurry_min, vis=mode)
        best, ctrl = _gate_best(out.get("logit_gate_precursor"), hold=hold)
        res[mode] = {"pools": out.get("pools"), "slots": out.get("slots"), "events": out.get("events"),
                     "event_rate": out.get("event_rate"), "gate_rows": (out.get("logit_gate_precursor") or {}).get("rows"),
                     "best": best, "control": ctrl,
                     "survivors": len((out.get("survivors") or {}).get(out.get("net_lead"), []))}
    return {"mode": "vis-experiment", "move": move, "cost": cost, "hold": hold,
            "usable_pools": len(usable), "tagged_pools": len(tagged), "results": res}


def render_vis_experiment(out):
    L = ["FRONT-RUN VISIBILITY EXPERIMENT  (does the box's ShredStream view carry the edge?)",
         "  >= %.0f%% up-move, cost %.2f%%, gate hold %d slots;  usable pools %d, visibility-tagged %d"
         % (out["move"] * 100, out["cost"] * 100, out["hold"], out["usable_pools"], out["tagged_pools"])]
    if out["tagged_pools"] < out["usable_pools"]:
        L.append("  NOTE: run is on the %d TAGGED pools (same set in every column). %d more flow pools are cached"
                 % (out["tagged_pools"], out["usable_pools"] - out["tagged_pools"]))
        L.append("        but not yet tagged -- rerun --fetch to grow the sample (it re-tags cached flows first).")
    L.append("")
    L.append("  %-11s %8s %8s %10s   %-28s %8s %8s" %
             ("visibility", "pools", "events", "evt-rate", "best precursor-gate cell (hold)", "control", "edge"))
    labels = {"full": "full (all)", "direct": "direct-only", "buyer_plus": "buyer-plus"}
    for mode in VIS_MODES:
        r = out["results"].get(mode) or {}
        b, ctrl = r.get("best"), r.get("control")
        if b:
            cell = "top%d%%  n%-4d %+6.1f%%  md%+5.1f win%2.0f" % (
                100 - b["pct"], b["n"], b["mean"] * 100,
                (b["median"] or 0) * 100, b["win"] or 0)
            edge = (b["mean"] - ctrl) * 100 if ctrl is not None else None
        else:
            cell, edge = "(no cell n>=30)", None
        L.append("  %-11s %8s %8s %9.2f%%   %-28s %7s %7s" % (
            labels[mode], r.get("pools", "-"), r.get("events", "-"),
            (r.get("event_rate") or 0) * 100, cell,
            ("%+.1f%%" % (ctrl * 100)) if ctrl is not None else "-",
            ("%+.1f" % edge) if edge is not None else "-"))
    L.append("")
    full = (out["results"].get("full") or {}).get("best")
    direct = (out["results"].get("direct") or {}).get("best")
    bplus = (out["results"].get("buyer_plus") or {}).get("best")

    def _edge(mode):
        r = out["results"].get(mode) or {}
        b, c = r.get("best"), r.get("control")
        return (b["mean"] - c) if (b and c is not None) else None

    ef, ed, eb = _edge("full"), _edge("direct"), _edge("buyer_plus")
    L.append("  => The target is the REAL full-flow move in every column; only the FEATURES are restricted to what")
    L.append("     the box can see. So a column with a positive edge is deployable from shreds as-is.")
    if ef is not None and ef > 0:
        keep = [m for m, e in (("direct", ed), ("buyer_plus", eb)) if e is not None and e > 0 and e >= 0.5 * ef]
        if keep:
            L.append("     The edge SURVIVES on %s (>= half the full-view edge) -- the box's view carries it; build/keep ShredStream."
                     % " and ".join(keep))
        elif (ed is not None and ed > 0) or (eb is not None and eb > 0):
            L.append("     The edge survives only WEAKLY on the restricted view (< half full) -- marginal; a live NET probe decides.")
        else:
            L.append("     The edge COLLAPSES on both restricted views -- it needs the aggregator (organic) trades the box can't")
            L.append("     see from shreds, so ShredStream cannot match the study. Drop the front-run unless the feed is widened.")
    else:
        L.append("     No full-view edge at this move/hold either -- nothing to transfer; this is a null result, not a feed gap.")
    L.append("")
    L.append("  Mid-to-mid, untouched pools (hypothesis per the invariant); the live arbiter is a real reactive NET entry.")
    return "\n".join(L)
