"""Rug-vs-win filter trained on the early books' own entries (11-06 study, no trading effect).

The launch scorer's rug-vs-win score (P(rug) - P(+50% win), second-4 on-chain features) separates early pools too,
but its kept half only broke even there and it skipped most of what the early model buys. This study trains the
same kind of score on what the early books see: GeckoTerminal's first-sight numbers (`model_book.early_features`)
for every eligible hand-made pool (seen within 4 min, liquid at first sight).

Walk-forward: each `BLOCK_H` block is scored by models fitted only on pools whose 5-minute snapshot came before the
block; the skip cut-offs are quantiles of the previous block's scores under the same model (as a live book ranks
against the recent past). Outcomes use the watcher's snapshots (`EXIT_AGES`, pool age in minutes; entry = the
first-sight price, 6% round trip, emptied pool = -100%), plus two conditional exits that hold only pools already up
at the 5-minute snapshot. Every result is compared with random skips of the same size.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

from tradingbot import watch_analysis as wa
from tradingbot.config import Config

BLOCK_H = 6
EXIT_AGES = (5, 15, 30, 60, 180)  # watcher snapshot ages: minute 5 is taken around pool minute 6
HOLD_IF_UP = 0.10  # conditional exits: keep holding past minute 5 only when already +10% after costs
SKIPS = (20, 30, 50)
MIN_TRAIN = 1000
MIN_CUT_POOLS = 30
RANDOM_DRAWS = 500
BIG = 0.5


def _exit_name(e) -> str:
    return f"snapshot {e} min" if isinstance(e, int) else e


EXITS = tuple(EXIT_AGES) + ("5 min, hold to 15 if up", "5 min, hold to 30 if up")


def load_pools(store, cfg: Config, since: float = 0.0) -> list[dict]:
    """Eligible early pools (as `model_book.early_candidates`), oldest first, with features and every exit's
    return (None = no usable snapshot)."""
    from tradingbot import model_book as mb
    pools = {}
    for p in store.query(
            "SELECT pool, mint, dex, name, created_ts, seen_ts, first FROM watch_pools WHERE dex = 'pumpswap' "
            "AND first IS NOT NULL AND seen_ts >= ? ORDER BY seen_ts", (since,)):
        if not mb.is_handmade(p["dex"], p["mint"]) or not p["created_ts"] or not p["seen_ts"]:
            continue
        if not 0 <= p["seen_ts"] - p["created_ts"] <= mb.EARLY_MAX_SEEN_MIN * 60:
            continue
        first = json.loads(p["first"])
        pf, lf = wa._f(first.get("price")), wa._f(first.get("liq")) or 0.0
        if not pf or lf < cfg.model_book_min_liq_usd:
            continue
        pools[p["pool"]] = {"pool": p["pool"], "created_ts": p["created_ts"], "seen_ts": p["seen_ts"], "pf": pf,
                            "lf": lf, "x": {k: v for k, v in mb.early_features(dict(p), first, p["seen_ts"]).items()
                                            if v is not None}, "snaps": {}}
    if not pools:
        return []
    marks = ",".join("?" * len(EXIT_AGES))
    for s in store.query(f"SELECT s.pool, s.age_min, s.ts, s.data FROM watch_snaps s JOIN watch_pools w "
                         f"ON w.pool = s.pool WHERE w.dex = 'pumpswap' AND w.seen_ts >= ? AND s.age_min IN ({marks})",
                         (since, *EXIT_AGES)):
        p = pools.get(s["pool"])
        if p is not None and s["data"] and s["data"] != "{}":
            p["snaps"][s["age_min"]] = (s["ts"], json.loads(s["data"]))
    out = []
    for p in pools.values():
        rets, pulled = {}, False
        for age in EXIT_AGES:
            snap = p["snaps"].get(age)
            if snap and (wa._f(snap[1].get("liq")) or 0.0) < wa.RUG_LIQ_SHARE * p["lf"]:
                pulled = True
            if pulled:
                rets[age] = -1.0
            elif snap and wa._f(snap[1].get("price")):
                rets[age] = min(wa._f(snap[1]["price"]) / p["pf"] - 1 - wa.ROUND_TRIP_COST, 3.0)
            else:
                rets[age] = None
        r5 = rets[5]
        for name, later in (("5 min, hold to 15 if up", 15), ("5 min, hold to 30 if up", 30)):
            rets[name] = None if r5 is None else rets[later] if r5 >= HOLD_IF_UP else r5
        p["rets"] = rets
        p["ts5"] = p["snaps"][5][0] if 5 in p["snaps"] else None
        p["exit_min"] = {age: (p["snaps"][age][0] - p["created_ts"]) / 60 for age in EXIT_AGES if age in p["snaps"]}
        del p["snaps"]
        out.append(p)
    out.sort(key=lambda r: r["seen_ts"])
    return out


def _summary(rets: list[float]) -> dict:
    if not rets:
        return {"n": 0}
    a = np.array(rets)
    return {"n": len(a), "mean_pct": round(100 * float(a.mean()), 1), "median_pct": round(100 * float(np.median(a)), 1),
            "rugged_pct": round(100 * float((a <= -0.999).mean())), "big_win_pct": round(100 * float((a >= BIG).mean()))}


def _scores(model, rows) -> np.ndarray:
    return np.array([model.score(r["x"]) for r in rows])


def study(store, cfg: Config, hours: float | None = None, now: float | None = None, seed: int = 0) -> dict:
    """Walk-forward test of a rug-vs-win score on first-sight features, over every exit in EXITS."""
    from tradingbot import model_book as mb
    from tradingbot.launch_live import RugWinModel, _fit_capped
    now = now if now is not None else time.time()
    pools = load_pools(store, cfg)
    if not pools:
        return {"error": "no eligible pools"}
    test_from = now - hours * 3600 if hours else 0.0
    block_s = BLOCK_H * 3600
    first_block = int(pools[0]["seen_ts"] // block_s + 1) * block_s
    tested: list[dict] = []
    blocks = []
    for start in range(first_block, int(now // block_s + 1) * block_s, block_s):
        if start + block_s <= test_from:
            continue
        test = [p for p in pools if start <= p["seen_ts"] < start + block_s]
        train = [p for p in pools if p["ts5"] is not None and p["ts5"] < start and p["rets"][5] is not None]
        prev = [p for p in pools if start - block_s <= p["seen_ts"] < start]
        if not test or len(train) < MIN_TRAIN or len(prev) < MIN_CUT_POOLS:
            continue
        rug = np.array([float(p["rets"][5] <= -0.999) for p in train])
        big = np.array([float(p["rets"][5] >= BIG) for p in train])
        if rug.min() == rug.max() or big.min() == big.max():
            continue
        model = RugWinModel(_fit_capped(train, rug), _fit_capped(train, big), {})
        prev_sc = _scores(model, prev)
        cuts = {k: float(np.quantile(prev_sc, 1 - k / 100)) for k in SKIPS}
        recent = [p for p in train if p["created_ts"] >= start - cfg.model_train_days * wa.DAY_S]
        early = mb.fit_early([{"pool": p["pool"], "created_ts": p["created_ts"], "ret": p["rets"][5],
                               "up": float(p["rets"][5] >= mb.EARLY_GAIN), "first_liq": p["lf"], **p["x"]}
                              for p in recent], cfg)
        sc = _scores(model, test)
        for p, s in zip(test, sc):
            tested.append({**p, "score": float(s), "block": start,
                           "kept": {k: bool(s < cuts[k]) for k in SKIPS},
                           "pick": early is not None and early.score(p["x"]) >= early.threshold})
        blocks.append({"block (UTC)": datetime.fromtimestamp(start, timezone.utc).strftime("%m-%d %H:00"),
                       "train": len(train), "pools": len(test)})
    if not tested:
        return {"error": "too little history for a walk-forward test", "pools": len(pools)}
    rng = np.random.default_rng(seed)
    universes = {"every eligible pool": tested, "early model picks (top 5%, walk-forward)":
                 [p for p in tested if p["pick"]]}
    out = {"pools": len(pools), "tested": len(tested), "blocks": len(blocks),
           "from": blocks[0]["block (UTC)"], "to": blocks[-1]["block (UTC)"],
           "exit taken at pool minute (median)": {_exit_name(a): round(float(np.median(
               [p["exit_min"][a] for p in tested if a in p["exit_min"]])), 1) for a in EXIT_AGES
               if any(a in p["exit_min"] for p in tested)}}
    r5 = [p for p in tested if p["rets"][5] is not None]
    if len(r5) >= 20:
        s = np.array([p["score"] for p in r5])
        out["AUC at the 5-min exit (score = riskiness)"] = {
            "rugs (want > 0.5)": _r3(wa.auc(s, np.array([p["rets"][5] <= -0.999 for p in r5], dtype=int))),
            "big wins +50% (want < 0.5)": _r3(wa.auc(s, np.array([p["rets"][5] >= BIG for p in r5], dtype=int))),
            "losers (want > 0.5)": _r3(wa.auc(s, np.array([p["rets"][5] <= 0 for p in r5], dtype=int)))}
    for uname, rows in universes.items():
        u = {}
        for e in EXITS:
            have = [p for p in rows if p["rets"][e] is not None]
            if not have:
                continue
            a = np.array([p["rets"][e] for p in have])
            ex = {"with outcome": f"{len(have)} of {len(rows)}", "all": _summary(list(a))}
            for k in SKIPS:
                kept = [p["rets"][e] for p in have if p["kept"][k]]
                skipped = [p["rets"][e] for p in have if not p["kept"][k]]
                if len(kept) < 5 or len(skipped) < 5:
                    ex[f"skip riskiest {k}%"] = {"note": "too few"}
                    continue
                rand = np.array([rng.choice(a, len(kept), replace=False).mean() for _ in range(RANDOM_DRAWS)])
                by_block: dict = {}
                for p in have:
                    by_block.setdefault(p["block"], ([], []))[0 if p["kept"][k] else 1].append(p["rets"][e])
                gaps = [np.mean(kb) - np.mean(sb) for kb, sb in by_block.values() if kb and sb]
                ex[f"skip riskiest {k}%"] = {
                    "kept": _summary(kept), "skipped": _summary(skipped),
                    "kept minus skipped (pts)": round(100 * (float(np.mean(kept)) - float(np.mean(skipped))), 1),
                    "random keeps: mean / share worse than kept": [round(100 * float(rand.mean()), 1),
                                                                   round(float(np.mean(rand < np.mean(kept))), 3)],
                    "blocks where kept beat skipped": f"{sum(g > 0 for g in gaps)} of {len(gaps)}"}
            u[_exit_name(e)] = ex
        out[uname] = u
    day = lambda ts: time.strftime("%Y-%m-%d", time.gmtime(ts))
    out["by day (every eligible pool, 5-min exit, skip riskiest 50%)"] = [
        {"day": d, "all": _summary([p["rets"][5] for p in tested if day(p["seen_ts"]) == d and p["rets"][5] is not None])
         .get("mean_pct"), "kept": _summary([p["rets"][5] for p in tested if day(p["seen_ts"]) == d
                                              and p["rets"][5] is not None and p["kept"][50]])}
        for d in sorted({day(p["seen_ts"]) for p in tested})]
    out["note"] = ("walk-forward by 6 h block; cut-offs from the previous block's scores; entry = first-sight price, "
                   "exits = watcher snapshots (minute 5 ~ pool minute 6), 6% round trip, emptied pool (liquidity < "
                   "10% of first sight) = -100% from then on; snapshot prices ran 8-20 pts above real trades on "
                   "11-02, so a kept half needs a clear margin")
    return out


def _r3(x):
    return None if x is None else round(x, 3)
