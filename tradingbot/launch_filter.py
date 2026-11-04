"""11-04 filter study: rebuild the early-pull filter for the history model's bottom 20% (D1-D2).

Live, the early-pull model (label: LP pulled < 90 s) skipped the D1-D2 pools that made the money: the riskiest
30% / 50% kept no big winners, because heavy buying in the first seconds marks both early pulls and big movers.
This walk-forward study tests five other ways to use a risk score, all on the scorer's own finished liquid pools
(`watch_launch_live`), each judged against random choices of the same size:

  now     the live model (pulled < 90 s, every second-4 feature + creator record)
  opt 2   label = rugged before our minute-6 exit (pulled or drained), every feature
  opt 3   label = rugged before the exit, operator signs only (creator record, seed, creation-slot bundling)
  opt 4   two models, P(rugged before the exit) - P(+50% or more): skip only when rug odds beat win odds
  opt 5   the live model, but pools whose first-4 s buy SOL is in the training set's top third are never skipped
  opt 1   (every risk score above) sell the riskiest k% at second 60 or 90 instead of skipping them

Walk-forward: PULL_BLOCK_H test blocks, trained on pools whose minute-6 outcome was known by the block start.
Option 1 needs each D1-D2 pool's price at seconds 60 and 90 (one Helius call each, cached in
`watch_launch_path`). Trades are the scorer's ideal ones: vault price at second ~6.5 in, 6% round trip."""
from __future__ import annotations

import collections
import json
import math
import time
from datetime import datetime, timezone

import numpy as np

from tradingbot import launch_analysis as la
from tradingbot import launch_live as ll
from tradingbot import watch_analysis as wa
from tradingbot.data import pumpswap as ps

SKIPS = (10, 20, 30, 50)  # % of a block's pools skipped (or sold early) as riskiest
EARLY_EXITS_S = (60, 90)  # option 1: the riskiest pools are sold at these seconds after creation
RANDOM_DRAWS = 500
HEAVY_BUY_Q = 2 / 3  # option 5: buy SOL at or above this training quantile = heavy early buying, never skipped
BIG_WIN = 0.5
# operator signs: who launched the pool and how, not how its price moved (option 3)
OPERATOR_KEYS = ("op_prior", "op_early_pulls", "op_early_share", "op_pull_share", "op_min_since_last",
                 "log_seed_sol", "repeat_creator", "log_prior_pools", "creator_buys", "first_slot_txs")


def _rows(store, since: float) -> list[dict]:
    rows = [dict(r) for r in store.query(
        "SELECT pool, token_mint, creator, create_ts, feat, score_h, ret, pulled_ts, fill_px, fill_sol "
        "FROM watch_launch_live WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND feat IS NOT NULL "
        "AND create_ts >= ? ORDER BY create_ts", (since,))]
    hist = ll._creator_history(rows)
    for r in rows:
        r["x"] = ll.pull_inputs(json.loads(r["feat"]), hist[r["pool"]])
        r["early"] = float(bool(r["pulled_ts"]) and r["pulled_ts"] - r["create_ts"] < ll.EARLY_RUG_S)
        r["rug"] = float(r["ret"] <= -0.999)
        r["big"] = float(r["ret"] >= BIG_WIN)
    return rows


def _fit(train: list[dict], test: list[dict], label: str, keys=None) -> np.ndarray | None:
    """Probabilities for `test` from a logistic on `train` (inputs standardised and capped, as the live model)."""
    y = np.array([r[label] for r in train])
    if y.min() == y.max():
        return None
    names = sorted({k for r in train for k in r["x"] if keys is None or k in keys})
    xs, med = wa._matrix([r["x"] for r in train], names)
    mu, sd = xs.mean(0), xs.std(0) + 1e-9
    z = lambda m: np.clip((m - mu) / sd, -la.CLIP_Z, la.CLIP_Z)
    w, b = wa.logistic(z(xs), y)
    return 1 / (1 + np.exp(-(z(wa._matrix([r["x"] for r in test], names, med)[0]) @ w + b)))


def _variants(train: list[dict], test: list[dict]) -> dict:
    """Risk score per test pool for each variant (higher = skip / sell early first); None = can't fit."""
    early = _fit(train, test, "early")
    rug = _fit(train, test, "rug")
    out = {"now: pulled < 90 s (live model)": early,
           "opt 2: rugged before our exit": rug,
           "opt 3: operator signs only": _fit(train, test, "rug", OPERATOR_KEYS)}
    big = _fit(train, test, "big")
    out["opt 4: P(rug) - P(+50% win)"] = rug - big if rug is not None and big is not None else None
    if early is not None:
        buys = [r["x"].get("log_buy_sol") for r in train if r["x"].get("log_buy_sol") is not None]
        cut = float(np.quantile(buys, HEAVY_BUY_Q)) if buys else math.inf
        heavy = np.array([(r["x"].get("log_buy_sol") or 0.0) >= cut for r in test])
        out["opt 5: live model, heavy early buying never skipped"] = np.where(heavy, -np.inf, early)
    else:
        out["opt 5: live model, heavy early buying never skipped"] = None
    return out


def _fetcher(store, helius, max_credits: int, stats: collections.Counter):
    have = {(r["pool"], r["sec"]): r for r in store.query("SELECT pool, sec, sol, tok FROM watch_launch_path")}
    start = helius.credits_used if helius is not None else 0

    def fetch(r, sec):
        if (r["pool"], sec) in have:
            return have[(r["pool"], sec)]
        if helius is None:
            stats["no Helius client"] += 1
            return None
        if helius.credits_used - start + 10 > max_credits:
            stats["left for a rerun (credit cap)"] += 1
            return None
        tx = helius.last_before(r["pool"], r["create_ts"] + sec)
        if tx is False:
            stats["call failed"] += 1
            return None
        st = {"sol": ll._raw_vault(tx, r["pool"], ps.WSOL) if tx else None,
              "tok": ll._raw_vault(tx, r["pool"], r["token_mint"]) if tx else None}
        store.execute("INSERT OR REPLACE INTO watch_launch_path(pool, sec, sol, tok, ts) VALUES (?,?,?,?,?)",
                      (r["pool"], sec, st["sol"], st["tok"], time.time()))
        have[(r["pool"], sec)] = st
        stats["fetched"] += 1
        return st

    fetch.used = lambda: (helius.credits_used - start) if helius is not None else 0
    return fetch


def _ret_at(r, sec, fetch) -> float | None:
    """The scorer's trade sold at `sec` instead of minute 6: -100% once pulled or drained; None = not fetched."""
    if r["pulled_ts"] and r["pulled_ts"] <= r["create_ts"] + sec:
        return -1.0
    if not r["fill_px"] or not r["fill_sol"]:
        return None
    st = fetch(r, sec)
    if st is None:
        return None
    if not st["sol"] or not st["tok"] or st["sol"] < la.PULLED_SHARE * r["fill_sol"]:
        return -1.0
    return min((st["sol"] / st["tok"]) / r["fill_px"] - 1 - la.COST, 3.0)


def _pct(x) -> float:
    return round(100 * float(x), 1)


def _skip_eval(group: list[dict], key: str, rng) -> dict:
    """Skip the riskiest k% within each block vs random skips of the same size."""
    rets = np.array([g["ret"] for g in group])
    by_block = collections.defaultdict(list)
    for i, g in enumerate(group):
        by_block[g["block"]].append(i)
    out = {}
    for k in SKIPS:
        keep = []
        for idx in by_block.values():
            idx = sorted(idx, key=lambda i: group[i][key])
            keep += idx[:len(idx) - round(len(idx) * k / 100)]
        kept = rets[keep]
        skipped = np.setdiff1d(np.arange(len(group)), keep)
        rand = np.array([rets[sum((list(rng.choice(idx, len(idx) - round(len(idx) * k / 100), replace=False))
                                   for idx in by_block.values()), [])].mean() for _ in range(RANDOM_DRAWS)])
        out[f"skip riskiest {k}%"] = {
            "kept": len(kept), "mean_pct": _pct(kept.mean()), "rugged_pct": round(100 * (kept <= -0.999).mean()),
            "big_win_pct": round(100 * (kept >= BIG_WIN).mean()),
            "skipped: rugs / big winners": f"{int((rets[skipped] <= -0.999).sum())} / "
                                           f"{int((rets[skipped] >= BIG_WIN).sum())}",
            "random_mean_pct": _pct(rand.mean()),
            "random_90pct_range": [_pct(np.percentile(rand, 5)), _pct(np.percentile(rand, 95))],
            "beats_random_pct": round(100 * (rand < kept.mean()).mean())}
    return out


def _exit_eval(group: list[dict], key: str, sec: int, rng) -> dict:
    """Sell the riskiest k% within each block at `sec`, the rest at minute 6, vs random pools sold at `sec`."""
    base = np.array([g["ret"] for g in group])
    early = np.array([g["at"][sec] for g in group])
    by_block = collections.defaultdict(list)
    for i, g in enumerate(group):
        by_block[g["block"]].append(i)
    out = {}
    for k in SKIPS:
        sel = []
        for idx in by_block.values():
            sel += sorted(idx, key=lambda i: -group[i][key])[:round(len(idx) * k / 100)]
        pol = base.copy()
        pol[sel] = early[sel]
        rand = []
        for _ in range(RANDOM_DRAWS):
            r = base.copy()
            pick = sum((list(rng.choice(idx, round(len(idx) * k / 100), replace=False))
                        for idx in by_block.values()), [])
            r[pick] = early[pick]
            rand.append(r.mean())
        rand = np.array(rand)
        out[f"sell riskiest {k}% early"] = {
            "mean_pct": _pct(pol.mean()), "rugged_pct": round(100 * (pol <= -0.999).mean()),
            "big_win_pct": round(100 * (pol >= BIG_WIN).mean()),
            "random_mean_pct": _pct(rand.mean()),
            "random_90pct_range": [_pct(np.percentile(rand, 5)), _pct(np.percentile(rand, 95))],
            "beats_random_pct": round(100 * (rand < pol.mean()).mean())}
    return out


def study(store, helius=None, hours: float | None = None, max_credits: int = 6_000, now: float | None = None,
          seed: int = 7) -> dict:
    ll.ensure_schema(store)
    store.execute(ll.PATH_SCHEMA)
    now = now if now is not None else time.time()
    rows = _rows(store, now - hours * 3600 if hours else 0)
    if len(rows) < ll.PULL_MIN_TRAIN + 50:
        return {"pools": len(rows), "note": f"needs {ll.PULL_MIN_TRAIN + 50}+ finished liquid pools"}
    from tradingbot.model_book import decile_index
    block = ll.PULL_BLOCK_H * 3600
    tested, blocks = [], []
    t = rows[0]["create_ts"] + block
    while t < rows[-1]["create_ts"]:
        # train only on pools whose minute-6 outcome was known when the block started
        train = [r for r in rows if r["create_ts"] < t - ll.EXIT_S - 60]
        test = [r for r in rows if t <= r["create_ts"] < t + block]
        t += block
        if len(train) < ll.PULL_MIN_TRAIN or not test:
            continue
        risks = _variants(train, test)
        dec = decile_index([r["score_h"] for r in test]) if all(r["score_h"] is not None for r in test) else None
        info = {"from": datetime.fromtimestamp(t - block, timezone.utc).strftime("%m-%d %H:%M"),
                "train": len(train), "test": len(test), "auc": {}}
        for name, p in risks.items():
            if p is None:
                continue
            for label in ("early", "rug", "big"):
                a = wa.auc(np.nan_to_num(p, neginf=-1e9), np.array([r[label] for r in test]))
                info["auc"].setdefault(name, {})[label] = round(a, 2) if a is not None else None
        blocks.append(info)
        for i, r in enumerate(test):
            row = {"pool": r["pool"], "ret": r["ret"], "block": len(blocks) - 1,
                   "low": dec is not None and dec[i] <= 1, "src": r}
            for name, p in risks.items():
                row[name] = float(p[i]) if p is not None else None
            tested.append(row)
    if not tested:
        return {"pools": len(rows), "note": "no block had enough training rows"}
    names = [n for n in tested[0] if n.startswith(("now", "opt"))]
    rng = np.random.default_rng(seed)
    low = [g for g in tested if g["low"]]

    # option 1 prices: D1-D2 test pools only
    stats = collections.Counter()
    fetch = _fetcher(store, helius, max_credits, stats)
    for g in low:
        g["at"] = {sec: _ret_at(g["src"], sec, fetch) for sec in EARLY_EXITS_S}
    stats["credits_used"] = fetch.used()

    def keep_all(group):
        rets = np.array([g["ret"] for g in group])
        return {"n": len(group), "mean_pct": _pct(rets.mean()) if len(rets) else None,
                "rugged_pct": round(100 * (rets <= -0.999).mean()) if len(rets) else None,
                "big_win_pct": round(100 * (rets >= BIG_WIN).mean()) if len(rets) else None}

    out = {"pools": len(rows), "tested": len(tested), "blocks": blocks, "fetch": dict(stats),
           "history D1-D2 (test blocks)": {"keep all": keep_all(low)},
           "every liquid pool (test blocks)": {"keep all": keep_all(tested)}}
    for sec in EARLY_EXITS_S:
        priced = [g for g in low if g["at"][sec] is not None]
        if priced:
            out["history D1-D2 (test blocks)"][f"sell every pool at {sec} s"] = {
                "n": len(priced), "mean_pct": _pct(np.mean([g["at"][sec] for g in priced])),
                "same pools at minute 6": _pct(np.mean([g["ret"] for g in priced]))}
    for name in names:
        lowv = [g for g in low if g[name] is not None]
        allv = [g for g in tested if g[name] is not None]
        res = {"skip (D1-D2)": _skip_eval(lowv, name, rng) if lowv else {},
               "skip (every liquid pool)": _skip_eval(allv, name, rng) if allv else {}}
        if not name.startswith("opt 5"):  # 'never skip' has no meaning for an early sale ranking
            for sec in EARLY_EXITS_S:
                priced = [g for g in lowv if g["at"][sec] is not None]
                if priced:
                    res[f"opt 1: sell early at {sec} s (D1-D2, n={len(priced)})"] = _exit_eval(priced, name, sec, rng)
        out[name] = res
    out["note"] = ("walk-forward, " f"{ll.PULL_BLOCK_H} h test blocks, trained on pools finished before each block; "
                   "D1-D2 = history-model bottom 20% ranked within each block; trades = the scorer's ideal ones "
                   "(second ~6.5 in, minute 6 out, 6% cost); beats_random = share of random choices of the same "
                   "size that did worse; auc per block: how well each score ranks early pulls / rugs before the "
                   "exit / +50% wins (for a good filter: high for rugs, low or 0.5 for wins); several variants were "
                   "tried, so expect one to look good by chance: trust only a clear, consistent lead")
    return out
