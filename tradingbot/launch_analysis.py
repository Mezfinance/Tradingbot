"""Would buying in a pool's first seconds pay? (launch data from `Enricher.poll_launch`)

For each sampled hand-made pool, the on-chain price path of its first minutes (pool SOL / pool tokens after
each trade) gives the price at any second X after creation. The exit is the 5-minute snapshot (~pool minute
6.3, the yardstick of the early-book backtests): emptied (liquidity < 10% of first sight) = -100%. The chain
price only links the entry second to first sight, so no SOL/USD conversion is needed:

    return(X) = (exit price / first-sight price) * (chain price at first sight / chain price at X) - 1 - cost

A pool pulled on-chain between X and first sight is -100% for the second-X buyer (the first-sight buyer
skips it: its pool is already empty)."""
from __future__ import annotations

import json

import numpy as np

from tradingbot import watch_analysis as wa
from tradingbot.config import Config

COST = 0.06
ENTRY_SECONDS = (5, 10, 20, 30, 45, 60, 90)
PULLED_SHARE = 0.10  # pool SOL below this share of its peak = pulled
MAX_SEEN_S = 240


def chain_price(txs: list, t: float) -> float | None:
    """Pool SOL per token after the last trade at or before t seconds (None before the pool has both sides)."""
    p = None
    for r in txs:
        if r[0] > t:
            break
        if r[5] > 0 and r[6] > 0:
            p = r[6] / r[5]
    return p


def pulled_by(txs: list, t: float) -> bool:
    peak = 0.0
    for r in txs:
        if r[0] > t:
            break
        peak = max(peak, r[6])
        if peak > 0 and r[6] < PULLED_SHARE * peak:
            return True
    return False


def rows(store, cfg: Config, since: float = 0.0) -> list[dict]:
    out = []
    for r in store.query(
            """SELECT p.pool, p.mint, p.created_ts, p.seen_ts, p.first, e.data AS launch, s.data AS snap5
               FROM watch_enrich e JOIN watch_pools p ON p.pool=e.pool
               LEFT JOIN watch_snaps s ON s.pool=p.pool AND s.age_min=5
               WHERE e.kind='launch' AND p.created_ts >= ? AND json_extract(e.data, '$.txs') IS NOT NULL""",
            (since,)):
        launch, first = json.loads(r["launch"]), json.loads(r["first"] or "{}")
        snap = json.loads(r["snap5"]) if r["snap5"] else None
        txs = launch["txs"]
        seen = r["seen_ts"] - launch["create_ts"]
        row = {"pool": r["pool"], "seen_s": seen, "first_liq": wa._f(first.get("liq")) or 0.0,
               "sample": launch.get("selection", "sample") == "sample"}
        if not txs or seen > MAX_SEEN_S or seen < 0:
            row["drop"] = "first sight outside the window"
        elif not launch.get("complete") and txs[-1][0] < seen:
            row["drop"] = "over 100 txs before first sight"
        elif not snap or not wa._f(snap.get("price")) or not wa._f(first.get("price")):
            row["drop"] = "no 5-minute snapshot"
        else:
            p_seen = chain_price(txs, seen)
            gone = row["first_liq"] > 0 and (wa._f(snap.get("liq")) or 0) < PULLED_SHARE * row["first_liq"]
            seen_ret = -1.0 if gone else min(snap["price"] / first["price"] - 1 - COST, 3.0)
            row["eligible"] = row["first_liq"] >= cfg.model_book_min_liq_usd
            row["ret"] = {"first sight": None if pulled_by(txs, seen) else seen_ret}
            for x in ENTRY_SECONDS:
                if x >= seen:
                    continue
                p_x = chain_price(txs, x)
                if p_x is None or p_seen is None:
                    continue
                if pulled_by(txs, seen):
                    row["ret"][x] = -1.0
                else:
                    row["ret"][x] = -1.0 if gone else min((1 + seen_ret + COST) * p_seen / p_x - 1 - COST, 3.0)
        out.append(row)
    return out


def backtest(store, cfg: Config, since: float = 0.0, picks: set[str] | None = None) -> dict:
    data = rows(store, cfg, since)
    out = {"pools": len(data), "dropped": {}}
    for r in data:
        if "drop" in r:
            out["dropped"][r["drop"]] = out["dropped"].get(r["drop"], 0) + 1
    rng = np.random.default_rng(0)

    def summary(vals):
        a = np.array(vals)
        if len(a) == 0:
            return {"n": 0}
        lo, hi = np.percentile([rng.choice(a, len(a)).mean() for _ in range(1000)], [5, 95])
        return {"n": len(a), "mean_pct": round(a.mean() * 100, 1), "ci90": [round(lo * 100, 1), round(hi * 100, 1)],
                "median_pct": round(float(np.median(a)) * 100, 1), "rugged_pct": round(float(np.mean(a <= -0.99)) * 100)}

    groups = {"all sampled hand-made": [r for r in data if "ret" in r and r["sample"]],
              "eligible (first-sight liquidity >= min)": [r for r in data if r.get("eligible") and r["sample"]]}
    if picks:
        groups["early-book picks"] = [r for r in data if r["pool"] in picks and "ret" in r]
    for name, rs in groups.items():
        g = {"first sight": summary([r["ret"]["first sight"] for r in rs if r["ret"].get("first sight") is not None])}
        for x in ENTRY_SECONDS:
            both = [r for r in rs if x in r["ret"] and r["ret"].get("first sight") is not None]
            g[f"second {x}"] = {**summary([r["ret"][x] for r in rs if x in r["ret"]]),
                                "same pools at first sight": summary([r["ret"]["first sight"] for r in both])["mean_pct"]
                                if both else None}
        g["first sight seconds (median)"] = round(float(np.median([r["seen_s"] for r in rs])), 1) if rs else None
        out[name] = g
    return out
