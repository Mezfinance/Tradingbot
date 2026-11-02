"""Unbiased history of hand-made pumpswap pools, straight from the chain (no GeckoTerminal).

The launch backtest only saw pools GeckoTerminal listed (~70 s after creation, ~1 in 5 of all hand-made pools live
on 11-02), so it filtered on something that happens after second 4. This rebuilds the last few days from Helius:

1. `backfill`: Helius getProgramAccountsV2 lists the hand-made pumpswap pools (coin creator all zeros) whose
   account changed since the slot N days ago (a pool account changes on every swap, so this is the recent pools
   plus the few old ones still trading), each pool's LP mint gives its creation time (first signature, 1
   credit), and pools created in the window get their first `SPAN_S` seconds of history (10 credits per 100 txs),
   parsed by the same `launch_record` the live scorer uses. Rows in watch.db table `watch_launch_hist`.
2. `test`: per pool the live scorer's second-4 features (`launch_live.live_features`) and a chain-only outcome:
   bought at the second-6 chain price, sold at the pool's own price at pool minute 6, -100% if the pool's SOL
   fell below 10% of its peak by then, 6% round trip. Reported for the current live model (`launch4.json`) and
   for a fresh walk-forward model trained on these rows, each split by listed / never listed on GeckoTerminal."""
from __future__ import annotations

import collections
import json
import logging
import time

import numpy as np

from tradingbot import launch_analysis as la
from tradingbot.config import Config
from tradingbot.data import pumpswap as ps
from tradingbot.data.helius import Helius, b58encode, launch_record

log = logging.getLogger(__name__)

SPAN_S = 400  # history kept per pool: past the minute-6 exit
MAX_PAGES = 6  # 600 txs; busier pools are marked incomplete
SLOTS_PER_S = 2.5 * 1.15  # Solana's ~2.5 slots/s plus a margin, so the slot window covers the whole time window
DECIDE_S, FILL_S, EXIT_S = 4, 6, 360

SCHEMA = """CREATE TABLE IF NOT EXISTS watch_launch_hist(
    pool TEXT PRIMARY KEY, token_mint TEXT, lp_mint TEXT, create_ts REAL, status TEXT, data TEXT, ts REAL)"""


def candidates(helius: Helius, days: float) -> list[dict] | None:
    """Hand-made SOL-pair pumpswap pools whose account changed in the last `days` days."""
    slot = helius.slot()
    if slot is None:
        return None
    since = max(0, slot - int(days * 86400 * SLOTS_PER_S))
    accts = helius.program_accounts_changed(
        ps.PROGRAM, [{"dataSize": 301}, {"memcmp": {"offset": 211, "bytes": ps.NO_KEY}}],
        {"offset": 43, "length": 96}, since)
    if accts is None:
        return None
    out = []
    for pool, raw in accts:
        if len(raw) < 96:
            continue
        base, quote, lp = b58encode(raw[0:32]), b58encode(raw[32:64]), b58encode(raw[64:96])
        if ps.WSOL not in (base, quote):
            continue
        out.append({"pool": pool, "token_mint": quote if base == ps.WSOL else base, "lp_mint": lp})
    return out


def backfill(store, helius: Helius, days: float, max_credits: int, now: float | None = None,
             progress=None) -> dict:
    now = now if now is not None else time.time()
    store.execute(SCHEMA)
    stats: collections.Counter = collections.Counter()
    base = helius.credits_used  # the listing calls count toward the cap too
    pools = candidates(helius, days)
    if pools is None:
        return {"error": "could not list pools (getSlot / getProgramAccountsV2 failed; needs HELIUS_API_KEY)"}
    stats["candidates"] = len(pools)
    done = {r[0] for r in store.query("SELECT pool FROM watch_launch_hist")}
    start, end = now - days * 86400, now - SPAN_S - 60
    save = lambda p, status, create=None, data=None: store.execute(
        "INSERT OR REPLACE INTO watch_launch_hist(pool, token_mint, lp_mint, create_ts, status, data, ts) "
        "VALUES(?,?,?,?,?,?,?)", (p["pool"], p["token_mint"], p["lp_mint"], create, status, data, now))
    for i, p in enumerate(pools):
        if progress and i % 100 == 0:
            progress(i, len(pools), helius.credits_used - base)
        if p["pool"] in done:
            stats["already fetched"] += 1
            continue
        if helius.credits_used - base + 10 * MAX_PAGES + 1 > max_credits:
            stats["stopped: credit cap"] = len(pools) - i
            break
        t = helius.first_time(p["lp_mint"])
        if t is None:
            stats["no LP mint history (left for a rerun)"] += 1
            continue
        if not start <= t <= end:
            save(p, "outside window", t)
            stats["outside window"] += 1
            continue
        txs, reached, ok = helius.history_from_start(p["pool"], SPAN_S, MAX_PAGES)
        if not ok:
            stats["history call failed (left for a rerun)"] += 1
            continue
        if not txs:
            save(p, "no transactions", t)
            stats["no transactions"] += 1
            continue
        rec = launch_record(txs, p["pool"], p["token_mint"], window_s=SPAN_S, exhausted=reached)
        if not start <= rec["create_ts"] <= end:
            save(p, "outside window", rec["create_ts"])
            stats["outside window"] += 1
            continue
        save(p, "ok", rec["create_ts"], json.dumps(rec))
        stats["fetched"] += 1
    stats["credits_used"] = helius.credits_used - base
    return dict(stats)


def rows(store, cfg: Config) -> tuple[list[dict], dict]:
    """One row per fetched pool: second-4 features (as live), liquid at second 4, chain-only minute-6 return."""
    from tradingbot.launch_live import live_features
    store.execute(SCHEMA)
    listed = {r[0] for r in store.query("SELECT pool FROM watch_pools")}
    dropped: collections.Counter = collections.Counter()
    out = []
    for r in store.query("SELECT pool, data FROM watch_launch_hist WHERE status='ok'"):
        rec = json.loads(r["data"])
        txs, create = rec.get("txs") or [], rec["create_ts"]
        if not txs:
            dropped["no trades parsed"] += 1
            continue
        if not rec.get("complete") and txs[-1][0] < EXIT_S:
            dropped["history pages ended before minute 6"] += 1
            continue
        upto4 = [t for t in txs if t[0] <= DECIDE_S]
        if not upto4:
            dropped["nothing on chain by second 4"] += 1
            continue
        sol4 = upto4[-1][6]
        sol_usd = la._sol_usd(store, create)
        liquid = (sol_usd is None or 2 * sol4 * sol_usd >= cfg.model_book_min_liq_usd) and not la.pulled_by(txs, DECIDE_S)
        entry, exit_p = la.chain_price(txs, FILL_S), la.chain_price(txs, EXIT_S)
        path = [t for t in txs if t[0] <= EXIT_S]
        peak = max(t[6] for t in path)
        rugged = peak > 0 and path[-1][6] < la.PULLED_SHARE * peak
        ret = -1.0 if rugged or not entry or not exit_p else min(exit_p / entry - 1 - la.COST, 3.0)
        feat = live_features(rec, create, la._prior_pools(store, rec.get("creator"), create))
        out.append({"pool": r["pool"], "created_ts": create, "listed": r["pool"] in listed, "liquid": liquid,
                    "liq_usd": 2 * sol4 * sol_usd if sol_usd else None,
                    "feat": {DECIDE_S: feat}, "ret": {FILL_S: ret}})
    return out, dict(dropped)


def _split(rs, rng) -> dict:
    return {"all": la._summary([r["ret"][FILL_S] for r in rs], rng),
            "listed on GeckoTerminal": la._summary([r["ret"][FILL_S] for r in rs if r["listed"]], rng),
            "never listed": la._summary([r["ret"][FILL_S] for r in rs if not r["listed"]], rng)}


def test(store, cfg: Config) -> dict:
    from tradingbot.launch_live import model_path
    from tradingbot.model_book import Model
    rng = np.random.default_rng(0)
    data, dropped = rows(store, cfg)
    status = dict(collections.Counter(r[0] for r in store.query("SELECT status FROM watch_launch_hist")))
    liquid = sorted([r for r in data if r["liquid"]], key=lambda r: r["created_ts"])
    out: dict = {"fetched_rows": status, "dropped": dropped, "liquid at second 4": len(liquid),
                 "listed share of liquid pools (%)":
                     round(100 * sum(r["listed"] for r in liquid) / len(liquid), 1) if liquid else None,
                 "every liquid pool": _split(liquid, rng)}
    try:
        m = Model.from_json(json.loads(model_path(cfg).read_text()))
        cut2, cut5 = m.meta.get("cut2"), m.meta.get("cut5")
        scores = {id(r): m.score(r["feat"][DECIDE_S]) for r in liquid}
        pick = lambda c: [r for r in liquid if c is not None and scores[id(r)] >= c - 1e-9]
        out["live model (launch4.json; in-sample for listed pools it was trained on)"] = {
            "share above cut2 (%)": round(100 * len(pick(cut2)) / len(liquid), 1) if liquid else None,
            "share above cut5 (%)": round(100 * len(pick(cut5)) / len(liquid), 1) if liquid else None,
            "top 2%": _split(pick(cut2), rng), "top 5%": _split(pick(cut5), rng)}
    except (OSError, ValueError, KeyError):
        out["live model (launch4.json; in-sample for listed pools it was trained on)"] = None
    days = sorted({la._day(r) for r in liquid})
    picks = {k: [] for k in la.TIMING_TOP_PCT}
    base, by_day = [], {}
    for d in days:
        train = [r for r in liquid if la._day(r) < d]
        test_rows = [r for r in liquid if la._day(r) == d]
        if len(train) < la.RANK_MIN_TRAIN or not test_rows:
            continue
        fit = la._fit_score(train, test_rows, DECIDE_S, FILL_S, la.LIVE_DROP)
        if fit is None:
            continue
        tr, te = fit
        base += test_rows
        row = {"pools": len(test_rows), "all_mean_pct": round(100 * float(np.mean([r["ret"][FILL_S] for r in test_rows])), 1)}
        for k in la.TIMING_TOP_PCT:
            cut = float(np.quantile(tr, 1 - k / 100))
            chosen = [r for r, s in zip(test_rows, te) if s >= cut - 1e-9]
            picks[k] += chosen
            row[f"top {k:g}%"] = {"n": len(chosen), "mean_pct": round(100 * float(np.mean(
                [r["ret"][FILL_S] for r in chosen])), 1) if chosen else None}
        by_day[d] = row
    out["walk-forward on these rows (train on earlier days, no names)"] = {
        "days_scored": len(by_day), "every liquid pool (test days)": _split(base, rng),
        **{f"top {k:g}%": _split(picks[k], rng) for k in la.TIMING_TOP_PCT}, "by day": by_day,
        **({} if by_day else {"note": f"needs {la.RANK_MIN_TRAIN}+ liquid pools on earlier days"})}
    return out
