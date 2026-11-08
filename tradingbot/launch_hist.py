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
from tradingbot.data.helius import Helius, _vault, b58encode, launch_record

log = logging.getLogger(__name__)

SPAN_S = 400  # history kept per pool: past the minute-6 exit
MAX_PAGES = 6  # fallback paging (600 txs) when the blockTime filter isn't available
HEAD_S, HEAD_PAGES = 30, 2  # the first seconds (features at 4, entry at 6) come from at most 2 pages
RESERVE = 1 + 10 * HEAD_PAGES + 10 + 10 * MAX_PAGES  # worst-case credits for one pool
SLOTS_PER_S = 2.5 * 1.15  # Solana's ~2.5 slots/s plus a margin, so the slot window covers the whole time window
DECIDE_S, FILL_S, EXIT_S = 4, 6, 360

SCHEMA = """CREATE TABLE IF NOT EXISTS watch_launch_hist(
    pool TEXT PRIMARY KEY, token_mint TEXT, lp_mint TEXT, create_ts REAL, status TEXT, data TEXT, ts REAL)"""


def candidates(helius: Helius, days: float, progress=None, max_pages: int = 5000) -> list[dict] | None:
    """Hand-made SOL-pair pumpswap pools whose account changed in the last `days` days."""
    slot = helius.slot()
    if slot is None:
        return None
    since = max(0, slot - int(days * 86400 * SLOTS_PER_S))
    accts = helius.program_accounts_changed(
        ps.PROGRAM, [{"dataSize": 301}, {"memcmp": {"offset": 211, "bytes": ps.NO_KEY}}],
        {"offset": 43, "length": 96}, since, max_pages=max_pages,
        progress=(lambda page, n: progress(f"listing page {page}", n, helius.credits_used)) if progress else None)
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


def _fetch_one(helius: Helius, p: dict, start: float, end: float) -> tuple[str, float | None, str | None, str]:
    """One pool: (status to save or '' to leave it for a rerun, create_ts, data JSON, stats key)."""
    act = helius.lp_activity(p["lp_mint"])
    if act is None:
        return "", None, None, "no LP mint history (left for a rerun)"
    t = act["first"]
    if not start <= t <= end:
        return "outside window", t, None, "outside window"
    txs, reached, ok = helius.history_from_start(p["pool"], HEAD_S, HEAD_PAGES)
    if not ok:
        return "", None, None, "history call failed (left for a rerun)"
    if not txs:
        return "no transactions", t, None, "no transactions"
    rec = launch_record(txs, p["pool"], p["token_mint"], window_s=SPAN_S, exhausted=reached)
    if not start <= rec["create_ts"] <= end:
        return "outside window", rec["create_ts"], None, "outside window"
    last_t = (txs[-1].get("blockTime") or 0) - rec["create_ts"]
    # a tx at or after minute 6 settles the state by then; a history that ended inside the head is complete
    covered = last_t >= EXIT_S or (reached and last_t < HEAD_S)
    key = "fetched"
    if not covered:
        # the pool's state at minute 6 = the balances after its last transaction by then (one call)
        last = helius.last_before(p["pool"], rec["create_ts"] + EXIT_S)
        if last is False:  # no blockTime filter: page through instead
            key = "fetched|exit by paging (filter unavailable)"
            txs, reached, ok = helius.history_from_start(p["pool"], SPAN_S, MAX_PAGES)
            if not ok:
                return "", None, None, "history call failed (left for a rerun)"
            rec = launch_record(txs, p["pool"], p["token_mint"], window_s=SPAN_S, exhausted=reached)
        elif last:
            rec["exit"] = [(last.get("blockTime") or 0) - rec["create_ts"],
                           _vault(last, p["pool"], p["token_mint"], "postTokenBalances"),
                           _vault(last, p["pool"], ps.WSOL, "postTokenBalances")]
    pulled, checked = helius.pull_time(p["lp_mint"], act)
    rec["pulled_ts"], rec["pull_checked"] = pulled, checked
    return "ok", rec["create_ts"], json.dumps(rec), key


def pulls(store, helius: Helius, workers: int = 1, max_credits: int = 10**9, progress=None) -> dict:
    """Pass over saved pools that have no pull time checked yet (fetched before 11-08): LP mint signatures + a
    supply read when the LP moved after creation -> `pulled_ts` in the row (same meaning as live: LP supply 0)."""
    import concurrent.futures as cf
    stats: collections.Counter = collections.Counter()
    base = helius.credits_used
    todo = [dict(r) for r in store.query("SELECT pool, lp_mint, data FROM watch_launch_hist WHERE status='ok'")]
    todo = [r for r in todo if not json.loads(r["data"]).get("pull_checked")]
    stats["to check"] = len(todo)

    def one(r):
        act = helius.lp_activity(r["lp_mint"])
        if act is None:
            return r, None, False
        pulled, checked = helius.pull_time(r["lp_mint"], act)
        return r, pulled, checked

    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, (r, pulled, checked) in enumerate(ex.map(one, todo)):
            if progress and i % 500 == 0:
                progress(i, len(todo), helius.credits_used - base)
            if not checked:
                stats["failed (left for a rerun)"] += 1
                continue
            rec = json.loads(r["data"])
            rec["pulled_ts"], rec["pull_checked"] = pulled, True
            store.execute("UPDATE watch_launch_hist SET data=? WHERE pool=?", (json.dumps(rec), r["pool"]))
            stats["pulled" if pulled else "not pulled"] += 1
            if helius.credits_used - base > max_credits:
                stats["stopped: credit cap"] = len(todo) - i - 1
                break
    stats["credits_used"] = helius.credits_used - base
    return dict(stats)


CREDITS_PER_POOL = 22  # typical: LP signatures 1 + first page 10 + minute-6 state 10 (+ LP supply 1 when it moved)


def backfill(store, helius: Helius, days: float, max_credits: int, now: float | None = None,
             progress=None, workers: int = 1, estimate: bool = False, seed: int | None = None) -> dict:
    """Fetch every hand-made SOL-pair pumpswap pool created in the last `days` days. `workers` threads share the
    Helius client's pace; `seed` shuffles the pools so a run stopped early leaves an even sample over the window;
    `estimate` only lists the candidates and prints the expected credits. Resumable: saved pools are skipped."""
    import concurrent.futures as cf
    import random
    now = now if now is not None else time.time()
    store.execute(SCHEMA)
    stats: collections.Counter = collections.Counter()
    base = helius.credits_used  # the listing calls count toward the cap too
    pools = candidates(helius, days, progress, max_pages=max(1, max_credits // 3 // 10))  # listing <= 1/3 of the cap
    if pools is None:
        return {"error": "could not list pools (getSlot / getProgramAccountsV2 failed; check the Helius key)"}
    stats["candidates"] = len(pools)
    stats["listing pages"] = getattr(helius, "last_pages", None)
    start, end = now - days * 86400, now - SPAN_S - 60
    # A pool saved as 'outside window' by a shorter run is fetched again once a longer window covers it.
    done = {r[0] for r in store.query("SELECT pool, status, create_ts FROM watch_launch_hist")
            if r[1] != "outside window" or r[2] is None or not start <= r[2] <= end}
    todo = [p for p in pools if p["pool"] not in done]
    stats["already fetched"] = len(pools) - len(todo)
    if estimate:
        stats["to fetch"] = len(todo)
        stats["estimated credits (upper: every candidate in the window)"] = len(todo) * CREDITS_PER_POOL
        stats["listing credits"] = helius.credits_used - base
        return dict(stats)
    if seed is not None:
        random.Random(seed).shuffle(todo)
    save = lambda p, status, create=None, data=None: store.execute(
        "INSERT OR REPLACE INTO watch_launch_hist(pool, token_mint, lp_mint, create_ts, status, data, ts) "
        "VALUES(?,?,?,?,?,?,?)", (p["pool"], p["token_mint"], p["lp_mint"], create, status, data, now))
    it = iter(enumerate(todo))
    stopped = False
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        running: dict = {}
        while True:
            while not stopped and len(running) < max(1, workers):
                nxt = next(it, None)
                if nxt is None:
                    break
                i, p = nxt
                if helius.credits_used - base + RESERVE * max(1, workers) > max_credits:
                    stats["stopped: credit cap"] = len(todo) - i
                    stopped = True
                    break
                if progress and i % 100 == 0:
                    progress(i, len(todo), helius.credits_used - base)
                running[ex.submit(_fetch_one, helius, p, start, end)] = p
            if not running:
                break
            fin, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for f in fin:
                p = running.pop(f)
                try:
                    status, create, data, key = f.result()
                except Exception as exc:  # noqa: BLE001  one bad pool must not end a days-long run
                    status, create, data, key = "", None, None, f"error (left for a rerun): {type(exc).__name__}"
                if status:
                    save(p, status, create, data)
                for k in key.split("|"):
                    stats[k] += 1
    stats["credits_used"] = helius.credits_used - base
    return dict(stats)


def fill_sol_prices(store, since: float, until: float, http=None) -> int:
    """Hourly SOL/USD (Binance SOLUSDT closes, public data mirror) into this store's `watch_market`, so pool
    liquidity in USD can be worked out for history older than the watcher's own SOL prices. Returns rows added."""
    import httpx
    http = http or httpx.Client(timeout=30)
    store.execute("CREATE TABLE IF NOT EXISTS watch_market(ts REAL PRIMARY KEY, sol_usd REAL, new_pools INTEGER)")
    store.execute("CREATE INDEX IF NOT EXISTS watch_market_ts ON watch_market(ts)")  # one lookup per pool
    have = {int(r[0]) for r in store.query("SELECT ts FROM watch_market WHERE ts >= ? AND ts <= ?", (since, until))}
    rows, t = [], since - since % 3600
    while t < until:
        try:
            d = http.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": "SOLUSDT", "interval": "1h", "startTime": int(t * 1000), "limit": 1000}).json()
        except Exception:  # noqa: BLE001
            break
        if not isinstance(d, list) or not d:
            break
        rows += [(x[0] / 1000, float(x[4])) for x in d if int(x[0] / 1000) not in have]
        t = d[-1][0] / 1000 + 3600
    if rows:
        store.executemany("INSERT OR IGNORE INTO watch_market(ts, sol_usd) VALUES (?, ?)", rows)
    return len(rows)


def rows(store, cfg: Config) -> tuple[list[dict], dict]:
    """One row per fetched pool: second-4 features (as live), liquid at second 4, chain-only minute-6 return."""
    from tradingbot.launch_live import live_features
    store.execute(SCHEMA)
    listed = {r[0] for r in store.query("SELECT pool FROM watch_pools")}
    dropped: collections.Counter = collections.Counter()
    out = []
    scan = getattr(store, "iterate", store.query)  # a year of rows: stream them instead of holding every JSON at once
    for r in scan("SELECT pool, data FROM watch_launch_hist WHERE status='ok'"):
        rec = json.loads(r["data"])
        txs, create = rec.get("txs") or [], rec["create_ts"]
        if not txs:
            dropped["no trades parsed"] += 1
            continue
        exit_state = rec.get("exit")
        if not exit_state and not rec.get("complete") and txs[-1][0] < EXIT_S:
            dropped["history pages ended before minute 6"] += 1
            continue
        upto4 = [t for t in txs if t[0] <= DECIDE_S]
        if not upto4:
            dropped["nothing on chain by second 4"] += 1
            continue
        sol4 = upto4[-1][6]
        sol_usd = la._sol_usd(store, create)
        liquid = (sol_usd is None or 2 * sol4 * sol_usd >= cfg.model_book_min_liq_usd) and not la.pulled_by(txs, DECIDE_S)
        entry = la.chain_price(txs, FILL_S)
        path = [t for t in txs if t[0] <= EXIT_S]
        if exit_state:  # [seconds after creation, pool tokens, pool SOL] after the last tx by minute 6
            _, tok6, sol6 = exit_state
            exit_p = sol6 / tok6 if tok6 and sol6 else None
            peak = max([t[6] for t in path] + [sol6])
        else:
            exit_p, sol6 = la.chain_price(txs, EXIT_S), path[-1][6]
            peak = max(t[6] for t in path)
        rugged = peak > 0 and sol6 < la.PULLED_SHARE * peak
        ret = -1.0 if rugged or not entry or not exit_p else min(exit_p / entry - 1 - la.COST, 3.0)
        feat = live_features(rec, create, la._prior_pools(store, rec.get("creator"), create))
        out.append({"pool": r["pool"], "created_ts": create, "listed": r["pool"] in listed, "liquid": liquid,
                    "creator": rec.get("creator"), "pulled_ts": rec.get("pulled_ts"),
                    "liq_usd": 2 * sol4 * sol_usd if sol_usd else None,
                    "feat": {DECIDE_S: feat}, "ret": {FILL_S: ret}})
    return out, dict(dropped)


def _split(rs, rng) -> dict:
    return {"all": la._summary([r["ret"][FILL_S] for r in rs], rng),
            "listed on GeckoTerminal": la._summary([r["ret"][FILL_S] for r in rs if r["listed"]], rng),
            "never listed": la._summary([r["ret"][FILL_S] for r in rs if not r["listed"]], rng)}


def _shifted(train: list[dict], test_rows: list[dict], drop: tuple = (), top: int = 3) -> dict:
    """The features whose test-day mean is furthest from the training mean, in training standard deviations."""
    out = {}
    for k in sorted({k for r in train for k in r["feat"][DECIDE_S]} - set(la.LIVE_DROP) - set(drop)):
        a = [r["feat"][DECIDE_S][k] for r in train if r["feat"][DECIDE_S].get(k) is not None]
        b = [r["feat"][DECIDE_S][k] for r in test_rows if r["feat"][DECIDE_S].get(k) is not None]
        if len(a) > 1 and b:
            out[k] = (float(np.mean(b)) - float(np.mean(a))) / (float(np.std(a)) + 1e-9)
    best = sorted(out, key=lambda k: -abs(out[k]))[:top]
    return {k: round(out[k], 2) if abs(out[k]) < 1e6 else "constant in training" for k in best}


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
            "top 2%": _split(pick(cut2), rng), "top 5%": _split(pick(cut5), rng),
            # the live model was trained on these days (listed pools): one good day can carry the whole mean
            "by day": {d: {f"top {k}%": _day_mean([r for r in pick(c) if la._day(r) == d])
                           for k, c in ((2, cut2), (5, cut5))}
                       for d in sorted({la._day(r) for r in liquid})}}
    except (OSError, ValueError, KeyError):
        out["live model (launch4.json; in-sample for listed pools it was trained on)"] = None
    out["walk-forward on these rows (train on earlier days, no names; picks = top k% of each day's scores)"] = \
        _walk(liquid, rng, tuple(la.LIVE_DROP))
    # Hour and weekend can only learn day effects from a few days (a near-constant weekend flag broke 11-02).
    out["walk-forward without hour/weekend features"] = _walk(liquid, rng, tuple(la.LIVE_DROP) + CALENDAR)
    # 11-04: the live launch4.json (trained only on GeckoTerminal-listed pools, i.e. ones that survived ~70 s)
    # beat the all-pools history model on the same live pools. Does training on survivors cause it? Same test
    # pools (every liquid pool), only the training rows differ.
    out["survivors test: trained on listed pools only (as launch4.json), tested on every pool"] = {
        "with hour/weekend (like launch4.json)": _walk(liquid, rng, tuple(la.LIVE_DROP), lambda r: r["listed"]),
        "without hour/weekend": _walk(liquid, rng, tuple(la.LIVE_DROP) + CALENDAR, lambda r: r["listed"])}
    return out


CALENDAR = ("hour_sin", "hour_cos", "weekend")


def _day_mean(rows: list[dict]) -> dict:
    return {"n": len(rows), "mean_pct": round(100 * float(np.mean([r["ret"][FILL_S] for r in rows])), 1) if rows else None,
            "rugged": sum(r["ret"][FILL_S] <= -0.8 for r in rows)}


def _walk(liquid: list[dict], rng, drop: tuple, train_filter=None) -> dict:
    """Walk-forward by day: train on earlier days (only rows passing `train_filter`, if given), pick the top k%
    of each test day's own scores among every liquid pool."""
    from tradingbot.model_book import decile_table
    picks = {k: [] for k in la.TIMING_TOP_PCT}
    base, by_day, day_scores = [], {}, []
    for d in sorted({la._day(r) for r in liquid}):
        train = [r for r in liquid if la._day(r) < d and (train_filter is None or train_filter(r))]
        test_rows = [r for r in liquid if la._day(r) == d]
        if len(train) < la.RANK_MIN_TRAIN or not test_rows:
            continue
        fit = la._fit_score(train, test_rows, DECIDE_S, FILL_S, drop)
        if fit is None:
            continue
        tr, te = fit
        base += test_rows
        day_scores.append([(float(s), r["ret"][FILL_S]) for r, s in zip(test_rows, te)])
        row = {"pools": len(test_rows), "all_mean_pct": round(100 * float(np.mean([r["ret"][FILL_S] for r in test_rows])), 1),
               # A cut-off from the training scores picked 99% of 11-02's pools: some feature shifts day to day.
               # Picks are the top k% of the day's own scores; this shows how far the training cut-off was off.
               "share above the training top-2% cut (%)": round(100 * float(np.mean(
                   te >= np.quantile(tr, 0.98) - 1e-9)), 1),
               "most shifted features (z vs training)": _shifted(train, test_rows, drop)}
        for k in la.TIMING_TOP_PCT:
            cut = float(np.quantile(te, 1 - k / 100))
            chosen = [r for r, s in zip(test_rows, te) if s >= cut - 1e-9]
            picks[k] += chosen
            row[f"top {k:g}%"] = {"n": len(chosen), "mean_pct": round(100 * float(np.mean(
                [r["ret"][FILL_S] for r in chosen])), 1) if chosen else None,
                "rugged": sum(r["ret"][FILL_S] <= -0.8 for r in chosen)}
        by_day[d] = row
    return {"days_scored": len(by_day), "every liquid pool (test days)": _split(base, rng),
            **{f"top {k:g}%": _split(picks[k], rng) for k in la.TIMING_TOP_PCT},
            "by score decile (ranked within each day; emptied = rugged)": decile_table(day_scores, rng),
            "by day": by_day,
            **({} if by_day else {"note": f"needs {la.RANK_MIN_TRAIN}+ liquid pools on earlier days"})}
