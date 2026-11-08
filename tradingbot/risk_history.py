"""Supermeme books replayed over the pool history (`risk-model --history`, 11-08, read only).

Pools: the year backfill (`hist.db`, `launch_hist.rows`: chain-only second-4 features, second-6 entry, minute-6 exit,
LP pull time) joined with the live scorer's finished liquid pools (`watch_launch_live`, preferred where both exist:
its features and vault outcome are what live used). A creator's earlier pools are recounted over the whole set
(the backfill's own `prior_pools` came from watch_creators, which only covers 10-14 on), then the creator record
(`launch_live._creator_history`) and the rug-vs-win inputs (`pull_inputs`) are built as live builds them.

Walk-forward, as the live rule: a `RugWinModel` (rug before the exit / +50% win, two capped logistics) is fitted every
`retrain_h` hours on pools finished before then (at most `MAX_TRAIN` of them, a fixed random sample of all earlier
pools, to keep a year's replay under an hour); each pool is scored by the model of its time, `rw_pct` = its rank
among the past `RW_BLOCK_S` (6 h) pools' scores (training quantiles until ROLLING_MIN), `rwb_pct` = its rank among
the past 24 h's pools of its BAND_RULES band. Rules then pick as the books do (rw50: rw_pct < 50; rwband:
`band_flags`), trades are the ideal result minus each book's measured execution gap (real minus ideal on the same
tokens, from the journal when it has `GAP_MIN` matched trades, else `DEFAULT_GAP`), sized like the book (capped at
`model_book_liq_share` of the pool), and go through `risk_model` (stats, trade-size bootstrap) plus a monthly table.
The live books' own rw_pct is compared with the replay's on the pools both scored (`check`)."""
from __future__ import annotations
import collections
import json
import math
import time
from bisect import bisect_left
from datetime import datetime, timezone

import numpy as np

RETRAIN_H = 24
MAX_TRAIN = 60_000
FIT_ITERS = 1500        # the first fit
WARM_ITERS = 250        # later fits start from the previous day's weights (same inputs, a little more data)
DEFAULT_GAP = 0.024     # 11-06 fill-gap study: launchrw50 real minus ideal, mean per trade
GAP_MIN = 30
BAND_WINDOW_S = 86400
RULES = (("every liquid pool", None, "flat"), ("launchrw50", "rw50", "flat"), ("launchrw50_sz", "rw50", "launchrw50_sz"),
         ("launchrwband", "rwband", "flat"), ("launchrwband_sz", "rwband", "launchrwband_sz"))


def load_pools(hist_store, watch_store, cfg) -> tuple[list[dict], dict]:
    from tradingbot import launch_hist as lh
    from tradingbot import launch_live as ll
    pools: dict = {}
    info = {"history rows": 0, "scorer rows": 0}
    if hist_store is not None:
        hist, dropped = lh.rows(hist_store, cfg)
        info["history dropped"] = dropped
        for r in hist:
            pools[r["pool"]] = {"pool": r["pool"], "create_ts": r["created_ts"], "creator": r.get("creator"),
                                "pulled_ts": r.get("pulled_ts"), "liquid": r["liquid"], "liq_usd": r["liq_usd"],
                                "feat": r["feat"][lh.DECIDE_S], "ret": r["ret"][lh.FILL_S], "src": "history"}
            info["history rows"] += 1
    if watch_store is not None:
        ll.ensure_schema(watch_store)
        for r in watch_store.query(
                "SELECT pool, creator, create_ts, feat, ret, pulled_ts, liq_usd, rw_pct FROM watch_launch_live "
                "WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND feat IS NOT NULL"):
            pools[r["pool"]] = {"pool": r["pool"], "create_ts": r["create_ts"], "creator": r["creator"],
                                "pulled_ts": r["pulled_ts"], "liquid": True, "liq_usd": r["liq_usd"],
                                "feat": json.loads(r["feat"]), "ret": r["ret"], "src": "scorer", "live_rw_pct": r["rw_pct"]}
            info["scorer rows"] += 1
    rows = sorted(pools.values(), key=lambda r: r["create_ts"])
    # a creator's earlier pools over the whole set (liquid or not), as live counts every earlier pool it knows
    seen: collections.Counter = collections.Counter()
    for r in rows:
        n = seen[r["creator"]] if r["creator"] else 0
        r["feat"] = {**r["feat"], "repeat_creator": float(n > 0), "log_prior_pools": math.log1p(n)}
        if r["creator"]:
            seen[r["creator"]] += 1
    op = ll._creator_history(rows)
    liquid = [r for r in rows if r["liquid"] and r["ret"] is not None]
    for r in liquid:
        r["x"] = ll.pull_inputs(r["feat"], op[r["pool"]])
    for r in rows:
        r.pop("feat", None)  # memory: a year of pools
    info["liquid pools"] = len(liquid)
    info["from"] = _fmt(liquid[0]["create_ts"]) if liquid else None
    info["to"] = _fmt(liquid[-1]["create_ts"]) if liquid else None
    return liquid, info


def _fmt(t) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _logistic(z: np.ndarray, y: np.ndarray, w0=None, b0=None, iters: int = FIT_ITERS, l2: float = 1.0, lr: float = 0.1):
    """watch_analysis.logistic (L2, gradient descent on standardised inputs), optionally warm-started."""
    w = np.zeros(z.shape[1]) if w0 is None else w0.copy()
    b = math.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)) if b0 is None else b0
    n = len(y)
    zt = np.ascontiguousarray(z.T)
    for _ in range(iters):
        g = 1 / (1 + np.exp(-(z @ w + b))) - y
        w -= lr * (zt @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


class _Fit:
    """A capped logistic on columns of the shared matrix (median-filled, standardised, clipped at CLIP_Z)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, clip: float, warm: "_Fit | None" = None):
        self.med = np.nanmedian(x, 0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        z = np.where(np.isfinite(x), x, self.med)
        self.mu, self.sd = z.mean(0), z.std(0) + 1e-9
        self.clip = clip
        self.w, self.b = _logistic(self._z(x), y, warm.w if warm is not None else None,
                                   warm.b if warm is not None else None, WARM_ITERS if warm is not None else FIT_ITERS)

    def _z(self, x):
        return np.clip((np.where(np.isfinite(x), x, self.med) - self.mu) / self.sd, -self.clip, self.clip)

    def prob(self, x):
        return 1 / (1 + np.exp(-np.clip(self._z(x) @ self.w + self.b, -50, 50)))


def replay(rows: list[dict], retrain_h: float = RETRAIN_H, max_train: int = MAX_TRAIN, seed: int = 7) -> dict:
    """Score every pool walk-forward and rank it as live does; sets r['rw'], r['rw_pct'], r['rwb_pct']."""
    from tradingbot import launch_analysis as la
    from tradingbot import launch_live as ll
    names = sorted({k for r in rows for k in r["x"]})
    X = np.array([[r["x"].get(n, np.nan) if r["x"].get(n) is not None else np.nan for n in names] for r in rows],
                 dtype=float)
    ret = np.array([r["ret"] for r in rows], dtype=float)
    done_at = np.array([r["create_ts"] + ll.EXIT_S for r in rows])
    created = np.array([r["create_ts"] for r in rows])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))  # fixed random order: the training sample is the first max_train earlier pools
    rank = np.empty(len(rows), dtype=int); rank[order] = np.arange(len(rows))
    step = retrain_h * 3600
    t = created[0] - created[0] % step
    fits = 0
    scores = np.full(len(rows), np.nan)
    fallback = np.full(len(rows), -1, dtype=int)  # which fit's training scores rank a pool when the 6 h window is thin
    fit_scores: list = []
    prev = (None, None)
    while t <= created[-1]:
        block = np.where((created >= t) & (created < t + step))[0]
        if len(block):
            train = np.where(done_at <= t)[0]
            if len(train) > max_train:
                train = train[np.argsort(rank[train])[:max_train]]
            yr, yb = (ret[train] <= -0.999).astype(float), (ret[train] >= ll.RW_BIG).astype(float)
            if len(train) >= ll.PULL_MIN_TRAIN and 0 < yr.mean() < 1 and 0 < yb.mean() < 1:
                rug, big = _Fit(X[train], yr, la.CLIP_Z, prev[0]), _Fit(X[train], yb, la.CLIP_Z, prev[1])
                prev = (rug, big)
                fit_scores.append(np.sort(rug.prob(X[train]) - big.prob(X[train])))
                fallback[block] = len(fit_scores) - 1
                scores[block] = rug.prob(X[block]) - big.prob(X[block])
                fits += 1
        t += step
    # rank against the past 6 h (all bands) and the past 24 h of the same band, like the live scorer
    past_t, past_s = [], []
    thin = 0
    band_t: dict = collections.defaultdict(list); band_s: dict = collections.defaultdict(list)
    for i, r in enumerate(rows):
        s = scores[i]
        r["rw"] = None if np.isnan(s) else float(s)
        if r["rw"] is None:
            r["rw_pct"] = r["rwb_pct"] = None
            continue
        lo = bisect_left(past_t, r["create_ts"] - ll.RW_BLOCK_S)
        window = past_s[lo:]
        if len(window) >= ll.ROLLING_MIN:
            r["rw_pct"] = 100.0 * float(np.mean(np.array(window) < r["rw"]))
        else:  # thin hours: rank among the training pools of the model that scored it (ties fall below)
            thin += 1
            ts_ = fit_scores[fallback[i]]
            r["rw_pct"] = 100.0 * bisect_left(ts_, r["rw"]) / len(ts_)
        b = ll.band_of(r["liq_usd"])
        if b:
            lo = bisect_left(band_t[b], r["create_ts"] - BAND_WINDOW_S)
            w = band_s[b][lo:]
            r["rwb_pct"] = 100.0 * float(np.mean(np.array(w) < r["rw"])) if len(w) >= ll.ROLLING_MIN else None
            band_t[b].append(r["create_ts"]); band_s[b].append(r["rw"])
        else:
            r["rwb_pct"] = None
        past_t.append(r["create_ts"]); past_s.append(r["rw"])
    return {"fits": fits, "retrain_h": retrain_h, "max_train": max_train, "features": len(names),
            "scored": int(np.isfinite(scores).sum()), "ranked on training scores (thin 6 h windows)": thin}


def check(rows: list[dict]) -> dict:
    """Replayed vs live ranking on the pools the live scorer ranked (scorer rows with a live rw_pct)."""
    from tradingbot import launch_live as ll
    both = [r for r in rows if r.get("live_rw_pct") is not None and r.get("rw_pct") is not None]
    if len(both) < 30:
        return {"pools": len(both), "note": "too few pools ranked by both"}
    a = np.array([r["rw_pct"] for r in both]); b = np.array([r["live_rw_pct"] for r in both])
    keep = 100 - ll.RW_SKIP
    return {"pools": len(both), "rank correlation": round(float(np.corrcoef(a, b)[0, 1]), 2),
            "same rw50 decision %": round(100 * float(np.mean((a < keep) == (b < keep))), 1),
            "note": "the live model differs (retrained every 6 h on the scorer's pools only, ranked live), so expect "
                    "agreement well above 50% but not 100%"}


def gaps(journal, watch_store, now: float) -> dict:
    """Each book's real minus ideal per trade on the same tokens (from the journal), DEFAULT_GAP if too few."""
    from tradingbot import band_compare as bc
    out = {}
    if journal is None or watch_store is None:
        return {name: {"gap": DEFAULT_GAP, "source": "default"} for name, _, _ in RULES}
    scorer: dict = {}
    for r in watch_store.query("SELECT token_mint, create_ts, liq_usd, ret FROM watch_launch_live "
                               "WHERE token_mint IS NOT NULL ORDER BY create_ts"):
        scorer.setdefault(r["token_mint"], []).append(dict(r))
    for name, _, _ in RULES:
        if name == "every liquid pool":
            out[name] = {"gap": DEFAULT_GAP, "source": "default (no book buys every pool)"}
            continue
        m = [x for x in bc._trades(watch_store, journal, name, 0.0, now, scorer) if x["ideal"] is not None]
        if len(m) >= GAP_MIN:
            out[name] = {"gap": round(-float(np.mean([x["ret"] - x["ideal"] for x in m])), 4),
                         "source": f"journal ({len(m)} trades)"}
        else:
            out[name] = {"gap": DEFAULT_GAP, "source": f"default ({len(m)} matched trades)"}
    return out


def rule_trades(rows: list[dict], flag, sizing: str, gap: float, liq_share: float) -> list[dict]:
    from tradingbot import launch_live as ll
    from tradingbot import model_book as mb
    out = []
    for r in rows:
        if flag == "rw50":
            take = r.get("rw_pct") is not None and r["rw_pct"] < 100 - ll.RW_SKIP
        elif flag == "rwband":
            take = r.get("rw") is not None and ll.band_flags(r["liq_usd"], r["rw_pct"], r["rwb_pct"])["rwband"]
        else:
            take = r.get("rw") is not None   # every pool the replay could score (same span as the rules)
        if not take:
            continue
        size = (mb.band_usd(sizing, r["liq_usd"]) if sizing != "flat" else None) or 100.0
        if r["liq_usd"]:
            size = min(size, liq_share * r["liq_usd"])
        if size < 5:
            continue
        out.append({"ret": max(-1.0, r["ret"] - gap), "cost": size, "ts": r["create_ts"] + ll.EXIT_S,
                    "opened": r["create_ts"] + 10, "liq": r["liq_usd"]})
    return out


def monthly(trades: list[dict]) -> dict:
    by: dict = collections.defaultdict(list)
    for t in trades:
        by[datetime.fromtimestamp(t["ts"], timezone.utc).strftime("%Y-%m")].append(t)
    out = {}
    for m, ts in sorted(by.items()):
        r = np.array([t["ret"] for t in ts]); c = np.array([t["cost"] for t in ts])
        out[m] = {"trades": len(ts), "pnl_usd": round(float((r * c).sum())),
                  "sized_pct": round(100 * float((r * c).sum() / c.sum()), 1), "rugged_pct": round(100 * float(np.mean(r <= -0.8)))}
    return out


def model(hist_store, watch_store, journal, cfg, retrain_h: float = RETRAIN_H, max_train: int = MAX_TRAIN,
          now: float | None = None, sims: int | None = None) -> dict:
    from tradingbot import risk_model as rm
    now = now if now is not None else time.time()
    t0 = time.time()
    rows, info = load_pools(hist_store, watch_store, cfg)
    if len(rows) < 500:
        return {"pools": info, "note": "fewer than 500 liquid pools: run history-backfill first"}
    meta = replay(rows, retrain_h, max_train)
    g = gaps(journal, watch_store, now)
    capital, rng = cfg.model_book_cash_usd, np.random.default_rng(0)
    books, months, corr_src = {}, {}, {}
    for name, flag, sizing in RULES:
        ts = sorted(rule_trades(rows, flag, sizing, g[name]["gap"], cfg.model_book_liq_share), key=lambda t: t["ts"])
        if len(ts) < rm.MIN_TRADES:
            continue
        st = rm.book_stats(ts, capital, rng)
        pnl, capped = rm.day_tables(ts, cfg.model_book_liq_share)
        sizing_rows = [rm.simulate(pnl, capped, c, f, np.random.default_rng(1), sims=sims or rm.SIMS)
                       for c in rm.CAPITALS for f in rm.FRACTIONS]
        st["actual_trade_usd_median"] = round(float(np.median([t["cost"] for t in ts])), 2)
        st["actual_pct_of_capital"] = round(100 * st["actual_trade_usd_median"] / capital, 2)
        st["with_liquidity_pct"] = round(100 * sum(1 for t in ts if t["liq"]) / len(ts))
        for r in sizing_rows:
            r["cash_short"] = r["pct_per_trade"] * st["peak_open"] > 100
        books[name] = {"stats": st, "sizing": sizing_rows, "best": rm._best(sizing_rows)}
        months[name] = monthly(ts)
        corr_src[name] = ts
    corr = portfolio = None
    names = [n for n in corr_src if n != "every liquid pool"]
    if len(names) >= 2:
        first = min(rm._day(corr_src[n][0]["ts"]) for n in names); last = max(rm._day(corr_src[n][-1]["ts"]) for n in names)
        series = {n: rm._daily(corr_src[n], first, last) / capital for n in names}
        m = np.corrcoef(np.array([series[n] for n in names]))
        corr = {a: {b: round(float(m[i, j]), 2) for j, b in enumerate(names)} for i, a in enumerate(names)}
        port = np.mean([series[n] for n in names], axis=0) * capital
        portfolio = {"books": len(names), **rm._curve_stats(port, capital),
                     "mean single-book sharpe": round(float(np.mean([books[n]["stats"]["sharpe"] or 0 for n in names])), 2)}
    return {"window": f"history {info['from']} .. {info['to']}", "built": _fmt(now), "capital": capital,
            "liq_share": cfg.model_book_liq_share, "pools": info, "replay": meta, "check": check(rows), "gaps": g,
            "books": books, "monthly": months, "too few trades": {}, "correlation": corr,
            "equal_weight_portfolio": portfolio, "took_s": round(time.time() - t0),
            "note": ("Replayed books: walk-forward rug-vs-win model retrained every "
                     f"{retrain_h:g} h on at most {max_train:,} earlier finished pools, ranked like live (6 h; band "
                     "24 h); trades = ideal result (second ~6 in, minute 6 out, 6% cost, +300% cap, rug -100%) minus "
                     "each book's execution gap, sized like the book and capped at the pool share. History pools: "
                     "a creator's earlier pulls are seen from the LP pull times the backfill records. "
                     "Then the same risk measures and trade-size bootstrap as risk-model.")}


def render(out: dict) -> str:
    from tradingbot import risk_model as rm
    if "books" not in out:
        return json.dumps(out, indent=2, default=str)
    p, r, c = out["pools"], out["replay"], out["check"]
    L = [f"SUPERMEME HISTORY REPLAY ({out['window']} UTC; built in {out['took_s']} s)",
         f"  pools: {p['liquid pools']:,} liquid ({p['history rows']:,} from the backfill, {p['scorer rows']:,} from the live "
         f"scorer); model fits {r['fits']} (every {r['retrain_h']:g} h, <= {r['max_train']:,} pools), scored {r['scored']:,} "
         f"({r['ranked on training scores (thin 6 h windows)']:,} ranked on training scores: fewer than 50 pools in their 6 h)",
         "  replay vs live ranking: " + (f"{c['pools']:,} pools, rank correlation {c['rank correlation']}, same rw50 decision "
                                        f"{c['same rw50 decision %']}%" if "rank correlation" in c else c.get("note", "")),
         "  execution gap per trade: " + ", ".join(f"{k} {100 * v['gap']:.1f} pts ({v['source']})" for k, v in out["gaps"].items()),
         "", "BY MONTH (size-weighted % per trade / $ at the book's sizes / rugged %)"]
    names = list(out["monthly"])
    months = sorted({m for n in names for m in out["monthly"][n]})
    L.append(f"  {'month':8}" + "".join(f"{n[:18]:>26}" for n in names))
    for m in months:
        cell = lambda d: "-" if not d else f"{d['sized_pct']:+.1f}% {d['pnl_usd']:>+8,} {d['rugged_pct']:>2}%"
        L.append(f"  {m:8}" + "".join(f"{cell(out['monthly'][n].get(m)):>26}" for n in names))
    L += ["", rm.render(out)]
    return "\n".join(L)
