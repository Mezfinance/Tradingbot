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
import math
import time

import numpy as np

from tradingbot import watch_analysis as wa
from tradingbot.config import Config

COST = 0.06
ENTRY_SECONDS = (5, 10, 20, 30, 45, 60, 90)
TIMING = ((4, 4), (4, 5), (4, 6), (4, 7), (5, 5), (5, 6), (5, 7))  # (decide on features at second d, fill at second f)
TIMING_TOP_PCT = (2.0, 5.0)
# What the live scorer can't know at second 4: the token's name (GeckoTerminal lists a pool ~70 s after creation).
LIVE_DROP = ("name_len", "name_has_digit", "name_all_caps", "name_words")
FEATURE_SECONDS = sorted(set(ENTRY_SECONDS) | {d for d, f in TIMING} | {f for d, f in TIMING})
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


def candle_price(candles: list, t: float) -> float | None:
    """Close of the last 1-minute candle that started before t (GeckoTerminal lists only minutes with trades, so
    a quiet stretch keeps the last price). The close lands within a minute of t either side."""
    p = None
    for c in candles:
        if c[0] >= t:
            break
        p = c[4]
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


def launch_features(txs: list, x: float, creator: str | None, prior_pools: int, seed_sol: float | None) -> dict:
    """What the chain shows by second x: seed and current SOL, trades, traders, buy/sell flow, the creator's own
    buys, bundling in the creation slot, price move since creation, and whether the creator has made pools before."""
    upto = [r for r in txs if r[0] <= x]
    if not upto:
        return {}
    seed_row = next((r for r in upto if r[4] > 0 and r[6] == r[4]), None)  # the creator funding the empty pool
    trades = [r for r in upto if r is not seed_row]
    buys = [r for r in trades if r[4] > 0]  # SOL into the pool = someone bought tokens
    sells = [r for r in trades if r[4] < 0]
    first_slot = upto[0][1]
    p0, px = chain_price(txs, 0), chain_price(txs, x)
    sol_x = upto[-1][6]
    seed = seed_sol or upto[0][6] or 0.0
    buy_sol, sell_sol = sum(r[4] for r in buys), -sum(r[4] for r in sells)
    return {
        "log_seed_sol": math.log1p(seed), "log_sol_now": math.log1p(max(sol_x, 0.0)),
        "sol_growth": (sol_x / seed) if seed > 0 else 1.0,
        "log_txs": math.log1p(len(trades)), "log_buys": math.log1p(len(buys)), "log_sells": math.log1p(len(sells)),
        "buy_share": len(buys) / (len(buys) + len(sells)) if buys or sells else 0.5,
        "log_buy_sol": math.log1p(buy_sol), "log_sell_sol": math.log1p(sell_sol),
        "net_flow_share": (buy_sol - sell_sol) / seed if seed > 0 else 0.0,
        "log_traders": math.log1p(len({r[2] for r in trades})),
        "log_buyers": math.log1p(len({r[2] for r in buys})),
        "creator_buys": float(sum(1 for r in buys if r[2] == creator)),
        "first_slot_txs": float(sum(1 for r in upto if r[1] == first_slot)),
        "log_price_move": math.log(px / p0) if p0 and px else 0.0,
        "repeat_creator": float(prior_pools > 0), "log_prior_pools": math.log1p(prior_pools),
    }


def _sol_usd(store, ts: float) -> float | None:
    r = store.query("SELECT sol_usd FROM watch_market WHERE ts <= ? AND sol_usd > 0 ORDER BY ts DESC LIMIT 1", (ts,))
    return float(r[0]["sol_usd"]) if r else None


def _prior_pools(store, creator: str | None, created_ts: float) -> int:
    if not creator:
        return 0
    r = store.query("SELECT COUNT(*) AS n FROM watch_creators c JOIN watch_pools p ON p.pool=c.pool "
                    "WHERE c.creator=? AND p.created_ts < ?", (creator, created_ts))
    return int(r[0]["n"]) if r else 0


def rows(store, cfg: Config, since: float = 0.0) -> list[dict]:
    out = []
    for r in store.query(
            """SELECT p.pool, p.mint, p.dex, p.name, p.created_ts, p.seen_ts, p.first, e.data AS launch,
                      s.data AS snap5, s.ts AS snap_ts, COALESCE(h.data, c.data) AS candles
               FROM watch_enrich e JOIN watch_pools p ON p.pool=e.pool
               LEFT JOIN watch_snaps s ON s.pool=p.pool AND s.age_min=5
               LEFT JOIN watch_enrich c ON c.pool=p.pool AND c.kind='candles15'
               LEFT JOIN watch_enrich h ON h.pool=p.pool AND h.kind='candles_hist'
               WHERE e.kind='launch' AND p.created_ts >= ? AND json_extract(e.data, '$.txs') IS NOT NULL""",
            (since,)):
        launch, first = json.loads(r["launch"]), json.loads(r["first"] or "{}")
        snap = json.loads(r["snap5"]) if r["snap5"] else None
        txs = launch["txs"]
        seen = r["seen_ts"] - launch["create_ts"]
        row = {"pool": r["pool"], "seen_s": seen, "first_liq": wa._f(first.get("liq")) or 0.0,
               "sample": launch.get("selection", "sample") == "sample", "created_ts": r["created_ts"]}
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
            sol_usd = _sol_usd(store, r["created_ts"])
            prior = _prior_pools(store, launch.get("creator"), r["created_ts"])
            static = wa._static_features({"name": r["name"], "dex": r["dex"], "created_ts": r["created_ts"]})
            row["liquid"], row["feat"] = {}, {}
            create = launch["create_ts"]
            row.update(chain_create=create, gone=gone, pulled_before_seen=pulled_by(txs, seen))
            row["snap_min"] = (r["snap_ts"] - create) / 60 if r["snap_ts"] else None
            candles = sorted((json.loads(r["candles"]) or {}).get("candles") or [], key=lambda c: c[0]) \
                if r["candles"] else []
            c_seen = candle_price(candles, r["seen_ts"])
            if c_seen and p_seen:  # exit-timing check: the same trade sold at exactly pool minute 6
                row["cx"] = {"c_seen": c_seen, "c6": candle_price(candles, create + 360),
                             "c_snap": candle_price(candles, r["snap_ts"]), "gone": gone,
                             "pulled_before_seen": pulled_by(txs, seen), "link": {}}
            for x in FEATURE_SECONDS:
                if x >= seen:
                    continue
                p_x = chain_price(txs, x)
                if p_x is None or p_seen is None:
                    continue
                if pulled_by(txs, seen):
                    row["ret"][x] = -1.0
                else:
                    row["ret"][x] = -1.0 if gone else min((1 + seen_ret + COST) * p_seen / p_x - 1 - COST, 3.0)
                # what a second-x buyer can see: the pool's own SOL then (both sides ~ 2 x SOL side), not first sight
                sol_x = [t[6] for t in txs if t[0] <= x][-1]
                row["liquid"][x] = (sol_usd is None or 2 * sol_x * sol_usd >= cfg.model_book_min_liq_usd) \
                    and not pulled_by(txs, x)
                row["feat"][x] = {**static, **launch_features(txs, x, launch.get("creator"), prior,
                                                               launch.get("seed_sol"))}
                if "cx" in row:
                    row["cx"]["link"][x] = p_seen / p_x
                if x == EXIT_CHECK_CELL[1]:
                    row["link6"] = round(p_seen / p_x, 6)
        out.append(row)
    return out


RANK_SECONDS = (5, 10, 20, 30)
RANK_TOP_PCT = (2.0, 5.0, 10.0)
RANK_MIN_TRAIN = 100  # pools a day's model must be trained on before it is scored


def _day(r) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(r["created_ts"]))


def _fit_score(train: list[dict], test: list[dict], feat_x: int, label_x: int, drop: tuple = ()):
    """Logistic on features at second feat_x, label: the trade filled at second label_x made money. Returns
    (training scores, test scores) or None when the training labels are all one class."""
    names = sorted({k for r in train for k in r["feat"][feat_x]} - set(drop))
    xs, med = wa._matrix([r["feat"][feat_x] for r in train], names)
    y = np.array([float(r["ret"][label_x] > 0) for r in train])
    if y.min() == y.max():
        return None
    mu, sd = xs.mean(0), xs.std(0) + 1e-9
    w, b = wa.logistic((xs - mu) / sd, y)
    te = (wa._matrix([r["feat"][feat_x] for r in test], names, med)[0] - mu) / sd @ w + b
    return ((xs - mu) / sd) @ w + b, te


EXIT_CHECK_CELL = (4, 6)  # the live rule: decide on second-4 features, fill at second 6


def candle_ret(r: dict, x: int, at: str) -> float | None:
    """The second-x trade sold at the 1-minute candle price `at` ('c6' = pool minute 6, 'c_snap' = when the
    backtest's snapshot was taken), linked to the entry by the chain price; emptied pools stay -100%."""
    cx = r.get("cx")
    if not cx or x not in cx["link"] or not cx.get(at):
        return None
    if cx["pulled_before_seen"] or cx["gone"]:
        return -1.0
    return min(cx[at] / cx["c_seen"] * cx["link"][x] - 1 - COST, 3.0)


def exit_check(rows: list[dict], f: int, rng) -> dict:
    """Does the backtest's exit time matter? The snapshot exit vs 1-minute candle prices at the snapshot's own time
    (same moment, other price source) and at exactly pool minute 6, on the pools that have candles; and the
    snapshot exit of pools without candles (if it differs, the candle subset is not representative)."""
    have = [r for r in rows if candle_ret(r, f, "c6") is not None and candle_ret(r, f, "c_snap") is not None]
    rest = [r for r in rows if r not in have]
    mins = [r["snap_min"] for r in rows if r.get("snap_min") is not None]
    kept = [r for r in have if not (r["cx"]["gone"] or r["cx"]["pulled_before_seen"])]
    return {
        "snapshot taken at pool minute": {
            "median": round(float(np.median(mins)), 2) if mins else None,
            "p10": round(float(np.percentile(mins, 10)), 2) if mins else None,
            "p90": round(float(np.percentile(mins, 90)), 2) if mins else None},
        "pools with candles": f"{len(have)} of {len(rows)}",
        "snapshot exit (backtest), pools with candles": _summary([r["ret"][f] for r in have], rng),
        "snapshot exit (backtest), pools without candles": _summary([r["ret"][f] for r in rest], rng),
        "candle price at the snapshot time": _summary([candle_ret(r, f, "c_snap") for r in have], rng),
        "candle price at pool minute 6": _summary([candle_ret(r, f, "c6") for r in have], rng),
        "not emptied: candle at snapshot time": _summary([candle_ret(r, f, "c_snap") for r in kept], rng),
        "not emptied: candle at minute 6": _summary([candle_ret(r, f, "c6") for r in kept], rng),
    }


HIST_CANDLE_WINDOW_S = 900  # historical 1-minute candles: the pool's first 15 minutes


def fetch_history_candles(store, pools: list[dict], gecko, progress=None) -> int:
    """GeckoTerminal's 1-minute candles of each pool's first 15 minutes, fetched after the fact (it serves them days
    later) and saved as kind `candles_hist`; pools already fetched are skipped, failed calls are left for a rerun."""
    done = {r[0] for r in store.query("SELECT pool FROM watch_enrich WHERE kind='candles_hist'")}
    fetched = 0
    for i, r in enumerate(pools):
        if r["pool"] in done:
            continue
        rows = gecko.ohlcv(r["pool"], minutes=1, limit=20, before=int(r["created_ts"] + HIST_CANDLE_WINDOW_S))
        if rows is None:
            continue
        store.execute("INSERT INTO watch_enrich(pool, kind, ts, data) VALUES(?,?,?,?)",
                      (r["pool"], "candles_hist", time.time(), json.dumps({"candles": rows})))
        done.add(r["pool"])
        fetched += 1
        if progress:
            progress(i + 1, len(pools))
    return fetched


def export_exit_rows(data: list[dict], test_from: str | None, no_name: bool, sample_n: int) -> dict:
    """What the exit timing check needs, for checking elsewhere (GeckoTerminal's rate limit is shared with the
    running bot on Fly): the decide-4/fill-6 picks and a random sample of the other liquid pools, each with its
    times, the backtest's snapshot return, the chain link from second 6 to first sight and the emptied flags."""
    got: dict = {}
    timing(data, np.random.default_rng(0), test_from, no_name, collect=got)
    d, f = EXIT_CHECK_CELL
    cut_of = {}
    for k in sorted(TIMING_TOP_PCT, reverse=True):
        for r in got.get("by_cut", {}).get(k, []):
            cut_of[r["pool"]] = k
    rest = [r for r in got.get("base", []) if r["pool"] not in cut_of]
    sample = [rest[i] for i in np.random.default_rng(1).permutation(len(rest))[:sample_n]]
    seen_s = lambda r: r["seen_s"]
    out = []
    for r in [r for r in got.get("base", []) if r["pool"] in cut_of] + sample:
        create = r["created_ts"]
        out.append({"pool": r["pool"], "top": cut_of.get(r["pool"]), "created": create,
                    "chain_create": r.get("chain_create"), "seen_s": round(seen_s(r), 1),
                    "snap_min": round(r["snap_min"], 3) if r.get("snap_min") else None,
                    "ret": round(r["ret"][f], 5), "link": r.get("link6"), "gone": r.get("gone"),
                    "pulled_before_seen": r.get("pulled_before_seen"), "day": _day(r)})
    return {"cell": f"decide {d}s, fill {f}s", "rows": out}


def timing(data: list[dict], rng, test_from: str | None = None, no_name: bool = False,
           collect: dict | None = None) -> dict:
    """Live entries can't fill at the decision second. For each (decide at d, fill at f): walk-forward by day, a
    model on features at second d (label: the second-f trade made money) picks the top 2% / 5% of the training
    scores; each pick earns its second-f return. Pools must be liquid and not pulled at d. Days before
    `test_from` (YYYY-MM-DD) only train. Per-day results show whether a cell rests on one or two days."""
    out = {}
    for d, f in TIMING:
        pool = sorted([r for r in data if r.get("sample") and d in r.get("feat", {}) and f in r.get("ret", {})
                       and r["liquid"].get(d)], key=lambda r: r["created_ts"])
        days = sorted({_day(r) for r in pool})
        picks = {k: [] for k in TIMING_TOP_PCT}
        pick_rows = {k: [] for k in TIMING_TOP_PCT}
        by_day = {}
        base, base_rows = [], []
        for dd in days:
            if test_from and dd < test_from:
                continue
            train = [r for r in pool if _day(r) < dd]
            test = [r for r in pool if _day(r) == dd]
            if len(train) < RANK_MIN_TRAIN or not test:
                continue
            fit = _fit_score(train, test, d, f, LIVE_DROP if no_name else ())
            if fit is None:
                continue
            tr, te = fit
            base += [r["ret"][f] for r in test]
            base_rows += test
            row = {"pools": len(test), "all_mean_pct": round(float(np.mean([r["ret"][f] for r in test])) * 100, 1)}
            for k in TIMING_TOP_PCT:
                cut = float(np.quantile(tr, 1 - k / 100))
                chosen = [r for r, sc in zip(test, te) if sc >= cut - 1e-9]  # ties count
                got = [r["ret"][f] for r in chosen]
                picks[k] += got
                pick_rows[k] += chosen
                row[f"top {k:g}%"] = {"n": len(got), "mean_pct": round(float(np.mean(got)) * 100, 1) if got else None}
            by_day[dd] = row
        res = {"pools": len(pool), "days_scored": len(by_day), "every liquid pool": _summary(base, rng)}
        for k in TIMING_TOP_PCT:
            res[f"top {k:g}%"] = _summary(picks[k], rng)
        res["by day"] = by_day
        if (d, f) == EXIT_CHECK_CELL and collect is not None:
            collect.update(base=base_rows, picks=pick_rows[max(TIMING_TOP_PCT)], by_cut=pick_rows)
        if (d, f) == EXIT_CHECK_CELL:
            res["exit timing check"] = {"every liquid pool": exit_check(base_rows, f, rng),
                                        **{f"top {k:g}%": exit_check(pick_rows[k], f, rng) for k in TIMING_TOP_PCT}}
        out[f"decide {d}s, fill {f}s"] = res
    return out


def rank(data: list[dict], rng) -> dict:
    """Walk-forward by day: for each test day a logistic model trained on earlier days' sampled pools (liquid at
    second x by the chain's own numbers; label: the second-x trade made money) scores the day's pools; picks are
    those above the top-k% quantile of the training scores. Compared with buying every liquid pool that day."""
    day = lambda r: time.strftime("%m-%d", time.gmtime(r["created_ts"]))
    out = {}
    for x in RANK_SECONDS:
        pool = sorted([r for r in data if r.get("sample") and x in r.get("ret", {}) and r["liquid"].get(x)],
                      key=lambda r: r["created_ts"])
        days = sorted({day(r) for r in pool})
        picks = {k: [] for k in RANK_TOP_PCT}
        base, scored = [], 0
        for d in days:
            train = [r for r in pool if day(r) < d]
            test = [r for r in pool if day(r) == d]
            if len(train) < RANK_MIN_TRAIN or not test:
                continue
            names = sorted({k for r in train for k in r["feat"][x]})
            xs, med = wa._matrix([r["feat"][x] for r in train], names)
            y = np.array([float(r["ret"][x] > 0) for r in train])
            if y.min() == y.max():
                continue
            mu, sd = xs.mean(0), xs.std(0) + 1e-9
            w, b = wa.logistic((xs - mu) / sd, y)
            tr_scores = ((xs - mu) / sd) @ w + b
            te = (wa._matrix([r["feat"][x] for r in test], names, med)[0] - mu) / sd @ w + b
            base += [r["ret"][x] for r in test]
            scored += 1
            for k in RANK_TOP_PCT:
                cut = float(np.quantile(tr_scores, 1 - k / 100))
                picks[k] += [r["ret"][x] for r, s in zip(test, te) if s >= cut]
        res = {"pools": len(pool), "days_scored": scored,
               "every liquid pool (test days)": _summary(base, rng)}
        for k in RANK_TOP_PCT:
            res[f"top {k:g}%"] = _summary(picks[k], rng)
        if not base:
            res["note"] = f"needs {RANK_MIN_TRAIN}+ earlier sampled pools before a day can be scored"
        out[f"second {x}"] = res
    return out


def _summary(vals, rng) -> dict:
    a = np.array(vals)
    if len(a) == 0:
        return {"n": 0}
    lo, hi = np.percentile([rng.choice(a, len(a)).mean() for _ in range(1000)], [5, 95])
    return {"n": len(a), "mean_pct": round(a.mean() * 100, 1), "ci90": [round(lo * 100, 1), round(hi * 100, 1)],
            "median_pct": round(float(np.median(a)) * 100, 1), "rugged_pct": round(float(np.mean(a <= -0.99)) * 100)}


def backtest(store, cfg: Config, since: float = 0.0, picks: set[str] | None = None,
             test_from: str | None = None, only_timing: bool = False, no_name: bool = False,
             fetch_candles: int = 0, gecko=None, progress=None) -> dict:
    data = rows(store, cfg, since)
    if fetch_candles:  # candles for the live cell's top-5% picks (top 2% included) + a random sample of the rest
        got: dict = {}
        timing(data, np.random.default_rng(0), test_from, no_name, collect=got)
        pick_ids = {r["pool"] for r in got.get("picks", [])}
        rest = [r for r in got.get("base", []) if r["pool"] not in pick_ids]
        sample = [rest[i] for i in np.random.default_rng(1).permutation(len(rest))[:fetch_candles]]
        if gecko is None:
            from tradingbot.data.geckoterminal import GeckoTerminal
            gecko = GeckoTerminal()
        fetch_history_candles(store, got.get("picks", []) + sample, gecko, progress)
        data = rows(store, cfg, since)
    out = {"pools": len(data), "dropped": {}}
    for r in data:
        if "drop" in r:
            out["dropped"][r["drop"]] = out["dropped"].get(r["drop"], 0) + 1
    rng = np.random.default_rng(0)
    summary = lambda vals: _summary(vals, rng)
    if only_timing:
        out["timing (walk-forward: decide at d, fill at f)"] = timing(data, rng, test_from, no_name)
        return out

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
    # The fair comparison: sampled pools a second-x buyer would take (liquid by the chain's numbers at x, not yet
    # pulled), each pool's second-x return minus what the first-sight buyer got on it (0 when it skipped the pool
    # because it was already pulled or no longer liquid enough at first sight).
    fair = {}
    for x in ENTRY_SECONDS:
        rs = [r for r in data if r.get("sample") and x in r.get("ret", {}) and r["liquid"].get(x)]
        first = [(r["ret"]["first sight"] if r["ret"]["first sight"] is not None and r["eligible"] else 0.0)
                 for r in rs]
        fair[f"second {x}"] = {"second x": summary([r["ret"][x] for r in rs]),
                               "first sight on the same pools (0 = skipped)": summary(first),
                               "paired gain (second x - first sight)": summary(
                                   [r["ret"][x] - f for r, f in zip(rs, first)])}
    out["fair: liquid at second x"] = fair
    out["ranking (walk-forward, on-chain features at second x)"] = rank(data, rng)
    out["timing (walk-forward: decide at d, fill at f)"] = timing(data, rng, test_from, no_name)
    return out
