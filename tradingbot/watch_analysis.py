"""What predicts a pump? Statistics over the pump watcher's data (watcher.py).

One row per pool at a feature age A (minutes after the pool was created). A row holds only
what was knowable at A: the snapshot taken at A, earlier snapshots, token info fetched
before the snapshot, and the developer's track record from tokens whose 24h outcome had
already finished. The label is what happened AFTER the snapshot, over the following 24h,
across every pool of the same token (so a pump.fun launch that graduates and runs on
PumpSwap counts as a pump).

  pump2x / pump5x   best price after A reached 2x / 5x the price at A
  clean2x           reached 2x before falling 50% (a pump a stop-loss trader could catch)

Then:
  * univariate screen: each feature's single-variable AUC and pump rate by quintile;
  * L2-regularised logistic regression, trained on the earliest 70% of pools by time and
    scored on the latest 30% (only the held-out numbers count: with dozens of features,
    something always looks significant in-sample);
  * a trading check on the held-out pools: buy the model's top decile at the snapshot
    price, take profit at +100%, stop at -50%, else exit at 24h, after a round-trip cost.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import time
from collections import defaultdict

import numpy as np

from tradingbot.watch_features import Context, load_snaps
from tradingbot.watcher import LAUNCHPAD_DEXES, WatchStore

DAY_S = 86400
FEATURE_AGES = (5, 15, 30, 60)
TRAIN_SHARE = 0.7
ROUND_TRIP_COST = 0.06  # measured: a $50 simulated round trip on a real token cost 6.15%
TP_MULT, SL_MULT = 2.0, 0.5
MIN_ROWS = 200
REPORT_DAYS = 4  # finished days in the daily report: ~1.3 GB on a 2 GB machine (7 loaded days took 1.73 GB)
WARMUP_S = DAY_S  # history loaded before the first row, for look-back factors


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _log1p(x) -> float | None:
    v = _f(x)
    return math.log1p(max(v, 0.0)) if v is not None else None


def _ratio(a, b) -> float | None:
    a, b = _f(a), _f(b)
    if a is None or b is None or a + b <= 0:
        return None
    return a / (a + b)


# --- dataset -------------------------------------------------------------------------
def _future_paths(store: WatchStore, snaps: dict | None = None,
                  since: float | None = None) -> dict[str, list[tuple[float, float, float, float]]]:
    """mint -> [(ts, high, low, close)] from candles of every pool of that mint, plus snapshot prices.

    Only prices from a pool that was trading at the time count: a launch pool that has graduated keeps
    reporting its last (graduation) price, and mixing that stale price with the new pool's live one
    turned every post-graduation dump into a fake 2x.
    """
    paths: dict[str, list] = defaultdict(list)
    pools = store.query("SELECT pool, mint, created_ts, ohlcv IS NOT NULL AND ohlcv != '[]' AS has_c FROM watch_pools")
    mint_of = {r["pool"]: r["mint"] for r in pools}
    # Labels need every pool of a token, even one created before the window (an older pool of the same mint).
    mints = {r["mint"] for r in pools if r["created_ts"] >= (since or 0.0)}
    for r in pools:
        if not r["has_c"] or r["mint"] not in mints:
            continue
        for c in json.loads(store.query("SELECT ohlcv FROM watch_pools WHERE pool=?", (r["pool"],))[0]["ohlcv"]):
            if len(c) > 5 and not (_f(c[5]) or 0) > 0:
                continue  # no volume: no trade at this price
            ts, _o, h, low, close = (float(c[0]), c[1], c[2], c[3], c[4])
            paths[r["mint"]].append((ts + 300, float(h), float(low), float(close)))  # candle known at its close

    def add(pool, ts, d):
        price = _f(d.get("price"))
        if price and pool in mint_of and (d.get("buys_m5") or 0) + (d.get("sells_m5") or 0) > 0:
            paths[mint_of[pool]].append((ts, price, price, price))
    for pool, ss in (snaps if snaps is not None else load_snaps(store, since)).items():
        for ts, _age, d in ss:
            add(pool, ts, d)
    if since:  # snapshots of older pools of the same tokens, taken inside the window
        older = {r["pool"] for r in pools if r["created_ts"] < since and r["mint"] in mints}
        for r in store.iterate("SELECT s.pool, s.ts, s.data FROM watch_snaps s JOIN watch_pools p USING(pool) "
                               "WHERE p.created_ts < ? AND s.ts >= ? AND s.data IS NOT NULL AND s.data != '{}'",
                               (since, since)):
            if r["pool"] in older:
                add(r["pool"], r["ts"], json.loads(r["data"]))
    for v in paths.values():
        v.sort()
    return paths


FLAT_OUTCOME = {"best_mult": 1.0, "best_close_mult": 1.0, "worst_mult": 1.0, "pump2x": False, "pump5x": False,
                "clean2x": False, "trade_ret": -ROUND_TRIP_COST, "rugged": False}
# A pool whose liquidity falls below this share of its liquidity at entry, with no trading in the token
# afterwards on any pool, was emptied: holders lose everything. These used to fall into FLAT_OUTCOME
# (-6%): on 10-26, 240 of 246 'flat' top picks were pulled pools, which hid most of model5's losses.
RUG_LIQ_SHARE = 0.10
RUG_OUTCOME = {"best_mult": 1.0, "best_close_mult": 1.0, "worst_mult": 0.0, "pump2x": False, "pump5x": False,
               "clean2x": False, "trade_ret": -1.0, "rugged": True}


def rug_ts(pool_snaps, start_ts: float, entry_liq: float | None) -> float | None:
    """First snapshot after start_ts (within 24h) showing the pool's liquidity collapsed; `pool_snaps` is an
    iterable of (ts, snapshot)."""
    if not entry_liq:
        return None
    for ts, d in sorted(pool_snaps, key=lambda x: x[0]):
        if start_ts < ts <= start_ts + DAY_S:
            liq = _f(d.get("liq"))
            if liq is not None and liq < RUG_LIQ_SHARE * entry_liq:
                return ts
    return None


def label(path: list, start_ts: float, entry: float, pool_snaps, entry_liq: float | None,
          watched_after: bool) -> dict | None:
    """The 24h outcome of buying at start_ts, counting an emptied pool as a total loss."""
    pulled = rug_ts(pool_snaps, start_ts, entry_liq)
    out = _outcome(path, start_ts, entry, pulled)
    if out is None and entry and watched_after:
        out = RUG_OUTCOME if pulled is not None else FLAT_OUTCOME
    return out


def _outcome(path: list, start_ts: float, entry: float, pulled_ts: float | None = None) -> dict | None:
    """Best/worst after start_ts within 24h, the order of +100% / -50%, and a simulated trade.

    Pumps are judged on 5-minute CLOSES: a single wick in a thin pool is one trade nobody else
    could have sold into. The stop fills at the stop price or worse (the candle's close if it
    gapped through); if one candle spans both levels, the stop is assumed to come first.
    With `pulled_ts` (the pool was emptied then) and no trading in the token afterwards, a trade still
    open at that point loses everything.
    """
    window = [p for p in path if start_ts < p[0] <= start_ts + DAY_S]
    if not window or entry <= 0:
        return None
    rugged = pulled_ts is not None and not any(p[0] > pulled_ts for p in window)
    if rugged:
        window = [p for p in window if p[0] <= pulled_ts]
        if not window:
            return dict(RUG_OUTCOME)
    best_high = max(p[1] for p in window) / entry
    best_close = max(p[3] for p in window) / entry
    worst = min(p[2] for p in window) / entry
    clean, trade = False, None
    for _ts, _high, low, close in window:
        if low <= entry * SL_MULT:
            trade = min(SL_MULT, close / entry) - 1
            break
        if close >= entry * TP_MULT:
            clean, trade = True, TP_MULT - 1
            break
    if trade is None and rugged:
        return {"best_mult": best_high, "best_close_mult": best_close, "worst_mult": 0.0,
                "pump2x": best_close >= 2, "pump5x": best_close >= 5, "clean2x": False, "trade_ret": -1.0,
                "rugged": True}
    if trade is None:
        trade = window[-1][3] / entry - 1
    return {"best_mult": best_high, "best_close_mult": best_close, "worst_mult": 0.0 if rugged else worst,
            "pump2x": best_close >= 2, "pump5x": best_close >= 5, "clean2x": clean,
            "trade_ret": trade - ROUND_TRIP_COST, "rugged": rugged}


def _snap_features(d: dict, prefix: str = "") -> dict:
    out = {
        "log_liq": _log1p(d.get("liq")), "log_fdv": _log1p(d.get("fdv")),
        "liq_to_fdv": (_f(d.get("liq")) / _f(d.get("fdv"))) if _f(d.get("liq")) and _f(d.get("fdv")) else None,
    }
    for w in ("m5", "h1"):
        buys, sells, buyers, sellers = (d.get(f"{k}_{w}") for k in ("buys", "sells", "buyers", "sellers"))
        vol = _f(d.get(f"vol_{w}"))
        trades = (_f(buys) or 0) + (_f(sells) or 0)
        traders = (_f(buyers) or 0) + (_f(sellers) or 0)
        out.update({
            f"log_buys_{w}": _log1p(buys), f"log_sells_{w}": _log1p(sells),
            f"log_buyers_{w}": _log1p(buyers), f"log_sellers_{w}": _log1p(sellers),
            f"log_vol_{w}": _log1p(vol),
            f"buy_ratio_{w}": _ratio(buys, sells), f"buyer_ratio_{w}": _ratio(buyers, sellers),
            f"trades_per_trader_{w}": trades / traders if traders else None,  # high = bot churn
            f"log_avg_trade_{w}": _log1p(vol / trades) if vol and trades else None,
            f"vol_to_liq_{w}": vol / _f(d["liq"]) if vol and _f(d.get("liq")) else None,
            f"chg_{w}": _f(d.get(f"chg_{w}")),
        })
    return {prefix + k: v for k, v in out.items()}


def _static_features(pool: dict) -> dict:
    name = (pool["name"] or "").split(" / ")[0]
    created = time.gmtime(pool["created_ts"])
    hour = created.tm_hour + created.tm_min / 60
    return {
        "launchpad": float(pool["dex"] in LAUNCHPAD_DEXES),
        "dex_pumpfun": float(pool["dex"] == "pump-fun"), "dex_pumpswap": float(pool["dex"] == "pumpswap"),
        "dex_meteora": float((pool["dex"] or "").startswith("meteora")),
        "dex_raydium": float((pool["dex"] or "").startswith("raydium")),
        "dex_launchlab": float(pool["dex"] == "raydium-launchlab"), "dex_bags": float(pool["dex"] == "bags-fm"),
        "dex_meteora_dbc": float(pool["dex"] == "meteora-dbc"),
        "hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekend": float(created.tm_wday >= 5),
        "name_len": float(len(name)), "name_has_digit": float(any(c.isdigit() for c in name)),
        "name_all_caps": float(name.isupper()), "name_words": float(len(name.split())),
    }


def _info_features(info: dict) -> dict:
    if not info:
        return {}
    return {
        "has_website": float(bool(info.get("websites"))), "has_twitter": float(bool(info.get("twitter"))),
        "has_telegram": float(bool(info.get("telegram"))), "has_discord": float(bool(info.get("discord"))),
        "description_len": _log1p(info.get("description_len")), "gt_score": _f(info.get("gt_score")),
        "dev_holding_pct": _f(info.get("dev_holding_pct")), "log_holders": _log1p(info.get("holders")),
        "graduation_pct": _f(info.get("graduation_pct")),
        "mint_authority": float(info.get("mint_authority") == "yes"),
        "freeze_authority": float(info.get("freeze_authority") == "yes"),
        "honeypot_flag": float(info.get("is_honeypot") == "yes"),
        "info_graduated": float(bool(info.get("graduated"))),
    }


def _shared_inputs(store: WatchStore, since: float | None = None) -> dict:
    """Everything build_dataset reads that does not depend on the age (build once, reuse for every age)."""
    by_pool = load_snaps(store, since)  # parsed once: the factor context and the rows share the same dicts
    snaps: dict[str, dict[int, tuple[float, dict]]] = defaultdict(dict)
    for pool, ss in by_pool.items():
        for ts, age, d in ss:
            snaps[pool][age] = (ts, d)
    try:
        from tradingbot.config import Config
        cfg = Config.from_env()  # the operator's wallet / caller lists reach the factors
    except Exception:  # noqa: BLE001 (analysis must run even with odd env settings)
        cfg = None
    return {"snaps": snaps, "paths": _future_paths(store, by_pool, since),
            "ctx": Context(store, cfg, snaps=by_pool, since=since), "since": since}


def data_now(store: WatchStore) -> float:
    """The data's own clock: on a backup, pools whose 24h ran past the copy are not finished."""
    last = store.query("SELECT MAX(ts) AS t FROM watch_snaps")[0]["t"]
    return min(time.time(), last) if last else time.time()


def build_dataset(store: WatchStore, age: int, now: float | None = None, shared: dict | None = None,
                  since: float | None = None) -> list[dict]:
    """Rows for pools whose 24h outcome after `age` minutes is complete.

    Pass the same `shared` dict when building several ages: the snapshots, price paths and factor
    context are then loaded once. Factors with no value are left out of a row (read them with .get).
    With `since`, only data from pools created from then on is loaded, and rows start one day later
    (WARMUP_S), so look-back factors (developer record, theme heat, smart wallets) have some history.
    """
    if now is None:
        now = data_now(store)
    pools = {r["pool"]: dict(r) for r in store.query(
        "SELECT pool, mint, dex, name, created_ts, first, info, info_ts FROM watch_pools "
        "WHERE created_ts <= ? AND created_ts >= ?", (now - DAY_S - age * 60 - 1800, since or 0.0))}
    rows_from = since + WARMUP_S if since else -math.inf
    if shared is None:
        shared = {}
    if not shared:
        shared.update(_shared_inputs(store, since))
    snaps, paths, ctx = shared["snaps"], shared["paths"], shared["ctx"]

    # Developer track record: launches and pumps among the developer's EARLIER tokens.
    by_dev: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for q, pq in pools.items():
        dev = json.loads(pq["info"]).get("developer") if pq["info"] else None
        if dev:
            by_dev[dev].append((pq["created_ts"], q))
    for v in by_dev.values():
        v.sort()
    outcome_at_age: dict[str, dict] = {}
    rows = []
    for pool, p in sorted(pools.items(), key=lambda kv: kv[1]["created_ts"]):
        if age not in snaps[pool] or p["created_ts"] < rows_from:
            continue
        ts, d = snaps[pool][age]
        entry = _f(d.get("price"))
        # Watched afterwards but nobody traded it again: flat at the entry price, or lost if the pool was emptied.
        out = label(paths.get(p["mint"], []), ts, entry or 0, snaps[pool].values(), _f(d.get("liq")),
                    any(a > age for a in snaps[pool]))
        if out is None:
            continue
        outcome_at_age[pool] = out
        row = {"pool": pool, "mint": p["mint"], "dex": p["dex"], "created_ts": p["created_ts"], "age": age}
        row.update(_static_features(p))
        row.update(_snap_features(d))
        first = json.loads(p["first"]) if p["first"] else {}
        fp = _f(first.get("price"))
        row["log_price_vs_first"] = math.log(entry / fp) if entry and fp else None
        earlier = [a for a in snaps[pool] if a < age]
        if earlier:
            e_ts, e = snaps[pool][max(earlier)]
            ep = _f(e.get("price"))
            row["log_price_vs_prev_snap"] = math.log(entry / ep) if entry and ep else None
            row["buyer_growth_h1"] = (_log1p(d.get("buyers_h1")) or 0) - (_log1p(e.get("buyers_h1")) or 0)
        if p["info"] and p["info_ts"] and p["info_ts"] <= ts:  # no peeking: info fetched before the snapshot
            info = json.loads(p["info"])
            row.update(_info_features(info))
            dev = info.get("developer")
            if dev:
                prior = [q for c, q in by_dev[dev] if c < p["created_ts"]]
                done = [q for q in prior if pools[q]["created_ts"] + DAY_S + age * 60 <= ts and q in outcome_at_age]
                row["dev_prior_launches"] = _log1p(len(prior))
                row["dev_prior_pump_rate"] = (sum(outcome_at_age[q]["pump2x"] for q in done) / len(done)
                                              if done else None)
        row.update(ctx.features(pool, p, ts, info_ok=bool(p["info"] and p["info_ts"] and p["info_ts"] <= ts), snap=d))
        row["hel_selection"], row["x_selection"] = ctx.selection(pool, "helius", ts), ctx.selection(pool, "x", ts)
        row.update(out)
        rows.append({k: v for k, v in row.items() if v is not None})  # about half the factors are empty
    return rows


LABEL_KEYS = ("best_mult", "best_close_mult", "worst_mult", "pump2x", "pump5x", "clean2x", "trade_ret", "rugged")
ID_KEYS = ("pool", "mint", "dex", "created_ts", "age", "hel_selection", "x_selection")


def feature_names(rows: list[dict]) -> list[str]:
    keys = {k for r in rows for k in r} - set(LABEL_KEYS) - set(ID_KEYS)
    # Keep features present in at least 30% of rows.
    return sorted(k for k in keys if sum(r.get(k) is not None for r in rows) >= 0.3 * len(rows))


# --- statistics ----------------------------------------------------------------------
def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Probability a random positive outranks a random negative (Mann-Whitney U, ties averaged)."""
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _matrix(rows: list[dict], names: list[str], medians: dict | None = None):
    raw = np.array([[np.nan if r.get(n) is None else float(r[n]) for n in names] for r in rows], dtype=float)
    if medians is None:
        medians = {n: float(np.nanmedian(raw[:, i])) if np.isfinite(raw[:, i]).any() else 0.0
                   for i, n in enumerate(names)}
    for i, n in enumerate(names):
        col = raw[:, i]
        col[~np.isfinite(col)] = medians[n]
    return raw, medians


def logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 3000, lr: float = 0.1):
    """L2-regularised logistic regression by gradient descent on standardised inputs."""
    w, b = np.zeros(x.shape[1]), math.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))
    n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        g = p - y
        w -= lr * (x.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


SUBSETS = {
    None: lambda r: True,
    # Tokens with the paid wallet-level data (Helius): graduations and launches with real buyers at 5 min.
    "enriched": lambda r: r.get("hel_early_buyers") is not None or r.get("hel_top10_share") is not None,
    # Helius rows chosen without looking at the free score (random control slice, or everyone while the
    # budget covered all): the fair sample for asking whether the paid data adds anything.
    "control": lambda r: r.get("hel_selection") in UNBIASED,
}
UNBIASED = ("control", "all")


def analyse(store: WatchStore, age: int = 15, label: str = "pump2x", now: float | None = None,
            subset: str | None = None, rows: list[dict] | None = None) -> dict:
    rows = [r for r in (rows if rows is not None else build_dataset(store, age, now)) if SUBSETS[subset](r)]
    out: dict = {"age_min": age, "label": label, "subset": subset or "all", "rows": len(rows)}
    if len(rows) < MIN_ROWS:
        out["note"] = f"need at least {MIN_ROWS} finished rows; have {len(rows)} (pools finish 24h after creation)"
        return out
    names = feature_names(rows)
    y = np.array([float(r[label]) for r in rows])
    x_raw, medians = _matrix(rows, names)
    out["base_rate"] = float(y.mean())
    out["positives"] = int(y.sum())
    out["days"] = round((rows[-1]["created_ts"] - rows[0]["created_ts"]) / DAY_S, 1)

    # Univariate screen (whole sample): which single numbers separate pumps from the rest.
    uni = []
    for i, n in enumerate(names):
        col = x_raw[:, i]
        a = auc(col, y)
        if a is None:
            continue
        qs = np.quantile(col, [0.2, 0.4, 0.6, 0.8])
        bucket = np.searchsorted(qs, col, side="right")
        rates = [float(y[bucket == k].mean()) if (bucket == k).any() else None for k in range(5)]
        uni.append({"feature": n, "auc": round(a, 3), "pump_rate_by_quintile": [None if v is None else round(v, 3)
                                                                                  for v in rates]})
    uni.sort(key=lambda u: abs(u["auc"] - 0.5), reverse=True)
    out["univariate"] = uni

    # Time split: train on the earliest pools, judge on the latest.
    cut = int(len(rows) * TRAIN_SHARE)
    mu, sd = x_raw[:cut].mean(axis=0), x_raw[:cut].std(axis=0) + 1e-9
    x = (x_raw - mu) / sd
    if y[:cut].sum() == 0 or y[cut:].sum() == 0:
        out["note"] = "no pumps in the training or test period yet"
        return out
    w, b = logistic(x[:cut], y[:cut])
    p_test = 1 / (1 + np.exp(-(x[cut:] @ w + b)))
    y_test = y[cut:]
    out["train_rows"], out["test_rows"] = cut, len(rows) - cut
    out["test_auc"] = auc(p_test, y_test)
    top = p_test >= np.quantile(p_test, 0.9)
    out["test_base_rate"] = float(y_test.mean())
    out["test_top_decile_rate"] = float(y_test[top].mean()) if top.any() else None
    out["test_lift"] = (out["test_top_decile_rate"] / out["test_base_rate"]) if out["test_base_rate"] else None
    out["coefficients"] = sorted(({"feature": n, "coef": round(float(c), 3)} for n, c in zip(names, w)),
                                 key=lambda c: abs(c["coef"]), reverse=True)

    # Would trading the top decile have made money after costs?
    test_rows = rows[cut:]
    rets = np.array([r["trade_ret"] for r in test_rows])
    out["trade_all_test_mean"] = float(rets.mean())
    out["trade_top_decile"] = {
        "n": int(top.sum()), "mean_ret": float(rets[top].mean()) if top.any() else None,
        "median_ret": float(np.median(rets[top])) if top.any() else None,
        "win_rate": float((rets[top] > 0).mean()) if top.any() else None,
        "rule": f"buy at snapshot price, +{(TP_MULT - 1) * 100:.0f}% take profit, -{(1 - SL_MULT) * 100:.0f}% stop, "
                f"else exit at 24h; {ROUND_TRIP_COST * 100:.0f}% round-trip cost",
    }
    return out


PAID_SOURCES = {"helius": "hel_selection", "x": "x_selection"}
HAS_DATA = ("all", "ranked", "control", "legacy")


def _heldout(rows: list[dict], names: list[str], label: str) -> dict:
    """Held-out AUC and top-decile lift of the logistic model on `names` (earliest 70% trains)."""
    y = np.array([float(r[label]) for r in rows])
    cut = int(len(rows) * TRAIN_SHARE)
    if not names or y[:cut].sum() == 0 or y[cut:].sum() == 0:
        return {"test_auc": None, "test_lift": None}
    x_raw, _ = _matrix(rows, names)
    mu, sd = x_raw[:cut].mean(axis=0), x_raw[:cut].std(axis=0) + 1e-9
    x = (x_raw - mu) / sd
    w, b = logistic(x[:cut], y[:cut])
    p = 1 / (1 + np.exp(-(x[cut:] @ w + b)))
    top = p >= np.quantile(p, 0.9)
    base = y[cut:].mean()
    return {"test_auc": auc(p, y[cut:]), "test_lift": float(y[cut:][top].mean() / base) if base else None}


def paid_value(rows: list[dict], label: str = "pump2x") -> dict:
    """Is each paid source worth its cost? On tokens it was fetched for without looking at their free score
    (control slice), compare the model with and without the features only that source provides. Also: did
    ranking by free score pick tokens that pumped more often than the random control slice?"""
    out = {}
    for source, key in PAID_SOURCES.items():
        with_data = [r for r in rows if r.get(key) in HAS_DATA]
        without = [r for r in rows if r.get(key) not in HAS_DATA]
        present = lambda rs, n: any(r.get(n) is not None for r in rs)  # noqa: E731
        res: dict = {"rows_with_data": len(with_data)}
        for sel in ("ranked", "control"):
            got = [r[label] for r in rows if r.get(key) == sel]
            res[f"{sel}_n"], res[f"{sel}_pump_rate"] = len(got), (sum(got) / len(got) if got else None)
        fair = [r for r in rows if r.get(key) in UNBIASED]
        res["fair_rows"] = len(fair)
        if len(fair) < MIN_ROWS:
            res["note"] = f"need {MIN_ROWS} control/all rows with this source; have {len(fair)}"
            out[source] = res
            continue
        names = feature_names(fair)
        paid = [n for n in names if present(with_data, n) and not present(without, n)]
        res["paid_features"] = len(paid)
        full = _heldout(fair, names, label)
        free = _heldout(fair, [n for n in names if n not in paid], label)
        res.update({"auc_with": full["test_auc"], "auc_without": free["test_auc"],
                    "lift_with": full["test_lift"], "lift_without": free["test_lift"]})
        if full["test_auc"] is not None and free["test_auc"] is not None:
            res["auc_gain"] = full["test_auc"] - free["test_auc"]
        out[source] = res
    return out


def render_paid_value(pv: dict) -> str:
    f = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
    pct = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    lines = ["PAID DATA VALUE (control slice: tokens looked up without regard to their free score)"]
    for source, r in pv.items():
        lines.append(f"  {source}: {r['rows_with_data']} rows with data; pump rate ranked {pct(r['ranked_pump_rate'])} "
                     f"(n={r['ranked_n']}) vs control {pct(r['control_pump_rate'])} (n={r['control_n']})")
        if r.get("note"):
            lines.append(f"    {r['note']}")
            continue
        lines.append(f"    held-out AUC with its {r['paid_features']} features {f(r['auc_with'])} vs without "
                     f"{f(r['auc_without'])} (gain {f(r.get('auc_gain'))}); lift {f(r['lift_with'], 2)} vs "
                     f"{f(r['lift_without'], 2)} on {r['fair_rows']} rows")
    return "\n".join(lines)


def summary(store: WatchStore, label: str = "pump2x", now: float | None = None,
            subset: str | None = None, rows_by_age: dict | None = None) -> list[dict]:
    """Held-out quality per feature age, to pick when to decide."""
    return [summary_entry(analyse(store, age, label, now, subset, rows=(rows_by_age or {}).get(age)))
            for age in FEATURE_AGES]


def report_by_age(store: WatchStore, labels=("pump2x",), subsets=(None,), detail_age: int = 15,
                  now: float | None = None, on_rows=None, days: float | None = REPORT_DAYS) -> dict:
    """Analyse every feature age with one age's rows in memory at a time (a week of rows for all four
    ages does not fit a small machine). `on_rows(age, rows)` sees each age's rows before they are freed.
    Returns {"summary": {(label, subset): [entry per age]}, "detail": {(label, subset): analyse at detail_age}}."""
    shared: dict = {}
    if now is None:
        now = data_now(store)
    since = (now - DAY_S - 1800 - days * DAY_S - WARMUP_S) if days else None
    summ = {(lb, sb): [] for lb in labels for sb in subsets}
    detail: dict = {}
    ages = list(FEATURE_AGES) + ([detail_age] if detail_age not in FEATURE_AGES else [])
    for age in ages:
        rows = build_dataset(store, age, now, shared, since)
        for lb in labels:
            for sb in subsets:
                a = analyse(store, age, lb, now, sb, rows=rows)
                if age in FEATURE_AGES:
                    summ[(lb, sb)].append(summary_entry(a))
                if age == detail_age:
                    detail[(lb, sb)] = a
        if on_rows is not None:
            on_rows(age, rows)
        del rows
        gc.collect()
    return {"summary": summ, "detail": detail}


def summary_entry(a: dict) -> dict:
    """The one-line view of an analyse() result used in the by-age table."""
    return ({k: a.get(k) for k in ("age_min", "rows", "base_rate", "test_auc", "test_lift", "trade_all_test_mean",
                                   "note")}
            | {"top_decile_mean_ret": (a.get("trade_top_decile") or {}).get("mean_ret")})


def render(a: dict, summ: list[dict] | None = None) -> str:
    pct = lambda v: "n/a" if v is None else f"{v * 100:+.1f}%"  # noqa: E731
    lines = []
    if summ:
        lines += ["BY FEATURE AGE (held-out test set)",
                  f"{'age':>5}{'rows':>7}{'base':>8}{'AUC':>7}{'lift':>7}{'trade all':>11}{'trade top10%':>14}"]
        for s in summ:
            if s.get("note"):
                lines.append(f"{s['age_min']:>4}m{s['rows']:>7}  {s['note']}")
                continue
            lines.append(f"{s['age_min']:>4}m{s['rows']:>7}{s['base_rate'] * 100:>7.1f}%"
                         f"{(s['test_auc'] or 0):>7.3f}{(s['test_lift'] or 0):>7.2f}"
                         f"{pct(s['trade_all_test_mean']):>11}{pct(s['top_decile_mean_ret']):>14}")
        lines.append("")
    lines.append(f"DETAIL at {a['age_min']} min, label {a['label']}, tokens: {a.get('subset', 'all')}: "
                 f"{a['rows']} rows")
    if a.get("note"):
        lines.append(f"  {a['note']}")
        return "\n".join(lines)
    lines.append(f"  {a['positives']} pumps over {a['days']} days, base rate {a['base_rate'] * 100:.1f}%")
    if "test_auc" in a:
        lines.append(f"  held-out AUC {a['test_auc']:.3f} (0.5 = coin flip); top-decile pump rate "
                     f"{a['test_top_decile_rate'] * 100:.1f}% vs {a['test_base_rate'] * 100:.1f}% "
                     f"-> lift {a['test_lift']:.2f}x")
        t = a["trade_top_decile"]
        lines.append(f"  trading the top decile: n={t['n']}, mean {pct(t['mean_ret'])}, median "
                     f"{pct(t['median_ret'])}, win rate {(t['win_rate'] or 0) * 100:.0f}% "
                     f"(all test pools: mean {pct(a['trade_all_test_mean'])})")
        lines.append(f"  rule: {t['rule']}")
        lines.append("  strongest model weights (standardised; + = more likely to pump):")
        for c in a["coefficients"][:12]:
            lines.append(f"    {c['coef']:+.3f}  {c['feature']}")
    lines.append("  single features (whole sample; many tests, so treat as leads, not proof):")
    lines.append(f"    {'feature':<28}{'AUC':>6}   pump rate by quintile (low -> high)")
    for u in a["univariate"][:15]:
        q = " ".join("  n/a" if v is None else f"{v * 100:5.1f}" for v in u["pump_rate_by_quintile"])
        lines.append(f"    {u['feature']:<28}{u['auc']:>6.3f}   {q}")
    return "\n".join(lines)


def export_csv(store: WatchStore, path: str, age: int = 15, now: float | None = None,
               rows: list[dict] | None = None) -> int:
    rows = rows if rows is not None else build_dataset(store, age, now)
    if not rows:
        return 0
    cols = list(ID_KEYS) + feature_names(rows) + list(LABEL_KEYS)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)
