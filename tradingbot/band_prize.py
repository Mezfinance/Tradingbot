"""Is there a prize in one liquidity band if we filter it better? (11-08 study, read only; Helius only with --fetch.)

Asked for $40-80k: `band-rw-study` found no robust rug-vs-win filter there, so this looks wider:
1. ceiling: what the band's pools made at the ideal minute-6 trade, and what a perfect filter would have made
   (rugs removed; the best half / third in hindsight), plus where the money sits (how much the big wins carry);
2. single signals: for every second-4 input (the rug-vs-win inputs + creator record) and the live scores (rw_pct,
   launch4 `score`, history model `score_h`, liquidity), AUC for rugs / big wins / any win and the mean return of
   the top vs bottom quintile, in the first and second half of the band's pools by time (a real signal keeps its
   sign in both halves);
3. walk-forward filters with different targets, in BLOCK_H blocks, fitted only on pools created before each block:
   P(rug) / P(win) / P(+50%) logistics and the rug-vs-win score, each trained on the band only and on every pool,
   keep the best KEEPS share by past-score cut-offs, judged with `band_rw._judge` (kept vs every pool, vs skipped,
   vs random keeps of the same size, blocks won); the deployed rule (rw_pct < 50) for reference;
4. later exits: the same ideal trade sold at pool minute 10 / 15 / 30 (vault states from `Helius.last_before`,
   fetched by `fetch_states` with --fetch, cached in `watch_launch_path`), ceiling and every filter judged again
   at each exit on one fixed set of pools (`_exits`).
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

import numpy as np

from tradingbot import band_rw as br

BLOCK_H = 6
MIN_TRAIN = 150  # band-only models need this many finished earlier band pools
KEEPS = (30, 50, 70)  # % of the band kept as the most promising
CUT_WINDOW_H = 24
MIN_CUT_POOLS = 30
EXIT_S = 360
MIN_LIQ, MAX_LIQ = 40_000.0, 80_000.0
LIVE_SCORES = ("rw_pct", "score", "score_h", "log_liq")
TOP_SIGNALS = 15
EXITS = {10: 600, 15: 900, 30: 1800}  # later ideal exits (pool minute: seconds after creation)
CREDITS_PER_CALL = 10


def _rows(store, since: float) -> list[dict]:
    from tradingbot import launch_live as ll
    ll.ensure_schema(store)
    rows = ll._pull_rows(store, since)
    extra = {r["pool"]: dict(r) for r in store.query(
        "SELECT pool, liq_usd, rw_pct, score, fill_px, fill_sol, peak_sol, token_mint FROM watch_launch_live "
        "WHERE status='done' AND liquid=1")}
    for r in rows:
        e = extra.get(r["pool"], {})
        r["liq"], r["rw_pct"], r["score"] = e.get("liq_usd"), e.get("rw_pct"), e.get("score")
        r["fill_px"], r["fill_sol"], r["peak_sol"] = e.get("fill_px"), e.get("fill_sol"), e.get("peak_sol")
        r["token_mint"] = e.get("token_mint")
        r["log_liq"] = math.log(r["liq"]) if r["liq"] else None
    return [r for r in rows if r["liq"] is not None]


def _ceiling(rets: list[float]) -> dict:
    a = np.sort(np.array(rets))[::-1]
    n = len(a)
    no_rug = a[a > -0.999]
    big = a[a >= 0.5]
    out = {"every pool": br._summary(list(a)),
           "rugs removed (a perfect rug filter)": br._summary(list(no_rug)),
           "best half in hindsight": br._summary(list(a[: max(1, n // 2)])),
           "best third in hindsight": br._summary(list(a[: max(1, n // 3)]))}
    total = float(a.sum())
    out["money"] = {"total $ at $100": round(100 * total),
                    "$ from +50% wins": round(100 * float(big.sum())),
                    "$ lost to rugs": round(-100 * float((a <= -0.999).sum())),
                    "$ from everything else": round(100 * float(a[(a > -0.999) & (a < 0.5)].sum()))}
    return out


def _quintiles(v: np.ndarray, rets: np.ndarray) -> tuple[float, float] | None:
    ok = ~np.isnan(v)
    if ok.sum() < 25 or np.nanstd(v) == 0:
        return None
    v, r = v[ok], rets[ok]
    lo, hi = np.quantile(v, [0.2, 0.8])
    bot, top = r[v <= lo], r[v >= hi]
    if len(bot) < 5 or len(top) < 5 or lo == hi:
        return None
    return float(top.mean()), float(bot.mean())


def _signals(rows: list[dict]) -> list[dict]:
    from tradingbot.watch_analysis import auc
    names = sorted({k for r in rows for k in r["x"]}) + list(LIVE_SCORES)
    half = len(rows) // 2
    parts = {"all": rows, "first half": rows[:half], "second half": rows[half:]}
    out = []
    for name in names:
        entry: dict = {"signal": name}
        for label, part in parts.items():
            v = np.array([_val(r, name) for r in part], dtype=float)
            rets = np.array([r["ret"] for r in part])
            q = _quintiles(v, rets)
            ok = ~np.isnan(v)
            if q is None:
                entry[label] = None
                continue
            entry[label] = {"top - bottom quintile (pts)": round(100 * (q[0] - q[1]), 1),
                            "top quintile mean": round(100 * q[0], 1), "bottom quintile mean": round(100 * q[1], 1)}
            if label == "all":
                vv, rr = v[ok], rets[ok]
                entry["AUC"] = {"rug": br._r(auc(vv, (rr <= -0.999).astype(int))),
                                "+50%": br._r(auc(vv, (rr >= 0.5).astype(int))),
                                "win": br._r(auc(vv, (rr > 0).astype(int))),
                                "present %": round(100 * float(ok.mean()))}
        a, b = entry.get("first half"), entry.get("second half")
        entry["same sign both halves"] = bool(a and b and a["top - bottom quintile (pts)"]
                                               * b["top - bottom quintile (pts)"] > 0)
        out.append(entry)
    key = lambda e: (e["same sign both halves"],
                     min(abs(e["first half"]["top - bottom quintile (pts)"]),
                         abs(e["second half"]["top - bottom quintile (pts)"]))
                     if e.get("first half") and e.get("second half") else 0.0)
    return sorted([e for e in out if e.get("all")], key=key, reverse=True)


def _val(r: dict, name: str):
    v = r.get(name) if name in LIVE_SCORES else r["x"].get(name)
    return np.nan if v is None else float(v)


def _logit(rows, y):
    from tradingbot import launch_live as ll
    if y.min() == y.max():
        return None
    return ll._fit_capped(rows, y)


def _fits(train: list[dict]) -> dict:
    """Goodness scorers (higher = keep) fitted on `train`."""
    from tradingbot import launch_live as ll
    if len(train) < MIN_TRAIN:
        return {}
    ret = np.array([r["ret"] for r in train])
    out = {}
    rug = _logit(train, (ret <= -0.999).astype(float))
    if rug:
        out["not rug"] = lambda x, m=rug: -m.score(x)
    win = _logit(train, (ret > 0).astype(float))
    if win:
        out["win"] = lambda x, m=win: m.score(x)
    big = _logit(train, (ret >= ll.RW_BIG).astype(float))
    if big:
        out["+50%"] = lambda x, m=big: m.score(x)
    rw = br._fit(train)
    if rw:
        out["rug-vs-win"] = lambda x, m=rw: -m.score(x)
    return out


def _cut(fn, base: list[dict], t: float, keep: int) -> float:
    recent = [r for r in base if r["create_ts"] >= t - CUT_WINDOW_H * 3600]
    pool = recent if len(recent) >= MIN_CUT_POOLS else base
    return float(np.quantile([fn(r["x"]) for r in pool], 1 - keep / 100))


def _walk(rows: list[dict], band: list[dict]) -> tuple[list[dict], int]:
    """Walk-forward: every band pool gets, per model and keep share, `skip:<model>:<k>` (True = filtered out)."""
    from tradingbot import launch_live as ll
    block = BLOCK_H * 3600
    first = band[0]["create_ts"] // block * block
    tested, blocks = [], 0
    t = first + block
    while t <= band[-1]["create_ts"]:
        test = [r for r in band if t <= r["create_ts"] < t + block]
        if test:
            train_all = [r for r in rows if r["create_ts"] < t - EXIT_S]
            train_b = [r for r in band if r["create_ts"] < t - EXIT_S]
            models = {f"{k} (band)": f for k, f in _fits(train_b).items()}
            if len(train_b) >= MIN_CUT_POOLS:
                models.update({f"{k} (all pools)": f for k, f in _fits(train_all).items()})
            cuts = {(m, k): _cut(f, train_b, t, k) for m, f in models.items() for k in KEEPS}
            blocks += 1
            for r in test:
                rec = {"pool": r["pool"], "ret": r["ret"], "block": t, "create_ts": r["create_ts"],
                       "skip:live": None if r["rw_pct"] is None else r["rw_pct"] >= 100 - ll.RW_SKIP}
                for (m, k), c in cuts.items():
                    rec[f"skip:{m}:{k}"] = models[m](r["x"]) < c
                tested.append(rec)
        t += block
    return tested, blocks


def _order(band: list[dict]) -> list[dict]:
    """Stable random order (hash of the pool), so a credit cap leaves an even sample of the band."""
    import hashlib
    return sorted(band, key=lambda r: hashlib.sha1(f"prize:{r['pool']}".encode()).hexdigest())


def _have(store) -> dict:
    from tradingbot import launch_live as ll
    store.execute(ll.PATH_SCHEMA)
    return {(r["pool"], r["sec"]): {"sol": r["sol"], "tok": r["tok"]}
            for r in store.query("SELECT pool, sec, sol, tok FROM watch_launch_path")}


def _usable(r: dict) -> bool:
    return r["fill_px"] is not None and r["fill_sol"] is not None


def fetch_states(store, helius, band: list[dict], max_credits: int) -> dict:
    """Pool state after its last tx by minutes 10 / 15 / 30 (`Helius.last_before`, ~10 credits each, cached in
    `watch_launch_path` like the hold study), for every band pool still alive at minute 6, in `_order`. Stops at
    the first known death (later exits are -100% without a call) and before `max_credits`."""
    from tradingbot import hold_study as hs
    from tradingbot import launch_live as ll
    from tradingbot.data import pumpswap as ps
    have = _have(store)
    stats = {"fetched": 0, "call failed": 0, "credits_used": 0}
    start = helius.credits_used
    for r in _order(band):
        if not _usable(r) or r["ret"] <= -0.999:
            continue
        for sec in EXITS.values():
            p = hs.path(r, have)
            if p[sec] is not None:
                if p[sec] <= -0.999:
                    break  # dead: later exits follow without a call
                continue
            if helius.credits_used - start + CREDITS_PER_CALL > max_credits:
                stats["stopped at the credit cap (rerun to continue; nothing is fetched twice)"] = 1
                stats["credits_used"] = helius.credits_used - start
                return stats
            tx = helius.last_before(r["pool"], r["create_ts"] + sec)
            if tx is False:
                stats["call failed"] += 1
                break
            st = {"sol": ll._raw_vault(tx, r["pool"], ps.WSOL) if tx else None,
                  "tok": ll._raw_vault(tx, r["pool"], r["token_mint"]) if tx else None}
            store.execute("INSERT OR REPLACE INTO watch_launch_path(pool, sec, sol, tok, ts) VALUES (?,?,?,?,?)",
                          (r["pool"], sec, st["sol"], st["tok"], time.time()))
            have[(r["pool"], sec)] = st
            stats["fetched"] += 1
    stats["credits_used"] = helius.credits_used - start
    return stats


def _exits(store, band: list[dict]) -> dict:
    """Adds `ret_by` {6, 10, 15, 30} and `cause_by` to every band pool and returns the covered set: pools in
    `_order` up to the last one alive at minute 6 with every later exit known. Pools rugged by minute 6 need no
    state, so counting all of them while only some live ones were fetched would add rugs; the prefix keeps the
    set an even sample. Live pools inside the prefix still missing a state (a failed call) are left out."""
    from tradingbot import hold_study as hs
    try:
        have = _have(store)
    except Exception:  # noqa: BLE001 - optional part
        have = {}
    order = [r for r in _order(band) if _usable(r)]
    last = -1
    for i, r in enumerate(order):
        d = hs.path_detail(r, have)
        r["ret_by"] = {6: r["ret"], **{m: d[sec][0] for m, sec in EXITS.items()}}
        r["cause_by"] = {m: d[sec][1] for m, sec in EXITS.items()}
        if r["ret"] > -0.999 and all(v is not None for v in r["ret_by"].values()):
            last = i
    covered = [r for r in order[:last + 1] if all(v is not None for v in r["ret_by"].values())]
    alive = sum(r["ret"] > -0.999 for r in order)
    return {"covered": covered, "usable": len(order), "alive at minute 6": alive,
            "alive covered": sum(r["ret"] > -0.999 for r in covered)}


def _paired(a: list[float], b: list[float], rng) -> dict:
    d = np.array(a) - np.array(b)
    boots = [rng.choice(d, len(d)).mean() for _ in range(1000)]
    return {"mean": round(100 * float(d.mean()), 1),
            "ci90": [round(100 * float(np.quantile(boots, q)), 1) for q in (0.05, 0.95)]}


def _by_exit(cov: list[dict], tested: list[dict], keys: list[str], rng) -> dict:
    """Ideal trade at each exit on the covered pools: every pool, rugs removed, paired gap vs minute 6, rug
    causes, and every walk-forward filter judged at that exit."""
    import collections
    out = {}
    six = [r["ret_by"][6] for r in cov]
    by_pool = {r["pool"]: r for r in cov}
    for m in (6, *EXITS):
        rets = [r["ret_by"][m] for r in cov]
        e = {"every pool": br._summary(rets),
             "rugs removed": br._summary([x for x in rets if x > -0.999])}
        if m != 6:
            e["vs minute 6 (pts, paired)"] = _paired(rets, six, rng)
            e["rug causes"] = dict(collections.Counter(r["cause_by"][m] for r in cov if r["cause_by"][m]))
        t = [{**x, "ret": by_pool[x["pool"]]["ret_by"][m]} for x in tested if x["pool"] in by_pool]
        e["filters"] = {"live rule (rw_pct < 50)": br._judge(t, "skip:live", rng)}
        for k in keys:
            _, name, keep = k.split(":")
            e["filters"][f"{name}, keep best {keep}%"] = br._judge(t, k, rng)
        out[m] = e
    return out


# ---- minute 30: conditional holds and models trained on the minute-30 outcome (11-08) ----
HOLD_THRESHOLDS = (0.0, 0.10, 0.25)  # keep holding at a check only if the trade is at least this far up (after cost)
HOLD_FINALS = (15, 30)
LEAD_FILTER = "skip:+50% (all pools):30"  # the one robust minute-6 filter, to see the holds on its picks too
SETTLE_S = 120


def hold_result(ret_by: dict, thr: float, rising: bool, final: int) -> tuple[float, int]:
    """Buy at second 6, check at minutes 6 / 10 / 15 (before `final`): keep holding only while the pool is alive,
    the trade is at least `thr` up and (with `rising`) not below the previous check; else sell at that check.
    Returns (return, minute sold). Uses only states known at each check."""
    prev = None
    for cp in (6, 10, 15):
        if cp >= final:
            break
        r = ret_by[cp]
        if r <= -0.999:
            return -1.0, cp
        if r < thr or (rising and prev is not None and r < prev):
            return r, cp
        prev = r
    return ret_by[final], final


def _holds(cov: list[dict], tested: list[dict], rng) -> dict:
    """Every conditional hold rule on the covered pools, and on the lead filter's kept pools among them."""
    blocks = {r["pool"]: r["block"] for r in tested}
    kept = {r["pool"] for r in tested if r.get(LEAD_FILTER) is False}
    groups = {"every pool": cov, "lead filter's picks (+50% model, all pools, best 30%)":
              [r for r in cov if r["pool"] in kept]}
    out = {}
    for gname, pools in groups.items():
        if len(pools) < 20:
            out[gname] = {"pools": len(pools), "note": "too few pools"}
            continue
        six = [r["ret_by"][6] for r in pools]
        thirty = [r["ret_by"][30] for r in pools]
        rows = {"sell at minute 6": {**br._summary(six)}, "hold to minute 30": {
            **br._summary(thirty), "vs minute 6 (pts, paired)": _paired(thirty, six, rng)}}
        for final in HOLD_FINALS:
            for thr in HOLD_THRESHOLDS:
                for rising in (False, True):
                    res = [hold_result(r["ret_by"], thr, rising, final) for r in pools]
                    rets = [x[0] for x in res]
                    label = f"hold to {final} while up {100 * thr:.0f}%+" + (" and rising" if rising else "")
                    won, n = 0, 0
                    by_block: dict = {}
                    for r, x in zip(pools, rets):
                        if r["pool"] in blocks:
                            by_block.setdefault(blocks[r["pool"]], []).append((x, r["ret_by"][6]))
                    for g in by_block.values():
                        if len(g) >= 3:
                            n += 1
                            won += np.mean([a for a, _ in g]) > np.mean([b for _, b in g])
                    rows[label] = {**br._summary(rets), "vs minute 6 (pts, paired)": _paired(rets, six, rng),
                                   "held to the end %": round(100 * sum(x[1] == final for x in res) / len(res)),
                                   "blocks beating minute 6": f"{won} of {n}"}
        out[gname] = {"pools": len(pools), "rules": rows}
    return out


def _fits30(train: list[dict]) -> dict:
    """Goodness scorers (higher = keep) fitted on the minute-30 outcome of `train`."""
    from tradingbot import launch_live as ll
    if len(train) < MIN_TRAIN:
        return {}
    r30 = np.array([r["ret_by"][30] for r in train])
    out = {}
    rug = _logit(train, (r30 <= -0.999).astype(float))
    big = _logit(train, (r30 >= 0.5).astype(float))
    win = _logit(train, (r30 > 0).astype(float))
    if rug:
        out["not rug by 30"] = lambda x, m=rug: -m.score(x)
    if win:
        out["win at 30"] = lambda x, m=win: m.score(x)
    if big:
        out["+50% at 30"] = lambda x, m=big: m.score(x)
    if rug and big:
        rw = ll.RugWinModel(rug, big, {})
        out["rug-vs-win at 30"] = lambda x, m=rw: -m.score(x)
    return out


def _walk30(cov: list[dict]) -> tuple[list[dict], int]:
    """Walk-forward on the covered band pools, models fitted only on pools whose minute-30 outcome was known
    before the block (created at least 30 min + SETTLE_S earlier)."""
    cov = sorted(cov, key=lambda r: r["create_ts"])
    block = BLOCK_H * 3600
    first = cov[0]["create_ts"] // block * block
    tested, blocks = [], 0
    t = first + block
    while t <= cov[-1]["create_ts"]:
        test = [r for r in cov if t <= r["create_ts"] < t + block]
        train = [r for r in cov if r["create_ts"] < t - EXITS[30] - SETTLE_S]
        models = _fits30(train) if test else {}
        if models:
            blocks += 1
            cuts = {(m, k): _cut(f, train, t, k) for m, f in models.items() for k in KEEPS}
            for r in test:
                rec = {"pool": r["pool"], "ret": r["ret_by"][30], "ret6": r["ret_by"][6], "block": t,
                       "create_ts": r["create_ts"]}
                for m, f in models.items():
                    rec[f"score:{m}"] = f(r["x"])
                for (m, k), c in cuts.items():
                    rec[f"skip:{m}:{k}"] = rec[f"score:{m}"] < c
                tested.append(rec)
        t += block
    return tested, blocks


def _models30(cov: list[dict], rng) -> dict:
    from tradingbot.watch_analysis import auc
    tested, blocks = _walk30(cov)
    out: dict = {"blocks": blocks, "tested pools": len(tested)}
    if len(tested) < 20:
        out["note"] = f"too few pools with a minute-30 outcome to train on (needs {MIN_TRAIN} before the first block)"
        return out
    rets = np.array([r["ret"] for r in tested])
    rug = (rets <= -0.999).astype(int)
    models = sorted({k.split(":", 1)[1] for r in tested for k in r if k.startswith("score:")})
    out["AUC for a rug by minute 30 (riskiness; want > 0.5)"] = {
        m: _r(auc(-np.array([r[f"score:{m}"] for r in tested]), rug)) for m in models
        if all(f"score:{m}" in r for r in tested)}
    filters = {}
    for m in models:
        for k in KEEPS:
            j = br._judge([r for r in tested if f"skip:{m}:{k}" in r], f"skip:{m}:{k}", rng)
            kept6 = [r["ret6"] for r in tested if r.get(f"skip:{m}:{k}") is False]
            j["kept at minute 6"] = br._summary(kept6)
            filters[f"{m}, keep best {k}%"] = j
    out["filters (judged at minute 30)"] = filters
    return out


def _r(x):
    return None if x is None else round(x, 3)


def study(store, hours: float | None = None, min_liq: float = MIN_LIQ, max_liq: float = MAX_LIQ,
          now: float | None = None, seed: int = 13, helius=None, max_credits: int = 20_000) -> dict:
    """`helius`: fetch the minute-10/15/30 states first (else only cached states are used)."""
    now = now if now is not None else time.time()
    rows = _rows(store, now - hours * 3600 if hours else 0.0)
    # a pool needs its 30-minute state to be due (and a little settling) before it counts
    band = [r for r in rows if min_liq <= r["liq"] < max_liq and r["create_ts"] <= now - EXITS[30] - 120]
    name = f"${min_liq / 1000:g}-{max_liq / 1000:g}k" if math.isfinite(max_liq) else f"${min_liq / 1000:g}k+"
    if len(band) < MIN_TRAIN + 50:
        return {"band": name, "pools": len(band), "note": f"needs {MIN_TRAIN + 50}+ finished liquid pools in the band"}
    rng = np.random.default_rng(seed)
    fetch = fetch_states(store, helius, band, max_credits) if helius is not None else None
    ex = _exits(store, band)
    tested, blocks = _walk(rows, band)
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M")
    out: dict = {"band": name, "pools": len(band), "every pool in the data": len(rows),
                 "window": f"{fmt(band[0]['create_ts'])} .. {fmt(band[-1]['create_ts'])} UTC",
                 "share of all liquid pools": round(100 * len(band) / len(rows), 1),
                 "ceiling": _ceiling([r["ret"] for r in band]), "signals": _signals(band)[:TOP_SIGNALS],
                 "walk-forward blocks": blocks, "tested pools": len(tested)}
    filters = {"live rule (rw_pct < 50)": br._judge(tested, "skip:live", rng)}
    keys = sorted({k for r in tested for k in r if k.startswith("skip:") and k != "skip:live"})
    for k in keys:
        _, m, keep = k.split(":")
        filters[f"{m}, keep best {keep}%"] = br._judge(tested, k, rng)
    out["filters"] = filters
    cov = ex.pop("covered")
    out["later exits"] = {**ex, "covered": len(cov), "fetch": fetch}
    if len(cov) >= 20:
        out["later exits"]["by exit"] = _by_exit(cov, tested, keys, rng)
        out["conditional holds"] = _holds(cov, tested, rng)
        out["minute-30 models"] = _models30(cov, rng)
    out["note"] = (f"ideal trade = the scorer's (second ~6.5 in, minute 6 out, 6% cost, +300% cap, rug -100%); "
                   "minutes 10/15/30 = vault price at the pool's last tx by then vs the same fill, same cost and cap, "
                   "-100% once pulled or SOL < 10% of the peak so far ('no balance found' counts as -100%: possibly a "
                   "data gap); "
                   f"walk-forward in {BLOCK_H} h blocks, models fitted on pools created before each block (band "
                   f"models need {MIN_TRAIN}), keep cut-offs from the band's past {CUT_WINDOW_H} h of training scores; "
                   "signals are in-sample (whole band), so trust only those with the same sign in both halves and a "
                   "walk-forward filter that beats random keeps in most blocks")
    return out


def render(out: dict) -> str:
    if "filters" not in out:
        return json.dumps(out, indent=2)
    f = lambda v: "-" if v is None else f"{v:+.1f}"
    lines = [f"PRIZE IN THE {out['band']} BAND ({out['window']}, {out['pools']} pools = "
             f"{out['share of all liquid pools']}% of liquid pools)", "", "CEILING (ideal minute-6 trade)"]
    for label, s in out["ceiling"].items():
        if label == "money":
            continue
        lines.append(f"  {label:<38} n {s['n']:>5}  mean {f(s.get('mean_pct')):>7}%  median "
                     f"{f(s.get('median_pct')):>7}%  win {s.get('win_pct')}%  rug {s.get('rugged_pct')}%  "
                     f"+50% {s.get('big_pct')}%  $ at $100 {s.get('usd_at_100')}")
    m = out["ceiling"]["money"]
    lines += [f"  money at $100 a pool: total {m['total $ at $100']}, +50% wins {m['$ from +50% wins']}, rugs "
              f"{m['$ lost to rugs']}, everything else {m['$ from everything else']}", "",
              f"SINGLE SIGNALS (top {TOP_SIGNALS}; top - bottom quintile mean return, pts)",
              f"  {'signal':<26}{'all':>8}{'1st half':>10}{'2nd half':>10}{'same sign':>11}{'AUC rug':>9}"
              f"{'AUC +50%':>10}{'AUC win':>9}{'present':>9}"]
    for s in out["signals"]:
        g = lambda k: (s.get(k) or {}).get("top - bottom quintile (pts)")
        a = s.get("AUC", {})
        lines.append(f"  {s['signal'][:25]:<26}{f(g('all')):>8}{f(g('first half')):>10}{f(g('second half')):>10}"
                     f"{('yes' if s['same sign both halves'] else 'no'):>11}{str(a.get('rug')):>9}"
                     f"{str(a.get('+50%')):>10}{str(a.get('win')):>9}{str(a.get('present %')) + '%':>9}")
    lines += ["", f"WALK-FORWARD FILTERS ({out['walk-forward blocks']} blocks, {out['tested pools']} pools)",
              f"  {'filter':<40}{'kept n':>7}{'kept':>8}{'skip':>8}{'vs all':>8}{'gap 90%':>16}{'rand<':>7}"
              f"{'blocks':>9}{'kept $100':>11}"]
    for label, j in out["filters"].items():
        k, s = j["kept"], j["skipped"]
        ci = (j.get("kept minus skipped (pts)") or {}).get("ci90")
        lines.append(f"  {label[:39]:<40}{k.get('n', 0):>7}{f(k.get('mean_pct')):>8}{f(s.get('mean_pct')):>8}"
                     f"{f(j.get('kept minus every pool (pts)')):>8}"
                     f"{(f'{ci[0]:+.1f}..{ci[1]:+.1f}' if ci else '-'):>16}"
                     f"{j.get('random keeps worse than kept', '-'):>7}"
                     f"{j.get('blocks where kept beat every pool', '-'):>9}{k.get('usd_at_100', '-'):>11}")
    le = out["later exits"]
    lines += ["", f"LATER EXITS (ideal trade at pool minute 6 / 10 / 15 / 30 on the same {le['covered']} pools; "
              f"{le['alive covered']} of {le['alive at minute 6']} pools alive at minute 6 have every state)"]
    if le.get("fetch"):
        lines.append(f"  fetch: {le['fetch']}")
    if "by exit" not in le:
        lines.append("  too few pools with minute-10/15/30 states: run with --fetch")
    else:
        lines.append(f"  {'exit':<10}{'n':>6}{'mean':>8}{'median':>8}{'win':>6}{'rug':>6}{'+50%':>6}{'$ at $100':>11}"
                     f"{'no rugs':>9}{'vs min 6':>10}{'90%':>16}")
        for m, e in le["by exit"].items():
            s_, nr, g = e["every pool"], e["rugs removed"], e.get("vs minute 6 (pts, paired)")
            ci = g["ci90"] if g else None
            lines.append(f"  {'minute ' + str(m):<10}{s_['n']:>6}{f(s_.get('mean_pct')):>8}{f(s_.get('median_pct')):>8}"
                         f"{s_.get('win_pct'):>5}%{s_.get('rugged_pct'):>5}%{s_.get('big_pct'):>5}%"
                         f"{s_.get('usd_at_100'):>11}{f(nr.get('mean_pct')):>9}{f(g['mean'] if g else None):>10}"
                         f"{(f'{ci[0]:+.1f}..{ci[1]:+.1f}' if ci else '-'):>16}")
        for m, e in le["by exit"].items():
            if e.get("rug causes"):
                lines.append(f"  rug causes by minute {m}: " + ", ".join(f"{k} {v}" for k, v in e["rug causes"].items()))
        for m, e in le["by exit"].items():
            lines += ["", f"  FILTERS AT MINUTE {m} (same pools)",
                      f"  {'filter':<40}{'kept n':>7}{'kept':>8}{'skip':>8}{'vs all':>8}{'gap 90%':>16}{'rand<':>7}"
                      f"{'blocks':>9}{'kept $100':>11}"]
            for label, j in e["filters"].items():
                k, sk = j["kept"], j["skipped"]
                ci = (j.get("kept minus skipped (pts)") or {}).get("ci90")
                lines.append(f"  {label[:39]:<40}{k.get('n', 0):>7}{f(k.get('mean_pct')):>8}{f(sk.get('mean_pct')):>8}"
                             f"{f(j.get('kept minus every pool (pts)')):>8}"
                             f"{(f'{ci[0]:+.1f}..{ci[1]:+.1f}' if ci else '-'):>16}"
                             f"{j.get('random keeps worse than kept', '-'):>7}"
                             f"{j.get('blocks where kept beat every pool', '-'):>9}{k.get('usd_at_100', '-'):>11}")
    lines += _render30(out, f)
    lines += ["", "vs all = kept mean - every pool's mean; gap = kept - skipped with 90% interval; rand< = share of "
              "random keeps of the same size that did worse; blocks = blocks where kept beat every pool",
              out["note"]]
    return "\n".join(lines)


def _render30(out: dict, f) -> list[str]:
    lines = []
    ch = out.get("conditional holds")
    if ch:
        lines += ["", "CONDITIONAL HOLDS (buy at second 6; at minutes 6 / 10 / 15 keep holding only while alive and up "
                  "by the threshold, else sell there; same pools as LATER EXITS)"]
        for gname, g in ch.items():
            lines.append(f"  {gname}: {g['pools']} pools")
            if "rules" not in g:
                lines.append(f"    {g['note']}")
                continue
            lines.append(f"    {'rule':<36}{'mean':>8}{'median':>8}{'rug':>6}{'$ at $100':>11}{'vs min 6':>10}"
                         f"{'90%':>16}{'held %':>8}{'blocks':>9}")
            for label, s in g["rules"].items():
                gap = s.get("vs minute 6 (pts, paired)")
                ci = gap["ci90"] if gap else None
                lines.append(f"    {label[:35]:<36}{f(s.get('mean_pct')):>8}{f(s.get('median_pct')):>8}"
                             f"{s.get('rugged_pct'):>5}%{s.get('usd_at_100'):>11}{f(gap['mean'] if gap else None):>10}"
                             f"{(f'{ci[0]:+.1f}..{ci[1]:+.1f}' if ci else '-'):>16}"
                             f"{str(s.get('held to the end %', '-')):>8}{s.get('blocks beating minute 6', '-'):>9}")
    m30 = out.get("minute-30 models")
    if m30:
        lines += ["", f"MINUTE-30 MODELS (trained on the band's minute-30 outcome, walk-forward; {m30['blocks']} blocks, "
                  f"{m30['tested pools']} pools)"]
        if "note" in m30:
            lines.append(f"  {m30['note']}")
        else:
            lines.append("  AUC for a rug by minute 30: " + ", ".join(
                f"{m} {v}" for m, v in m30["AUC for a rug by minute 30 (riskiness; want > 0.5)"].items()))
            lines.append(f"  {'filter (minute-30 exit)':<38}{'kept n':>7}{'kept':>8}{'rug':>6}{'skip':>8}{'vs all':>8}"
                         f"{'gap 90%':>16}{'rand<':>7}{'blocks':>9}{'kept $100':>11}{'kept @6':>9}")
            for label, j in m30["filters (judged at minute 30)"].items():
                k, s = j["kept"], j["skipped"]
                ci = (j.get("kept minus skipped (pts)") or {}).get("ci90")
                lines.append(f"  {label[:37]:<38}{k.get('n', 0):>7}{f(k.get('mean_pct')):>8}"
                             f"{str(k.get('rugged_pct', '-')) + '%':>6}{f(s.get('mean_pct')):>8}"
                             f"{f(j.get('kept minus every pool (pts)')):>8}"
                             f"{(f'{ci[0]:+.1f}..{ci[1]:+.1f}' if ci else '-'):>16}"
                             f"{j.get('random keeps worse than kept', '-'):>7}"
                             f"{j.get('blocks where kept beat every pool', '-'):>9}{k.get('usd_at_100', '-'):>11}"
                             f"{f(j['kept at minute 6'].get('mean_pct')):>9}")
        lines.append("  many rules and models are tried here, so the best row is optimistic; trust a rule only if it "
                     "beats minute 6 in most blocks and a model only if it beats random keeps (rand< ~0.95+) in most blocks")
    return lines
