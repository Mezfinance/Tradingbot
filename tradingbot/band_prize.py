"""Is there a prize in one liquidity band if we filter it better? (11-08 study, read only, no Helius spend.)

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
4. minute 10: pools with cached hold-study states (`watch_launch_path`) get their minute-10 result too (only
   the rug-vs-win kept half, which the kept-only hold study fetched in full).
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
                rec = {"ret": r["ret"], "ret10": r.get("ret10"), "block": t, "create_ts": r["create_ts"],
                       "skip:live": None if r["rw_pct"] is None else r["rw_pct"] >= 100 - ll.RW_SKIP}
                for (m, k), c in cuts.items():
                    rec[f"skip:{m}:{k}"] = models[m](r["x"]) < c
                tested.append(rec)
        t += block
    return tested, blocks


def _minute10(store, band: list[dict]) -> None:
    """Adds `ret10` (minute-10 result) to band pools whose hold-study states are cached."""
    from tradingbot import hold_study as hs
    from tradingbot import launch_live as ll
    try:
        store.execute(ll.PATH_SCHEMA)
        have = {(r["pool"], r["sec"]): {"sol": r["sol"], "tok": r["tok"]}
                for r in store.query("SELECT pool, sec, sol, tok FROM watch_launch_path WHERE sec <= 600")}
    except Exception:  # noqa: BLE001 - optional part
        return
    cached = {pool for pool, _ in have}
    if not cached:
        return  # the hold study never ran here
    for r in band:
        # only the kept half: the kept-only hold study fetched every kept pool (a rug by minute 6 needs no state), so
        # the kept half is covered without picking; elsewhere only rugs would show up uncached
        if r["fill_px"] is None or r["fill_sol"] is None or r["rw_pct"] is None or r["rw_pct"] >= 50:
            continue
        if r["pool"] not in cached and not (r["ret"] <= -0.999):
            continue
        p = hs.path(r, have)
        r["ret10"] = p.get(600)


def study(store, hours: float | None = None, min_liq: float = MIN_LIQ, max_liq: float = MAX_LIQ,
          now: float | None = None, seed: int = 13) -> dict:
    now = now if now is not None else time.time()
    rows = _rows(store, now - hours * 3600 if hours else 0.0)
    band = [r for r in rows if min_liq <= r["liq"] < max_liq]
    name = f"${min_liq / 1000:g}-{max_liq / 1000:g}k" if math.isfinite(max_liq) else f"${min_liq / 1000:g}k+"
    if len(band) < MIN_TRAIN + 50:
        return {"band": name, "pools": len(band), "note": f"needs {MIN_TRAIN + 50}+ finished liquid pools in the band"}
    rng = np.random.default_rng(seed)
    _minute10(store, band)
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
    with10 = [r for r in tested if r.get("ret10") is not None]
    out["minute 10 (cached pools only)"] = {"pools": len(with10)}
    if len(with10) >= 20:
        t10 = [{**r, "ret": r["ret10"]} for r in with10]
        out["minute 10 (cached pools only)"].update({
            "minute 6 on the same pools": br._summary([r["ret"] for r in with10]),
            "minute 10": br._summary([r["ret10"] for r in with10]),
            "best filter at minute 10": _best(t10, keys, rng)})
    out["note"] = (f"ideal trade = the scorer's (second ~6.5 in, minute 6 out, 6% cost, +300% cap, rug -100%); "
                   f"walk-forward in {BLOCK_H} h blocks, models fitted on pools created before each block (band "
                   f"models need {MIN_TRAIN}), keep cut-offs from the band's past {CUT_WINDOW_H} h of training scores; "
                   "signals are in-sample (whole band), so trust only those with the same sign in both halves and a "
                   "walk-forward filter that beats random keeps in most blocks")
    return out


def _best(tested: list[dict], keys: list[str], rng) -> dict | None:
    best = None
    for k in keys:
        j = br._judge(tested, k, rng)
        if "note" in j:
            continue
        if best is None or j["kept"]["mean_pct"] > best[1]["kept"]["mean_pct"]:
            best = (k, j)
    return None if best is None else {"filter": best[0], **best[1]}


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
    m10 = out["minute 10 (cached pools only)"]
    lines += ["", f"MINUTE 10 (pools with cached hold-study states: {m10['pools']}; rug-vs-win kept half only)"]
    if "minute 10" in m10:
        for label in ("minute 6 on the same pools", "minute 10"):
            s = m10[label]
            lines.append(f"  {label:<28} n {s['n']}  mean {f(s.get('mean_pct'))}%  rug {s.get('rugged_pct')}%  "
                         f"$ at $100 {s.get('usd_at_100')}")
        b = m10.get("best filter at minute 10")
        if b:
            lines.append(f"  best walk-forward filter at minute 10: {b['filter'][5:]}: kept n {b['kept']['n']} mean "
                         f"{f(b['kept'].get('mean_pct'))}% vs every pool {f(b['every pool'].get('mean_pct'))}%, "
                         f"rand< {b.get('random keeps worse than kept')} (best of many, so optimistic)")
    lines += ["", "vs all = kept mean - every pool's mean; gap = kept - skipped with 90% interval; rand< = share of "
              "random keeps of the same size that did worse; blocks = blocks where kept beat every pool",
              out["note"]]
    return "\n".join(lines)
