"""By-band rug-vs-win books against the sized book (`band-compare`, 11-08, read only).

Compares the band family (`launchrwband` flat $100, `launchrwband_sz` $250 in $5-15k pools) with `launchrw50_sz`
(and `launchrw50` flat $100) over the same window, by default from the band books' first trade:
1. real trades of each book: size-weighted result, $, median, win, rugged, 90% interval; split by liquidity band
   at second 4 ($5-15k / $15-25k / $25-40k / above $40k);
2. what each rule picked on the scorer's own finished liquid pools (ideal trades, so execution is left out):
   every pool, the rw50 picks (rw_pct < 50), the band picks (`launch_live.band_flags`), picked by both, band only,
   rw50 only, plus each book's $ result at its own sizing on those ideal trades;
3. per 6 h block: ideal mean of every pool, the rw50 picks and the band picks, and each book's real result;
4. execution: each book's real trades against the scorer's ideal on the same tokens.
Judged on the same pools, so the difference between the rules is selection, and the difference between ideal and
real is execution."""
from __future__ import annotations
import time
from datetime import datetime, timezone

import numpy as np

BOOKS = ("launchrwband", "launchrwband_sz", "launchrw50_sz", "launchrw50")
BANDS = (("$5-15k", 5_000.0, 15_000.0), ("$15-25k", 15_000.0, 25_000.0), ("$25-40k", 25_000.0, 40_000.0),
         ("above $40k", 40_000.0, float("inf")))
BLOCK_H = 6
STUCK_S = 1800
RUG = -0.999
BIG = 0.5
DRAWS = 2000


def band(liq) -> str | None:
    if liq is None:
        return None
    return next((n for n, lo, hi in BANDS if lo <= liq < hi), None)


def _fmt_ts(t) -> str | None:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M") if t else None


def _trades(store, journal, book: str, since: float, now: float, scorer: dict) -> list[dict]:
    """Real trades of `book` opened since `since`: return (open > STUCK_S = -100%), cost, liquidity and the scorer's
    ideal return for the token (latest scorer row created before the buy)."""
    out = []
    for p in journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ? AND opened_at < ?", (book, since, now)):
        if not p["cost_usd"]:
            continue
        if p["closed_at"]:
            ret = p["realized_usd"] / p["cost_usd"] - 1
        elif now - p["opened_at"] >= STUCK_S:
            ret = -1.0
        else:
            continue
        rows = [r for r in scorer.get(p["token"], []) if r["create_ts"] <= p["opened_at"]]
        r = rows[-1] if rows else None
        out.append({"ret": ret, "cost": p["cost_usd"], "ts": p["opened_at"], "token": p["token"],
                    "liq": r["liq_usd"] if r else None, "ideal": r["ret"] if r else None})
    return out


def _real(trades: list[dict], rng) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t["ret"] for t in trades]); c = np.array([t["cost"] for t in trades])
    out = {"n": len(r), "traded_usd": round(float(c.sum())), "pnl_usd": round(float((r * c).sum())),
           "size_weighted_pct": round(100 * float((r * c).sum() / c.sum()), 1),
           "median_pct": round(100 * float(np.median(r)), 1), "win_pct": round(100 * float(np.mean(r > 0))),
           "rugged_pct": round(100 * float(np.mean(r <= -0.8))), "big_win_pct": round(100 * float(np.mean(r >= BIG)))}
    if len(r) >= 5:
        idx = rng.integers(0, len(r), (DRAWS, len(r)))
        boots = (r[idx] * c[idx]).sum(1) / c[idx].sum(1)
        out["ci90"] = [round(100 * float(np.quantile(boots, q)), 1) for q in (0.05, 0.95)]
    return out


def _ideal(rows: list[dict], rng, size=None) -> dict:
    """Equal-weight ideal result of a set of scorer rows, and the $ result at a book's sizing (size(liq) -> $)."""
    rets = np.array([r["ret"] for r in rows], dtype=float)
    if not len(rets):
        return {"n": 0}
    out = {"n": len(rets), "mean_pct": round(100 * float(rets.mean()), 1),
           "median_pct": round(100 * float(np.median(rets)), 1), "win_pct": round(100 * float(np.mean(rets > 0))),
           "rugged_pct": round(100 * float(np.mean(rets <= RUG))), "big_win_pct": round(100 * float(np.mean(rets >= BIG))),
           "usd_at_100": round(float(100 * rets.sum()))}
    if len(rets) >= 5:
        boots = rets[rng.integers(0, len(rets), (DRAWS, len(rets)))].mean(1)
        out["ci90"] = [round(100 * float(np.quantile(boots, q)), 1) for q in (0.05, 0.95)]
    if size is not None:
        usd = np.array([size(r["liq_usd"]) or 0.0 for r in rows])
        if usd.sum():
            out["usd_at_book_size"] = round(float((usd * rets).sum()))
            out["size_weighted_pct"] = round(100 * float((usd * rets).sum() / usd.sum()), 1)
    return out


def _gap(a: list[float], b: list[float], rng) -> dict | None:
    if len(a) < 5 or len(b) < 5:
        return None
    a, b = np.array(a), np.array(b)
    d = a[rng.integers(0, len(a), (DRAWS, len(a)))].mean(1) - b[rng.integers(0, len(b), (DRAWS, len(b)))].mean(1)
    return {"pts": round(100 * float(a.mean() - b.mean()), 1),
            "ci90": [round(100 * float(np.quantile(d, q)), 1) for q in (0.05, 0.95)]}


def compare(store, journal, hours: float | None = None, now: float | None = None) -> dict:
    from tradingbot import launch_live as ll
    from tradingbot import model_book as mb
    now = now if now is not None else time.time()
    ll.ensure_schema(store)
    if hours:
        since = now - hours * 3600
    else:
        first = journal.query("SELECT MIN(opened_at) t FROM positions WHERE strategy IN (?, ?)", BOOKS[:2])
        # a minute before the first buy, so the pool it came from (created seconds earlier) is in the window
        since = first[0]["t"] - 60 if first and first[0]["t"] else now - 24 * 3600
    rows = [dict(r) for r in store.query(
        "SELECT pool, token_mint, create_ts, liq_usd, ret, rw_pct, rwb_pct FROM watch_launch_live "
        "WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND create_ts >= ? AND create_ts < ? ORDER BY create_ts",
        (since, now))]
    scorer: dict = {}
    for r in store.query("SELECT token_mint, create_ts, liq_usd, ret FROM watch_launch_live "
                         "WHERE token_mint IS NOT NULL AND create_ts >= ? ORDER BY create_ts", (since - 3600,)):
        scorer.setdefault(r["token_mint"], []).append(dict(r))
    for r in rows:
        r["rw50"] = r["rw_pct"] is not None and r["rw_pct"] < 100 - ll.RW_SKIP
        r["rwband"] = ll.band_flags(r["liq_usd"], r["rw_pct"], r["rwb_pct"])["rwband"]
        r["band"] = band(r["liq_usd"])
    sizes = {b: (lambda liq, b=b: mb.band_usd(b, liq) or 100.0) for b in BOOKS}
    trades = {b: _trades(store, journal, b, since, now, scorer) for b in BOOKS}
    rng = np.random.default_rng(0)

    # 1. real trades
    real = {}
    for b in BOOKS:
        t = trades[b]
        real[b] = {"all": _real(t, rng), **{n: _real([x for x in t if x["liq"] is not None and band(x["liq"]) == n], rng)
                                           for n, _, _ in BANDS},
                   "no scorer row": sum(1 for x in t if x["liq"] is None)}

    # 2. what each rule picked, ideal trades on the same pools
    groups = {"every liquid pool": rows,
              "rw50 picks (launchrw50 / _sz)": [r for r in rows if r["rw50"]],
              "band picks (launchrwband / _sz)": [r for r in rows if r["rwband"]],
              "picked by both": [r for r in rows if r["rw50"] and r["rwband"]],
              "band only": [r for r in rows if r["rwband"] and not r["rw50"]],
              "rw50 only": [r for r in rows if r["rw50"] and not r["rwband"]]}
    picks = {}
    for name, g in groups.items():
        e = {"all": _ideal(g, rng)}
        if name.startswith("rw50 picks"):
            e["at launchrw50_sz sizes"] = _ideal(g, rng, sizes["launchrw50_sz"])
        if name.startswith("band picks"):
            e["at launchrwband_sz sizes"] = _ideal(g, rng, sizes["launchrwband_sz"])
        for n, _, _ in BANDS:
            e[n] = _ideal([r for r in g if r["band"] == n], rng)
        picks[name] = e
    unranked = sum(1 for r in rows if r["band"] in ("$5-15k", "$15-25k") and r["rwb_pct"] is None)
    pick_gap = _gap([r["ret"] for r in groups["band picks (launchrwband / _sz)"]],
                    [r["ret"] for r in groups["rw50 picks (launchrw50 / _sz)"]], rng)

    # 3. per 6 h block
    blocks, start = [], since - since % (BLOCK_H * 3600)
    won = played = 0
    for t0 in np.arange(start, now, BLOCK_H * 3600):
        t1 = t0 + BLOCK_H * 3600
        g = [r for r in rows if t0 <= r["create_ts"] < t1]
        if not g:
            continue
        rw = [r["ret"] for r in g if r["rw50"]]; bd = [r["ret"] for r in g if r["rwband"]]
        e = {"block": _fmt_ts(t0), "pools": len(g), "every pool %": round(100 * float(np.mean([r["ret"] for r in g])), 1),
             "rw50 picks": len(rw), "rw50 ideal %": round(100 * float(np.mean(rw)), 1) if rw else None,
             "band picks": len(bd), "band ideal %": round(100 * float(np.mean(bd)), 1) if bd else None}
        for b in BOOKS:
            x = [y for y in trades[b] if t0 <= y["ts"] < t1]
            c = sum(y["cost"] for y in x)
            e[f"{b} real %"] = round(100 * sum(y["ret"] * y["cost"] for y in x) / c, 1) if c else None
            e[f"{b} $"] = round(sum(y["ret"] * y["cost"] for y in x)) if x else None
        if len(rw) >= 5 and len(bd) >= 5:
            played += 1; won += float(np.mean(bd)) > float(np.mean(rw))
        blocks.append(e)

    # 4. execution: real vs the scorer's ideal on the same tokens
    execution = {}
    for b in BOOKS:
        m = [x for x in trades[b] if x["ideal"] is not None]
        if m:
            execution[b] = {"matched": len(m), "real mean %": round(100 * float(np.mean([x["ret"] for x in m])), 1),
                            "ideal mean %": round(100 * float(np.mean([x["ideal"] for x in m])), 1),
                            "real minus ideal (pts)": round(100 * float(np.mean([x["ret"] - x["ideal"] for x in m])), 1),
                            "median gap (pts)": round(100 * float(np.median([x["ret"] - x["ideal"] for x in m])), 1)}
    return {"window": f"last {hours:g} h" if hours else "since the band books' first trade",
            "from": _fmt_ts(since), "to": _fmt_ts(now), "pools": len(rows), "real": real, "picks": picks,
            "band picks minus rw50 picks (ideal)": pick_gap,
            "blocks": blocks, "blocks band beat rw50 (ideal, 5+ picks each)": f"{won} of {played}",
            "execution": execution, "band pools without a band rank yet": unranked,
            "note": ("ideal = the scorer's trade (second ~6.5 in, minute 6 out, 6% cost, +300% cap, rug -100%) on "
                     "finished liquid pools; real trades open 30 min after the buy count as -100%; band rule: $5-15k "
                     "band rank < 80, $15-25k band rank < 30, $25-40k rw_pct < 50, above $40k never; rw50: rw_pct < 50 "
                     "in every band. Sizes: launchrw50_sz $250 / $25 / $100 (< $15k / $15-25k / above), "
                     "launchrwband_sz $250 below $15k else $100, the others $100.")}


def _row(label: str, d: dict, cols) -> str:
    def f(k):
        v = d.get(k)
        if v is None:
            return "-"
        if k in ("n", "traded_usd", "pnl_usd", "usd_at_100", "usd_at_book_size"):
            return f"{v:,.0f}"
        if k == "ci90":
            return f"{v[0]:+.1f}..{v[1]:+.1f}"
        if k.endswith("_pct") and k in ("win_pct", "rugged_pct", "big_win_pct"):
            return f"{v:.0f}%"
        return f"{v:+.1f}%"
    return f"  {label:34}" + "".join(f"{f(k):>{w}}" for k, w in cols)


def render(out: dict) -> str:
    L = [f"BAND FAMILY vs SZ ({out['window']}: {out['from']} .. {out['to']} UTC, {out['pools']} finished liquid pools)", ""]
    rc = (("n", 6), ("traded_usd", 10), ("pnl_usd", 9), ("size_weighted_pct", 9), ("ci90", 15), ("median_pct", 8),
          ("win_pct", 6), ("rugged_pct", 7), ("big_win_pct", 6))
    head = f"  {'':34}{'n':>6}{'traded $':>10}{'P&L $':>9}{'sized %':>9}{'90% CI':>15}{'median':>8}{'win':>6}{'rugged':>7}{'+50%':>6}"
    L += ["1. REAL TRADES (size-weighted = P&L / money traded)", head]
    for b, e in out["real"].items():
        L.append(_row(b, e["all"], rc))
        for n, _, _ in BANDS:
            if e[n].get("n"):
                L.append(_row(f"   {n}", e[n], rc))
    ic = (("n", 6), ("mean_pct", 8), ("ci90", 15), ("median_pct", 8), ("win_pct", 6), ("rugged_pct", 7),
          ("big_win_pct", 6), ("usd_at_100", 10), ("size_weighted_pct", 9), ("usd_at_book_size", 11))
    L += ["", "2. WHAT EACH RULE PICKED (the scorer's ideal trade on the same pools)",
          f"  {'':34}{'n':>6}{'mean':>8}{'90% CI':>15}{'median':>8}{'win':>6}{'rugged':>7}{'+50%':>6}{'$ at $100':>10}{'sized %':>9}{'$ at size':>11}"]
    for name, e in out["picks"].items():
        L.append(_row(name, e["all"], ic))
        for k in ("at launchrw50_sz sizes", "at launchrwband_sz sizes"):
            if k in e:
                L.append(_row(f"   {k}", e[k], ic))
        for n, _, _ in BANDS:
            if e[n].get("n"):
                L.append(_row(f"   {n}", e[n], ic))
    g = out.get("band picks minus rw50 picks (ideal)")
    if g:
        L.append(f"  band picks minus rw50 picks: {g['pts']:+.1f} pts (90% CI {g['ci90'][0]:+.1f}..{g['ci90'][1]:+.1f})")
    if out.get("band pools without a band rank yet"):
        L.append(f"  $5-25k pools without a band rank yet (the band rule can't keep them): {out['band pools without a band rank yet']}")
    L += ["", f"3. PER {BLOCK_H} H BLOCK (ideal %, then real size-weighted %; band beat rw50 in "
              f"{out['blocks band beat rw50 (ideal, 5+ picks each)']} blocks)",
          f"  {'block':12}{'pools':>6}{'every':>7}{'rw50 n':>7}{'rw50':>7}{'band n':>7}{'band':>7}" +
          "".join(f"{b[:14]:>16}" for b in BOOKS)]
    for e in out["blocks"]:
        p = lambda v: "-" if v is None else f"{v:+.1f}"
        L.append(f"  {e['block']:12}{e['pools']:>6}{p(e['every pool %']):>7}{e['rw50 picks']:>7}{p(e['rw50 ideal %']):>7}"
                 f"{e['band picks']:>7}{p(e['band ideal %']):>7}" + "".join(f"{p(e[f'{b} real %']):>16}" for b in BOOKS))
    L += ["", "4. EXECUTION (real trades vs the scorer's ideal on the same tokens, equal weight)"]
    for b, e in out["execution"].items():
        L.append(f"  {b:18} matched {e['matched']:>4}  real {e['real mean %']:+.1f}%  ideal {e['ideal mean %']:+.1f}%  "
                 f"gap {e['real minus ideal (pts)']:+.1f} pts (median {e['median gap (pts)']:+.1f})")
    L += ["", out["note"]]
    return "\n".join(L)
