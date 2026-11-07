"""Rug-vs-win results by pool size (11-08): eight liquidity bands at second 4.

sz's edge sits in pools under $15k and fades above $25k, so every measure here is split by the pool's liquidity at
the scorer's decision (second 4). Two views share the band list:

* `hour_bands(done, trades)`: one hour's sums per band, stored in each market_hours row (key `bands`) so any window
  can be added up exactly (`aggregate(rows)`); the dashboard's Market tab shows it over 6 / 24 / 72 h.
* `study(store, journal)`: the whole life (or a window) straight from the scorer table and the journal, with 90%
  intervals, the random-keep test and AUCs per band (CLI `rw-bands`).

Read only; nothing here trades.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

BANDS = (("$5-15k", 5_000.0, 15_000.0), ("$15-25k", 15_000.0, 25_000.0), ("$25-40k", 25_000.0, 40_000.0),
         ("$40-80k", 40_000.0, 80_000.0), ("$80-150k", 80_000.0, 150_000.0), ("$150-300k", 150_000.0, 300_000.0),
         ("$300-600k", 300_000.0, 600_000.0), ("$600k+", 600_000.0, float("inf")))
NAMES = tuple(b[0] for b in BANDS)
RUG = -0.999
BIG_WIN = 0.5
STUCK_S = 1800  # a real trade still open this long after its buy never sold: -100%
BOOKS = ("launchrw50_sz", "launchrw50")  # real trades shown per band (sz sized by band, launchrw50 flat $100)
TOTAL = "All bands"  # the running tally row aggregate() adds after the eight bands
KEYS = ("n", "ret_sum", "rugs", "big", "wins", "kept_n", "kept_sum", "kept_rugs", "kept_big", "kept_wins", "skip_n",
        "skip_sum", "sz_n", "sz_pnl", "sz_cost")


def band_of(liq: float | None) -> str | None:
    if liq is None:
        return None
    for name, lo, hi in BANDS:
        if lo <= liq < hi:
            return name
    return None


def _kept(p, skip: float) -> bool | None:
    return None if p.get("rw_pct") is None else p["rw_pct"] < 100 - skip


def hour_bands(done: list[dict], trades: list[tuple[float, float, float | None]], skip: float) -> dict:
    """Sums per band for one hour. `done` = finished liquid scorer rows (ret, liq_usd, rw_pct); `trades` = the sized
    book's real results as (return, cost, liquidity at second 4). Bands with nothing in them are left out."""
    out: dict = {}

    def slot(name):
        return out.setdefault(name, dict.fromkeys(KEYS, 0))
    for p in done:
        b = band_of(p.get("liq_usd"))
        if b is None:
            continue
        s, r = slot(b), p["ret"]
        s["n"] += 1
        s["ret_sum"] += r
        s["rugs"] += r <= RUG
        s["big"] += r >= BIG_WIN
        s["wins"] += r > 0
        k = _kept(p, skip)
        if k is True:
            s["kept_n"] += 1
            s["kept_rugs"] += r <= RUG
            s["kept_big"] += r >= BIG_WIN
            s["kept_wins"] += r > 0
            s["kept_sum"] += r
        elif k is False:
            s["skip_n"] += 1
            s["skip_sum"] += r
    for r, cost, liq in trades:
        b = band_of(liq)
        if b is None:
            continue
        s = slot(b)
        s["sz_n"] += 1
        s["sz_pnl"] += r * cost
        s["sz_cost"] += cost
    for s in out.values():
        for k in ("ret_sum", "kept_sum", "skip_sum", "sz_pnl", "sz_cost"):
            s[k] = round(s[k], 4)
    return out


def aggregate(rows: list[dict]) -> list[dict]:
    """market_hours rows (with `bands`) added up per band, plus an `All bands` total row: share of pools, ideal mean,
    rugged / +50% %, kept share, kept / skipped ideal means and gap, sz's real trades (size-weighted %, $ P&L and $
    traded), and running $ totals of the ideal trade at $100 a pool for every pool and for the kept half."""
    tot = {n: dict.fromkeys(KEYS, 0.0) for n in NAMES}
    for row in rows:
        for name, s in (row.get("bands") or {}).items():
            if name in tot:
                for k in KEYS:
                    tot[name][k] += s.get(k) or 0
    pools = sum(t["n"] for t in tot.values())
    tot[TOTAL] = {k: sum(tot[n][k] for n in NAMES) for k in KEYS}
    out = []
    for name in NAMES + (TOTAL,):
        t = tot[name]
        n = int(t["n"])
        mean = lambda s, c: round(100 * s / c, 1) if c else None
        kept, skipped = mean(t["kept_sum"], t["kept_n"]), mean(t["skip_sum"], t["skip_n"])
        out.append({"band": name, "pools": n, "share_pct": round(100 * n / pools, 1) if pools else None,
                    # every pool in the band (no filter), the ideal trade
                    "mean_pct": mean(t["ret_sum"], n), "win_pct": mean(t["wins"], n), "rugged_pct": mean(t["rugs"], n),
                    "big_pct": mean(t["big"], n), "all_100_usd": round(100 * t["ret_sum"]),
                    # the rug-vs-win kept half, same measures
                    "kept_n": int(t["kept_n"]), "kept_share_pct": mean(t["kept_n"], t["kept_n"] + t["skip_n"]),
                    "kept_pct": kept, "kept_win_pct": mean(t["kept_wins"], t["kept_n"]),
                    "kept_rugged_pct": mean(t["kept_rugs"], t["kept_n"]), "kept_big_pct": mean(t["kept_big"], t["kept_n"]),
                    "kept_100_usd": round(100 * t["kept_sum"]),
                    "kept_vs_all": round(kept - mean(t["ret_sum"], n), 1) if kept is not None and n else None,
                    "skipped_pct": skipped, "gap": round(kept - skipped, 1) if kept is not None and skipped is not None
                    else None,
                    "sz_n": int(t["sz_n"]), "sz_pct": mean(t["sz_pnl"], t["sz_cost"]),
                    "sz_pnl": round(t["sz_pnl"], 2), "sz_cost": round(t["sz_cost"], 2)})
    return out


# --- whole-life study (CLI rw-bands) ------------------------------------------------------------------------------
def _real_trades(store, journal, book: str, since: float, now: float) -> list[dict]:
    """Each real trade of `book` with its return (open > STUCK_S = -100%) and the scorer's liquidity for the token."""
    liq_by_mint: dict = {}
    for r in store.query("SELECT token_mint, create_ts, liq_usd FROM watch_launch_live WHERE liq_usd IS NOT NULL "
                         "AND token_mint IS NOT NULL ORDER BY create_ts"):
        liq_by_mint.setdefault(r["token_mint"], []).append((r["create_ts"], r["liq_usd"]))
    out = []
    for p in journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ?", (book, since)):
        if not p["cost_usd"]:
            continue
        if p["closed_at"]:
            ret = p["realized_usd"] / p["cost_usd"] - 1
        elif now - p["opened_at"] >= STUCK_S:
            ret = -1.0
        else:
            continue
        match = [liq for ts, liq in liq_by_mint.get(p["token"], []) if ts <= p["opened_at"]]
        out.append({"ret": ret, "cost": p["cost_usd"], "liq": match[-1] if match else None})
    return out


def _real_summary(trades: list[dict], rng) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t["ret"] for t in trades])
    c = np.array([t["cost"] for t in trades])
    out = {"n": len(trades), "pnl_usd": round(float((r * c).sum())), "traded_usd": round(float(c.sum())),
           "size_weighted_pct": round(100 * float((r * c).sum() / c.sum()), 1),
           "rugged_pct": round(100 * float(np.mean(r <= -0.8)))}
    if len(trades) >= 5:
        idx = [rng.integers(0, len(r), len(r)) for _ in range(2000)]
        boots = [float((r[i] * c[i]).sum() / c[i].sum()) for i in idx]
        out["ci90"] = [round(100 * float(np.quantile(boots, q)), 1) for q in (0.05, 0.95)]
    return out


def study(store, journal, hours: float | None = None, now: float | None = None) -> dict:
    """Per band, over the whole life of the scorer's finished liquid pools (or the last `hours`): every pool, the
    rug-vs-win kept and skipped halves (study rule, live `rw_pct`) with the gap, its 90% interval, random keeps of
    the same size and AUCs; launch4's top 2% / 5% for scale; and the rug-vs-win books' real trades."""
    from tradingbot import launch_live as ll
    now = now if now is not None else time.time()
    ll.ensure_schema(store)
    since = now - hours * 3600 if hours else 0.0
    rows = [dict(r) for r in store.query(
        "SELECT * FROM watch_launch_live WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND create_ts >= ? "
        "ORDER BY create_ts", (since,))]
    keep = lambda r: None if r["rw_pct"] is None else r["rw_pct"] < 100 - ll.RW_SKIP
    real = {b: _real_trades(store, journal, b, since, now) for b in BOOKS} if journal is not None else {}
    total = sum(1 for r in rows if band_of(r["liq_usd"]))
    bands = []
    for name in NAMES + (TOTAL,):
        rng = np.random.default_rng(0)
        g = [r for r in rows if band_of(r["liq_usd"]) == name or (name == TOTAL and band_of(r["liq_usd"]))]
        entry: dict = {"band": name, "pools": len(g), "share_pct": round(100 * len(g) / total, 1) if total else None}
        if g:
            scored = [r for r in g if keep(r) is not None]
            entry["every pool"] = ll._group(g, rng)
            entry["kept share of scored %"] = round(100 * sum(keep(r) for r in scored) / len(scored)) if scored else None
            test = ll._keep_test(g, keep, "rw", rng)
            for k in ("kept", "skipped", "kept minus skipped (pts)", "kept minus every pool (pts)", "random keeps of the same size",
                      "AUC (score = riskiness)", "blocks where kept beat skipped", "note"):
                if k in test:
                    entry[k] = test[k]
            entry["launch4 top 2%"] = ll._group([r for r in g if r.get("top2")], rng)
            entry["launch4 top 5%"] = ll._group([r for r in g if r.get("top5")], rng)
        for b, ts in real.items():
            entry[f"{b} real"] = _real_summary([t for t in ts if band_of(t["liq"]) == name
                                                or (name == TOTAL and band_of(t["liq"]))], rng)
        bands.append(entry)
    return {"window": f"last {hours:g} h" if hours else "whole life",
            "from": datetime.fromtimestamp(rows[0]["create_ts"], timezone.utc).strftime("%m-%d %H:%M") if rows else None,
            "pools": len(rows), "bands": bands,
            "unbanded real trades": {b: sum(1 for t in ts if band_of(t["liq"]) is None) for b, ts in real.items()},
            "note": ("liquidity = 2 x pool SOL x SOL price at second 4; ideal = the scorer's trade (second ~6.5 in, "
                     "minute 6 out, 6% cost, +300% cap, rug -100%); kept = rug-vs-win percentile < "
                     f"{100 - ll.RW_SKIP} among the past 6 h's pools (all bands ranked together); real trades still "
                     f"open {STUCK_S // 60} min after the buy count as -100%")}


def render(out: dict) -> str:
    def f(v, unit="%"):
        return "-" if v is None else f"{v:+.1f}{unit}" if unit in ("%", " pts") else f"{v}"
    lines = [f"RUG-VS-WIN BY POOL SIZE ({out['window']}, from {out['from']}, {out['pools']} finished liquid pools)", ""]
    hdr = (f"{'band':<10}{'pools':>6}{'share':>7} | {'every pool: mean':>16}{'median':>8}{'win%':>6}{'rug%':>6}"
           f"{'+50%':>6} | {'kept: n':>8}{'mean':>8}{'median':>8}{'win%':>6}{'rug%':>6}{'+50%':>6}{'vs all':>8}"
           f" | {'skip':>8}{'gap':>8}{'gap 90%':>15}{'rand<':>7} | {'sz n':>6}{'sz':>8}{'sz $':>8}{'rw50 n':>8}{'rw50':>8}")
    lines.append(hdr)
    for b in out["bands"]:
        e, k = b.get("every pool", {}), b.get("kept", {})
        g = b.get("kept minus skipped (pts)", {})
        ci = g.get("ci90")
        sz, r5 = b.get("launchrw50_sz real", {}), b.get("launchrw50 real", {})
        big = lambda x: (x.get("returns (% of trades)") or {}).get("+50% or more", "-")
        lines.append(
            f"{b['band']:<10}{b['pools']:>6}{(str(b['share_pct']) + '%') if b['share_pct'] is not None else '-':>7} | "
            f"{f(e.get('mean_pct')):>16}{f(e.get('median_pct')):>8}{e.get('win_pct', '-'):>6}{e.get('rugged_pct', '-'):>6}"
            f"{big(e):>6} | {k.get('n', 0):>8}{f(k.get('mean_pct')):>8}{f(k.get('median_pct')):>8}"
            f"{k.get('win_pct', '-'):>6}{k.get('rugged_pct', '-'):>6}{big(k):>6}"
            f"{f(b.get('kept minus every pool (pts)'), ' pts') if b.get('kept minus every pool (pts)') is not None else '-':>8} | "
            f"{f(b.get('skipped', {}).get('mean_pct')):>8}"
            f"{f(g.get('mean'), ' pts') if g else '-':>8}{(f'{ci[0]:+.1f}..{ci[1]:+.1f}' if ci else '-'):>15}"
            f"{b.get('random keeps of the same size', {}).get('share worse than kept', '-'):>7} | "
            f"{sz.get('n', 0):>6}{f(sz.get('size_weighted_pct')):>8}{sz.get('pnl_usd', '-'):>8}"
            f"{r5.get('n', 0):>8}{f(r5.get('size_weighted_pct')):>8}")
    lines += ["", "every pool = no filter, the ideal trade on every liquid pool in the band; kept = the rug-vs-win kept half; vs all = kept mean - every-pool mean; gap = kept - skipped "
              "(ideal); rand< = share of random keeps of the same size that did worse than kept; sz / rw50 = real "
              "trades, size-weighted (P&L / money traded)", out["note"]]
    return "\n".join(lines)
