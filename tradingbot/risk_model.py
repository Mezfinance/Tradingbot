"""Supermeme risk-adjusted return model (`risk-model`, 11-08, read only).

Part 1, per book, from its real trades (closed, plus open positions whose exit was blocked or a pool-minute book's
position still open 30 min after the buy = -100%): daily P&L on its starting capital, then annualised Sharpe and
Sortino (daily, sqrt 365), max drawdown, worst day, return / max drawdown, profit factor, trades per day, size-weighted mean per
trade with a 90% interval. Plus the correlation of the books' daily results and an equal-weight portfolio of them.

Part 2, per book: what per-trade size gives the best risk-adjusted growth. A year is simulated `SIMS` times by
drawing whole historical days with replacement (a bad day keeps all its trades). Each trade is sized at a share
`f` of current equity, capped at `model_book_liq_share` (2%) of the pool's liquidity at second 4; a trade bigger
than the one really made pays extra constant-product impact, about 4 x (extra size) / liquidity for the round trip
(in on the SOL side, out again), and loses at most its stake. Per capital and share: 1-year median and p10/p90,
median Sharpe, the 90th-percentile drawdown, chance of a 30%+ drawdown, chance equity falls below half (ruin),
median $ per year and the share of trades held back by the 2% cap. Trades with no liquidity on record (non-launch
books) keep their return at any size, so their big sizes read optimistic.

To keep it fast, each historical day's P&L is tabulated once against the trade size on a log grid and the
simulations interpolate in it."""
from __future__ import annotations
import time
from datetime import datetime, timezone

import numpy as np

MIN_TRADES = 30
STUCK_S = 1800
SIMS = 400
DAYS = 365
FRACTIONS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)   # % of equity per trade
CAPITALS = (10_000.0, 100_000.0, 1_000_000.0)
IMPACT_RT = 4.0      # round-trip impact ~ IMPACT_RT x size / liquidity (constant product, both legs)
GRID = np.concatenate([[0.0], np.geomspace(1.0, 2e6, 80)])
RUIN = 0.5
DRAWS = 2000


def _day(ts: float) -> int:
    return int(ts // 86400)


def trades(journal, store, book: str, since: float, now: float, pool_minute: bool) -> list[dict]:
    """Real trades of `book`: return, stake, close time, liquidity at second 4 (scorer row for the token)."""
    liq: dict = {}
    if store is not None:
        for r in store.query("SELECT token_mint, create_ts, liq_usd FROM watch_launch_live WHERE liq_usd IS NOT NULL "
                             "AND token_mint IS NOT NULL AND create_ts >= ? ORDER BY create_ts", (since - 3600,)):
            liq.setdefault(r["token_mint"], []).append((r["create_ts"], r["liq_usd"]))
    blocked = {(r["token"]): r["t"] for r in journal.query(
        "SELECT token, MIN(ts) t FROM events WHERE strategy=? AND kind='exit_blocked' AND ts >= ? GROUP BY token",
        (book, since))}
    out = []
    for p in journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ? AND opened_at < ?", (book, since, now)):
        if not p["cost_usd"]:
            continue
        if p["closed_at"]:
            ret, ts = p["realized_usd"] / p["cost_usd"] - 1, p["closed_at"]
        elif p["token"] in blocked and blocked[p["token"]] >= p["opened_at"]:
            ret, ts = -1.0, blocked[p["token"]]
        elif pool_minute and now - p["opened_at"] >= STUCK_S:
            ret, ts = -1.0, p["opened_at"] + STUCK_S
        else:
            continue
        m = [l for t, l in liq.get(p["token"], []) if t <= p["opened_at"]]
        out.append({"ret": max(-1.0, ret), "cost": p["cost_usd"], "ts": ts, "liq": m[-1] if m else None,
                    "opened": p["opened_at"]})
    return sorted(out, key=lambda t: t["ts"])


def _daily(ts_list: list[dict], first: int, last: int) -> np.ndarray:
    d = np.zeros(last - first + 1)
    for t in ts_list:
        d[_day(t["ts"]) - first] += t["ret"] * t["cost"]
    return d


def _curve_stats(daily_usd: np.ndarray, capital: float) -> dict:
    eq = capital + np.cumsum(daily_usd)
    prev = np.concatenate([[capital], eq[:-1]])
    r = np.where(prev > 0, daily_usd / prev, 0.0)
    sd = r.std()
    down = r[r < 0]
    dsd = float(np.sqrt(np.mean(down ** 2))) if len(down) else 0.0
    peak = np.maximum.accumulate(np.concatenate([[capital], eq]))
    dd = float((np.concatenate([[capital], eq]) / peak - 1).min())
    total = eq[-1] / capital - 1
    return {"days": len(r), "return_pct": round(100 * total, 1),
            "sharpe": round(float(r.mean() / sd * np.sqrt(365)), 2) if sd else None,
            "sortino": round(float(r.mean() / dsd * np.sqrt(365)), 2) if dsd else None,
            "max_drawdown_pct": round(100 * dd, 1), "worst_day_pct": round(100 * float(r.min()), 1),
            "best_day_pct": round(100 * float(r.max()), 1),
            "return_per_dd": round(total / -dd, 2) if dd <= -0.005 else None}


def peak_open(ts: list[dict]) -> int:
    """Most positions open at the same time (a share of equity per trade needs share x this <= 100%)."""
    ev = sorted([(t["opened"], 1) for t in ts] + [(t["ts"], -1) for t in ts], key=lambda e: (e[0], e[1]))
    n = best = 0
    for _, d in ev:
        n += d; best = max(best, n)
    return best


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(len(x) - 1, int(np.searchsorted(cw, q * cw[-1])))])


def book_stats(ts: list[dict], capital: float, rng, per_trade: list[dict] | None = None) -> dict:
    """Daily measures from `ts`; per-trade measures (mean, median, win, rugged, profit factor, interval) from
    `per_trade` when given (the sampled history's own trades, each weighted by `w` = pools it stands for)."""
    first, last = _day(ts[0]["ts"]), _day(ts[-1]["ts"])
    daily = _daily(ts, first, last)
    pt = per_trade if per_trade is not None else ts
    r = np.array([t["ret"] for t in pt]); c = np.array([t["cost"] for t in pt])
    w = np.array([t.get("w", 1.0) for t in pt])
    rc = r * c * w
    gains, losses = float(rc[r > 0].sum()), float(-rc[r < 0].sum())
    ct = np.array([t["cost"] for t in ts])
    out = {"trades": len(ts), "trades_per_day": round(len(ts) / max(1, last - first + 1), 1),
           "pnl_usd": round(float(daily.sum())), "traded_usd": round(float(ct.sum())),
           "mean_trade_pct": round(100 * float(rc.sum() / (c * w).sum()), 2),
           "median_trade_pct": round(100 * (float(np.median(r)) if np.all(w == 1.0) else _wquantile(r, w, 0.5)), 1),
           "win_pct": round(100 * float(w[r > 0].sum() / w.sum())),
           "rugged_pct": round(100 * float(w[r <= -0.8].sum() / w.sum())),
           "profit_factor": round(gains / losses, 2) if losses else None, "peak_open": peak_open(ts),
           **_curve_stats(daily, capital)}
    if per_trade is not None:
        out["sampled_trades"] = len(pt)
    chunk = max(1, min(DRAWS, 20_000_000 // max(1, len(r))))   # keeps the index matrix under ~160 MB
    p = None if np.all(w == 1.0) else w / w.sum()
    boots = []
    for i in range(0, DRAWS, chunk):
        m = min(chunk, DRAWS - i)
        idx = rng.integers(0, len(r), (m, len(r))) if p is None else rng.choice(len(r), (m, len(r)), p=p)
        boots.append((r[idx] * c[idx]).sum(1) / c[idx].sum(1))
    boots = np.concatenate(boots)
    out["mean_trade_ci90"] = [round(100 * float(np.quantile(boots, q)), 2) for q in (0.05, 0.95)]
    return out


def day_tables(ts: list[dict], liq_share: float) -> tuple[np.ndarray, np.ndarray]:
    """Per historical day (including days with no trade): P&L when the book's AVERAGE trade is x, for every x in
    GRID, and the share of that day's trades held back by the liquidity cap at x. Each trade keeps its own size
    relative to the average (11-08: every trade used to get the same size, which threw away a sized book's sizing:
    launchrw50_sz's $250 thin-pool / $25 mid-pool split is its whole edge, and at one size it read as a loser)."""
    first, last = _day(ts[0]["ts"]), _day(ts[-1]["ts"])
    pnl = np.zeros((last - first + 1, len(GRID))); capped = np.zeros_like(pnl); count = np.zeros(last - first + 1)
    ref = float(np.mean([t["cost"] for t in ts])) or 1.0
    for t in ts:
        d = _day(t["ts"]) - first
        count[d] += 1
        want = GRID * (t["cost"] / ref)
        if t["liq"]:
            cap = liq_share * t["liq"]
            s = np.minimum(want, cap)
            ret = np.maximum(-1.0, t["ret"] - IMPACT_RT * (s - t["cost"]) / t["liq"])
            capped[d] += want > cap
        else:
            s, ret = want, np.full(len(GRID), t["ret"])
        pnl[d] += s * ret
    with np.errstate(invalid="ignore", divide="ignore"):
        capped = np.where(count[:, None] > 0, capped / np.maximum(count[:, None], 1), 0.0)
    return pnl, capped


def _interp(table: np.ndarray, rows: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, GRID[-1])
    j = np.clip(np.searchsorted(GRID, x), 1, len(GRID) - 1)
    w = (x - GRID[j - 1]) / (GRID[j] - GRID[j - 1])
    return table[rows, j - 1] * (1 - w) + table[rows, j] * w


def simulate(pnl: np.ndarray, capped: np.ndarray, capital: float, frac: float, rng, sims: int = SIMS,
             days: int = DAYS) -> dict:
    eq = np.full(sims, capital)
    peak = eq.copy(); dd = np.zeros(sims); rets = np.zeros((sims, days)); cap_share = np.zeros(sims)
    for d in range(days):
        rows = rng.integers(0, len(pnl), sims)
        x = np.where(eq > 0, frac / 100 * eq, 0.0)
        p = _interp(pnl, rows, x)
        cap_share += _interp(capped, rows, x)
        new = np.maximum(eq + p, 0.0)
        rets[:, d] = np.where(eq > 0, (new - eq) / np.where(eq > 0, eq, 1), 0.0)
        eq = new
        peak = np.maximum(peak, eq); dd = np.minimum(dd, eq / peak - 1)
    total = eq / capital - 1
    sd = rets.std(1)
    sharpe = np.where(sd > 0, rets.mean(1) / np.where(sd > 0, sd, 1) * np.sqrt(365), 0.0)
    growth = np.log(np.maximum(eq, 1e-9) / capital)
    return {"capital": capital, "pct_per_trade": frac,
            "ret_p10": round(100 * float(np.quantile(total, 0.1)), 1),
            "ret_median": round(100 * float(np.median(total)), 1),
            "ret_p90": round(100 * float(np.quantile(total, 0.9)), 1),
            "usd_per_year_median": round(float(np.median(eq - capital))),
            "sharpe_median": round(float(np.median(sharpe)), 2),
            "dd_p90": round(100 * float(np.quantile(dd, 0.1)), 1),
            "p_dd30": round(100 * float(np.mean(dd <= -0.3))), "p_ruin": round(100 * float(np.mean(eq < RUIN * capital))),
            "p_loss": round(100 * float(np.mean(total < 0))), "median_log_growth": round(float(np.median(growth)), 3),
            "capped_pct": round(100 * float(np.mean(cap_share / days)))}


def _best(rows: list[dict]) -> dict:
    """Per capital: the share with the highest median log growth (Kelly-like), and the highest return per unit of
    bad-case drawdown (median return / |90th-percentile drawdown|) among shares with ruin under 5%."""
    out = {}
    for cap in sorted({r["capital"] for r in rows}):
        g = [r for r in rows if r["capital"] == cap]
        g = [r for r in g if not r.get("cash_short")] or g
        growth = max(g, key=lambda r: r["median_log_growth"])
        safe = [r for r in g if r["p_ruin"] < 5 and r["dd_p90"] < 0]
        calm = max(safe, key=lambda r: r["ret_median"] / -r["dd_p90"]) if safe else None
        out[f"${cap:,.0f}"] = {"best growth": growth["pct_per_trade"],
                               "best return per drawdown (ruin < 5%)": calm["pct_per_trade"] if calm else None}
    return out


def model(journal, store, cfg, books: list[str] | None = None, hours: float | None = None,
          now: float | None = None, sims: int = SIMS) -> dict:
    from tradingbot import model_book as mb
    now = now if now is not None else time.time()
    since = now - hours * 3600 if hours else 0.0
    books = books or mb.active_books(cfg)
    capital = cfg.model_book_cash_usd
    rng = np.random.default_rng(0)
    per, skipped, daily = {}, {}, {}
    for b in books:
        ts = trades(journal, store, b, since, now, b in mb.POOL_MINUTE_EXIT)
        if len(ts) < MIN_TRADES:
            skipped[b] = len(ts)
            continue
        st = book_stats(ts, capital, rng)
        pnl, capped = day_tables(ts, cfg.model_book_liq_share)
        sims_out = [simulate(pnl, capped, c, f, np.random.default_rng(1), sims=sims) for c in CAPITALS for f in FRACTIONS]
        st["actual_trade_usd"] = round(float(np.mean([t["cost"] for t in ts])), 2)
        st["actual_pct_of_capital"] = round(100 * st["actual_trade_usd"] / capital, 2)
        st["with_liquidity_pct"] = round(100 * sum(1 for t in ts if t["liq"]) / len(ts))
        for r in sims_out:
            r["cash_short"] = r["pct_per_trade"] * st["peak_open"] > 100   # more open at once than the book holds
        per[b] = {"stats": st, "sizing": sims_out, "best": _best(sims_out)}
        daily[b] = {_day(t["ts"]) for t in ts}, ts
    corr, portfolio = None, None
    if len(per) >= 2:
        first = min(_day(v[1][0]["ts"]) for v in daily.values()); last = max(_day(v[1][-1]["ts"]) for v in daily.values())
        series = {b: _daily(v[1], first, last) / capital for b, v in daily.items()}
        names = list(series)
        m = np.corrcoef(np.array([series[b] for b in names]))
        corr = {a: {b: round(float(m[i, j]), 2) for j, b in enumerate(names)} for i, a in enumerate(names)}
        port = np.mean([series[b] for b in names], axis=0) * capital
        portfolio = {"books": len(names), **_curve_stats(port, capital),
                     "mean single-book sharpe": round(float(np.mean([per[b]["stats"]["sharpe"] or 0 for b in names])), 2)}
    return {"window": f"last {hours:g} h" if hours else "whole life",
            "built": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "capital": capital, "liq_share": cfg.model_book_liq_share, "books": per,
            "too few trades": skipped, "correlation": corr, "equal_weight_portfolio": portfolio,
            "note": ("Sharpe/Sortino from daily P&L on each book's starting capital (sqrt 365); stuck trades count "
                     "-100%; sizing simulation: whole historical days drawn with replacement, average trade = share of "
                     f"equity (each trade keeps its size relative to the book's average), capped at {100 * cfg.model_book_liq_share:g}% of the pool, extra round-trip impact "
                     f"{IMPACT_RT:g} x extra size / liquidity, {sims} one-year paths. Few days of history make the "
                     "year paths repeat the same days: read them as the shape of the trade-off, not a forecast.")}


def render(out: dict) -> str:
    L = [f"SUPERMEME RISK MODEL ({out['window']}, built {out['built']} UTC; capital ${out['capital']:,.0f} per book)", "",
         "1. RISK-ADJUSTED RESULTS (real trades, daily P&L on the book's capital)",
         f"  {'book':20}{'trades':>7}{'/day':>6}{'return':>8}{'Sharpe':>7}{'Sortino':>8}{'max DD':>8}{'worst day':>10}"
         f"{'ret/DD':>7}{'PF':>6}{'mean/trade':>11}{'90% CI':>15}{'rugged':>7}{'days':>5}"]
    f = lambda v, d=2: "-" if v is None else (">99" if v > 99 else "<-99" if v < -99 else f"{v:.{d}f}")
    ranked = sorted(out["books"].items(), key=lambda kv: -(kv[1]["stats"]["sharpe"] or -99))
    for b, e in ranked:
        s = e["stats"]; ci = s["mean_trade_ci90"]
        L.append(f"  {b[:20]:20}{s['trades']:>7}{s['trades_per_day']:>6.0f}{s['return_pct']:>+7.1f}%{f(s['sharpe']):>7}"
                 f"{f(s['sortino']):>8}{s['max_drawdown_pct']:>7.1f}%{s['worst_day_pct']:>9.1f}%{f(s['return_per_dd']):>7}"
                 f"{f(s['profit_factor']):>6}{s['mean_trade_pct']:>+10.2f}%{f'{ci[0]:+.1f}..{ci[1]:+.1f}':>15}"
                 f"{s['rugged_pct']:>6}%{s['days']:>5}")
    if out["too few trades"]:
        L.append("  fewer than %d trades: %s" % (MIN_TRADES, ", ".join(f"{b} ({n})" for b, n in out["too few trades"].items())))
    if out["equal_weight_portfolio"]:
        p = out["equal_weight_portfolio"]
        L.append(f"  equal-weight portfolio of {p['books']} books: Sharpe {f(p['sharpe'])} (single books average "
                 f"{f(p['mean single-book sharpe'])}), max DD {p['max_drawdown_pct']:.1f}%, worst day {p['worst_day_pct']:.1f}%")
    if out["correlation"]:
        names = list(out["correlation"])
        L += ["", "  correlation of daily results", "  " + " " * 20 + "".join(f"{n[:9]:>10}" for n in names)]
        for a in names:
            L.append(f"  {a[:20]:20}" + "".join(f"{out['correlation'][a][b]:>10.2f}" for b in names))
    L += ["", "2. TRADE SIZE (average trade as a share of equity, each trade keeping its size relative to the average, capped at %g%% of the pool; 1-year bootstrap of real days)"
          % (100 * out["liq_share"])]
    for b, e in ranked:
        s = e["stats"]
        L += ["", f"  {b}: average trade today ${s['actual_trade_usd']:,.0f} = {s['actual_pct_of_capital']:.2f}% of capital; "
                  f"{s['with_liquidity_pct']}% of trades have pool liquidity on record; best: " +
              "; ".join(f"{k} growth {v['best growth']}% / per-DD {v['best return per drawdown (ruin < 5%)']}%"
                        for k, v in e["best"].items()),
              f"  {'capital':>11}{'%/trade':>8}{'p10':>8}{'median':>9}{'p90':>9}{'$/yr median':>13}{'Sharpe':>7}"
              f"{'DD p90':>8}{'DD>30%':>7}{'ruin':>6}{'loss':>6}{'capped':>7}"]
        L[-2] += f"; up to {s['peak_open']} positions open at once (* = share x that > 100%: needs more cash than the book has)"
        for r in e["sizing"]:
            L.append(f"  {r['capital']:>11,.0f}{r['pct_per_trade']:>7.2f}%{r['ret_p10']:>+7.0f}%{r['ret_median']:>+8.0f}%"
                     f"{r['ret_p90']:>+8.0f}%{r['usd_per_year_median']:>13,.0f}{r['sharpe_median']:>7.2f}{r['dd_p90']:>7.0f}%"
                     f"{r['p_dd30']:>6}%{r['p_ruin']:>5}%{r['p_loss']:>5}%{r['capped_pct']:>6}%" + (" *" if r.get("cash_short") else ""))
    L += ["", out["note"]]
    return "\n".join(L)
