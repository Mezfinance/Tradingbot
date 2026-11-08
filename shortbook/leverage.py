"""Live leverage profiles for MZF Short Only: what the book would have made at 1x, 2x, 3x, 4x ... on the same shorts.

The book itself trades at 1x (each short up to `trade_pct` of NAV, open shorts <= `gross_cap_pct` of NAV). A profile at
leverage L takes every short the book opened and scales it, from the same entry and exit prices and funding:
- notional = min(L x size, max(size, the short's open-interest cap)): a short that was already at the 2%-of-OI cap
  can't grow, one held back by the cap or by gross room grows up to L times;
- costs = notional x (round trip + 2 x impact_k x sqrt(notional / 24h volume)); funding scales with notional;
- isolated margin = notional / L; liquidated once the highest price seen since entry (`positions.max_px`, read every
  manage pass, so spikes between passes are missed) is 1/L - MM above the entry: the loss is the margin plus
  LIQ_FEE of notional, and the short stops counting for that profile.
Profits are not reinvested: each profile = starting cash + the sum of its shorts' results. NAV per profile is
stored hourly in `lev_equity` (from `Book.snapshot`), which gives drawdown and a Sharpe ratio."""
from __future__ import annotations
import json, math, statistics

MM = 0.025
LIQ_FEE = 0.01
DEFAULT_IMPACT = 0.005
MAX_IMPACT = 0.05


def ensure(store) -> None:
    cols = {r["name"] for r in store.q("PRAGMA table_info(positions)")}
    if "max_px" not in cols:
        store.x("ALTER TABLE positions ADD COLUMN max_px REAL")
    store.x("CREATE TABLE IF NOT EXISTS lev_equity (ts REAL, lev REAL, nav REAL, open_usd REAL, PRIMARY KEY (ts, lev))")


def levels(cfg) -> list[float]:
    out = []
    for x in str(cfg.leverage_profiles).split(","):
        try:
            v = float(x)
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    return sorted(set(out))


def position(p, L: float, cfg) -> dict:
    """One short at leverage L: notional, P&L now (open) or final (closed), liquidated or not."""
    data = json.loads(p["data"] or "{}")
    size, entry = p["size_usd"], p["entry_px"]
    cap = data.get("oi_cap_usd") or size
    notional = min(L * size, max(size, cap))
    vol = ((data.get("perps") or {}).get(p["venue"]) or {}).get("vol24_usd")
    impact = min(MAX_IMPACT, cfg.impact_k * math.sqrt(notional / vol)) if vol else DEFAULT_IMPACT
    fees = notional * (cfg.fee_round_trip_pct / 100 + 2 * impact)
    hi = max(x for x in (p["max_px"], p["last_px"], p["exit_px"], entry) if x)
    liquidated = hi / entry - 1 >= 1 / L - MM
    if liquidated:
        pnl = -notional / L - LIQ_FEE * notional
    else:
        px = p["exit_px"] if p["status"] == "closed" else p["last_px"]
        funding = (p["funding_usd"] or 0) * notional / size if size else 0
        pnl = notional * (entry - px) / entry + funding - fees
    return {"notional": notional, "pnl": pnl, "liquidated": liquidated,
            "open": p["status"] == "open" and not liquidated, "ret": pnl / (notional / L) if notional else 0}


def navs(book) -> dict[float, dict]:
    ensure(book.s)
    rows = book.s.q("SELECT * FROM positions")
    out = {}
    for L in levels(book.cfg):
        ps = [position(p, L, book.cfg) for p in rows]
        out[L] = {"nav": book.cfg.cash_usd + sum(x["pnl"] for x in ps),
                  "open_usd": sum(x["notional"] for x in ps if x["open"]),
                  "liquidated": sum(x["liquidated"] for x in ps), "positions": ps}
    return out


def snapshot(book, ts: float) -> None:
    for L, v in navs(book).items():
        book.s.x("INSERT OR REPLACE INTO lev_equity (ts, lev, nav, open_usd) VALUES (?, ?, ?, ?)", (ts, L, v["nav"], v["open_usd"]))


def track_high(book, p, price: float) -> None:
    if price and (p["max_px"] is None or price > p["max_px"]):
        book.s.x("UPDATE positions SET max_px=? WHERE id=?", (price, p["id"]))


def _curve_stats(points: list[tuple[float, float]], start: float) -> dict:
    if not points:
        return {"max_drawdown_pct": 0.0, "sharpe": None, "vol_pct": None}
    peak, dd = start, 0.0
    for _, v in points:
        peak = max(peak, v); dd = min(dd, v / peak - 1 if peak else 0)
    daily, last_day = [], None
    for ts, v in points:              # last value of each UTC day
        d = int(ts // 86400)
        if d != last_day:
            daily.append(v); last_day = d
        else:
            daily[-1] = v
    rets = [b / a - 1 for a, b in zip(daily, daily[1:]) if a > 0]
    sharpe = vol = None
    if len(rets) >= 5:
        sd = statistics.pstdev(rets)
        vol = sd * math.sqrt(365) * 100
        sharpe = (statistics.mean(rets) * 365) / (sd * math.sqrt(365)) if sd else None
    return {"max_drawdown_pct": round(dd * 100, 2), "sharpe": round(sharpe, 2) if sharpe is not None else None,
            "vol_pct": round(vol, 1) if vol is not None else None}


EQUITY_POINTS = 500


def profiles(book) -> dict:
    """Per leverage: NAV, return, drawdown, Sharpe (daily, needs 5+ days), liquidations, trades, mean and win of
    closed shorts on their margin, gross exposure, and the hourly curve."""
    ensure(book.s)
    start = book.cfg.cash_usd
    cur = navs(book)
    out = []
    for L, v in cur.items():
        pts = [(r["ts"], r["nav"]) for r in book.s.q("SELECT ts, nav FROM lev_equity WHERE lev=? ORDER BY ts", (L,))]
        closed = [x for x in v["positions"] if not x["open"]]
        rets = [x["ret"] for x in closed]
        step = max(1, len(pts) // EQUITY_POINTS)
        curve = [[t, round(n, 2)] for t, n in pts[::step]]
        out.append({"lev": L, "nav_usd": round(v["nav"], 2), "return_pct": round((v["nav"] / start - 1) * 100, 3),
                    "pnl_usd": round(v["nav"] - start, 2), "open_usd": round(v["open_usd"], 2),
                    "gross_x": round(v["open_usd"] / v["nav"], 3) if v["nav"] > 0 else None,
                    "shorts": len(v["positions"]), "closed": len(closed), "liquidated": v["liquidated"],
                    "mean_on_margin_pct": round(100 * sum(rets) / len(rets), 2) if rets else None,
                    "win_pct": round(100 * sum(r > 0 for r in rets) / len(rets)) if rets else None,
                    **_curve_stats(pts, start), "equity": curve})
    return {"start_usd": start, "mm_pct": MM * 100, "liq_fee_pct": LIQ_FEE * 100, "profiles": out,
            "backtest": BACKTEST}


# research/listings/s1_leverage_results.txt, $1M, sized on starting capital, isolated margin, half edge (the planning
# case): Sharpe / max drawdown / 1-year median / chance of a 30%+ drawdown in a year / chance NAV falls below half
BACKTEST = {"1": [1.27, -27, 43, 6, 0], "1.5": [1.43, -32, 64, 17, 0], "2": [1.60, -26, 95, 18, 1],
            "3": [1.50, -26, 95, 47, 3], "4": [1.40, -40, 107, 54, 6]}


def render(out: dict) -> str:
    lines = [f"LEVERAGE PROFILES (same shorts as the 1x book, isolated margin, liquidated at 1/L - {out['mm_pct']:.1f}%"
             f" above entry, profits not reinvested; start ${out['start_usd']:,.0f})",
             f"{'L':>4} {'NAV $':>12} {'return':>8} {'max DD':>7} {'Sharpe':>6} {'shorts':>6} {'closed':>6} {'liq':>4} "
             f"{'mean on margin':>14} {'win':>5} {'gross':>6}   backtest half edge: Sharpe / DD / 1y median / P(DD>30%)"]
    for p in out["profiles"]:
        b = out["backtest"].get(f"{p['lev']:g}")
        f = lambda x, s="": "-" if x is None else f"{x:+.1f}{s}"
        lines.append(f"{p['lev']:>4g} {p['nav_usd']:>12,.0f} {p['return_pct']:>+7.2f}% {p['max_drawdown_pct']:>6.1f}% "
                     f"{('-' if p['sharpe'] is None else format(p['sharpe'], '.2f')):>6} {p['shorts']:>6} {p['closed']:>6} "
                     f"{p['liquidated']:>4} {f(p['mean_on_margin_pct'], '%'):>14} {('-' if p['win_pct'] is None else str(p['win_pct']) + '%'):>5} "
                     f"{('-' if p['gross_x'] is None else format(p['gross_x'], '.2f') + 'x'):>6}"
                     + (f"   {b[0]:.2f} / {b[1]}% / +{b[2]}% / {b[3]}%" if b else ""))
    lines.append("Sharpe needs 5+ days of hourly NAV. Judge after 30+ closed shorts.")
    return "\n".join(lines)
