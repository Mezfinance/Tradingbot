"""Strategy 1 leverage backtest. Replays the shortable core trades (cex/s1_trades.json, perp on Binance/Bybit/Gate at
the listing hour) at gross leverage L = 1..6 and asks whether more leverage gives a better risk-adjusted result.

Leverage L scales every position: each short's notional = min(L x W x NAV, CAP_OI x open interest), total open
notional <= L x NAV (W = 10%, as the live book). Each short is backed by its own margin = notional / L (isolated, as
the trades sit on three venues that can't share collateral). It is liquidated when the move against it reaches
1/L - MM (maintenance margin MM, conservative for new listings), losing its margin plus LIQ_FEE of notional; at 1x
that caps a squeeze at about -100% of the margin. Cross-margin variant: all shorts share the fund's NAV and every
open short is closed when NAV < MM x open notional. Costs: 0.2% round trip (in the trade), impact 0.10 x sqrt(notional
/ 24h volume) per side, 7-day funding on notional. Marked to market hourly on each trade's hourly path.

Edge cases: 'full' = the recorded paths; 'half' = every path shifted so each trade earns half the average edge less
(r_h + h/168 x 0.5 x mean edge). Bootstrapped forward runs (Poisson arrivals, trades resampled) give the spread.
Output cex/s1_leverage.json and a table."""
import json, sys, time, numpy as np

W, CAP_OI, IMPACT = 0.10, 0.02, 0.10
MM, LIQ_FEE = 0.025, 0.01
LEVERS = [1, 1.5, 2, 3, 4, 5, 6]
SIZES = [1e6, 1e7]
H = 3600.0
SIMS = 300
FIXED = "--compound" not in sys.argv   # default: size on the starting capital, so leverage is not mixed up with compounding
RUIN = 0.5            # 'ruin' = NAV ever below half the start


def load():
    T = [t for t in json.load(open("cex/s1_trades.json")) if set(t["venues"]) & {"binance", "bybit", "gate"}]
    edge = float(np.mean([t["short"] for t in T]))
    for t in T:
        t["dense"] = dense(t["path"])
    return T, edge


def dense(path):
    """Move since entry at whole hours 0..168 (last known value carried forward)."""
    out, j, m = np.zeros(169), 0, 0.0
    for h in range(169):
        while j < len(path) and path[j][0] <= h:
            m = path[j][1]; j += 1
        out[h] = m
    return out


def path_at(d, h, shift=0.0):
    i = min(int(h), 168)
    return d[i] + shift * i / 168   # half edge: the price drifts up against the short by shift over 7 days


def run(trades, nav0, L, margin="isolated", shift=0.0, start=None, fixed=None):
    """Hourly replay. Returns (curve [(ts, nav)], stats)."""
    start = start or trades[0]["ts"]
    end = max(t["ts"] for t in trades) + 8 * 86400
    cash, opn, curve = nav0, [], []
    st = {"trades": 0, "liquidated": 0, "capped": 0, "notional": [], "gross": [], "wiped": False}
    ti, n = 0, len(trades)
    for now in np.arange(start, end, H):
        # open
        while ti < n and trades[ti]["ts"] <= now:
            t = trades[ti]; ti += 1
            if t["ts"] < now - H: continue
            nav = cash + sum(p["pnl"] for p in opn)
            if nav <= 0: continue
            base = nav0 if (FIXED if fixed is None else fixed) else nav   # fixed: size on the starting capital (profits paid out)
            room = L * min(base, nav) - sum(p["notional"] for p in opn)
            want, cap = L * W * min(base, nav), CAP_OI * t["oi"]
            size = min(want, cap, max(room, 0.0))
            if size < 100: continue
            st["trades"] += 1; st["capped"] += cap < want; st["notional"].append(size)
            imp = 2 * IMPACT * np.sqrt(size / t["vol24"])
            opn.append({"t": t, "notional": size, "margin": size / L, "imp": imp, "pnl": 0.0})
        # mark, liquidate, close
        for p in list(opn):
            t = p["t"]; h = (now - t["ts"]) / H
            move = path_at(t["dense"], h, shift)          # price change since entry (+ = against the short)
            p["pnl"] = -move * p["notional"]
            if margin == "isolated" and move >= 1.0 / L - MM:
                cash += -p["margin"] - LIQ_FEE * p["notional"]; opn.remove(p); st["liquidated"] += 1
                continue
            if h >= 168:
                r = t["short"] - shift + t["funding"] - p["imp"]   # final 7-day result, costs and funding included
                cash += r * p["notional"]; opn.remove(p)
        nav = cash + sum(p["pnl"] for p in opn)
        if margin == "cross" and opn and nav < MM * sum(p["notional"] for p in opn):
            cash = nav - LIQ_FEE * sum(p["notional"] for p in opn)
            st["liquidated"] += len(opn); opn = []
            nav = cash
        if nav <= 0:
            st["wiped"] = True; curve.append((now, 0.0)); break
        st["gross"].append(sum(p["notional"] for p in opn) / nav)
        curve.append((now, nav))
    return curve, st


def metrics(curve, nav0):
    ts = np.array([c[0] for c in curve]); navs = np.array([c[1] for c in curve])
    yrs = max((ts[-1] - ts[0]) / (365 * 86400), 1e-9)
    daily = navs[::24]
    rets = np.diff(daily) / daily[:-1] if len(daily) > 2 else np.array([0.0])
    vol = float(np.std(rets) * np.sqrt(365))
    down = rets[rets < 0]
    dvol = float(np.sqrt(np.mean(down ** 2)) * np.sqrt(365)) if len(down) else 0.0
    final = navs[-1]
    cagr = (final / nav0) ** (1 / yrs) - 1 if final > 0 else -1.0
    peak = np.maximum.accumulate(navs); dd = float((navs / peak - 1).min())
    monthly = daily[::30]
    worst_m = float(np.min(np.diff(monthly) / monthly[:-1])) if len(monthly) > 2 else 0.0
    mean_ann = float(np.mean(rets) * 365)
    return {"total": final / nav0 - 1, "cagr": cagr, "vol": vol, "max_dd": dd,
            "sharpe": mean_ann / vol if vol else None, "sortino": mean_ann / dvol if dvol else None,
            "calmar": cagr / -dd if dd < 0 else None, "worst_month": worst_m, "worst_day": float(rets.min()),
            "ruined": bool(navs.min() < RUIN * nav0), "years": yrs}


def mc(T, nav0, L, margin, shift, rng, years=1):
    span = T[-1]["ts"] - T[0]["ts"]; rate = len(T) / span
    out = []
    for _ in range(SIMS):
        k = rng.poisson(rate * years * 365 * 86400)
        tss = np.sort(rng.uniform(0, years * 365 * 86400, k)) + 1.9e9
        tr = [dict(T[i], ts=float(x)) for i, x in zip(rng.integers(0, len(T), k), tss)]
        if not tr:
            continue
        c, s = run(tr, nav0, L, margin, shift, start=1.9e9)
        m = metrics(c, nav0)
        out.append((m["total"], m["max_dd"], m["ruined"], s["liquidated"]))
    a = np.array(out, dtype=float)
    return {"ret_p10": float(np.percentile(a[:, 0], 10)), "ret_p50": float(np.percentile(a[:, 0], 50)),
            "ret_p90": float(np.percentile(a[:, 0], 90)), "dd_p50": float(np.percentile(a[:, 1], 50)),
            "dd_p90": float(np.percentile(a[:, 1], 10)), "p_loss": float(np.mean(a[:, 0] < 0)),
            "p_ruin": float(np.mean(a[:, 2])), "p_dd30": float(np.mean(a[:, 1] < -0.3)), "liq_mean": float(np.mean(a[:, 3]))}


def main():
    T, edge = load()
    print(f"{len(T)} trades {time.strftime('%Y-%m-%d', time.gmtime(T[0]['ts']))}..{time.strftime('%Y-%m-%d', time.gmtime(T[-1]['ts']))}, "
          f"mean 7-day short {edge * 100:+.1f}%, worst {min(t['short'] for t in T) * 100:+.0f}%, worst move against "
          f"{max(max(r for h, r in t['path']) for t in T) * 100:+.0f}%")
    print("sizing on the starting capital (profits paid out)" if FIXED else "sizing on current NAV (compounding)")
    res = {"params": {"fixed": FIXED, "W": W, "CAP_OI": CAP_OI, "IMPACT": IMPACT, "MM": MM, "LIQ_FEE": LIQ_FEE, "trades": len(T),
                      "edge": edge}, "replay": [], "mc": []}
    rng = np.random.default_rng(1)
    sims = "--mc" in sys.argv
    for nav0 in SIZES:
        for margin in ("isolated", "cross"):
            for case, shift in (("full", 0.0), ("half", 0.5 * edge)):
                print(f"\n== ${nav0:,.0f}, {margin} margin, {case} edge ==")
                print(f"{'L':>4} {'CAGR':>7} {'vol':>6} {'maxDD':>7} {'Sharpe':>6} {'Sortino':>7} {'Calmar':>6} "
                      f"{'worst mo':>8} {'liq':>4} {'capped':>6} {'avg gross':>9}" + ("   MC 1y: p10/p50/p90  DD p90  P(loss) P(ruin) P(DD>30%)" if sims else ""))
                for L in LEVERS:
                    c, s = run(T, nav0, L, margin, shift)
                    m = metrics(c, nav0)
                    row = {"nav0": nav0, "margin": margin, "case": case, "L": L, **m, "liquidated": s["liquidated"],
                           "trades": s["trades"], "capped": s["capped"], "avg_gross": float(np.mean(s["gross"])) if s["gross"] else 0}
                    res["replay"].append(row)
                    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"
                    line = (f"{L:>4} {m['cagr'] * 100:>6.0f}% {m['vol'] * 100:>5.0f}% {m['max_dd'] * 100:>6.0f}% {f(m['sharpe']):>6} "
                            f"{f(m['sortino']):>7} {f(m['calmar']):>6} {m['worst_month'] * 100:>7.0f}% {s['liquidated']:>4} {s['capped']:>6} "
                            f"{row['avg_gross']:>8.2f}x")
                    if sims and nav0 == SIZES[0]:
                        q = mc(T, nav0, L, margin, shift, rng)
                        res["mc"].append({"nav0": nav0, "margin": margin, "case": case, "L": L, **q})
                        line += (f"   {q['ret_p10'] * 100:+.0f}/{q['ret_p50'] * 100:+.0f}/{q['ret_p90'] * 100:+.0f}%  {q['dd_p90'] * 100:.0f}%"
                                 f"  {q['p_loss'] * 100:4.0f}%  {q['p_ruin'] * 100:4.0f}%  {q['p_dd30'] * 100:4.0f}%")
                    print(line, flush=True)
    json.dump(res, open("cex/s1_leverage.json", "w"), default=float)


if __name__ == "__main__":
    main()
