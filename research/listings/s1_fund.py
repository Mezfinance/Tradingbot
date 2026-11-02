"""Strategy 1 fund model. Replays the shortable core trades (cex/s1_trades.json, venues binance/bybit/gate at the
listing hour) in time order for each fund size, then a forward Monte Carlo.
Rules: each trade = min(W x NAV, CAP_OI x the token's Binance open interest); total open shorts <= GROSS x NAV;
cost = 0.2% round trip (in the trade) + market impact 0.10 x sqrt(size / 24h perp volume) per side; funding as
recorded (7-day Binance funding, group median when unknown); no stop. Daily mark-to-market on hourly paths.
Writes cex/s1_fund.json. Fund fees: 2% management + 20% performance over a high-water mark, charged yearly."""
import json, numpy as np, time
SIZES = [1e5, 2.5e5, 5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8]
W, CAP_OI, GROSS, IMPACT = 0.10, 0.02, 1.0, 0.10
T = [t for t in json.load(open("cex/s1_trades.json")) if set(t["venues"]) & {"binance", "bybit", "gate"}]
def run(trades, nav0, haircut=1.0, start=None, fixed=False):
    start = start or trades[0]["ts"]
    end = max(t["ts"] for t in trades) + 8 * 86400
    days = np.arange(start, end, 86400.0)
    cash, opn, curve, stats = nav0, [], [], {"n": 0, "capped": 0, "size": [], "impact": [], "deployed": []}
    ti = 0
    for d in days:
        # close trades whose 7 days are over
        for p in [p for p in opn if p["t"]["ts"] + 168 * 3600 <= d]:
            t = p["t"]; r = t["short"] * haircut + t["funding"] - p["imp"]
            cash += p["size"] * r; opn.remove(p)
        def mtm():
            v = 0.0
            for p in opn:
                h = (d - p["t"]["ts"]) / 3600; m = 0.0
                for hh, mm in p["t"]["path"]:
                    if hh <= h: m = mm
                v += p["size"] * (-m)
            return v
        nav = cash + mtm()
        while ti < len(trades) and trades[ti]["ts"] < d + 86400:
            t = trades[ti]; ti += 1
            if t["ts"] < d: continue
            base = nav0 if fixed else nav
            room = GROSS * base - sum(p["size"] for p in opn)
            want = W * base; cap = CAP_OI * t["oi"]
            size = min(want, cap, max(room, 0))
            if size <= 0: continue
            stats["capped"] += cap < want; stats["n"] += 1; stats["size"].append(size)
            imp = 2 * IMPACT * np.sqrt(size / t["vol24"]); stats["impact"].append(imp)
            opn.append({"t": t, "size": size, "imp": imp})
        stats["deployed"].append(sum(p["size"] for p in opn) / nav if nav > 0 else 0)
        curve.append((d, nav))
    return curve, stats
def fees(curve, nav0):
    """2/20 yearly with a high-water mark; returns the investor's curve."""
    out, hwm, paid, base = [], nav0, 0.0, nav0
    last = curve[0][0]
    for d, nav in curve:
        v = nav - paid
        if d - last >= 365 * 86400:
            mgmt = 0.02 * v; perf = 0.2 * max(0, v - mgmt - hwm)
            paid += mgmt + perf; v -= mgmt + perf; hwm = max(hwm, v); last = d
        out.append((d, v))
    return out
def summ(curve, nav0):
    navs = np.array([n for d, n in curve]); yrs = (curve[-1][0] - curve[0][0]) / (365 * 86400)
    peak = np.maximum.accumulate(navs); dd = (navs / peak - 1).min()
    return {"final": navs[-1], "total": navs[-1] / nav0 - 1, "cagr": (navs[-1] / nav0) ** (1 / yrs) - 1, "max_dd": dd, "years": yrs}
res = {"params": {"W": W, "CAP_OI": CAP_OI, "GROSS": GROSS, "IMPACT": IMPACT, "trades": len(T),
                  "from": time.strftime("%Y-%m-%d", time.gmtime(T[0]["ts"])), "to": time.strftime("%Y-%m-%d", time.gmtime(T[-1]["ts"] + 7 * 86400))},
       "replay": {}, "mc": {}}
print(res["params"])
print(f"{'fund':>12} {'trades':>6} {'capped':>6} {'med size':>10} {'impact':>7} {'deployed':>8} {'total':>8} {'CAGR':>7} {'maxDD':>7} {'CAGR 2/20':>9} {'CAGR @50%edge':>13}")
for nav0 in SIZES:
    c, s = run(T, nav0); sm = summ(c, nav0); sf = summ(fees(c, nav0), nav0)
    c2, _ = run(T, nav0, haircut=0.5); s2 = summ(c2, nav0)
    q = {time.strftime("%Y-%m", time.gmtime(d)): n for d, n in c}
    res["replay"][nav0] = {"gross": sm, "net_2_20": sf, "half_edge": s2, "trades": s["n"], "capped": s["capped"],
                           "median_size": float(np.median(s["size"])), "median_impact": float(np.median(s["impact"])),
                           "avg_deployed": float(np.mean(s["deployed"])), "monthly": q}
    print(f"{nav0:>12,.0f} {s['n']:>6} {s['capped']:>6} {np.median(s['size']):>10,.0f} {np.median(s['impact'])*100:>6.2f}% {np.mean(s['deployed'])*100:>7.0f}% "
          f"{sm['total']*100:>7.0f}% {sm['cagr']*100:>6.0f}% {sm['max_dd']*100:>6.0f}% {sf['cagr']*100:>8.0f}% {s2['cagr']*100:>12.0f}%")
# Profits paid out (sizing on the starting capital): yearly profit in $ and %, full and half edge
res["paid_out"] = {}
print(f"\n{'fund':>12} {'yearly profit (full edge)':>26} {'(half edge)':>22}  worst month (full)")
for nav0 in SIZES:
    row = {}
    for hc in (1.0, 0.5):
        c, _ = run(T, nav0, haircut=hc, fixed=True); sm = summ(c, nav0)
        navs = np.array([n for d, n in c]); m = navs[::30]; worst_m = float(np.min(np.diff(m)) / nav0)
        row[hc] = {"per_year_pct": sm["total"] / sm["years"], "per_year_usd": (sm["final"] - nav0) / sm["years"], "max_dd": sm["max_dd"], "worst_month": worst_m}
    res["paid_out"][nav0] = {"full": row[1.0], "half": row[0.5]}
    print(f"{nav0:>12,.0f} {row[1.0]['per_year_usd']:>12,.0f} ({row[1.0]['per_year_pct']*100:>4.0f}%/yr) {row[0.5]['per_year_usd']:>12,.0f} ({row[0.5]['per_year_pct']*100:>4.0f}%/yr)  {row[1.0]['worst_month']*100:+.1f}%", flush=True)
# Forward Monte Carlo: Poisson event arrivals at the historical rate, trades resampled with their capacity.
rng = np.random.default_rng(1)
span = (T[-1]["ts"] - T[0]["ts"]); rate = len(T) / span
for nav0 in SIZES:
    out = {1: [], 2: [], 3: []}; dds = []
    for sim in range(300):
        n = rng.poisson(rate * 3 * 365 * 86400)
        ts = np.sort(rng.uniform(0, 3 * 365 * 86400, n)) + 1.9e9
        tr = [dict(T[i], ts=float(x)) for i, x in zip(rng.integers(0, len(T), n), ts)]
        c, _ = run(tr, nav0, start=1.9e9)
        navs = np.array([v for d, v in c])
        for y in (1, 2, 3): out[y].append(navs[min(len(navs) - 1, y * 365)] / nav0 - 1)
        dds.append((navs / np.maximum.accumulate(navs) - 1).min())
    res["mc"][nav0] = {str(y): [float(np.percentile(out[y], p)) for p in (10, 50, 90)] for y in out}
    res["mc"][nav0]["dd"] = [float(np.percentile(dds, p)) for p in (50, 90)]
    res["mc"][nav0]["p_loss_1y"] = float(np.mean(np.array(out[1]) < 0))
    print(f"MC {nav0:>12,.0f}: " + "  ".join(f"{y}y p10/p50/p90 {res['mc'][nav0][str(y)][0]*100:+.0f}/{res['mc'][nav0][str(y)][1]*100:+.0f}/{res['mc'][nav0][str(y)][2]*100:+.0f}%" for y in (1, 2, 3))
          + f"  DD median {res['mc'][nav0]['dd'][0]*100:.0f}% p90 {res['mc'][nav0]['dd'][1]*100:.0f}%  P(loss 1y) {res['mc'][nav0]['p_loss_1y']*100:.0f}%", flush=True)
json.dump(res, open("cex/s1_fund.json", "w"), default=float)
