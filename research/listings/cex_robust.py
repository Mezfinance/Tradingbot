"""Robustness of the 7-day short: by half-year, worst trades, and a buy-back stop (+X% on an hourly close)."""
import json, collections, time, numpy as np
exec(open("scripts/cex_analyse.py").read().split("HOR = ")[0])
GROUPS = [("B", "upbit"), ("B", "binance"), ("B", "coinbase"), ("A", "upbit"), ("A", "coinbase"), ("A", "binance"), ("A", "okx"), ("A", "gate")]
half = lambda ts: time.strftime("%Y", time.gmtime(ts)) + ("H1" if time.gmtime(ts).tm_mon <= 6 else "H2")
rows = collections.defaultdict(list)
for k, v in px.items():
    e = ev.get(k); s = v["series"]
    if not e or len(s) < 10 or (e["kind"], e["exchange"]) not in GROUPS: continue
    if e["kind"] == "A":
        h0 = min((h for h, p in s if h >= 0), default=None)
        if h0 is None or h0 > 2: continue
    else: h0 = 0
    p0 = at(s, h0, 0.01 if e["kind"] == "A" else 1.5)
    p7 = at(s, h0 + 168)
    if not p0 or not p7: continue
    path = [p for h, p in s if h0 < h <= h0 + 168]
    r = {"ts": e["ts"], "base": e["base"], "ret": -(p7 / p0 - 1) - 0.002, "peak": max(path) / p0 - 1 if path else 0}
    for stop in (0.15, 0.30, 0.50):
        hit = next((p for p in path if p >= p0 * (1 + stop)), None)
        r[f"stop{int(stop*100)}"] = (-(hit / p0 - 1) - 0.002) if hit else r["ret"]
    rows[(e["kind"], e["exchange"])].append(r)
for g in GROUPS:
    rs = rows[g]
    if not rs: continue
    a = np.array([r["ret"] for r in rs])
    byh = collections.defaultdict(list)
    for r in rs: byh[half(r["ts"])].append(r["ret"])
    worst = sorted(rs, key=lambda r: r["ret"])[:3]
    print(f"\n{g[0]} {g[1]}  n={len(a)}  7d short mean {a.mean()*100:+.1f}% median {np.median(a)*100:+.1f}%  "
          f"peak above entry within the week: median {np.median([r['peak'] for r in rs])*100:+.0f}%, >50% in {np.mean([r['peak']>0.5 for r in rs])*100:.0f}%")
    print("   by half-year: " + "  ".join(f"{h} {np.mean(v)*100:+.1f}% (n={len(v)})" for h, v in sorted(byh.items())))
    print("   with buy-back stop: " + "  ".join(f"+{s}% {np.mean([r[f'stop{s}'] for r in rs])*100:+.1f}%" for s in (15, 30, 50)))
    print("   worst: " + ", ".join(f"{r['base']} {r['ret']*100:+.0f}%" for r in worst))
