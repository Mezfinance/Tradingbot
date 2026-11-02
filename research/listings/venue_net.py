import json, collections, numpy as np
exec(open("scripts/cex_analyse.py").read().split("HOR = ")[0])
fd = json.load(open("cex/venue_funding.json"))
g = collections.defaultdict(list)
for k, f in fd.items():
    e, v = ev.get(k), px.get(k)
    if not e or not v or "binance" not in f: continue
    s = v["series"]
    if e["kind"] == "A":
        h0 = min((h for h, p in s if h >= 0), default=None)
        if h0 is None or h0 > 2: continue
        p0 = at(s, h0, 0.01)
    else: h0, p0 = 0, at(s, 0)
    p7 = at(s, h0 + 168)
    if p0 and p7: g[(e["kind"], e["exchange"])].append((-(p7 / p0 - 1) - 0.002, f["binance"]))
for k, rs in sorted(g.items()):
    s, f = np.array([r[0] for r in rs]), np.array([r[1] for r in rs])
    print(f"{k[0]} {k[1]:9} n={len(rs):3} short {s.mean()*100:+.1f}%  Binance funding {f.mean()*100:+.1f}% (median {np.median(f)*100:+.2f}%)  net {np.mean(s+f)*100:+.1f}% {ci(list(s+f))}  win {np.mean(s+f>0)*100:.0f}%")
