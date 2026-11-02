"""Which perp venue could short each listing at the listing hour, and the 7-day short (spot path) on the events each
venue covers. Day-granular listing dates (Binance, Bybit) count as available from that UTC midnight."""
import json, collections, numpy as np
exec(open("scripts/cex_analyse.py").read().split("HOR = ")[0])
pv = {k: v for k, v in json.load(open("cex/perp_venues.json")).items() if not k.startswith("_")}
VEN = ["binance", "bybit", "okx", "gate", "mexc", "kucoin", "aster", "hyperliquid"]
GROUPS = [("B", "upbit"), ("B", "coinbase"), ("B", "binance"), ("B", "okx"), ("A", "upbit"), ("A", "coinbase"), ("A", "binance"), ("A", "okx")]
def norm(b):
    for p in ("1000000", "100000", "10000", "1000", "1M"):
        if b.startswith(p) and len(b) > len(p) + 1: return b[len(p):]
    return b
res = collections.defaultdict(list)
for k, v in px.items():
    e = ev.get(k)
    if not e or (e["kind"], e["exchange"]) not in GROUPS: continue
    s = v["series"]
    if e["kind"] == "A":
        h0 = min((h for h, p in s if h >= 0), default=None)
        if h0 is None or h0 > 2: continue
        p0 = at(s, h0, 0.01)
    else: h0, p0 = 0, at(s, 0)
    p7 = at(s, h0 + 168)
    if not p0 or not p7: continue
    b = norm(e["base"])
    avail = {vn for vn in VEN if b in pv.get(vn, {}) and pv[vn][b] <= e["ts"] + 3600}
    res[(e["kind"], e["exchange"])].append((-(p7 / p0 - 1) - 0.002, avail))
for g in GROUPS:
    rs = res[g]
    if not rs: continue
    print(f"\n{g[0]} {g[1]}: n={len(rs)}  all {np.mean([r for r, a in rs])*100:+.1f}%  any perp at t0: {sum(1 for r, a in rs if a)} ({np.mean([r for r, a in rs if a])*100:+.1f}%)  none: {np.mean([r for r, a in rs if not a])*100 if any(not a for r, a in rs) else 0:+.1f}%")
    line = []
    for vn in VEN:
        sub = [r for r, a in rs if vn in a]
        if sub: line.append(f"{vn} {len(sub)} ({np.mean(sub)*100:+.1f}%)")
    print("   " + " | ".join(line))
# Coverage of the main tradeable set by single venues and pairs
core = [a for g in [("B", "upbit"), ("B", "coinbase"), ("B", "binance"), ("A", "upbit")] for r, a in res[g]]
print(f"\nCore set (B Upbit + B Coinbase + B Binance + A Upbit) n={len(core)}: share shortable at t0")
single = sorted(((sum(vn in a for a in core) / len(core), vn) for vn in VEN), reverse=True)
print("   single venue: " + ", ".join(f"{vn} {s*100:.0f}%" for s, vn in single))
pairs = sorted(((sum((x in a) or (y in a) for a in core) / len(core), x, y) for i, x in enumerate(VEN) for y in VEN[i+1:]), reverse=True)[:6]
print("   best pairs: " + ", ".join(f"{x}+{y} {s*100:.0f}%" for s, x, y in pairs))
print(f"   any venue: {sum(bool(a) for a in core)/len(core)*100:.0f}%")
