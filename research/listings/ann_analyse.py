"""Step 1 of the short fund's next research: enter at the Binance announcement instead of the listing hour.
Matches each Binance spot listing event (cex/events.json) to the latest catalog-48 announcement naming
'(SYM)' up to 30 days before it ('Will List', 'Will Add', HODLer, Launchpool...; futures-only notices skipped).
B events (token already trading elsewhere): reference-venue hourly path (cex/prices.json, t0-7d..t0+7d), so
the short can start at the announcement. Short P&L = -(return) - 0.2%, funding ignored."""
import json, re, collections, numpy as np
rng = np.random.default_rng(0)
ann = json.load(open("cex/ann_binance.json"))
ev = [e for e in json.load(open("cex/events.json")) if e["exchange"] == "binance"]
px = json.load(open("cex/prices.json"))
def syms(title):
    return set(re.findall(r"\(([^()]+)\)", title))
spot = [a for a in ann if not a["title"].startswith("Binance Futures") and "Perpetual" not in a["title"]]
def at(series, h, tol=1.5):
    best = None
    for hh, p in series:
        if hh <= h + 1e-9 and h - hh <= tol: best = p
    return best
def ci(a):
    a = np.array(a)
    return [round(float(x) * 100, 1) for x in np.percentile([rng.choice(a, len(a)).mean() for _ in range(2000)], [5, 95])]
def fmt(a):
    a = np.array(a); return f"n={len(a):3} mean {a.mean()*100:+6.1f}% CI {ci(a)} median {np.median(a)*100:+6.1f}% win {(a>0).mean()*100:3.0f}%"
lead = collections.defaultdict(list); rows = []; unmatched = collections.Counter()
for e in ev:
    m = [a for a in spot if e["base"] in syms(a["title"]) and 0 < e["ts"] - a["ts"] < 30 * 86400]
    if not m: unmatched[e["kind"]] += 1; continue
    a = min(m, key=lambda a: a["ts"])          # first announcement naming the token
    L = (e["ts"] - a["ts"]) / 3600
    lead[e["kind"]].append(L)
    rows.append((e, a, L))
print("matched", {k: len(v) for k, v in lead.items()}, "unmatched", dict(unmatched))
for k, v in lead.items():
    v = np.array(v); print(f"{k}: lead announcement -> listing hours p10 {np.percentile(v,10):.1f} median {np.median(v):.1f} p90 {np.percentile(v,90):.1f}")
out = collections.defaultdict(list)
for e, a, L in rows:
    if e["kind"] != "B": continue
    s = px.get(f"B:binance:{e['base']}:{int(e['ts'])}", {}).get("series")
    if not s: continue
    ha = -np.ceil(L)                              # first hourly close after the announcement
    pa, p0 = at(s, ha), at(s, 0)
    pr = at(s, -168) if L < 168 else None
    if not pa or not p0: continue
    out["pre: 7d before -> announcement (rise)"].append(pa / pr - 1) if pr else None
    out["run-up: announcement -> listing (rise)"].append(p0 / pa - 1)
    for H in (24, 72, 168):
        ph = at(s, H)
        if ph:
            out[f"short ann -> listing+{H}h"].append(-(ph / pa - 1) - 0.002)
            out[f"short listing -> listing+{H}h"].append(-(ph / p0 - 1) - 0.002)
    for H in (1, 6, 24):                          # short at the announcement, cover at/before the listing
        if H <= L:
            ph = at(s, ha + H)
            if ph: out[f"short ann -> ann+{H}h (before listing)"].append(-(ph / pa - 1) - 0.002)
print("\nB Binance (token already trading elsewhere), reference-venue prices")
for k, v in out.items(): print(f"{k:42} {fmt(v)}")
