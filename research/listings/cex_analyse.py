"""Part 4: listing returns per exchange and kind, raw and net of BTC over the same window.
A: entry at the first hourly close after listing; B: entry at the reference venue's close at the listing hour,
plus the run-up into it. Short P&L = -(return) - 0.2% round trip (funding ignored)."""
import json, sys, collections, time, numpy as np, httpx
rng = np.random.default_rng(0)
ev = {f"{e['kind']}:{e['exchange']}:{e['base']}:{int(e['ts'])}": e for e in json.load(open("cex/events.json"))}
px = json.load(open("cex/prices.json"))
btc = np.load("liquid/btc_1h.npy")
c = httpx.Client(timeout=30, verify="/root/.ccr/ca-bundle.crt")
t, extra = btc[-1, 0] + 3600, []
while t < time.time():
    d = c.get("https://data-api.binance.vision/api/v3/klines", params={"symbol": "BTCUSDT", "interval": "1h", "startTime": int(t * 1000), "limit": 1000}).json()
    if not d: break
    extra += [(x[0] / 1000, float(x[4])) for x in d]; t = extra[-1][0] + 3600
if extra: btc = np.vstack([btc, np.array(extra)])
def btc_at(ts):
    i = np.searchsorted(btc[:, 0], ts, side="right") - 1
    return btc[i, 1] if i >= 0 else None
def at(series, h, tol=1.5):
    best = None
    for hh, p in series:
        if hh <= h + 1e-9 and h - hh <= tol: best = p
    return best
def ci(a):
    a = np.array(a)
    return [round(float(x) * 100, 1) for x in np.percentile([rng.choice(a, len(a)).mean() for _ in range(2000)], [5, 95])]
HOR = (1, 24, 72, 168)
res = collections.defaultdict(lambda: collections.defaultdict(list))
pre = collections.defaultdict(lambda: collections.defaultdict(list))
for k, v in px.items():
    e = ev.get(k)
    s = v["series"]
    if not e or len(s) < 10: continue
    grp = (e["kind"], e["exchange"])
    if e["kind"] == "A":
        h0 = min((h for h, p in s if h >= 0), default=None)
        if h0 is None or h0 > 2: continue
        p0 = at(s, h0, 0.01)
    else:
        h0 = 0; p0 = at(s, 0)
        if p0 is None: continue
        for back in (168, 24):
            pb = at(s, -back)
            if pb:
                b0, bb = btc_at(e["ts"]), btc_at(e["ts"] - back * 3600)
                pre[grp][back].append((p0 / pb - 1, (p0 / pb) - (b0 / bb)))
    b0 = btc_at(e["ts"] + h0 * 3600)
    for h in HOR:
        ph = at(s, h0 + h)
        if ph is None or not p0: continue
        bh = btc_at(e["ts"] + (h0 + h) * 3600)
        r = ph / p0 - 1
        res[grp][h].append((-r - 0.002, -(r - (bh / b0 - 1)) - 0.002))
print("SHORT from listing (mean % per trade, 90% CI; 'net' = minus BTC's move)")
for grp in sorted(res):
    line = []
    for h in HOR:
        a = res[grp][h]
        if len(a) < 5: continue
        raw, net = [x[0] for x in a], [x[1] for x in a]
        line.append(f"{h:>3}h n={len(a):4} raw {np.mean(raw)*100:+6.1f}% {ci(raw)} med {np.median(raw)*100:+6.1f}% win {np.mean(np.array(raw)>0)*100:3.0f}% | net {np.mean(net)*100:+6.1f}% {ci(net)}")
    print(f"\n{grp[0]} {grp[1]}"); print("\n".join("   " + l for l in line))
print("\nRUN-UP before B listings (reference venue, mean %, 90% CI; net = minus BTC)")
for grp in sorted(pre):
    for back in (168, 24):
        a = pre[grp][back]
        if len(a) >= 5:
            raw, net = [x[0] for x in a], [x[1] for x in a]
            print(f"   {grp[1]:9} -{back:>3}h..0 n={len(a):4} raw {np.mean(raw)*100:+6.1f}% {ci(raw)} med {np.median(raw)*100:+6.1f}% | net {np.mean(net)*100:+6.1f}% {ci(net)}")
