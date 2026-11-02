"""Strategy 1 trade list + half-year stability of the venue-shortable sets.
Core groups: B Upbit, B Coinbase, B Binance, A Upbit. A trade = short at the listing hour on a perp that existed
at that hour, cover 7 days later (spot path, 0.2% round trip). Funding: the event's own 7-day Binance funding when
known, else the group median. Capacity: the event's Binance open interest (median of first 48 h) when known, else
the group's 25th percentile (conservative). Writes cex/s1_trades.json."""
import json, collections, time, numpy as np
exec(open("scripts/cex_analyse.py").read().split("HOR = ")[0])
pv = {k: v for k, v in json.load(open("cex/perp_venues.json")).items() if not k.startswith("_")}
fd = json.load(open("cex/venue_funding.json")); cap = json.load(open("cex/capacity.json"))
GROUPS = [("B", "upbit"), ("B", "coinbase"), ("B", "binance"), ("A", "upbit")]
SETS = {"binance+bybit": {"binance", "bybit"}, "binance+bybit+gate": {"binance", "bybit", "gate"}}
def norm(b):
    for p in ("1000000", "100000", "10000", "1000", "1M"):
        if b.startswith(p) and len(b) > len(p) + 1: return b[len(p):]
    return b
def half(ts):
    t = time.gmtime(ts); return f"{t.tm_year}H{1 if t.tm_mon <= 6 else 2}"
trades = []
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
    venues = sorted(vn for vn in pv if b in pv.get(vn, {}) and pv[vn][b] <= e["ts"] + 3600)
    path = [(h - h0, p / p0 - 1) for h, p in s if h0 <= h <= h0 + 168]
    trades.append({"key": k, "group": f"{e['kind']} {e['exchange']}", "ts": e["ts"] + h0 * 3600, "half": half(e["ts"]),
                   "short": -(p7 / p0 - 1) - 0.002, "worst": -max(r for h, r in path), "venues": venues,
                   "funding": fd.get(k, {}).get("binance"), "oi": (cap.get(k) or {}).get("oi_usd_median"),
                   "vol_min_hour": (cap.get(k) or {}).get("vol_usd_min_hour"), "vol24": (cap.get(k) or {}).get("vol_usd_24h"), "path": path})
for g in {t["group"] for t in trades}:
    sub = [t for t in trades if t["group"] == g]
    fmed = float(np.median([t["funding"] for t in sub if t["funding"] is not None] or [0]))
    ois = [t["oi"] for t in sub if t["oi"]]; o25 = float(np.percentile(ois, 25)) if ois else 2e6
    vs_ = [t["vol24"] for t in sub if t["vol24"]]; v25 = float(np.percentile(vs_, 25)) if vs_ else 3e7
    for t in sub:
        t["funding_est"] = t["funding"] is None; t["funding"] = fmed if t["funding"] is None else t["funding"]
        t["oi_est"] = not t["oi"]; t["oi"] = o25 if not t["oi"] else t["oi"]
        t["vol24"] = t["vol24"] or v25
        t["net"] = t["short"] + t["funding"]
json.dump(sorted(trades, key=lambda t: t["ts"]), open("cex/s1_trades.json", "w"))
halves = sorted({t["half"] for t in trades})
for sname, vs in SETS.items():
    print(f"\n=== shortable on {sname} at the listing hour: 7-day short net of funding, mean % (n) per half-year ===")
    print(f"{'group':12}" + "".join(f"{h:>14}" for h in halves) + f"{'all':>22}")
    for g in [f"{a} {b}" for a, b in GROUPS] + ["core"]:
        row = f"{g:12}"
        sub = [t for t in trades if (g == "core" or t["group"] == g) and set(t["venues"]) & vs]
        for h in halves:
            x = [t["net"] for t in sub if t["half"] == h]
            row += f"{(f'{np.mean(x)*100:+.1f} ({len(x)})' if x else '-'):>14}"
        x = [t["net"] for t in sub]
        row += f"   {np.mean(x)*100:+.1f} ({len(x)}) {ci(x)}"
        print(row)
    sub = [t for t in trades if set(t["venues"]) & vs]
    print(f"core: win {np.mean([t['net']>0 for t in sub])*100:.0f}%, median {np.median([t['net'] for t in sub])*100:+.1f}%, "
          f"worst trade {min(t['net'] for t in sub)*100:+.0f}%, worst intra-hold move against {min(t['worst'] for t in sub)*100:+.0f}%, "
          f"funding estimated for {sum(t['funding_est'] for t in sub)}, OI estimated for {sum(t['oi_est'] for t in sub)}")
