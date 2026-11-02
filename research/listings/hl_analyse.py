"""Step 1 results: per group, how many events were shortable on Hyperliquid at the listing hour (or within a day),
the 7-day short on those (spot price path; exit at listing + 7 days) and what Hyperliquid funding added."""
import json, collections, numpy as np
exec(open("scripts/cex_analyse.py").read().split("HOR = ")[0])
sv = json.load(open("cex/short_venues.json"))["events"]
GROUPS = [("B", "upbit"), ("B", "binance"), ("B", "coinbase"), ("B", "okx"), ("A", "upbit"), ("A", "coinbase"), ("A", "binance"), ("A", "okx"), ("A", "gate")]
rows = collections.defaultdict(list)
for k, v in px.items():
    e, info = ev.get(k), sv.get(k)
    if not e or not info or (e["kind"], e["exchange"]) not in GROUPS: continue
    s = v["series"]; t0 = e["ts"]
    if e["kind"] == "A":
        h0 = min((h for h, p in s if h >= 0), default=None)
        if h0 is None or h0 > 2: continue
    else: h0 = 0
    p7 = at(s, 168)
    hl_ts = info.get("hl_list_ts")
    hl_delay = None if hl_ts is None else (hl_ts - t0) / 3600
    bn = info.get("bn_perp_ts"); okx = info.get("okx_swap_ts")
    r = {"hl_delay_h": hl_delay, "perp_any_at_t0": any(x is not None and x <= t0 + 3600 for x in (hl_ts, okx)) or (bn is not None and bn <= t0)}
    if p7:
        for label, start in (("t0", h0), ("hl_entry", max(h0, hl_delay) if hl_delay is not None else None)):
            if start is None or start > 167: continue
            ps = at(s, start, 1.5 if label == "hl_entry" else (0.01 if e["kind"] == "A" else 1.5))
            if ps: r[label] = -(p7 / ps - 1) - 0.002
    r["funding"] = info.get("hl_funding_7d")
    rows[(e["kind"], e["exchange"])].append(r)
def m(a): return f"{np.mean(a)*100:+.1f}% {ci(a)}" if len(a) >= 5 else f"n<5"
for g in GROUPS:
    rs = rows[g]
    if not rs: continue
    at0 = [r for r in rs if r["hl_delay_h"] is not None and r["hl_delay_h"] <= 1]
    d24 = [r for r in rs if r["hl_delay_h"] is not None and 1 < r["hl_delay_h"] <= 24]
    later = [r for r in rs if r["hl_delay_h"] is not None and r["hl_delay_h"] > 24]
    never = [r for r in rs if r["hl_delay_h"] is None]
    print(f"\n{g[0]} {g[1]}: n={len(rs)}  on Hyperliquid at the listing hour {len(at0)}, within 24h {len(d24)}, later {len(later)}, never {len(never)}; any perp (HL/Binance/OKX) at t0 {sum(r['perp_any_at_t0'] for r in rs)}")
    s_all = [r["t0"] for r in rs if "t0" in r]
    s_hl = [r["t0"] for r in at0 if "t0" in r]
    f_hl = [r["funding"] for r in at0 if r.get("funding") is not None and "t0" in r]
    net = [r["t0"] + r["funding"] for r in at0 if r.get("funding") is not None and "t0" in r]
    late = [r["hl_entry"] + (r["funding"] or 0) for r in d24 if "hl_entry" in r]
    print(f"   7d short, all events: {m(s_all)}")
    print(f"   shortable on HL at t0: short {m(s_hl)}  | HL funding received {m(f_hl)} (median {np.median(f_hl)*100 if f_hl else 0:+.2f}%)  | short + funding {m(net)}")
    print(f"   HL listed 1-24h after: short from the HL listing hour + funding {m(late)}")
