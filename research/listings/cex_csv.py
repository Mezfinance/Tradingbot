import json, csv, time
exec(open("scripts/cex_analyse.py").read().split("HOR = ")[0])
out = csv.writer(open("/home/user/cex_listings_study.csv", "w", newline=""))
out.writerow(["kind", "listing_exchange", "token", "listing_time_utc", "first_cex_listing_utc", "venues_7d_before",
              "price_venue", "runup_7d_pct", "runup_24h_pct", "move_1h_pct", "move_24h_pct", "move_72h_pct", "move_7d_pct",
              "peak_within_7d_pct", "short_7d_pct_after_0.2pct_cost"])
fmt = lambda x: "" if x is None else round(x * 100, 1)
n = 0
for k, v in sorted(px.items(), key=lambda kv: ev.get(kv[0], {}).get("ts", 0)):
    e = ev.get(k); s = v["series"]
    if not e or len(s) < 10: continue
    if e["kind"] == "A":
        h0 = min((h for h, p in s if h >= 0), default=None)
        if h0 is None or h0 > 2: continue
        p0 = at(s, h0, 0.01)
    else:
        h0, p0 = 0, at(s, 0)
    if not p0: continue
    mv = lambda h: (at(s, h0 + h) / p0 - 1) if at(s, h0 + h) else None
    ru = lambda b: (p0 / at(s, -b) - 1) if e["kind"] == "B" and at(s, -b) else None
    path = [p for h, p in s if h0 < h <= h0 + 168]
    m7 = mv(168)
    out.writerow([e["kind"], e["exchange"], e["base"], time.strftime("%Y-%m-%d %H:%M", time.gmtime(e["ts"])),
                  time.strftime("%Y-%m-%d %H:%M", time.gmtime(e["first_ts"])), " ".join(e["prior_venues"]), v["venue"],
                  fmt(ru(168)), fmt(ru(24)), fmt(mv(1)), fmt(mv(24)), fmt(mv(72)), fmt(m7),
                  fmt(max(path) / p0 - 1 if path else None), fmt(-m7 - 0.002 if m7 is not None else None)])
    n += 1
print(n)
