"""Market size around each core event on Binance USD-M (data.binance.vision daily archives): open interest (USD,
5-min metrics) and perp quote volume (1h klines) over the first 48 h from the listing hour. Output
cex/capacity.json {event_key: {"oi_usd_median", "oi_usd_min", "vol_usd_24h", "vol_usd_min_hour"}}."""
import httpx, json, io, zipfile, csv, time, datetime as dt
c = httpx.Client(timeout=40, verify="/root/.ccr/ca-bundle.crt", follow_redirects=True)
ev = json.load(open("cex/events.json")); fd = json.load(open("cex/venue_funding.json"))
try: out = json.load(open("cex/capacity.json"))
except Exception: out = {}
def rows(url):
    try:
        r = c.get(url)
        if r.status_code != 200: return []
        z = zipfile.ZipFile(io.BytesIO(r.content))
        return [row for row in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))) if row and row[0][:1].isdigit()]
    except Exception: return []
B = "https://data.binance.vision/data/futures/um/daily"
for k, f in fd.items():
    if "binance" not in f or k in out: continue
    kind, ex, base, t0 = k.split(":"); t0 = float(t0); sym = base + "USDT"
    days = [dt.datetime.fromtimestamp(t0 + d * 86400, dt.timezone.utc).strftime("%Y-%m-%d") for d in (0, 1, 2)]
    oi, vol = [], []
    for d in days:
        for r in rows(f"{B}/metrics/{sym}/{sym}-metrics-{d}.zip"):
            try:
                ts = dt.datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
                if t0 <= ts < t0 + 48 * 3600: oi.append(float(r[3]))
            except Exception: pass
        for r in rows(f"{B}/klines/{sym}/1h/{sym}-1h-{d}.zip"):
            ts = int(r[0]) / 1000
            ts = ts / 1000 if ts > 1e11 else ts
            if t0 <= ts < t0 + 24 * 3600: vol.append(float(r[7]))
    out[k] = {"oi_usd_median": sorted(oi)[len(oi) // 2] if oi else None, "oi_usd_min": min(oi) if oi else None,
              "vol_usd_24h": sum(vol) if vol else None, "vol_usd_min_hour": min(vol) if vol else None}
    json.dump(out, open("cex/capacity.json", "w"))
import numpy as np
for key in ("oi_usd_median", "oi_usd_min", "vol_usd_24h", "vol_usd_min_hour"):
    a = [v[key] for v in out.values() if v[key]]
    print(f"{key:18} n={len(a)} p10 ${np.percentile(a,10):,.0f}  median ${np.median(a):,.0f}  p90 ${np.percentile(a,90):,.0f}")
