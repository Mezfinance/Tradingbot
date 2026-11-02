"""7-day funding a short received (+) or paid (-) from the listing hour, on Gate (history API) and Binance
(monthly fundingRate archives on data.binance.vision), for the core events (B Upbit/Coinbase/Binance, A Upbit).
Output cex/venue_funding.json {event_key: {"gate": x, "binance": y}}."""
import httpx, json, io, zipfile, csv, time, datetime as dt
c = httpx.Client(timeout=40, verify="/root/.ccr/ca-bundle.crt", follow_redirects=True)
ev = json.load(open("cex/events.json"))
pv = json.load(open("cex/perp_venues.json"))
try: out = json.load(open("cex/venue_funding.json"))
except Exception: out = {}
CORE = {("B", "upbit"), ("B", "coinbase"), ("B", "binance"), ("A", "upbit")}
bn_cache = {}
def bn_month(sym, ym):
    if (sym, ym) in bn_cache: return bn_cache[(sym, ym)]
    url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"
    rows = []
    try:
        r = c.get(url)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            for row in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))):
                if row and row[0].isdigit(): rows.append((int(row[0]) / 1000, float(row[-1])))
    except Exception: pass
    bn_cache[(sym, ym)] = rows
    return rows
for i, e in enumerate(ev):
    if (e["kind"], e["exchange"]) not in CORE: continue
    key = f"{e['kind']}:{e['exchange']}:{e['base']}:{int(e['ts'])}"
    if key in out: continue
    t0, t1, rec = e["ts"], e["ts"] + 7 * 86400, {}
    if e["base"] in pv.get("gate", {}) and pv["gate"][e["base"]] <= t0 + 3600:
        try:
            d = c.get("https://api.gateio.ws/api/v4/futures/usdt/funding_rate", params={"contract": f"{e['base']}_USDT", "from": int(t0), "to": int(t1), "limit": 1000}).json()
            if isinstance(d, list): rec["gate"] = sum(float(x["r"]) for x in d); rec["gate_n"] = len(d)
        except Exception: pass
        time.sleep(0.15)
    if e["base"] in pv.get("binance", {}) and pv["binance"][e["base"]] <= t0 + 3600:
        sym = e["base"] + "USDT"
        months = sorted({dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m") for t in (t0, t1)})
        rows = [x for m in months for x in bn_month(sym, m)]
        sel = [r for t, r in rows if t0 <= t < t1]
        if sel: rec["binance"] = sum(sel); rec["binance_n"] = len(sel)
    out[key] = rec
    if i % 20 == 0: json.dump(out, open("cex/venue_funding.json", "w"))
json.dump(out, open("cex/venue_funding.json", "w"))
import numpy as np
for v in ("gate", "binance"):
    a = [r[v] for r in out.values() if v in r]
    print(v, len(a), f"mean {np.mean(a)*100:+.2f}% median {np.median(a)*100:+.2f}% p10 {np.percentile(a,10)*100:+.2f}% p90 {np.percentile(a,90)*100:+.2f}%")
