"""Exact listing hour for Coinbase / Upbit events (catalog_b only had the first day): earliest hourly candle within
the first two days. Rewrites those events' ts in cex/events.json and drops their old price paths."""
import httpx, json, time, datetime as dt
c = httpx.Client(timeout=30, verify="/root/.ccr/ca-bundle.crt", follow_redirects=True)
def get(url, params=None):
    for a in range(5):
        try:
            r = c.get(url, params=params)
            if r.status_code == 429: time.sleep(2 * (a + 1)); continue
            return r.json() if r.status_code == 200 else None
        except Exception: time.sleep(2)
def iso(ts): return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
ev = json.load(open("cex/events.json")); px = json.load(open("cex/prices.json"))
done = 0
for e in ev:
    if e["exchange"] not in ("coinbase", "upbit") or e.get("hour_fixed"): continue
    day = e["ts"]
    if e["exchange"] == "coinbase":
        d = get(f"https://api.exchange.coinbase.com/products/{e['symbol']}/candles", {"granularity": 3600, "start": iso(day - 86400), "end": iso(day + 2 * 86400)})
        first = min((x[0] for x in d or []), default=None)
    else:
        d = get("https://api.upbit.com/v1/candles/minutes/60", {"market": e["symbol"], "count": 72, "to": iso(day + 2 * 86400).replace("T", " ").replace("Z", "")})
        first = min((dt.datetime.strptime(x["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp() for x in d or []), default=None)
    time.sleep(0.15)
    old = f"{e['kind']}:{e['exchange']}:{e['base']}:{int(e['ts'])}"
    if first:
        px.pop(old, None); e["ts"] = float(first); e["hour_fixed"] = True; done += 1
json.dump(ev, open("cex/events.json", "w")); json.dump(px, open("cex/prices.json", "w"))
print("fixed", done)
