"""Part 2b: exchanges without a published listing time: first daily candle of each market (Coinbase USD/USDC,
Upbit KRW, KuCoin USDT). Checkpointed to cex/catalog_b.json."""
import httpx, json, time, datetime as dt
c = httpx.Client(timeout=30, verify="/root/.ccr/ca-bundle.crt", follow_redirects=True)
try: out = json.load(open("cex/catalog_b.json"))
except Exception: out = {}
def get(url, params=None):
    for a in range(5):
        try:
            r = c.get(url, params=params)
            if r.status_code == 429: time.sleep(2 * (a + 1)); continue
            return r.json() if r.status_code == 200 else None
        except Exception: time.sleep(2)
    return None
def iso(ts): return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
START = 1704067200  # 2024-01-01: anything trading before then is an old listing
# Coinbase: daily candles in 300-day windows from 2024-01-01 forward; first non-empty = listing day
for p in get("https://api.exchange.coinbase.com/products") or []:
    if p["quote_currency"] not in ("USD", "USDC") or f"coinbase:{p['id']}" in out: continue
    first = None
    t = START
    while t < time.time() and first is None:
        d = get(f"https://api.exchange.coinbase.com/products/{p['id']}/candles",
                {"granularity": 86400, "start": iso(t), "end": iso(t + 299 * 86400)}); time.sleep(0.15)
        if d: first = min(x[0] for x in d)
        t += 299 * 86400
    out[f"coinbase:{p['id']}"] = {"exchange": "coinbase", "base": p["base_currency"], "quote": p["quote_currency"],
                                  "symbol": p["id"], "first_day": first, "status": p.get("status")}
    if len(out) % 50 == 0: json.dump(out, open("cex/catalog_b.json", "w")); print("coinbase", len(out), flush=True)
json.dump(out, open("cex/catalog_b.json", "w"))
# Upbit KRW: page daily candles backwards (200 per call) until the start of history
for m in get("https://api.upbit.com/v1/market/all") or []:
    if not m["market"].startswith("KRW-") or f"upbit:{m['market']}" in out: continue
    to, first = None, None
    for _ in range(6):
        d = get("https://api.upbit.com/v1/candles/days", {"market": m["market"], "count": 200, **({"to": to} if to else {})})
        time.sleep(0.12)
        if not d: break
        first = d[-1]["candle_date_time_utc"]
        if len(d) < 200: break
        to = first.replace("T", " ")
    ts = dt.datetime.strptime(first, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp() if first else None
    out[f"upbit:{m['market']}"] = {"exchange": "upbit", "base": m["market"][4:], "quote": "KRW", "symbol": m["market"],
                                   "first_day": ts, "complete": True}
    if len(out) % 50 == 0: json.dump(out, open("cex/catalog_b.json", "w")); print("upbit", len(out), flush=True)
json.dump(out, open("cex/catalog_b.json", "w"))
print("done", len(out), flush=True)
