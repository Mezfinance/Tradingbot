"""Part 3b: hourly closes around each event. A: the listing venue's own candles, t0 .. t0+7d.
B: a venue where the token already traded (binance > okx > gate > coinbase > bitget > upbit), t0-7d .. t0+7d.
Saved as {event_id: {"venue", "series": [[hours from t0, close], ...]}} in cex/prices.json (checkpointed)."""
import httpx, json, time, random, datetime as dt
c = httpx.Client(timeout=30, verify="/root/.ccr/ca-bundle.crt", follow_redirects=True)
H = 3600
def get(url, params=None):
    for a in range(5):
        try:
            r = c.get(url, params=params)
            if r.status_code == 429: time.sleep(2 * (a + 1)); continue
            return r.json() if r.status_code == 200 else None
        except Exception: time.sleep(2)
    return None
def iso(ts): return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def binance(base, a, b):
    d = get("https://data-api.binance.vision/api/v3/klines", {"symbol": base + "USDT", "interval": "1h", "startTime": int(a * 1000), "endTime": int(b * 1000), "limit": 1000})
    return [(x[0] / 1000, float(x[4])) for x in d or []]
def okx(base, a, b):
    out, after = [], int(b * 1000)
    for _ in range(6):
        d = get("https://www.okx.com/api/v5/market/history-candles", {"instId": f"{base}-USDT", "bar": "1H", "after": after, "limit": 100})
        rows = (d or {}).get("data") or []
        if not rows: break
        out += [(int(x[0]) / 1000, float(x[4])) for x in rows]
        after = int(rows[-1][0])
        if after / 1000 <= a: break
        time.sleep(0.12)
    return [x for x in out if a <= x[0] <= b]
def gate(base, a, b):
    d = get("https://api.gateio.ws/api/v4/spot/candlesticks", {"currency_pair": f"{base}_USDT", "interval": "1h", "from": int(a), "to": int(b)})
    return [(int(x[0]), float(x[2])) for x in d or [] if isinstance(x, list)]
def coinbase(sym, a, b):
    out = []
    for s in range(int(a), int(b), 300 * H):
        d = get(f"https://api.exchange.coinbase.com/products/{sym}/candles", {"granularity": H, "start": iso(s), "end": iso(min(b, s + 299 * H))})
        out += [(x[0], float(x[4])) for x in d or []]; time.sleep(0.12)
    return out
def bitget(base, a, b):
    out, end = [], int(b * 1000)
    for _ in range(3):
        d = get("https://api.bitget.com/api/v2/spot/market/history-candles", {"symbol": base + "USDT", "granularity": "1h", "endTime": end, "limit": 200})
        rows = (d or {}).get("data") or []
        if not rows: break
        out += [(int(x[0]) / 1000, float(x[4])) for x in rows]
        end = min(int(x[0]) for x in rows) - 1
        if end / 1000 <= a: break
        time.sleep(0.12)
    return [x for x in out if a <= x[0] <= b]
def upbit(sym, a, b):
    out, to = [], b
    for _ in range(3):
        d = get("https://api.upbit.com/v1/candles/minutes/60", {"market": sym, "count": 200, "to": iso(to).replace("T", " ").replace("Z", "")})
        if not d: break
        out += [(dt.datetime.strptime(x["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp(), float(x["trade_price"])) for x in d]
        to = min(x[0] for x in out)
        if to <= a or len(d) < 200: break
        time.sleep(0.12)
    return [x for x in out if a <= x[0] <= b]
FETCH = {"binance": lambda e, a, b: binance(e["base"], a, b), "okx": lambda e, a, b: okx(e["base"], a, b),
         "gate": lambda e, a, b: gate(e["base"], a, b), "coinbase": lambda e, a, b: coinbase(e["symbol"], a, b),
         "bitget": lambda e, a, b: bitget(e["base"], a, b), "upbit": lambda e, a, b: upbit(e["symbol"], a, b)}
REF = ["binance", "okx", "gate", "bitget"]  # reference venues fetched by base ticker
events = json.load(open("cex/events.json"))
rng = random.Random(1)
bitget_a = [e for e in events if e["kind"] == "A" and e["exchange"] == "bitget"]
keep_bitget = {id(e) for e in rng.sample(bitget_a, min(400, len(bitget_a)))}
try: out = json.load(open("cex/prices.json"))
except Exception: out = {}
todo = [e for e in events if not (e["kind"] == "A" and e["exchange"] == "bitget")]  # bitget A sample already fetched
for i, e in enumerate(todo):
    key = f"{e['kind']}:{e['exchange']}:{e['base']}:{int(e['ts'])}"
    if key in out: continue
    t0 = e["ts"]
    if e["kind"] == "A":
        venue, a, b = e["exchange"], t0 - H, t0 + 7 * 86400 + H
        rows = FETCH[venue](e, a, b)
    else:
        rows, venue = [], None
        for v in [v for v in REF if v in e["prior_venues"]]:
            rows = FETCH[v](e, t0 - 7 * 86400 - H, t0 + 7 * 86400 + H)
            if len(rows) > 50: venue = v; break
    rows = sorted(set(rows))
    out[key] = {"venue": venue, "series": [[round((t - t0) / H, 2), p] for t, p in rows]}
    if i % 25 == 0:
        json.dump(out, open("cex/prices.json", "w")); print(i, len(todo), flush=True)
json.dump(out, open("cex/prices.json", "w")); print("done", len(out), flush=True)
