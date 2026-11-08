"""Rebuild liquid/binance_listings.json ([symbol, first-trade ts, source]) for every Binance spot USDT symbol ever
listed (delisted ones included): symbols from the data.binance.vision S3 listing, first day = earliest 1h daily file,
refined to the first 1-minute candle via data-api when the symbol still trades. Also liquid/btc_1h.npy."""
import httpx, json, re, time, numpy as np, os
c = httpx.Client(timeout=30, verify="/root/.ccr/ca-bundle.crt")
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
syms, marker = [], None
while True:
    p = {"delimiter": "/", "prefix": "data/spot/daily/klines/", "max-keys": 1000, **({"marker": marker} if marker else {})}
    x = c.get(S3, params=p).text
    got = re.findall(r"<Prefix>data/spot/daily/klines/([A-Z0-9]+)/</Prefix>", x)
    syms += got
    if "<IsTruncated>true" not in x or not got: break
    marker = f"data/spot/daily/klines/{got[-1]}/"
syms = sorted({s for s in syms if s.endswith("USDT")})
print("usdt symbols", len(syms), flush=True)
path = "liquid/binance_listings.json"
done = {r[0]: r for r in json.load(open(path))} if os.path.exists(path) else {}
for i, s in enumerate(syms):
    if s in done: continue
    ts, src = None, None
    try:
        d = c.get("https://data-api.binance.vision/api/v3/klines", params={"symbol": s, "interval": "1m", "startTime": 0, "limit": 1}).json()
        if isinstance(d, list) and d: ts, src = d[0][0] / 1000, "1m"
    except Exception: pass
    if ts is None:
        try:
            x = c.get(S3, params={"delimiter": "/", "prefix": f"data/spot/daily/klines/{s}/1h/", "max-keys": 2}).text
            m = re.search(r"-1h-(\d{4}-\d{2}-\d{2})\.zip", x)
            if m: ts, src = time.mktime(time.strptime(m.group(1), "%Y-%m-%d")) - time.timezone, "s3 day"
        except Exception: pass
    if ts: done[s] = [s, ts, src]
    if i % 100 == 0: json.dump(list(done.values()), open(path, "w")); print("bn", i, len(syms), flush=True)
json.dump(list(done.values()), open(path, "w")); print("binance listings", len(done), flush=True)
t, rows = 1717200000, []
while t < time.time():
    d = c.get("https://data-api.binance.vision/api/v3/klines", params={"symbol": "BTCUSDT", "interval": "1h", "startTime": int(t * 1000), "limit": 1000}).json()
    if not d: break
    rows += [(x[0] / 1000, float(x[4])) for x in d]; t = rows[-1][0] + 3600
np.save("liquid/btc_1h.npy", np.array(rows)); print("btc hours", len(rows), flush=True)
