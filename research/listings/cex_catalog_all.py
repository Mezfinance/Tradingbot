"""Part 2a: listing catalogue from exchanges that publish a listing time (OKX spot+swap, Gate, Bitget) plus the
Binance spot list built earlier (first 1-minute candle). Since 2024-10-01, USDT/USDC/USD quotes."""
import httpx, json, time, re
c = httpx.Client(timeout=60, verify="/root/.ccr/ca-bundle.crt", follow_redirects=True)
SINCE = 1
rows = []
skip = re.compile(r"(\d+[LS]$|UP$|DOWN$|BULL$|BEAR$)")
def add(ex, market, base, quote, sym, ts, status):
    if isinstance(ts, str) and '-' in ts:
        import datetime as dt; ts = dt.datetime.strptime(ts, '%Y-%m-%d').replace(tzinfo=dt.timezone.utc).timestamp()
    ts = float(ts or 0)
    if ts and ts >= SINCE and quote in ("USDT", "USDC", "USD") and not skip.search(base):
        rows.append({"exchange": ex, "market": market, "base": base.upper(), "quote": quote, "symbol": sym,
                     "listed_ts": int(ts), "status": status})
for it in c.get("https://www.okx.com/api/v5/public/instruments?instType=SPOT").json()["data"]:
    add("okx", "spot", it["baseCcy"], it["quoteCcy"], it["instId"], int(it["listTime"] or 0) / 1000, it["state"])
for it in c.get("https://www.okx.com/api/v5/public/instruments?instType=SWAP").json()["data"]:
    base, quote = it["instFamily"].split("-")[:2] if it.get("instFamily") else (it["ctValCcy"], it["settleCcy"])
    add("okx", "perp", base, quote, it["instId"], int(it["listTime"] or 0) / 1000, it["state"])
for it in c.get("https://api.gateio.ws/api/v4/spot/currency_pairs").json():
    add("gate", "spot", it["base"], it["quote"], it["id"], it.get("buy_start") or it.get("sell_start"), it["trade_status"])
for it in c.get("https://api.bitget.com/api/v2/spot/public/symbols").json()["data"]:
    add("bitget", "spot", it["baseCoin"], it["quoteCoin"], it["symbol"], int(it["openTime"] or 0) / 1000, it["status"])
for sym, ts, _ in json.load(open("liquid/binance_listings.json")):
    if sym.endswith("USDT"):
        add("binance", "spot", sym[:-4], "USDT", sym, ts, "listed (incl. delisted)")
json.dump(rows, open("cex/catalog_all.json", "w"))
from collections import Counter
print(Counter((r["exchange"], r["market"]) for r in rows))
