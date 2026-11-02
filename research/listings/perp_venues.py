"""Perp listing time per venue and base token: Gate/MEXC/KuCoin futures and Aster publish it; Bybit = first daily
trade file in public.bybit.com/trading/<SYMBOL>/; Binance (data.binance.vision), OKX and Hyperliquid come from
earlier files. Output cex/perp_venues.json {venue: {base: ts}} (checkpointed)."""
import httpx, json, re, time, datetime as dt
c = httpx.Client(timeout=30, verify="/root/.ccr/ca-bundle.crt", follow_redirects=True)
try: out = json.load(open("cex/perp_venues.json"))
except Exception: out = {}
def norm(base):
    for p in ("1000000", "100000", "10000", "1000", "1M"):
        if base.startswith(p) and len(base) > len(p) + 1: return base[len(p):]
    return base[1:] if base.startswith("k") and base[1:].isupper() else base
def put(v, base, ts):
    if not ts: return
    b = norm(base.upper() if not base.startswith("k") else base)
    out.setdefault(v, {})
    out[v][b] = min(out[v].get(b, 9e12), ts)
for x in c.get("https://api.gateio.ws/api/v4/futures/usdt/contracts").json():
    put("gate", x["name"].split("_")[0], x.get("launch_time") or x.get("create_time"))
for x in c.get("https://contract.mexc.com/api/v1/contract/detail").json()["data"]:
    if x["symbol"].endswith("_USDT"): put("mexc", x["symbol"][:-5], x["createTime"] / 1000)
for x in c.get("https://api-futures.kucoin.com/api/v1/contracts/active").json()["data"]:
    if x["symbol"].endswith("USDTM"): put("kucoin", x["baseCurrency"].replace("XBT", "BTC"), (x.get("firstOpenDate") or 0) / 1000)
for x in c.get("https://fapi.asterdex.com/fapi/v1/exchangeInfo").json()["symbols"]:
    if x["symbol"].endswith("USDT"): put("aster", x["symbol"][:-4], x.get("onboardDate", 0) / 1000)
sv = json.load(open("cex/short_venues.json"))
for coin, v in sv["hl_list"].items(): put("hyperliquid", coin, v["ts"])
for base, ts in sv["bn_perp"].items(): put("binance", base, ts)
for r in json.load(open("cex/catalog_all.json")):
    if r["exchange"] == "okx" and r["market"] == "perp": put("okx", r["base"], r["listed_ts"])
json.dump(out, open("cex/perp_venues.json", "w"))
print({k: len(v) for k, v in out.items()}, flush=True)
# Bybit: directory listing of linear perps, then the first file of each symbol we care about
need = {e["base"] for e in json.load(open("cex/events.json")) if e["exchange"] in ("upbit", "binance", "coinbase", "okx", "gate")}
idx = c.get("https://public.bybit.com/trading/").text
syms = sorted(set(re.findall(r'href="([A-Z0-9]+USDT)/"', idx)))
done = set(out.get("_bybit_checked", []))
for i, s in enumerate(syms):
    base = s[:-4]
    if norm(base) not in need or s in done: continue
    try:
        page = c.get(f"https://public.bybit.com/trading/{s}/").text
        dates = re.findall(r"(\d{4}-\d{2}-\d{2})\.csv\.gz", page)
        if dates:
            put("bybit", base, dt.datetime.strptime(min(dates), "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
    except Exception: pass
    done.add(s); time.sleep(0.1)
    if i % 50 == 0:
        out["_bybit_checked"] = sorted(done); json.dump(out, open("cex/perp_venues.json", "w")); print("bybit", i, len(syms), flush=True)
out["_bybit_checked"] = sorted(done); json.dump(out, open("cex/perp_venues.json", "w"))
print("done", {k: len(v) for k, v in out.items() if not k.startswith("_")}, flush=True)
