"""Step 1: shortability + funding. For every event: could it be shorted on Hyperliquid at the listing hour (perp
listed by then, first funding entry = listing), how long until it could; was there a Binance USD-M perp (first
daily 1h-kline file on data.binance.vision) or an OKX swap; and Hyperliquid funding a short received (+) or paid
(-) over the 7 days from the listing hour. Saved to cex/short_venues.json (checkpointed)."""
import httpx, json, re, time
c = httpx.Client(timeout=30, verify="/root/.ccr/ca-bundle.crt")
U = "https://api.hyperliquid.xyz/info"
def hl(body):
    for a in range(6):
        try:
            r = c.post(U, json=body)
            if r.status_code == 429: time.sleep(3 * (a + 1)); continue
            return r.json()
        except Exception: time.sleep(2)
    return None
try: out = json.load(open("cex/short_venues.json"))
except Exception: out = {"hl_list": {}, "bn_perp": {}, "events": {}}
meta = hl({"type": "meta"})["universe"]
for x in meta:
    if x["name"] in out["hl_list"]: continue
    f = hl({"type": "fundingHistory", "coin": x["name"], "startTime": 0}) or []
    out["hl_list"][x["name"]] = {"ts": f[0]["time"] / 1000 if f else None, "delisted": bool(x.get("isDelisted"))}
    time.sleep(0.25)
json.dump(out, open("cex/short_venues.json", "w")); print("hl coins", len(out["hl_list"]), flush=True)
def hl_coin(base):
    for n in (base, "k" + base, "1000" + base):
        if n in out["hl_list"] and out["hl_list"][n]["ts"]: return n
    return None
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
def bn_perp(base):
    if base in out["bn_perp"]: return out["bn_perp"][base]
    ts = None
    for sym in (base + "USDT", "1000" + base + "USDT"):
        try:
            x = c.get(S3, params={"delimiter": "/", "prefix": f"data/futures/um/daily/klines/{sym}/1h/", "max-keys": 3}).text
        except Exception: x = ""
        m = re.search(r"-1h-(\d{4}-\d{2}-\d{2})\.zip", x)
        if m:
            ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d")) - time.timezone; break
    out["bn_perp"][base] = ts
    return ts
okx_swap = {}
for r in json.load(open("cex/catalog_all.json")):
    if r["exchange"] == "okx" and r["market"] == "perp":
        okx_swap[r["base"]] = min(okx_swap.get(r["base"], 9e12), r["listed_ts"])
ev = json.load(open("cex/events.json"))
want = [e for e in ev if e["exchange"] in ("upbit", "binance", "coinbase", "okx", "gate")]
for i, e in enumerate(want):
    key = f"{e['kind']}:{e['exchange']}:{e['base']}:{int(e['ts'])}"
    if key in out["events"]: continue
    t0, coin = e["ts"], hl_coin(e["base"])
    rec = {"hl_coin": coin, "hl_list_ts": out["hl_list"][coin]["ts"] if coin else None,
           "bn_perp_ts": bn_perp(e["base"]), "okx_swap_ts": okx_swap.get(e["base"])}
    if coin and rec["hl_list_ts"] <= t0 + 7 * 86400:
        start = max(t0, rec["hl_list_ts"])
        f = hl({"type": "fundingHistory", "coin": coin, "startTime": int(start * 1000), "endTime": int((t0 + 7 * 86400) * 1000)}) or []
        rec["hl_funding_7d"] = sum(float(x["fundingRate"]) for x in f)  # + = longs paid shorts
        rec["hl_funding_hours"] = len(f)
        time.sleep(0.2)
    out["events"][key] = rec
    if i % 50 == 0: json.dump(out, open("cex/short_venues.json", "w")); print(i, len(want), flush=True)
json.dump(out, open("cex/short_venues.json", "w")); print("done", len(out["events"]), flush=True)
