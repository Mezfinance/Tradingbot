"""Fetch Binance 'New Cryptocurrency Listing' announcements (catalog 48) back to 2024-09.
Writes cex/ann_binance.json: [{id, title, ts}] (ts = release time, seconds)."""
import httpx, json, time
c = httpx.Client(timeout=20, headers={"User-Agent": "Mozilla/5.0"})
U = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId=48&pageNo={}&pageSize=50"
out, stop = [], 1725148800  # 2024-09-01
for p in range(1, 200):
    for t in range(4):
        try:
            arts = c.get(U.format(p)).json()["data"]["catalogs"][0]["articles"]; break
        except Exception as e:
            time.sleep(2 ** t)
    else:
        raise SystemExit(f"page {p} failed")
    if not arts: break
    out += [{"id": a["id"], "title": a["title"], "ts": a["releaseDate"] / 1000} for a in arts]
    print(p, len(out), time.strftime("%Y-%m-%d", time.gmtime(out[-1]["ts"])), flush=True)
    if out[-1]["ts"] < stop: break
    time.sleep(0.5)
json.dump(out, open("cex/ann_binance.json", "w"))
