"""Part 3a: events. Earliest listing per token across all venues -> type A (the token's first CEX listings, within
3 days of its first) or type B (a major venue listing a token already on another venue for 7+ days)."""
import json, collections, time
SINCE, UNTIL = 1727740800, time.time() - 8 * 86400
STABLE = {"USDT", "USDC", "USD", "FDUSD", "TUSD", "DAI", "USDE", "PYUSD", "EUR", "EURC", "USD1", "RLUSD", "USDD", "BUSD", "WBTC", "WETH", "STETH", "WSTETH", "CBBTC", "BTC", "ETH"}
first = collections.defaultdict(list)  # base -> [(ts, exchange, market, symbol)]
for r in json.load(open("cex/catalog_all.json")):
    first[r["base"]].append((r["listed_ts"], r["exchange"], r["market"], r["symbol"]))
for k, v in json.load(open("cex/catalog_b.json")).items():
    if v["first_day"]:
        first[v["base"].upper()].append((v["first_day"], v["exchange"], "spot", v["symbol"]))
events = []
for base, ls in first.items():
    if base in STABLE or base.startswith("R") and False: continue
    ls.sort()
    t_first = ls[0][0]
    seen = set()
    for ts, ex, mk, sym in ls:
        if mk != "spot" or ex in seen or not (SINCE <= ts <= UNTIL): continue
        seen.add(ex)
        kind = "A" if ts - t_first <= 3 * 86400 else ("B" if ts - t_first >= 7 * 86400 else None)
        if kind is None: continue
        prior = sorted({e for t, e, m, s in ls if m == "spot" and t <= ts - 7 * 86400})
        perps = [(t, e) for t, e, m, s in ls if m == "perp"]
        events.append({"base": base, "exchange": ex, "symbol": sym, "ts": ts, "kind": kind, "first_ts": t_first,
                       "prior_venues": prior, "perp_ts": min((t for t, e in perps), default=None)})
json.dump(events, open("cex/events.json", "w"))
c = collections.Counter((e["kind"], e["exchange"]) for e in events)
for k in sorted(c): print(k, c[k])
