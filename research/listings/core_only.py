"""Keep only the events Strategy 1 trades (B Upbit / Coinbase / Binance, A Upbit) to save fetches."""
import json
CORE = {("B", "upbit"), ("B", "coinbase"), ("B", "binance"), ("A", "upbit")}
ev = json.load(open("cex/events.json")); json.dump(ev, open("cex/events_all.json", "w"))
ev = [e for e in ev if (e["kind"], e["exchange"]) in CORE]; json.dump(ev, open("cex/events.json", "w"))
print("core events", len(ev))
