# Rebuild Strategy 1 trades from public data (about 45 min). Run from an empty working folder holding scripts/ = copies of research/listings/*.py; outputs cex/s1_trades.json.
set -x
cd "$(dirname "$0")"
step() { [ -f "done.$1" ] && return 0; python scripts/$2 > logs/$1.txt 2>&1 && touch "done.$1" || { echo "FAILED $1"; exit 1; }; }
mkdir -p logs
step 1 bn_list.py
step 2 cex_catalog_all.py
step 3 cex_catalog_b.py
step 4 cex_events.py
step 5 core_only.py
[ -f cex/prices.json ] || echo '{}' > cex/prices.json
step 6 cex_hours.py
step 7 hl_short.py
step 8 cex_prices.py
step 9 perp_venues.py
step 10 venue_funding.py
step 11 venue_capacity.py
step 12 s1_trades.py
echo ALL_DONE
