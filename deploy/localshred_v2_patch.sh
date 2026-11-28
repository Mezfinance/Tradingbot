cd /root/ls && ~/sv/bin/python - <<'PY'
import py_compile, shutil, sys, time
p = "localshred_consumer.py"
s = open(p).read()
if "SELL_V2" in s:
    print("already applied: localshred_consumer.py has SELL_V2; nothing written"); sys.exit(0)
A1 = "BUY_DISCS = frozenset({BUY_EXACT_QUOTE_IN, BUY_BASE_OUT})\n"
B1 = A1 + '''# pump_amm v2 swaps (2026-11-28): Anchor sha256("global:<name>")[:8]. The big routers' flow moved to v2 (router
# study, docs/ROUTER_DECODE.md); a v1-only decode silently drops every direct v2 swap. Same account layout as v1
# (verified on the raw top-26 router fetch, 5880 v2 ixs, 17 accounts each: 0 = pool (the vault owner on all 200
# successful txs), 1 = user (== fee payer), 2 = global_config, 3 = base_mint, 4 = quote_mint), and the disc
# direction matched the payer's token-balance change on 474/474 txs, so the same orientation flip applies.
# The study's visibility tag (helius._top_level_pumpswap) is log-based and disc-agnostic, so the gate was fit
# WITH top-level v2 swaps; decoding them here closes a train/serve gap rather than opening one.
SELL_V2 = bytes.fromhex("5df6823ce7e940b2")                 # sell_v2
BUY_V2 = bytes.fromhex("b817ee6167c5d33d")                  # buy_v2
BUY_EXACT_QUOTE_IN_V2 = bytes.fromhex("c2ab1c46684d5b2f")   # buy_exact_quote_in_v2
SELL_DISCS = frozenset({SELL, SELL_V2})
BUY_DISCS_V2 = frozenset({BUY_V2, BUY_EXACT_QUOTE_IN_V2})
DIRECT_BUY_DISCS = BUY_DISCS | BUY_DISCS_V2                 # what classify_instruction decodes as a buy
'''
A2 = "        if disc == SELL:\n            inst_buy = False\n        elif disc in BUY_DISCS:\n"
B2 = "        if disc in SELL_DISCS:\n            inst_buy = False\n        elif disc in DIRECT_BUY_DISCS:\n"
for a in (A1, A2):
    n = s.count(a)
    if n != 1:
        print("ABORT: anchor found %d times (need 1), nothing written:\n%r" % (n, a)); sys.exit(1)
s = s.replace(A1, B1).replace(A2, B2)
bak = p + ".bak-prev2-" + time.strftime("%Y%m%d%H%M%S")
shutil.copy2(p, bak)
tmp = p + ".tmp"
open(tmp, "w").write(s)
py_compile.compile(tmp, doraise=True)
shutil.move(tmp, p)
import localshred_consumer as lc
keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
ok = [lc.classify_instruction(lc.PROGRAM, [1,2,3,4,5], d + bytes(16), keys, "PAYER").is_buy
      for d in (lc.BUY_V2, lc.SELL_V2, lc.BUY_EXACT_QUOTE_IN_V2, lc.BUY_BASE_OUT, lc.SELL)]
assert ok == [True, False, True, True, False], ok
print("patched OK; backup at", bak, "; self-check", ok)
PY
