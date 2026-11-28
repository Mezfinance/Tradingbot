cd /root/ls && ~/sv/bin/python - <<'PY'
import py_compile, shutil, sys, time
p = sys.argv[1] if len(sys.argv) > 1 else "localshred_consumer.py"
s = open(p).read()
orig = s

def step(s, edits, label):
    for a, b in edits:
        n = s.count(a)
        if n != 1:
            print("ABORT (%s): anchor found %d times (need 1), nothing written:\n%r" % (label, n, a)); sys.exit(1)
    for a, b in edits:
        s = s.replace(a, b)
    return s

# Step A: decode pump_amm v2 swaps (skipped if already applied)
if "SELL_V2" not in s:
    A1 = "BUY_DISCS = frozenset({BUY_EXACT_QUOTE_IN, BUY_BASE_OUT})\n"
    s = step(s, [
        (A1, A1 + '''# pump_amm v2 swaps (2026-11-28): Anchor sha256("global:<name>")[:8]. The big routers' flow moved to v2 (router
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
'''),
        ("        if disc == SELL:\n            inst_buy = False\n        elif disc in BUY_DISCS:\n",
         "        if disc in SELL_DISCS:\n            inst_buy = False\n        elif disc in DIRECT_BUY_DISCS:\n"),
    ], "v2 decode")
    print("step A: v2 decode added")
else:
    print("step A: v2 decode already present")

# Step B: v2 counter + a 'v2 decoded' info line every 60 s (skipped if already applied)
if "v2_swaps" not in s:
    D = "DIRECT_BUY_DISCS = BUY_DISCS | BUY_DISCS_V2                 # what classify_instruction decodes as a buy\n"
    P = "    payer: str     # tx fee payer (account_keys[0]) = the trading wallet, per the study\n"
    R = "    return Swap(pool=pool, token=token, is_buy=token_buy, payer=fee_payer)\n"
    I = "        self.direct_swaps = 0\n        self.cpi_txs = 0\n"
    T = "        self.direct_swaps += len(swaps)\n"
    s = step(s, [
        (D, D + "V2_DISCS = frozenset({SELL_V2}) | BUY_DISCS_V2                # the v2 subset (for the v2 info line)\n"),
        (P, P + "    v2: bool = False  # decoded from a pump_amm v2 instruction (buy_v2 / sell_v2 / buy_exact_quote_in_v2)\n"),
        (R, "    return Swap(pool=pool, token=token, is_buy=token_buy, payer=fee_payer, v2=disc in V2_DISCS)\n"),
        (I, I + "        # v2_swaps = the direct swaps that came from a v2 instruction. Every v2_log_s seconds on_tx prints one\n"
                "        # 'v2 decoded' info line (cumulative counts) so the box log shows v2 working; 0 = off.\n"
                "        self.v2_swaps = 0\n"
                "        self.v2_log_s = 60.0\n"
                "        self._v2_last = time.monotonic()\n"),
        (T, T + "        self.v2_swaps += sum(1 for sw in swaps if sw.v2)\n"
                "        if self.v2_log_s:\n"
                "            now = time.monotonic()\n"
                "            if now - self._v2_last >= self.v2_log_s:\n"
                "                self._v2_last = now\n"
                "                print(self.v2_line(), flush=True)\n"),
    ], "v2 info line")
    M = "    def router_top(self, n: int = 5):\n"
    s = step(s, [(M, "    def v2_line(self) -> str:\n"
                     "        \"\"\"The 'v2 decoded' info line: cumulative v2 swaps out of all direct swaps decoded.\"\"\"\n"
                     "        pct = 100.0 * self.v2_swaps / self.direct_swaps if self.direct_swaps else 0.0\n"
                     "        return \"v2 decoded: %d of %d direct swaps (%.1f%%)\" % (self.v2_swaps, self.direct_swaps, pct)\n\n" + M)],
             "v2 info line")
    print("step B: v2 info line added")
else:
    print("step B: v2 info line already present")

if s == orig:
    print("already applied; nothing written")
else:
  bak = p + ".bak-v2-" + time.strftime("%Y%m%d%H%M%S")
  shutil.copy2(p, bak)
  tmp = p + ".tmp"
  open(tmp, "w").write(s)
  py_compile.compile(tmp, doraise=True)
  shutil.move(tmp, p)
  print("patched OK; backup at", bak)
import importlib, localshred_consumer as lc
importlib.reload(lc)
keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
got = [(sw.is_buy, sw.v2) for d in (lc.BUY_V2, lc.SELL_V2, lc.BUY_EXACT_QUOTE_IN_V2, lc.BUY_BASE_OUT, lc.SELL)
       for sw in [lc.classify_instruction(lc.PROGRAM, [1,2,3,4,5], d + bytes(16), keys, "PAYER")]]
assert got == [(True, True), (False, True), (True, True), (True, False), (False, False)], got
print("self-check OK", got)
PY
