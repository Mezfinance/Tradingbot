# Box patch (idempotent): v2 decode + v2 line + router-rule measure-only (--route-rules / --route-feed) for
# /root/ls/localshred_consumer.py and /root/ls/run_localshred.py. Needs route_rules.py + router_rules.json in /root/ls first.
~/sv/bin/python - /root/ls <<'PY'
import os, py_compile, shutil, sys, time
BOX_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/ls"
os.chdir(BOX_DIR)
for need in ("route_rules.py", "router_rules.json"):
    if not os.path.exists(need):
        print("ABORT: %s/%s missing -- scp it first; nothing written" % (BOX_DIR, need)); sys.exit(1)

def step(s, edits, label):
    for a, b in edits:
        n = s.count(a)
        if n != 1:
            print("ABORT (%s): anchor found %d times (need 1), nothing written:\n%r" % (label, n, a)); sys.exit(1)
    for a, b in edits:
        s = s.replace(a, b)
    return s


# ---------------- localshred_consumer.py ----------------
s = open("localshred_consumer.py").read()
orig = s
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

# Step C: route measure-only mode + a 'routed' info line (skipped if already applied)
if "route_measure_only" not in s:
    C1 = "                 is_armed: Optional[Callable] = None, route_classify: Optional[Callable] = None):\n"
    C2 = "        self.cpi_txs_armed = 0\n        self.cpi_routers_armed: dict = {}\n"
    C3 = ("        verdict = self.route_classify(program_id, data, accts, account_keys, token)   # token for mint_order rules\n"
          "        if verdict not in (\"buy\", \"sell\", \"buyer_plus\"):\n")
    C4 = "                print(self.v2_line(), flush=True)\n"
    C5 = "    def router_top(self, n: int = 5):\n"
    s = step(s, [
        (C1, "                 is_armed: Optional[Callable] = None, route_classify: Optional[Callable] = None,\n"
             "                 route_measure_only: bool = False):\n"),
        (C2, C2 + "        # Router recovery tally (route_classify set): one count per router instruction on a TRACKED pool, by\n"
                  "        # verdict (buy / sell / buyer_plus = tail, counted a buy / drop). route_measure_only=True tallies\n"
                  "        # but never feeds on_swap, so the gate and its fires are exactly as with recovery off.\n"
                  "        self.route_measure_only = route_measure_only\n"
                  "        self.route_counts: dict = {}\n"),
        (C3, "        verdict = self.route_classify(program_id, data, accts, account_keys, token)   # token for mint_order rules\n"
             "        self.route_counts[verdict] = self.route_counts.get(verdict, 0) + 1\n"
             "        if self.route_measure_only:\n"
             "            return None                                      # measure only: count, never feed the gate\n"
             "        if verdict not in (\"buy\", \"sell\", \"buyer_plus\"):\n"),
        (C4, C4 + "                if self.route_classify is not None:\n"
                  "                    print(self.route_line(), flush=True)\n"),
        (C5, "    def route_line(self) -> str:\n"
             "        \"\"\"The 'routed' info line: router instructions on tracked pools, by verdict (cumulative).\"\"\"\n"
             "        c = self.route_counts\n"
             "        n = sum(c.values())\n"
             "        rec = c.get(\"buy\", 0) + c.get(\"sell\", 0)\n"
             "        pct = 100.0 * rec / n if n else 0.0\n"
             "        return (\"routed (tracked pools): buy=%d sell=%d tail=%d drop=%d -> rule-decoded %d of %d (%.1f%%)%s\"\n"
             "                % (c.get(\"buy\", 0), c.get(\"sell\", 0), c.get(\"buyer_plus\", 0), c.get(\"drop\", 0), rec, n, pct,\n"
             "                   \" [measure only]\" if self.route_measure_only else \" [feeding the gate]\"))\n\n" + C5),
    ], "route measure")
    print("step C: route measure-only + routed line added")
else:
    print("step C: route measure-only already present")


# Step E: name the routers behind the tail count on the routed line (skipped if already applied)
if "route_tail" not in s:
    E1 = "        self.route_counts: dict = {}\n"
    E2 = "        self.route_counts[verdict] = self.route_counts.get(verdict, 0) + 1\n"
    E3 = "                   \" [measure only]\" if self.route_measure_only else \" [feeding the gate]\"))\n"
    s = step(s, [
        (E1, E1 + "        self.route_tail: dict = {}             # buyer_plus verdicts by router program (route_tail_line)\n"),
        (E2, E2 + "        if verdict == \"buyer_plus\":                         # which unruled routers fill the tail\n"
                  "            self.route_tail[program_id] = self.route_tail.get(program_id, 0) + 1\n"),
        (E3, "                   \" [measure only]\" if self.route_measure_only else \" [feeding the gate]\")\n"
             "                + self.route_tail_line())\n\n"
             "    def route_tail_line(self, n: int = 6) -> str:\n"
             "        \"\"\"Top unruled routers behind the tail count (short ids), so the next raw fetch targets them.\"\"\"\n"
             "        top = sorted(self.route_tail.items(), key=lambda kv: kv[1], reverse=True)[:n]\n"
             "        return (\"  tail top: \" + \" \".join(\"%s=%d\" % (k[:8], v) for k, v in top)) if top else \"\"\n"),
    ], "route tail")
    print("step E: tail-by-router on the routed line added")
else:
    print("step E: tail-by-router already present")


new_lc, orig_lc = s, orig

# ---------------- run_localshred.py: --route-rules / --route-feed (step D) ----------------
s = open("run_localshred.py").read()
orig = s
if "route_rules" not in s:
    D1 = '    ap.add_argument("--arm-delay", type=float, default=4.0, help="seconds after first sight to read the pool account")\n'
    D2 = "    consumer = lc.Consumer(gate, tier=args.tier, on_fire=on_fire,"
    s = step(s, [
        (D1, D1 + '    ap.add_argument("--route-rules", default=None, metavar="PATH", help="router_rules.json: classify routed (CPI) swaps on tracked pools and print a routed line every 60 s (measure only unless --route-feed)")\n'
                  '    ap.add_argument("--route-feed", action="store_true", help="--route-rules: ALSO feed recovered routed swaps into the gate (changes fires); default off = measure only")\n'),
        (D2, "    _route_cls = None\n"
             "    if args.route_rules:\n"
             "        import route_rules as _rr\n"
             "        _route_cls, _n_rules = _rr.make_classifier(args.route_rules)\n"
             "        print(\"[route] %d router rules from %s (%s)\" % (_n_rules, args.route_rules,\n"
             "              \"FEEDING the gate\" if args.route_feed else \"measure only\"), flush=True)\n"
             "    consumer = lc.Consumer(gate, route_classify=_route_cls, route_measure_only=not args.route_feed, tier=args.tier, on_fire=on_fire,"),
    ], "run_localshred flags")
    print("step D: run_localshred --route-rules/--route-feed added")
else:
    print("step D: run_localshred route flags already present")
new_rl, orig_rl = s, orig

# ---------------- write (only after every anchor matched) ----------------
ts = time.strftime("%Y%m%d%H%M%S")
for p, new, old in (("localshred_consumer.py", new_lc, orig_lc), ("run_localshred.py", new_rl, orig_rl)):
    if new == old:
        print("%s: already applied; nothing written" % p); continue
    tmp = p + ".tmp"
    open(tmp, "w").write(new)
    py_compile.compile(tmp, doraise=True)
for p, new, old in (("localshred_consumer.py", new_lc, orig_lc), ("run_localshred.py", new_rl, orig_rl)):
    if new != old:
        shutil.copy2(p, p + ".bak-route-" + ts)
        shutil.move(p + ".tmp", p)
        print("%s: patched OK; backup at %s.bak-route-%s" % (p, p, ts))
py_compile.compile("route_rules.py", doraise=True)

sys.path.insert(0, BOX_DIR)
import localshred_consumer as lc, route_rules as rr
cls, n = rr.make_classifier("router_rules.json")
keys = ["PAYER", "POOL", "USER", "GCONF", "TOKEN", lc.WSOL]
got = [(sw.is_buy, sw.v2) for d in (lc.BUY_V2, lc.SELL_V2, lc.BUY_EXACT_QUOTE_IN_V2, lc.BUY_BASE_OUT, lc.SELL)
       for sw in [lc.classify_instruction(lc.PROGRAM, [1,2,3,4,5], d + bytes(16), keys, "PAYER")]]
assert got == [(True, True), (False, True), (True, True), (True, False), (False, False)], got
c = lc.Consumer(None, route_classify=cls, route_measure_only=True, runner_factory=lambda: None)
assert c.route_measure_only and n > 0, n
c.route_tail["GYLVJVVfKTuEKbxcmJnpi1v6ygsRost1sx1pjeoYxoyk"] = 3
assert "tail top: GYLVJVVf=3" in c.route_line(), c.route_line()
src = open("run_localshred.py").read()
assert "--route-rules" in src and "route_measure_only=not args.route_feed" in src
print("self-check OK: %d router rules load, %d denylisted routers; v2 decode %s; measure-only consumer builds; tail line OK" % (n, len(rr.DENYLIST), got))
PY
