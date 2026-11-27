"""Per-router swap-direction decode for the top aggregator/sniper-bot routers (2026-11-27).

The front-run gate is symbol-flip fragile: one mis-classified buyer/seller flips a slot and the fire. So
for the routed pumpswap flow (43-46% of fresh-pool txs, spread over ~170 programs) we do NOT guess
direction. Two tiers, set by the owner:

  - TOP routers (PIN_TARGETS, ~80% of routed volume): a per-router rule that gives the TRUE buy/sell
    direction, validated against Helius ground truth to ~100% PRECISION (never wrong; it may abstain).
  - TAIL (everything else): `buyer_plus` -- recover (pool, buyer=fee payer) and count it a buy (snipers on
    fresh pools overwhelmingly buy). The accepted approximation for the long tail only.
  - A pin-target with no validated rule yet -> DROP (not a guess). A drop is a known, recoverable miss; a
    wrong direction breaks the gate.
  - DENYLIST (flash/arb programs like FLASHX): DROP -- their fee-payer is a wash buy+sell, pure noise.

How a rule is found (`find_rule`): the box logs, per routed tx on a pin-target, the ROUTER instruction's
pre-execution view (data + account indices + resolved keys) plus the signature. Helius `getTransaction`
gives the TRUE direction from the inner pumpswap CPI (`label_from_tx`, reusing our own discriminator +
base/quote flip). We then fit the shred-visible router features to that truth: almost every router is an
Anchor program with separate buy/sell instructions, so the 8-byte discriminator maps cleanly to direction
(`disc` rule); a few use one swap instruction and direction is "which mint is WSOL" at a fixed account
index (`wsol_index` rule, the Jupiter case). A rule is accepted only when its held-out PRECISION is 1.0
(every fire correct); coverage < 1.0 just means it abstains on some txs, which is a safe drop. The
discriminator/layout is pool- and token-independent, so buy AND sell samples can be sourced from ANY of a
router's pumpswap swaps, not only fresh-pool captures where sells are scarce.

The pure parts (apply_rule / find_rule / label_from_tx / router_classify) are fully unit-tested offline;
the Helius fetch is exercised live like every other fetch path. Validated rules live in a json
(`load_rules`) so the live box picks them up without a code change.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .localshred_consumer import PROGRAM as PUMPSWAP, WSOL, SELL, BUY_DISCS

GETTX_CREDITS = 10
MIN_PER_DIR = 4            # a router needs at least this many buy AND sell samples to fit/validate a rule
PRECISION_FLOOR = 1.0      # a rule is accepted only when it is NEVER wrong on held-out (fragile gate)

# --- the top routers we commit to pinning (from the 2026-11-27 Phase 0 breakdown; refine with the box's
# top-20 print). A pin-target with no validated rule DROPS rather than guessing. ---------------------------
PIN_TARGETS = frozenset({
    "CtjS9o7rj3UuiNSjUdh4vHih5WiEoLXDk6XE1y3LN8Vg",
    "DiabLokxisGR8P4Qqp2QL6PmBhhVsdzPstCE4WoLxc5z",
    "BmGwqQXtzu8tLH6TiG5DEVakNs4chNo2boVSov5rr3DG",
    "3vHAVbqXVS4i9gqmfuxsWMb92HRncQo38uQCKDRJyBHg",
    "H16rUHa5roA8T8cqDpCqGWRQsXqDEgrWWiSww7GBWa5t",
    "9KGuApSJZmxHaFzk7pYM6W3feZvowVT1q6thnEGsTfzv",
    "JDk32MFrBeUpA9AN7CciPryokhSDxuWEAJdJfQxqDrpE",
    "9is6NreU9HdcLBdph8fe4CxfYwMu6fjoBB6BDZFBvKNh",
    "Fo31nF1C8wZKjqExaZQJ3sdPUL8vLMHDeB8mAFFYTfa7",
    "6Vo3245eszAb5wuqEMw8mGdbfRUdKbHhDHP5LcaGuTAB",
    "A91gCSL2VyJ7M4fsyaGQbKN9VouCVR71hpn6MLqhYkX",
    "3WTz7UpFBzSVEs7C6KcUeUgFhQDJVhJAKsdhBcsmEbcu",
    "J7cs9NCmeP7MrqTF1XcsP5UAE3S7NxUwszZCFFeRQnMx",
    "Cvf5Gb7wgS3LvCDeecw5yoaAWo8SyPG3624CzRFkrfZp",
    "HVi6VyyLvTtFTA8f8atavxVjUKi8WjmnydfKgoZKzt7H",
    "6MWVTis8rmmk6Vt9zmAJJbmb3VuLpzoQ1aHH4N6wQEGh",
    "PopStBcAMfoEXNr2mZDdinxZUgmYN2UNeEt9bti6aKY",
    "deadpz4wo5sqhEKWZkNFmqsjx1z4h6NaJWEuZVAcSCB",
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",   # Jupiter v6 -- public IDL, direction is free
})

# flash-loan / arbitrage programs: their fee-payer buys AND sells in one tx (wash), so counting it a buyer
# is pure noise for the gate. DROP, never decode. (FLASHX = flash/arb, seen at ~2% of routed.)
DENYLIST = frozenset({
    "FLASHX8DrLbgeR8FcfNV1F5krxYcYMUdBkrP1EPBtxB9",
})

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58IDX = {c: i for i, c in enumerate(_B58)}


def _b58decode(s: str) -> bytes:
    """Decode a base58 string to bytes (for a jsonParsed inner instruction's `data`). [] on a bad char."""
    n = 0
    for ch in s:
        v = _B58IDX.get(ch)
        if v is None:
            return b""
        n = n * 58 + v
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = 0
    for ch in s:
        if ch == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + body


# --- rule application (pure) ------------------------------------------------------------------------------

def apply_rule(rule: dict, data: bytes, accounts, account_keys) -> Optional[str]:
    """Apply a validated per-router rule to a routed instruction's pre-execution view. Returns "buy" / "sell",
    or None (abstain -> the caller DROPS, which is safe). `data` = the router ix data bytes; `accounts` = its
    account-index list; `account_keys` = the tx's resolved account keys."""
    kind = rule.get("kind")
    if kind == "disc":
        return rule["map"].get(data[:8].hex())                    # unknown discriminator -> None (abstain)
    if kind == "wsol_index":
        i = int(rule["index"])
        if i < 0 or i >= len(accounts):
            return None
        ai = int(accounts[i])
        if ai < 0 or ai >= len(account_keys):
            return None
        return rule["wsol_is"] if account_keys[ai] == WSOL else rule["other"]
    if kind == "byte_flag":                                        # a side/swap_base_in arg past the discriminator
        o = int(rule["offset"])
        if o < 0 or o >= len(data):
            return None
        return rule["map"].get("%02x" % data[o])                  # unseen byte value -> None (abstain)
    return None


# --- ground-truth labeling from the inner pumpswap CPI (Helius) -------------------------------------------

def _dir_from_pumpswap(disc: bytes, base_mint: str, quote_mint: str) -> Optional[str]:
    """True token direction of an inner pumpswap swap, reusing the box flip. SELL ix / BUY ix + the
    base/quote WSOL orientation -> is the TOKEN being bought or sold."""
    if disc == SELL:
        inst_buy = False
    elif disc in BUY_DISCS:
        inst_buy = True
    else:
        return None
    base_is_sol = base_mint == WSOL
    quote_is_sol = quote_mint == WSOL
    if base_is_sol == quote_is_sol:               # need exactly one WSOL side (a SOL-paired pool)
        return None
    token_is_base = not base_is_sol
    token_buy = (inst_buy == token_is_base)       # the box flip
    return "buy" if token_buy else "sell"


def label_from_tx(tx_json: dict, pool: Optional[str] = None) -> Optional[str]:
    """TRUE direction of the routed swap, from the tx's inner instructions (meta carries CPIs; the shred
    stream does not). Finds the inner pumpswap instruction (touching `pool`, or any if pool is None), reads
    its discriminator + base/quote mints, returns "buy"/"sell". None if not a clean SOL-paired pumpswap swap.

    `tx_json` is a Solana getTransaction result (jsonParsed): inner instructions for an unknown program
    come back as {"programId", "accounts": [pubkeys], "data": base58}."""
    meta = (tx_json or {}).get("meta") or {}
    for grp in meta.get("innerInstructions") or []:
        for ix in grp.get("instructions") or []:
            if ix.get("programId") != PUMPSWAP:
                continue
            accts = ix.get("accounts") or []                      # pubkeys (jsonParsed)
            if len(accts) < 5 or (pool is not None and pool not in accts):
                continue
            data = _b58decode(ix.get("data") or "")
            if len(data) < 8:
                continue
            d = _dir_from_pumpswap(data[:8], accts[3], accts[4])
            if d:
                return d
    return None


def _sig_of(tx: dict) -> Optional[str]:
    sigs = ((tx or {}).get("transaction") or {}).get("signatures") or []
    return sigs[0] if sigs else None


def _router_top_ix(tx: dict, router: str):
    """The router's top-level instruction (the shred-visible view): (data bytes, account pubkeys). None if
    the router has no top-level instruction here (e.g. it was itself a CPI)."""
    msg = ((tx or {}).get("transaction") or {}).get("message") or {}
    for ix in msg.get("instructions") or []:
        if ix.get("programId") != router:
            continue
        data = _b58decode(ix.get("data") or "")
        accts = list(ix.get("accounts") or [])                    # pubkeys (jsonParsed)
        if data and accts:
            return data, accts
    return None


def fetch_router_samples(helius, router: str, limit: int = 100) -> list:
    """Labeled samples for one router straight from Helius: getTransactionsForAddress(router) gives each
    tx's TOP-LEVEL router instruction (data + account pubkeys = the shred-visible view, for fitting) AND its
    inner pumpswap CPI (the TRUE direction, for the label) in one call. The sample is normalised to
    (accounts=identity indices, keys=the instruction's account pubkeys) so apply_rule reads the i-th account
    of the router instruction whether the source is this fetch (pubkey list) or the live shred path (indices
    into the resolved keys) -- the rule's index is instruction-relative in both."""
    out = []
    for tx in helius.history(router, limit) or []:
        d = label_from_tx(tx, None)                               # true direction from the inner pumpswap swap
        if not d:
            continue
        ix = _router_top_ix(tx, router)
        if ix is None:
            continue
        data, accts = ix
        out.append({"router": router, "sig": _sig_of(tx), "pool": None,
                    "payer": accts[0] if accts else None, "data": data,
                    "accounts": list(range(len(accts))), "keys": accts, "dir": d})
    return out


def fetch_label(helius, sig: str, pool: str) -> Optional[str]:
    """getTransaction(sig) -> label_from_tx. None on any failure (the sample is left unlabeled)."""
    res = helius._rpc("getTransaction",
                      [sig, {"maxSupportedTransactionVersion": 1, "encoding": "jsonParsed",
                             "commitment": "confirmed"}], GETTX_CREDITS)
    if not isinstance(res, dict):
        return None
    return label_from_tx(res, pool)


def _routed_sample(tx: dict) -> Optional[dict]:
    """From a getTransactionsForAddress(PUMPSWAP) tx: a labeled ROUTER sample when the pumpswap swap is an
    inner CPI under a non-pumpswap top-level instruction (a ROUTED swap). None for a direct swap (pumpswap is
    top-level) or a non-swap. The router's top-level instruction is the shred-visible view (for fitting); the
    inner pumpswap CPI is the TRUE label -- both from the one tx, naturally balanced and recent."""
    meta = (tx or {}).get("meta") or {}
    top = (((tx or {}).get("transaction") or {}).get("message") or {}).get("instructions") or []
    for grp in meta.get("innerInstructions") or []:
        idx = grp.get("index")
        if idx is None or idx >= len(top):
            continue
        for ins in grp.get("instructions") or []:
            if ins.get("programId") != PUMPSWAP:
                continue
            accts = ins.get("accounts") or []
            data = _b58decode(ins.get("data") or "")
            if len(accts) < 5 or len(data) < 8:
                continue
            d = _dir_from_pumpswap(data[:8], accts[3], accts[4])
            if not d:
                continue
            rix = top[idx]
            router = rix.get("programId")
            if router == PUMPSWAP:                                # pumpswap was itself top-level = direct, skip
                return None
            rdata = _b58decode(rix.get("data") or "")
            raccts = list(rix.get("accounts") or [])
            if not rdata or not raccts:
                return None
            return {"router": router, "sig": _sig_of(tx), "pool": None, "payer": raccts[0],
                    "data": rdata, "accounts": list(range(len(raccts))), "keys": raccts, "dir": d}
    return None


def fetch_pumpswap_routed(helius, max_pages: int = 6, per_page: int = 1000) -> list:
    """Recent routed pumpswap swaps, bucketed by their top-level router, straight from the pumpswap program.
    One desc-paginated getTransactionsForAddress(PUMPSWAP) sweep is a recent, balanced, volume-weighted sample
    of every router in proportion to its real pumpswap activity: the biggest routers saturate in the first
    pages, the tail stays thin (and the tail is buyer_plus anyway). Far better than one fetch per router --
    no genesis-window bias (the sortOrder=asc trap that starved the two biggest routers to 1 sample) and the
    buy/sell split is natural, not a one-sided slice. ~10 credits per 100 txs."""
    out: list = []
    token = None
    for _ in range(max_pages):
        opts = {"transactionDetails": "full", "sortOrder": "desc", "limit": per_page,
                "maxSupportedTransactionVersion": 1, "encoding": "jsonParsed"}
        if token:
            opts["paginationToken"] = token
        res = helius._rpc("getTransactionsForAddress", [PUMPSWAP, opts], max(10, 10 * per_page // 100))
        if not isinstance(res, dict):
            break
        page = res.get("data") or []
        for tx in page:
            s = _routed_sample(tx)
            if s:
                out.append(s)
        token = res.get("paginationToken")
        if not page or not token:
            break
    return out


# --- reverse-engineering: fit a direction rule to labeled router samples (held-out precision) -------------

def _disc_rule(train) -> Optional[dict]:
    """A discriminator->direction map, only if every seen 8-byte discriminator is pure (one direction)."""
    m: dict = {}
    for s in train:
        d = s["data"][:8].hex()
        prev = m.get(d)
        if prev is None:
            m[d] = s["dir"]
        elif prev != s["dir"]:
            return None                                           # a discriminator used for both -> not a disc rule
    return {"kind": "disc", "map": m} if m else None


def _wsol_rule(train) -> Optional[dict]:
    """A single account index whose being-WSOL separates direction cleanly over the training set."""
    max_len = max((len(s["accounts"]) for s in train), default=0)
    for i in range(max_len):
        mapping: dict = {}                                        # (mint==WSOL) -> direction
        ok = True
        for s in train:
            accts = s["accounts"]
            if i >= len(accts) or accts[i] >= len(s["keys"]):
                ok = False
                break
            is_wsol = s["keys"][accts[i]] == WSOL
            prev = mapping.get(is_wsol)
            if prev is None:
                mapping[is_wsol] = s["dir"]
            elif prev != s["dir"]:
                ok = False
                break
        if ok and len(mapping) == 2:                              # both WSOL and non-WSOL seen, each pure
            return {"kind": "wsol_index", "index": i,
                    "wsol_is": mapping[True], "other": mapping[False]}
    return None


def _byte_flag_rule(train) -> Optional[dict]:
    """A single data byte past the 8-byte discriminator whose value separates direction cleanly -- an Anchor
    `side`/`swap_base_in`/enum arg the disc rule (first 8 bytes only) misses. The holdout precision guard in
    find_rule rejects a spurious split (e.g. a byte of the amount)."""
    min_len = min((len(s["data"]) for s in train), default=0)
    for o in range(8, min_len):
        m: dict = {}                                              # byte value -> direction
        ok = True
        for s in train:
            v = s["data"][o]
            prev = m.get(v)
            if prev is None:
                m[v] = s["dir"]
            elif prev != s["dir"]:
                ok = False
                break
        if ok and len(m) >= 2:                                    # both directions seen at distinct byte values
            return {"kind": "byte_flag", "offset": o, "map": {("%02x" % k): v for k, v in m.items()}}
    return None


def _score(rule: dict, test) -> tuple[int, int, int]:
    """(correct, fired, n): how often the rule fires on held-out, and how often it is right when it fires."""
    correct = fired = 0
    for s in test:
        pred = apply_rule(rule, s["data"], s["accounts"], s["keys"])
        if pred is not None:
            fired += 1
            if pred == s["dir"]:
                correct += 1
    return correct, fired, len(test)


def find_rule(labeled, seed: int = 0) -> dict:
    """Fit a direction rule to one router's labeled samples. Returns a report:
    {rule, kind, n, buys, sells, fired, correct, coverage, precision, accepted, reason}. A rule is accepted
    only when held-out precision == PRECISION_FLOOR (never wrong) with both directions present. `labeled` =
    list of {"data": bytes, "accounts": [int], "keys": [str], "dir": "buy"/"sell"}."""
    labeled = [s for s in labeled if s.get("dir") in ("buy", "sell")]
    buys = sum(1 for s in labeled if s["dir"] == "buy")
    sells = len(labeled) - buys
    base = {"rule": None, "kind": None, "n": len(labeled), "buys": buys, "sells": sells,
            "fired": 0, "correct": 0, "coverage": 0.0, "precision": 0.0, "accepted": False}
    if buys < MIN_PER_DIR or sells < MIN_PER_DIR:
        return {**base, "reason": "too few samples per direction (need %d buy & %d sell)" % (MIN_PER_DIR, MIN_PER_DIR)}
    order = sorted(range(len(labeled)), key=lambda i: (hash((seed, i)) & 0xffffffff))
    cut = max(1, int(len(order) * 0.7))
    train = [labeled[i] for i in order[:cut]]
    test = [labeled[i] for i in order[cut:]] or train            # tiny n: score on train (precision still guards)
    best = None
    for fit in (_disc_rule, _wsol_rule, _byte_flag_rule):         # structural rules first, byte search last
        rule = fit(train)
        if rule is None:
            continue
        correct, fired, n = _score(rule, test)
        precision = (correct / fired) if fired else 0.0
        coverage = (fired / n) if n else 0.0
        cand = {**base, "rule": rule, "kind": rule["kind"], "fired": fired, "correct": correct,
                "coverage": round(coverage, 4), "precision": round(precision, 4),
                "accepted": precision >= PRECISION_FLOOR and fired > 0,
                "reason": "ok" if precision >= PRECISION_FLOOR and fired > 0 else "precision %.3f below floor" % precision}
        if cand["accepted"]:
            return cand                                           # prefer the first clean rule (disc before wsol)
        if best is None or cand["precision"] > best["precision"]:
            best = cand
    return best or {**base, "reason": "no discriminator or WSOL-index rule separates direction"}


# --- the validated decoder table + the two-tier classifier ------------------------------------------------

RULES: dict = {}            # program_id -> validated rule; empty until find_rule output is loaded


def load_rules(path: str) -> dict:
    """Load validated per-router rules from a json {program_id: rule}. Missing/bad file -> {} (fail safe:
    an empty table just means every pin-target DROPS, never a wrong direction)."""
    try:
        with open(path) as f:
            data = json.load(f)
        return {str(k): v for k, v in data.items() if isinstance(v, dict) and v.get("kind")}
    except (OSError, ValueError):
        return {}


def load_samples(path: str) -> list:
    """Load the box's router-samples JSONL. Each line: {"router","sig","pool","payer","data"(hex),
    "accounts"[int],"keys"[str]}. `data` is coerced to bytes. A bad line is skipped."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            raw = r.get("data")
            try:
                data = bytes.fromhex(raw) if isinstance(raw, str) else bytes(raw or b"")
            except ValueError:
                continue
            out.append({"router": r.get("router"), "sig": r.get("sig"), "pool": r.get("pool"),
                        "payer": r.get("payer"), "data": data,
                        "accounts": [int(a) for a in (r.get("accounts") or [])],
                        "keys": list(r.get("keys") or []), "dir": r.get("dir")})
    return out


def study(samples: list, counts: Optional[dict] = None, seed: int = 0) -> dict:
    """Fit a direction rule per router over labeled samples, and (with `counts` = {program: routed-tx
    volume}, from the box's full cpi_routers) report VOLUME-WEIGHTED COVERAGE: what share of the routed flow
    we can now decode with a validated true-direction rule, the denylisted (arb) share, and the uncovered
    share (-> buyer_plus / drop). Samples must already carry `dir` (buy/sell) from labeling."""
    counts = counts or {}
    by_router: dict = {}
    for s in samples:
        by_router.setdefault(s.get("router"), []).append(s)
    routers, accepted = [], {}
    for prog, ss in by_router.items():
        rep = find_rule(ss, seed=seed)
        rep["router"] = prog
        rep["samples"] = len(ss)
        rep["count"] = int(counts.get(prog, 0))
        rep["pin_target"] = prog in PIN_TARGETS
        rep["denylisted"] = prog in DENYLIST
        routers.append(rep)
        if rep["accepted"] and prog not in DENYLIST:
            accepted[prog] = rep["rule"]
    routers.sort(key=lambda r: -(r["count"] or r["samples"]))
    out = {"routers": routers, "accepted": accepted, "n_routers": len(by_router),
           "n_samples": len(samples), "n_accepted": len(accepted)}
    if counts:
        total = sum(counts.values())
        covered = sum(counts.get(p, 0) for p in accepted)
        deny = sum(v for p, v in counts.items() if p in DENYLIST)
        out.update({"total_volume": total, "covered_volume": covered,
                    "coverage": (covered / total) if total else 0.0,
                    "denylist_volume": deny, "uncovered_volume": total - covered - deny,
                    "total_routers": len(counts)})
    return out


def render(report: dict) -> str:
    head = "%d routers, %d samples, %d validated rules (precision 1.0)" % (
        report["n_routers"], report["n_samples"], report["n_accepted"])
    lines = ["ROUTER DIRECTION -- per-router buy/sell rule from Helius-truth labels", head]
    if "coverage" in report:
        t = report["total_volume"] or 1
        lines.append("COVERAGE: validated rules decode %.1f%% of routed volume (%d/%d txns across %d routers); "
                     "denylist(arb) %.1f%%; uncovered %.1f%% -> buyer_plus/drop" % (
                         100.0 * report["covered_volume"] / t, report["covered_volume"], report["total_volume"],
                         report.get("total_routers", report["n_routers"]),
                         100.0 * report["denylist_volume"] / t, 100.0 * report["uncovered_volume"] / t))
        un = [r for r in report["routers"] if not r["accepted"] and not r["denylisted"]]
        if un:
            lines.append("top UNCOVERED by volume (what blocks fuller coverage):")
            for r in un[:12]:
                lines.append("  %-44s vol=%-6d n=%-4d buy=%-4d sell=%-4d  %s" % (
                    r["router"] or "?", r["count"], r["n"], r["buys"], r["sells"], r.get("reason", "")))
    lines.append("\n%-8s %-5s %6s %5s %5s %5s %-11s %5s %5s  %s" % (
        "router", "flags", "vol", "n", "buy", "sell", "rule", "prec", "cov", "status"))
    for r in report["routers"]:
        flags = ("P" if r["pin_target"] else "-") + ("D" if r["denylisted"] else "-")
        tag = "ACCEPT" if r["accepted"] else r.get("reason", "")
        lines.append("%-8s %-5s %6d %5d %5d %5d %-11s %5.2f %5.2f  %s" % (
            (r["router"] or "?")[:8], flags, r["count"], r["n"], r["buys"], r["sells"],
            r["kind"] or "-", r["precision"], r["coverage"], tag))
    return "\n".join(lines)


def router_classify(program: str, data: bytes, accounts, account_keys,
                    rules: Optional[dict] = None,
                    pin_targets=PIN_TARGETS, denylist=DENYLIST) -> str:
    """Classify a routed top-level instruction into one of: "buy" / "sell" (a validated rule fired),
    "buyer_plus" (tail -> recover buyer, count a buy), "drop" (denylist, or a pin-target with no/abstaining
    rule -> never guess direction on the bulk). Encodes the owner's risk split exactly."""
    rules = RULES if rules is None else rules
    if program in denylist:
        return "drop"
    if program in rules:
        d = apply_rule(rules[program], data, accounts, account_keys)
        return d if d in ("buy", "sell") else "drop"              # rule abstains -> drop (safe)
    if program in pin_targets:
        return "drop"                                             # committed to pin; do not guess
    return "buyer_plus"                                           # the long tail (accepted approximation)
