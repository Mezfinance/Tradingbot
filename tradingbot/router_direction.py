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
from collections import Counter
from typing import Optional

from .localshred_consumer import PROGRAM as PUMPSWAP, WSOL, SELL, BUY_DISCS
from .route_rules import PIN_TARGETS, DENYLIST, apply_rule, load_rules, router_classify as _router_classify  # noqa: F401

GETTX_CREDITS = 10
MIN_PER_DIR = 4            # a router needs at least this many buy AND sell samples to fit/validate a rule
PRECISION_FLOOR = 1.0      # a rule is accepted only when it is NEVER wrong on held-out (fragile gate)

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


# --- ground-truth labeling from the inner pumpswap CPI (Helius) -------------------------------------------

# pump_amm v2 swaps (2026-11): Anchor sha256("global:<name>")[:8], checked against the raw top-26 fetch, where
# disc direction matched the payer's token-balance change on 474/474 txs. v1-only labelling silently dropped
# every router whose flow had moved to v2 (the 26 "empty" top routers).
# Defined once in localshred_consumer (the box decoder decodes them too); re-exported here.
from .localshred_consumer import BUY_V2, SELL_V2, BUY_EXACT_QUOTE_IN_V2  # noqa: E402
SELL_ALL = frozenset({SELL, SELL_V2})
BUY_ALL = frozenset(BUY_DISCS) | {BUY_V2, BUY_EXACT_QUOTE_IN_V2}


def _dir_from_pumpswap(disc: bytes, base_mint: str, quote_mint: str) -> Optional[str]:
    """True token direction of an inner pumpswap swap, reusing the box flip. SELL ix / BUY ix + the
    base/quote WSOL orientation -> is the TOKEN being bought or sold."""
    if disc in SELL_ALL:
        inst_buy = False
    elif disc in BUY_ALL:
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


def raw_record(tx: dict, router: str) -> dict:
    """Compact, label-free copy of one fetched tx for offline decoding (every tx, kept or not): the top-level
    instructions, every inner instruction (program, accounts, data), token-balance deltas per (owner, mint) and
    the fee payer's SOL delta. Enough to recover direction from balances without re-spending credits."""
    tx = tx or {}
    meta = tx.get("meta") or {}
    msg = (tx.get("transaction") or {}).get("message") or {}
    keys = [k.get("pubkey") if isinstance(k, dict) else k for k in msg.get("accountKeys") or []]

    def ixs(lst):
        return [{"p": ix.get("programId"), "a": ix.get("accounts") or [], "d": ix.get("data") or ""}
                for ix in lst or [] if "parsed" not in ix]
    bal: dict = {}
    for side, sign in (("preTokenBalances", -1), ("postTokenBalances", 1)):
        for b in meta.get(side) or []:
            k = (b.get("owner"), b.get("mint"))
            amt = int(((b.get("uiTokenAmount") or {}).get("amount")) or 0)
            bal[k] = bal.get(k, 0) + sign * amt
    pre, post = meta.get("preBalances") or [], meta.get("postBalances") or []
    return {"router": router, "sig": _sig_of(tx), "slot": tx.get("slot"), "err": meta.get("err"),
            "payer": keys[0] if keys else None,
            "payer_sol": (post[0] - pre[0]) if pre and post else None,
            "top": ixs(msg.get("instructions")),
            "inner": [{"i": g.get("index"), "ix": ixs(g.get("instructions"))}
                      for g in meta.get("innerInstructions") or []],
            "tok": [[o, m, d] for (o, m), d in bal.items() if d]}


def label_raw(rec: dict) -> Optional[tuple]:
    """(direction, token) of a raw_record, or None. Every SOL-paired pumpswap swap in the tx (v1 or v2) must
    agree (a buy+sell wash -> None), and when the fee payer's balance of that token moved, its sign must agree
    too (the payer received tokens = buy). Failed txs -> None."""
    if rec.get("err") is not None:
        return None
    dirs = set()
    tok = None
    for g in rec.get("inner") or []:
        for ix in g.get("ix") or []:
            if ix.get("p") != PUMPSWAP:
                continue
            a = ix.get("a") or []
            if len(a) < 5:
                continue
            d = _dir_from_pumpswap(_b58decode(ix.get("d") or "")[:8], a[3], a[4])
            if d:
                dirs.add(d)
                tok = a[4] if a[3] == WSOL else a[3]
    if len(dirs) != 1:
        return None
    d = next(iter(dirs))
    moved = sum(int(x[2]) for x in rec.get("tok") or [] if x[0] == rec.get("payer") and x[1] == tok)
    if moved and (moved > 0) != (d == "buy"):
        return None                                               # balances contradict the instruction: drop
    return d, tok


def samples_from_raw(path: str) -> list:
    """Labeled router samples from a --raw-out gzip JSONL (streamed): the router's first top-level instruction
    (the shred-visible view) + label_raw's direction. Same shape as fetch_router_samples / load_samples."""
    import gzip
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            lab = label_raw(rec)
            if lab is None:
                continue
            r = rec.get("router")
            ix = next((t for t in rec.get("top") or [] if t.get("p") == r and t.get("d") and t.get("a")), None)
            if ix is None:
                continue
            accts = list(ix["a"])
            out.append({"router": r, "sig": rec.get("sig"), "pool": None, "payer": rec.get("payer"),
                        "token": lab[1], "data": _b58decode(ix["d"]), "accounts": list(range(len(accts))),
                        "keys": accts, "dir": lab[0]})
    return out


def fetch_router_samples(helius, router: str, pages: int = 2, per_page: int = 1000, raw_sink=None) -> list:
    """Labeled samples for one router, MOST-RECENT-FIRST (desc) and paged DEEP. getTransactionsForAddress(
    router) gives each tx's TOP-LEVEL router instruction (data + account pubkeys = the shred-visible view, for
    fitting) AND its inner pumpswap CPI (the TRUE direction, for the label). Desc + deep is the fix for the
    sortOrder=asc trap that pulled a router's GENESIS window (1-2 one-sided samples); a deep recent fetch of a
    single router gives plenty of buy AND sell. ~10 credits per 100 txs."""
    out: list = []
    token = None
    for _ in range(max(1, pages)):
        opts = {"transactionDetails": "full", "sortOrder": "desc", "limit": per_page,
                "maxSupportedTransactionVersion": 1, "encoding": "jsonParsed"}
        if token:
            opts["paginationToken"] = token
        res = helius._rpc("getTransactionsForAddress", [router, opts], max(10, 10 * per_page // 100))
        if not isinstance(res, dict):
            break
        page = res.get("data") or []
        for tx in page:
            if raw_sink is not None:
                raw_sink(raw_record(tx, router))                  # every tx, before any filter
            d = label_from_tx(tx, None)                           # true direction from the inner pumpswap swap
            if not d:
                continue
            ix = _router_top_ix(tx, router)
            if ix is None:
                continue
            data, accts = ix
            out.append({"router": router, "sig": _sig_of(tx), "pool": None,
                        "payer": accts[0] if accts else None, "token": _inner_token(tx),
                        "data": data, "accounts": list(range(len(accts))), "keys": accts, "dir": d})
        token = res.get("paginationToken")
        if not page or not token:
            break
    return out


def why_dropped(tx: dict, router: str) -> str:
    """Why fetch_router_samples kept or dropped one tx (for diagnosing a router that yields 0 samples):
    'kept' / 'no top-level router ix' / 'no inner pumpswap ix' / 'pumpswap disc <hex>' (not a swap we label)
    / 'pumpswap not SOL-paired' / 'failed tx'."""
    if ((tx or {}).get("meta") or {}).get("err") is not None:
        return "failed tx"
    found = []
    for grp in ((tx or {}).get("meta") or {}).get("innerInstructions") or []:
        for ix in grp.get("instructions") or []:
            if ix.get("programId") == PUMPSWAP:
                found.append(ix)
    msg = ((tx or {}).get("transaction") or {}).get("message") or {}
    top = [ix.get("programId") for ix in msg.get("instructions") or []]
    if not found:
        return "no inner pumpswap ix" + (" (pumpswap top-level)" if PUMPSWAP in top else "")
    if label_from_tx(tx, None) is None:
        ix = found[0]
        data = _b58decode(ix.get("data") or "")
        if data[:8] not in SELL_ALL and data[:8] not in BUY_ALL:
            return "pumpswap disc %s" % data[:8].hex()
        return "pumpswap not SOL-paired"
    if _router_top_ix(tx, router) is None:
        return "no top-level router ix" + (" (router invoked by CPI)" if router not in top else " (empty data)")
    return "kept"


def diagnose_router(helius, router: str, limit: int = 100) -> dict:
    """ONE page of a router's recent txs (~limit/10 credits) and a tally of why_dropped, plus the top-level
    programs of the dropped ones -- shows why a router yields no labeled samples before spending more."""
    opts = {"transactionDetails": "full", "sortOrder": "desc", "limit": limit,
            "maxSupportedTransactionVersion": 1, "encoding": "jsonParsed"}
    res = helius._rpc("getTransactionsForAddress", [router, opts], max(10, 10 * limit // 100))
    page = (res or {}).get("data") or [] if isinstance(res, dict) else []
    reasons: Counter = Counter()
    tops: Counter = Counter()
    for tx in page:
        r = why_dropped(tx, router)
        reasons[r] += 1
        if r != "kept":
            msg = ((tx or {}).get("transaction") or {}).get("message") or {}
            for ix in msg.get("instructions") or []:
                tops[ix.get("programId")] += 1
    return {"router": router, "txs": len(page), "reasons": dict(reasons.most_common()),
            "top_programs_of_dropped": dict(tops.most_common(8))}


def _inner_token(tx: dict) -> Optional[str]:
    """The non-WSOL mint of the tx's first inner pumpswap swap (for the mint_order rule). None if absent."""
    for grp in ((tx or {}).get("meta") or {}).get("innerInstructions") or []:
        for ix in grp.get("instructions") or []:
            if ix.get("programId") != PUMPSWAP:
                continue
            accts = ix.get("accounts") or []
            if len(accts) >= 5:
                base, quote = accts[3], accts[4]
                if (base == WSOL) != (quote == WSOL):
                    return base if quote == WSOL else quote
    return None


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
            token = accts[3] if accts[4] == WSOL else accts[4]    # the non-WSOL side (for the mint_order rule)
            return {"router": router, "sig": _sig_of(tx), "pool": None, "payer": raccts[0], "token": token,
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

MAX_DISCS = 8              # a real program has a handful of swap instructions; more distinct prefixes = amounts


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
    if len(m) > MAX_DISCS:                                        # dozens of "discriminators" = a native ix tag +
        return None                                               # amount bytes memorised, not an instruction set
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


def _byte_flag_rule(train, start: int = 8, max_values: Optional[int] = 3) -> Optional[dict]:
    """A single data byte past the 8-byte discriminator whose value separates direction cleanly -- an Anchor
    `side`/`swap_base_in`/enum arg the disc rule (first 8 bytes only) misses. The holdout precision guard in
    find_rule rejects a spurious split (e.g. a byte of the amount)."""
    min_len = min((len(s["data"]) for s in train), default=0)
    for o in range(start, min_len):
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
        if ok and len(m) >= 2 and len(set(m.values())) == 2 and (max_values is None or len(m) <= max_values):
            return {"kind": "byte_flag", "offset": o, "map": {("%02x" % k): v for k, v in m.items()}}
    return None


def _mint_order_rule(train) -> Optional[dict]:
    """Direction from the relative position of WSOL vs the token mint in the router's account list (input mint
    before output mint, say) -- catches routers where WSOL is present but not at a fixed index, which a
    wsol_index rule misses. Token-aware: the live box supplies the pool's token. A sample with no token, or
    where either mint is absent from the account list, is not separable this way, so the rule is not fitted."""
    mapping: dict = {}                                            # (WSOL before token) -> direction
    for s in train:
        tok = s.get("token")
        if not tok:
            return None
        keys, accts = s["keys"], s["accounts"]
        wi = ti = None
        for p, a in enumerate(accts):
            if 0 <= int(a) < len(keys):
                k = keys[int(a)]
                if wi is None and k == WSOL:
                    wi = p
                if ti is None and k == tok:
                    ti = p
        if wi is None or ti is None:
            return None
        wf = wi < ti
        prev = mapping.get(wf)
        if prev is None:
            mapping[wf] = s["dir"]
        elif prev != s["dir"]:
            return None
    if len(mapping) == 2:                                         # both orderings seen, each pure
        return {"kind": "mint_order", "wsol_first": mapping[True], "other": mapping[False]}
    return None


BIT_MASKS = tuple(1 << k for k in range(8)) + (0xF0, 0x0F)


def _bit_flag_rule(train, start: int = 8) -> Optional[dict]:
    """A side bit inside a flags byte: (data[o] & mask) separates direction with <= 3 masked values, where the
    raw byte takes more (else byte_flag already has it). E.g. 3vHAV byte 11: 0x1x = buy, 0x2x = sell."""
    min_len = min((len(s["data"]) for s in train), default=0)
    for o in range(start, min_len):
        for mask in BIT_MASKS:
            m: dict = {}
            ok = True
            for s in train:
                v = s["data"][o] & mask
                if m.setdefault(v, s["dir"]) != s["dir"]:
                    ok = False
                    break
            if ok and len(set(m.values())) == 2 and len(m) <= 3:
                return {"kind": "bit_flag", "offset": o, "mask": mask,
                        "map": {("%02x" % k): v for k, v in m.items()}}
    return None


def _const_field_rule(train, start: int = 8) -> Optional[dict]:
    """One direction ALWAYS carries the same 8-byte value at an offset (a sentinel like u64::MAX for 'no limit'
    on sells) and the other direction never does. E.g. Bgo4v: u64 at 40 == ffff..ff -> sell, else buy."""
    min_len = min((len(s["data"]) for s in train), default=0)
    w = 8
    for o in range(start, min_len - w + 1):
        for side in ("buy", "sell"):
            vals = {s["data"][o:o + w] for s in train if s["dir"] == side}
            if len(vals) != 1:
                continue
            v = next(iter(vals))
            if any(s["data"][o:o + w] == v for s in train if s["dir"] != side):
                continue
            other = "sell" if side == "buy" else "buy"
            return {"kind": "const_field", "offset": o, "width": w, "value": v.hex(), "is": side, "other": other}
    return None


SENTINEL_MIN = 5           # a one-sided sentinel must be seen at least this often


def _sentinel_rule(train, start: int = 8) -> Optional[dict]:
    """ONE-SIDED: an 8-byte value that appears only with one direction (>= SENTINEL_MIN times, >= half of that
    side) -> that direction; anything else ABSTAINS. For routers whose side is partly decided at run time (Bgo4v:
    u64::MAX limit on most sells, but some sells carry a real limit), so a two-sided rule is never 100% right."""
    min_len = min((len(s["data"]) for s in train), default=0)
    w = 8
    best = None
    for o in range(start, min_len - w + 1):
        for side in ("buy", "sell"):
            mine = [s["data"][o:o + w] for s in train if s["dir"] == side]
            if not mine:
                continue
            v, n = Counter(mine).most_common(1)[0]
            if n < SENTINEL_MIN or n * 2 < len(mine):            # (zeros are fine: a 0 amount on one side only)
                continue
            if any(s["data"][o:o + w] == v for s in train if s["dir"] != side):
                continue
            if best is None or n > best[0]:
                best = (n, {"kind": "const_field", "offset": o, "width": w, "value": v.hex(), "is": side,
                            "other": None})
    return best[1] if best else None


MIN_GROUP = 3              # per_disc: an instruction needs this many training samples to be mapped at all
SUB_MAX_VALUES = 3         # per_disc byte flag: a real side/enum byte takes few values (an amount byte takes many)


def _per_disc_rule(train, prefix: int = 8) -> Optional[dict]:
    """One sub-rule per router instruction, keyed by the first `prefix` data bytes (8 = Anchor discriminator,
    1 = a native program's instruction tag). An instruction seen with only one direction maps straight to it;
    a mixed instruction (one `swap` ix for both sides) gets a side byte past the prefix, a WSOL account slot, or
    the WSOL/token order -- else it is left out (abstain -> drop). Catches routers the flat rules miss: e.g. a
    buy/sell ix carrying a side byte next to a sell-only ix, where a global byte_flag picks a spurious offset."""
    groups: dict = {}
    for s in train:
        if len(s["data"]) >= prefix:
            groups.setdefault(s["data"][:prefix].hex(), []).append(s)
    m: dict = {}
    for key, g in groups.items():
        if len(g) < MIN_GROUP:
            continue
        dirs = {s["dir"] for s in g}
        if len(dirs) == 1:
            m[key] = next(iter(dirs))
            continue
        if min(sum(1 for s in g if s["dir"] == d) for d in ("buy", "sell")) < 2:
            continue                                              # a lone counter-example: not enough to fit a split
        sub = (_byte_flag_rule(g, start=prefix, max_values=SUB_MAX_VALUES)
               or _bit_flag_rule(g, start=prefix) or _const_field_rule(g, start=prefix)
               or _wsol_rule(g) or _mint_order_rule(g))
        if sub is not None:
            m[key] = sub
    if not m or len(m) > MAX_DISCS:
        return None
    return {"kind": "per_disc", "prefix": prefix, "map": m}


def _per_disc8(train):
    return _per_disc_rule(train, 8)


def _per_disc1(train):
    return _per_disc_rule(train, 1)


FITTERS = (_disc_rule, _wsol_rule, _byte_flag_rule, _mint_order_rule, _per_disc8, _per_disc1,
           _bit_flag_rule, _const_field_rule, _sentinel_rule)      # one-sided sentinel last


def _score(rule: dict, test) -> tuple[int, int, int]:
    """(correct, fired, n): how often the rule fires on held-out, and how often it is right when it fires."""
    correct = fired = 0
    for s in test:
        pred = apply_rule(rule, s["data"], s["accounts"], s["keys"], s.get("token"))
        if pred is not None:
            fired += 1
            if pred == s["dir"]:
                correct += 1
    return correct, fired, len(test)


def find_rule(labeled, seed: int = 0, only=None) -> dict:
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
    seen: dict = {}
    data_blind = False                                            # identical ix bytes seen as buy AND sell: the
    for s in labeled:                                             # side is chosen at execution (arb/MEV bot), so
        prev = seen.setdefault(s["data"], s["dir"])               # no data-only rule can be trusted
        if prev != s["dir"]:
            data_blind = True
            break
    # structural first, token-aware, then per-instruction (Anchor disc, native tag); data-blind -> accounts only
    fits = (_wsol_rule, _mint_order_rule) if data_blind else FITTERS
    if only is not None:
        fits = tuple(f for f in fits if f in only)
    best = None
    for fit in fits:
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
            full = fit(labeled)                                   # held-out passed: refit on ALL samples so an ix
            if full is not None and full.get("kind") == rule["kind"]:   # seen only in the test split is mapped too
                fc, ff, _ = _score(full, labeled)
                if ff and fc == ff:                               # never wrong on everything it was fit on
                    cand = {**cand, "rule": full}
            return cand                                           # prefer the first clean rule (disc before wsol)
        if best is None or cand["precision"] > best["precision"]:
            best = cand
    if best is None and data_blind:
        return {**base, "reason": "data-blind: identical instruction data seen as buy and sell"}
    return best or {**base, "reason": "no discriminator or WSOL-index rule separates direction"}


STABLE_SEEDS = 5           # a rule must pass held-out precision 1.0 on this many different train/test splits


def find_rule_stable(labeled, seed: int = 0, seeds: int = STABLE_SEEDS) -> dict:
    """find_rule, but a fitter's rule is accepted only if that SAME fitter passes held-out precision 1.0 on
    EVERY one of `seeds` different train/test splits. Fitters are tried in find_rule's order; the first that
    is stable wins. One lucky split is not proof on a fragile gate: this rejects a rule that hinges on which
    samples landed in training (e.g. a byte that only looks like a side flag on 70% of the data)."""
    first = find_rule(labeled, seed=seed)
    if first["n"] and (first["buys"] < MIN_PER_DIR or first["sells"] < MIN_PER_DIR):
        return first
    if first.get("reason", "").startswith("data-blind"):
        return first
    unstable = None
    for fit in FITTERS:
        rep = find_rule(labeled, seed=seed, only=(fit,))
        if not rep["accepted"]:
            continue
        bad = next((k for k in range(1, seeds)
                    if not find_rule(labeled, seed=seed + k, only=(fit,))["accepted"]), None)
        if bad is None:
            return rep
        unstable = unstable or {**rep, "accepted": False,
                                "reason": "unstable: %s rejected on holdout split %d" % (rep["kind"], bad)}
    return unstable or {**first, "accepted": False}


def rule_fire_rate(rule: dict, samples) -> float:
    """Share of a router's samples on which its rule fires (does not abstain) -- the router's own coverage."""
    if not samples:
        return 0.0
    fired = sum(1 for s in samples
                if apply_rule(rule, s["data"], s["accounts"], s["keys"], s.get("token")) is not None)
    return fired / len(samples)


# --- the validated decoder table + the two-tier classifier ------------------------------------------------

RULES: dict = {}            # program_id -> validated rule; empty until find_rule output is loaded


def decode_report(samples: list, counts: Optional[dict] = None) -> str:
    """Per-router decode worksheet, sorted by volume: each router's buy vs sell discriminators and the fitted
    rule, so the direction encoding is visible at a glance. Different disc on buy vs sell -> disc rule; same
    disc -> the side is a data byte (byte_flag) or a WSOL account slot (wsol_index)."""
    by: dict = {}
    for s in samples:
        by.setdefault(s["router"], []).append(s)

    def vol(r):
        return (counts or {}).get(r, 0)

    lines = ["ROUTER DECODE -- buy vs sell instruction data (top-level router instruction)"]
    for r in sorted(by, key=lambda r: (-vol(r), -len(by[r]))):
        ss = by[r]
        buys = [s for s in ss if s["dir"] == "buy"]
        sells = [s for s in ss if s["dir"] == "sell"]
        bd = Counter(s["data"][:8].hex() for s in buys)
        sd = Counter(s["data"][:8].hex() for s in sells)
        rep = find_rule(ss)
        status = "ACCEPT %s" % (rep.get("kind") or "") if rep["accepted"] else rep.get("reason", "")
        lines += ["",
                  "%s  vol=%d  n=%d  buys=%d sells=%d  -> %s" % (r, vol(r), len(ss), len(buys), len(sells), status),
                  "  buy  disc: " + (", ".join("%s(%d)" % (k, v) for k, v in bd.most_common(4)) or "-"),
                  "  sell disc: " + (", ".join("%s(%d)" % (k, v) for k, v in sd.most_common(4)) or "-")]
        if rep["accepted"]:
            rule = rep["rule"]
            if rule["kind"] == "byte_flag":
                lines.append("  side = data byte[%d]: %s" % (rule["offset"], rule["map"]))
            elif rule["kind"] == "wsol_index":
                lines.append("  side = WSOL at account index %d (wsol->%s)" % (rule["index"], rule["wsol_is"]))
            elif rule["kind"] == "mint_order":
                lines.append("  side = WSOL vs token order (wsol-first->%s)" % rule["wsol_first"])
    return "\n".join(lines)


def _mark_accounts(keys, token: Optional[str]) -> str:
    """Account list with WSOL and the token mint marked, others truncated -- so the layout is readable."""
    out = []
    for k in keys:
        if k == WSOL:
            out.append("WSOL")
        elif token and k == token:
            out.append("TOK")
        else:
            out.append((k or "")[:4])
    return "[" + " ".join(out) + "]"


def dump_examples(samples: list, counts: Optional[dict] = None, per_dir: int = 6, data_bytes: int = 48) -> str:
    """Raw per-router buy/sell examples for HAND decoding -- no rule-fitting. Per router (sorted by volume): a
    few buys and a few sells, each as the top-level router instruction's data (hex, first `data_bytes` bytes =
    discriminator + args) and its account list with WSOL/token marked. Paste this; the discriminator and any
    side byte are visible, and the hand-written rule goes into router_rules.json (the box's lookup table)."""
    by: dict = {}
    for s in samples:
        by.setdefault(s["router"], []).append(s)

    def vol(r):
        return (counts or {}).get(r, 0)

    out = ["ROUTER TX DUMP -- paste for hand decoding (data = router ix, first %d bytes)" % data_bytes]
    for r in sorted(by, key=lambda r: (-vol(r), -len(by[r]))):
        ss = by[r]
        nb = sum(1 for s in ss if s["dir"] == "buy")
        ns = sum(1 for s in ss if s["dir"] == "sell")
        out.append("")
        out.append("=== %s  vol=%d  n=%d  buys=%d sells=%d ===" % (r, vol(r), len(ss), nb, ns))
        for label in ("BUY", "SELL"):
            for s in [x for x in ss if x["dir"] == label.lower()][:per_dir]:
                d = s["data"]
                hx = d[:data_bytes].hex() + (".." if len(d) > data_bytes else "")
                out.append("  %-4s %s  %s" % (label, hx, _mark_accounts(s["keys"], s.get("token"))))
    return "\n".join(out)


ROUTER_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "router_data")
SAMPLED_PATH = os.path.join(ROUTER_DATA, "router_sampled.json")    # {router: [buys, sells]} already in the JSONL
COUNTS_PATH = os.path.join(ROUTER_DATA, "router_counts.json")      # {router: routed-tx volume} (box cpi_routers)


def load_sampled(path: str) -> dict:
    """{router: [buys, sells]} of samples we ALREADY hold. Accepts the compact summary json (shipped in the
    image) or a raw samples JSONL. Missing file -> {} (nothing excluded)."""
    try:
        if path.endswith(".jsonl"):
            out: dict = {}
            for s in load_samples(path):
                c = out.setdefault(s["router"], [0, 0])
                c[0 if s["dir"] == "buy" else 1] += 1
            return out
        with open(path) as f:
            return {str(k): [int(v[0]), int(v[1])] for k, v in json.load(f).items()}
    except (OSError, ValueError):
        return {}


def routers_to_fetch(counts: dict, sampled: dict, denylist=DENYLIST) -> list:
    """Routers that still NEED samples, highest routed volume first: every router without at least
    MIN_PER_DIR buys AND MIN_PER_DIR sells already in hand (so a fitted rule can be validated), minus the
    denylist. Routers we can already fit are skipped, so a fetch never re-buys what we have."""
    need = []
    for r, v in counts.items():
        if r in denylist:
            continue
        b, s = sampled.get(r, (0, 0))
        if b >= MIN_PER_DIR and s >= MIN_PER_DIR:
            continue
        need.append(r)
    return sorted(need, key=lambda r: -int(counts.get(r, 0)))


def load_samples(path: str) -> list:
    """Load one or more router-samples JSONLs (comma-separated paths), de-duplicated by (router, signature) (a
    later file's copy is dropped), so a top-up fetch can be merged with the original file safely."""
    if "," in path:
        out, seen = [], set()
        for p in [x for x in path.split(",") if x]:
            for s in _load_samples_one(p):
                key = (s.get("router"), s.get("sig"))           # one tx can sit under 2 routers: keep both
                if s.get("sig") and key in seen:
                    continue
                seen.add(key)
                out.append(s)
        return out
    return _load_samples_one(path)


def _load_samples_one(path: str) -> list:
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
                        "payer": r.get("payer"), "token": r.get("token"), "data": data,
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
        rep = find_rule_stable(ss, seed=seed)
        rep["fire_rate"] = rule_fire_rate(rep["rule"], ss) if rep["accepted"] else 0.0
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
        fire = {r["router"]: r["fire_rate"] for r in routers}
        effective = sum(counts.get(p, 0) * fire.get(p, 0.0) for p in accepted)
        deny = sum(v for p, v in counts.items() if p in DENYLIST)
        out.update({"total_volume": total, "covered_volume": covered,
                    "coverage": (covered / total) if total else 0.0,
                    "effective_volume": int(round(effective)),
                    "effective_coverage": (effective / total) if total else 0.0,
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
        if "effective_volume" in report:
            lines.append("EFFECTIVE: %.1f%% of routed volume gets a direction (router volume x the share of its "
                         "txs its rule fires on; abstains drop)" % (100.0 * report["effective_volume"] / t))
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


def router_classify(program: str, data: bytes, accounts, account_keys, token: Optional[str] = None,
                    rules: Optional[dict] = None,
                    pin_targets=PIN_TARGETS, denylist=DENYLIST) -> str:
    """route_rules.router_classify with this module's RULES as the default table."""
    return _router_classify(program, data, accounts, account_keys, token,
                            rules=RULES if rules is None else rules, pin_targets=pin_targets, denylist=denylist)
