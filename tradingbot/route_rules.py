"""Per-router direction rules applied to a routed top-level instruction (the shred-visible view).

Flat-safe on purpose: no package-relative imports, so the Nuremberg box can run this file beside its flat copy of
localshred_consumer.py. tradingbot.router_direction fits the rules (from Helius truth) and re-exports these
functions, so there is one definition. `make_classifier(path)` loads a router_rules.json and returns the
route_classify callable localshred_consumer.Consumer takes.
"""
from __future__ import annotations

import json
from typing import Optional

WSOL = "So11111111111111111111111111111111111111112"   # == localshred_consumer.WSOL

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
    # 2026-11-28 raw fetch (10,000 recent txs each, `Router Data/router-raw8.jsonl.gz`): these reference pumpswap
    # in ~every tx but almost never swap on it. >=99.7% FAILED (spam / sniper retries):
    "CtjS9o7rj3UuiNSjUdh4vHih5WiEoLXDk6XE1y3LN8Vg",   # 9,974 failed / 10,000; 21 ok pumpswap swaps
    "JDk32MFrBeUpA9AN7CciPryokhSDxuWEAJdJfQxqDrpE",   # 9,996 failed; 4 ok swaps
    "BmGwqQXtzu8tLH6TiG5DEVakNs4chNo2boVSov5rr3DG",   # 9,976 failed; 6 ok swaps
    # ...or succeed WITHOUT a pumpswap swap (check-and-abort arb): <=0.12% of their txs are real swaps:
    "DiabLokxisGR8P4Qqp2QL6PmBhhVsdzPstCE4WoLxc5z",   # 9,701 ok, 12 swaps
    "2xKALZWKqW9KSH98jVJxnom5csv3FKYdiu6Z595vXKRv",   # 10,000 ok, 5 swaps
    "81eej9NUm883KBEbYFxxLx3cbduVC1VQU8KmgwCpCp3",    # 10,000 ok, 5 swaps
    "2TK4sdgGTG89Rh3TurnbViNzkJRyg8CaXhhAEQimX3ia",   # 10,000 ok, 8 swaps
    # Counting any of them as a buyer (tail) would add ~1,000 phantom buys per real one, so they DROP.
})


# --- rule application (pure) ------------------------------------------------------------------------------

def apply_rule(rule: dict, data: bytes, accounts, account_keys, token: Optional[str] = None) -> Optional[str]:
    """Apply a validated per-router rule to a routed instruction's pre-execution view. Returns "buy" / "sell",
    or None (abstain -> the caller DROPS, which is safe). `data` = the router ix data bytes; `accounts` = its
    account-index list; `account_keys` = the tx's resolved account keys; `token` = the pool's non-WSOL mint
    (the live box knows it from the tracked pool), needed only by the token-aware `mint_order` rule."""
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
    if kind == "mint_order":                                       # WSOL before/after the token in the account list
        if not token:
            return None                                            # token unknown -> abstain (safe)
        wi = ti = None
        for p, a in enumerate(accounts):
            ai = int(a)
            if ai < 0 or ai >= len(account_keys):
                continue
            k = account_keys[ai]
            if wi is None and k == WSOL:
                wi = p
            if ti is None and k == token:
                ti = p
        if wi is None or ti is None:
            return None                                            # one of the mints absent -> abstain
        return rule["wsol_first"] if wi < ti else rule["other"]
    if kind == "bit_flag":                                         # side packed in a bitfield byte (flags & mask)
        o = int(rule["offset"])
        if o < 0 or o >= len(data):
            return None
        return rule["map"].get("%02x" % (data[o] & int(rule["mask"])))
    if kind == "const_field":                                      # one side always carries a sentinel (u64::MAX)
        o, w = int(rule["offset"]), int(rule["width"])
        if o < 0 or o + w > len(data):
            return None
        if data[o:o + w].hex() == rule["value"]:
            return rule["is"]
        return rule.get("other")                                   # one-sided rule: other is None -> abstain
    if kind == "mint_index":                                       # the mint at a fixed account index (Jupiter
        i = int(rule["index"])                                     # route dest mint): WSOL -> one side, the pool's
        if i < 0 or i >= len(accounts) or not token:               # token -> the other, any other mint (a USDC
            return None                                            # leg, multi-hop) -> abstain
        ai = int(accounts[i])
        if ai < 0 or ai >= len(account_keys):
            return None
        k = account_keys[ai]
        if k == WSOL:
            return rule["wsol_is"]
        if k == token:
            return rule["token_is"]
        return None
    if kind == "per_disc":                                         # one sub-rule per instruction (prefix bytes)
        n = int(rule.get("prefix", 8))
        if len(data) < n:
            return None
        v = rule["map"].get(data[:n].hex())
        if isinstance(v, str):
            return v if v in ("buy", "sell") else None
        if isinstance(v, dict) and v.get("kind") != "per_disc":    # no nesting
            return apply_rule(v, data, accounts, account_keys, token)
        return None                                                # unseen instruction / undecodable -> abstain
    return None


def load_rules(path: str) -> dict:
    """Load validated per-router rules from a json {program_id: rule}. Missing/bad file -> {} (fail safe:
    an empty table just means every pin-target DROPS, never a wrong direction)."""
    try:
        with open(path) as f:
            data = json.load(f)
        return {str(k): v for k, v in data.items() if isinstance(v, dict) and v.get("kind")}
    except (OSError, ValueError):
        return {}


def router_classify(program: str, data: bytes, accounts, account_keys, token: Optional[str] = None,
                    rules: Optional[dict] = None,
                    pin_targets=PIN_TARGETS, denylist=DENYLIST) -> str:
    """Classify a routed top-level instruction into one of: "buy" / "sell" (a validated rule fired),
    "buyer_plus" (tail -> recover buyer, count a buy), "drop" (denylist, or a pin-target with no/abstaining
    rule -> never guess direction on the bulk). Encodes the owner's risk split exactly."""
    rules = {} if rules is None else rules
    if program in denylist:
        return "drop"
    if program in rules:
        d = apply_rule(rules[program], data, accounts, account_keys, token)
        return d if d in ("buy", "sell") else "drop"              # rule abstains -> drop (safe)
    if program in pin_targets:
        return "drop"                                             # committed to pin; do not guess
    return "buyer_plus"                                           # the long tail (accepted approximation)


def make_classifier(path: str):
    """route_classify for localshred_consumer.Consumer from a router_rules.json. Returns (callable, n_rules);
    a missing/bad file gives 0 rules, so every pin-target drops and only the tail counts as buyer_plus."""
    rules = load_rules(path)

    def classify(program, data, accounts, account_keys, token=None):
        return router_classify(program, data, accounts, account_keys, token, rules=rules)
    return classify, len(rules)
