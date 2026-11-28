"""route_rules (flat-safe rule application for the box) + the consumer's measure-only routed recovery."""
import json

from tradingbot import localshred_consumer as lc
from tradingbot import route_rules as rr
from tradingbot import router_direction as rd

ROUTER = "Rout3r1111111111111111111111111111111111111"
RULES = {ROUTER: {"kind": "disc", "map": {"01" * 8: "buy", "02" * 8: "sell"}}}


def test_router_direction_reexports_one_definition():
    assert rd.apply_rule is rr.apply_rule and rd.load_rules is rr.load_rules
    assert rd.PIN_TARGETS is rr.PIN_TARGETS and rd.DENYLIST is rr.DENYLIST
    assert rr.WSOL == lc.WSOL


def test_route_rules_has_no_package_relative_import():
    src = open(rr.__file__).read()
    assert "from ." not in src and "import tradingbot" not in src


def test_make_classifier(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(RULES))
    cls, n = rr.make_classifier(str(p))
    assert n == 1
    assert cls(ROUTER, b"\x01" * 8, [], []) == "buy" and cls(ROUTER, b"\x02" * 8, [], []) == "sell"
    assert cls(ROUTER, b"\x03" * 8, [], []) == "drop"                     # rule abstains -> drop
    assert cls(next(iter(rr.DENYLIST)), b"\x01" * 8, [], []) == "drop"
    assert cls("TailRouter", b"\x01" * 8, [], []) == "buyer_plus"
    cls0, n0 = rr.make_classifier(str(tmp_path / "missing.json"))
    assert n0 == 0 and cls0(ROUTER, b"\x01" * 8, [], []) == "buyer_plus"


class _Rec:
    def __init__(self):
        self.records = []

    def on_slot(self, rec):
        self.records.append(dict(rec))
        return None


def _routed_tx(disc):
    keys = ["PAYER", "POOL", lc.PROGRAM, ROUTER, "TOKEN", lc.WSOL]
    return keys, [(ROUTER, [1, 2, 4, 5], disc + bytes(8))]


def _consumer(measure_only):
    cls = lambda p, d, a, k, t=None: rd.router_classify(p, d, a, k, t, rules=RULES)   # noqa: E731
    c = lc.Consumer(gate=None, runner_factory=lambda: _Rec(), route_classify=cls, route_measure_only=measure_only)
    c.v2_log_s = 0
    c.add_pool("POOL", "TOKEN")
    return c


def test_measure_only_counts_but_never_feeds_the_gate():
    c = _consumer(True)
    for d in (b"\x01" * 8, b"\x02" * 8, b"\x03" * 8):
        c.on_tx(10, *_routed_tx(d))
    assert c.route_counts == {"buy": 1, "sell": 1, "drop": 1}
    assert "POOL" not in c.pools or (c.pools["POOL"].n_buys == 0 and c.pools["POOL"].n_sells == 0)
    assert c.route_line() == "routed (tracked pools): buy=1 sell=1 tail=0 drop=1 -> rule-decoded 2 of 3 (66.7%) [measure only]"


def test_feed_mode_still_feeds_and_counts():
    c = _consumer(False)
    c.on_tx(10, *_routed_tx(b"\x01" * 8))
    c.on_tx(10, *_routed_tx(b"\x02" * 8))
    assert c.route_counts == {"buy": 1, "sell": 1}
    assert c.pools["POOL"].n_buys == 1 and c.pools["POOL"].n_sells == 1
    assert c.route_line().endswith("[feeding the gate]")


def test_routed_line_printed_with_the_v2_line(capsys):
    c = _consumer(True)
    c.v2_log_s = 60.0
    c._v2_last -= 61
    c.on_tx(10, *_routed_tx(b"\x01" * 8))
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("v2 decoded: ") and out[1].startswith("routed (tracked pools): ")


def test_mint_index_rule_is_token_aware_and_abstains_on_other_mints():
    rule = {"kind": "mint_index", "index": 1, "wsol_is": "sell", "token_is": "buy"}
    keys = ["A", lc.WSOL, "TOKEN", "USDC"]
    assert rr.apply_rule(rule, b"", [0, 1], keys, "TOKEN") == "sell"
    assert rr.apply_rule(rule, b"", [0, 2], keys, "TOKEN") == "buy"
    assert rr.apply_rule(rule, b"", [0, 3], keys, "TOKEN") is None     # a USDC leg -> abstain
    assert rr.apply_rule(rule, b"", [0, 2], keys, None) is None        # token unknown -> abstain
    assert rr.apply_rule(rule, b"", [0], keys, "TOKEN") is None        # index out of range
    assert rr.apply_rule(rule, b"", [0, 9], keys, "TOKEN") is None


def test_shipped_rules_include_jupiter_and_deny_the_no_swap_bots():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "Router Data", "router_rules.json")
    cls, n = rr.make_classifier(path)
    jup = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"
    keys = ["PAYER", "x", "y", "z", "q", "TOKEN"]                     # route (e517...): dest mint at account 5
    data = bytes.fromhex("e517cb977ae3ad2a") + bytes(8)
    assert cls(jup, data, list(range(6)), keys, "TOKEN") == "buy"
    assert cls(jup, data, list(range(6)), keys[:5] + [lc.WSOL], "TOKEN") == "sell"
    for bot in ("CtjS9o7rj3UuiNSjUdh4vHih5WiEoLXDk6XE1y3LN8Vg", "DiabLokxisGR8P4Qqp2QL6PmBhhVsdzPstCE4WoLxc5z",
                "2xKALZWKqW9KSH98jVJxnom5csv3FKYdiu6Z595vXKRv", "81eej9NUm883KBEbYFxxLx3cbduVC1VQU8KmgwCpCp3",
                "2TK4sdgGTG89Rh3TurnbViNzkJRyg8CaXhhAEQimX3ia", "JDk32MFrBeUpA9AN7CciPryokhSDxuWEAJdJfQxqDrpE",
                "BmGwqQXtzu8tLH6TiG5DEVakNs4chNo2boVSov5rr3DG"):
        assert bot in rr.DENYLIST and cls(bot, b"\x01" * 16, [], [], "TOKEN") == "drop"


def test_mint_conds_first_match_wins_and_abstains():
    rule = {"kind": "mint_conds", "conds": [[2, "token", "buy"], [1, "token", "sell"], [3, "wsol", "sell"]]}
    keys = ["A", "TOKEN", "X", lc.WSOL]
    assert rr.apply_rule(rule, b"", [0, 1, 2, 3], keys, "TOKEN") == "sell"       # acct1 is the token
    assert rr.apply_rule(rule, b"", [0, 2, 1, 3], keys, "TOKEN") == "buy"        # acct2 is the token (first cond)
    assert rr.apply_rule(rule, b"", [0, 2, 2, 2], keys, "TOKEN") is None         # nothing matches -> abstain
    assert rr.apply_rule(rule, b"", [0, 1, 2, 3], keys, None) is None            # token unknown
    assert rr.apply_rule(rule, b"", [0, 1], keys, "TOKEN") == "sell"              # short account list: skip conds


def test_shipped_rules_have_the_raw15_routers():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "Router Data", "router_rules.json")
    rules = rr.load_rules(path)
    for r in ("DF1ow4tspfHX9JwWJsAb9epbkA8hmpSEAtxXy1V27QBH", "proVF4pMXVaYqmy4NjniPh4pqKNfMmsihgd4wdkCX3u",
              "Cy91y8eNt72wDxutUBgSQhrG5KGRqQa28T55suZNngsQ", "GdvhFdVAJxVXecXRyVRqEz1DyHA8YAC8ZW7s7UviMzL4"):
        assert r in rules


def test_shipped_prov_rule_decodes_sells_on_the_pool_carrying_ix():
    """proV's swap ix (any disc) carries the pool: token@4 = buy, token@3 = sell; WSOL in the other slot."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "Router Data", "router_rules.json")
    classify, _ = rr.make_classifier(path)
    prov = "proVF4pMXVaYqmy4NjniPh4pqKNfMmsihgd4wdkCX3u"
    tok = "TokenMint11111111111111111111111111111pump"
    sell = ["P0", "PAYER", "X", tok, "MINTX"]
    buy = ["P0", "PAYER", "X", "MINTX", tok]
    data = bytes.fromhex("0ebf2cf68ee1e09d")
    assert classify(prov, data, list(range(5)), sell, tok) == "sell"
    assert classify(prov, data, list(range(5)), buy, tok) == "buy"
    assert classify(prov, bytes.fromhex("aa2955b184501f35"), list(range(5)),
                    ["P0", "PAYER", "X", rr.WSOL, "MINTX"], tok) == "buy"
    assert classify(prov, data, list(range(5)), ["P0", "PAYER", "X", "A", "B"], tok) == "drop"
