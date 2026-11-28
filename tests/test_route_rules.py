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
