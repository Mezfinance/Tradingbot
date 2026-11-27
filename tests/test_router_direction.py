from tradingbot import router_direction as rd
from tradingbot.localshred_tap import _b58encode
from tradingbot.localshred_consumer import PROGRAM as PUMPSWAP, WSOL, SELL, BUY_EXACT_QUOTE_IN

JUP = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"
BOT = "CtjS9o7rj3UuiNSjUdh4vHih5WiEoLXDk6XE1y3LN8Vg"   # a pin-target
TAIL = "ZzTailRouter11111111111111111111111111111111"


def test_b58_roundtrips_with_the_encoder():
    for raw in (b"\x00\x01\x02", bytes(range(32)), b"\x00\x00hello"):
        assert rd._b58decode(_b58encode(raw)) == raw


def test_apply_disc_rule_fires_and_abstains():
    rule = {"kind": "disc", "map": {(b"\xaa" * 8).hex(): "buy", (b"\xbb" * 8).hex(): "sell"}}
    assert rd.apply_rule(rule, b"\xaa" * 8 + b"\x00\x00", [], []) == "buy"
    assert rd.apply_rule(rule, b"\xbb" * 8, [], []) == "sell"
    assert rd.apply_rule(rule, b"\xcc" * 8, [], []) is None           # unknown disc -> abstain


def test_apply_wsol_index_rule():
    # the mint at instruction-account index 2 is WSOL on a buy, the token on a sell
    rule = {"kind": "wsol_index", "index": 2, "wsol_is": "buy", "other": "sell"}
    keys = ["PAYER", "POOL", "SRCMINT", WSOL, "TOKEN"]
    assert rd.apply_rule(rule, b"", [0, 1, 3], keys) == "buy"          # accounts[2]=3 -> keys[3]=WSOL
    assert rd.apply_rule(rule, b"", [0, 1, 4], keys) == "sell"         # accounts[2]=4 -> keys[4]=TOKEN
    assert rd.apply_rule(rule, b"", [0, 1], keys) is None             # index out of range -> abstain


def _inner_tx(disc, base_mint, quote_mint, pool="POOL"):
    accts = [pool, "x", "x", base_mint, quote_mint, "y"]
    data = _b58encode(disc + bytes(16))
    return {"meta": {"innerInstructions": [
        {"index": 0, "instructions": [
            {"programId": "SomeOtherProg", "accounts": ["a", "b"], "data": "11"},
            {"programId": PUMPSWAP, "accounts": accts, "data": data}]}]}}


def test_label_from_tx_reads_the_inner_pumpswap_direction():
    # base=TOKEN, quote=WSOL: BUY disc -> token buy, SELL disc -> token sell
    assert rd.label_from_tx(_inner_tx(BUY_EXACT_QUOTE_IN, "TOKEN", WSOL), "POOL") == "buy"
    assert rd.label_from_tx(_inner_tx(SELL, "TOKEN", WSOL), "POOL") == "sell"
    # pumpswap CPI on a different pool is ignored
    assert rd.label_from_tx(_inner_tx(SELL, "TOKEN", WSOL, pool="OTHER"), "POOL") is None
    # no inner pumpswap instruction
    assert rd.label_from_tx({"meta": {"innerInstructions": []}}, "POOL") is None


def _disc_samples(n=20):
    out = []
    for i in range(n):
        d = "buy" if i % 2 == 0 else "sell"
        disc = (b"\x01" * 8) if d == "buy" else (b"\x02" * 8)
        out.append({"data": disc + bytes(8), "accounts": [0, 1, 2], "keys": ["A", "B", "C"], "dir": d})
    return out


def test_find_rule_accepts_a_clean_discriminator_split():
    r = rd.find_rule(_disc_samples(), seed=1)
    assert r["accepted"] and r["kind"] == "disc" and r["precision"] == 1.0
    assert rd.apply_rule(r["rule"], b"\x01" * 8, [], []) == "buy"
    assert rd.apply_rule(r["rule"], b"\x02" * 8, [], []) == "sell"


def test_find_rule_accepts_a_wsol_index_split():
    # one discriminator, direction encoded by WSOL at account index 1
    out = []
    for i in range(20):
        d = "buy" if i % 2 == 0 else "sell"
        keys = ["POOL", WSOL, "TOKEN"] if d == "buy" else ["POOL", "TOKEN", WSOL]
        out.append({"data": b"\x09" * 8, "accounts": [0, 1], "keys": keys, "dir": d})
    r = rd.find_rule(out, seed=2)
    assert r["accepted"] and r["kind"] == "wsol_index"


def test_find_rule_rejects_an_unseparable_router():
    # same discriminator and no WSOL-position signal -> no rule, not accepted
    out = [{"data": b"\x07" * 8, "accounts": [0], "keys": ["X"], "dir": ("buy" if i % 2 else "sell")}
           for i in range(20)]
    r = rd.find_rule(out, seed=3)
    assert not r["accepted"]


def test_find_rule_rejects_too_few_samples():
    out = [{"data": b"\x01" * 8, "accounts": [], "keys": [], "dir": "buy"} for _ in range(10)]
    r = rd.find_rule(out, seed=4)
    assert not r["accepted"] and "too few" in r["reason"]


def test_router_classify_encodes_the_two_tier_risk_split():
    rules = {JUP: {"kind": "disc", "map": {(b"\x01" * 8).hex(): "buy", (b"\x02" * 8).hex(): "sell"}}}
    # a validated router -> true direction
    assert rd.router_classify(JUP, b"\x01" * 8, [], [], rules=rules) == "buy"
    assert rd.router_classify(JUP, b"\x02" * 8, [], [], rules=rules) == "sell"
    # a validated router whose rule abstains (unknown disc) -> drop, never guess
    assert rd.router_classify(JUP, b"\x03" * 8, [], [], rules=rules) == "drop"
    # a denylisted arb program -> drop
    assert rd.router_classify(next(iter(rd.DENYLIST)), b"\x01" * 8, [], [], rules=rules) == "drop"
    # a pin-target with no rule yet -> drop (committed to pin, do not guess)
    assert rd.router_classify(BOT, b"\x01" * 8, [], [], rules=rules) == "drop"
    # the long tail -> buyer_plus (accepted approximation)
    assert rd.router_classify(TAIL, b"\x01" * 8, [], [], rules=rules) == "buyer_plus"


class _FakeHelius:
    def __init__(self, txs):
        self._txs = txs

    def history(self, address, limit=100):
        return self._txs


def _history_tx(router, disc, base_mint, quote_mint, sig="SIG", router_accts=("U", "SRC", "DST")):
    # a getTransactionsForAddress tx: top-level router ix + inner pumpswap CPI (the label)
    return {"transaction": {"signatures": [sig], "message": {"instructions": [
        {"programId": "ComputeBudget111111111111111111111111111111", "accounts": [], "data": "11"},
        {"programId": router, "accounts": list(router_accts), "data": _b58encode(b"\xab" * 8 + b"\x00\x00")}]}},
        "meta": {"innerInstructions": [{"index": 1, "instructions": [
            {"programId": PUMPSWAP, "accounts": ["POOL", "x", "x", base_mint, quote_mint, "y"],
             "data": _b58encode(disc + bytes(16))}]}]}}


def test_fetch_router_samples_labels_from_the_inner_pumpswap_cpi():
    router = BOT
    txs = []
    for i in range(12):
        # base=TOKEN/quote=WSOL: BUY disc -> buy, SELL disc -> sell. The router ix data is the SAME here,
        # so direction is NOT in the discriminator -> but the router's source account flips WSOL/TOKEN
        d_buy = i % 2 == 0
        disc = BUY_EXACT_QUOTE_IN if d_buy else SELL
        accts = ("U", WSOL, "TOK") if d_buy else ("U", "TOK", WSOL)   # WSOL at router account index 1 on a buy
        txs.append(_history_tx(router, disc, "TOKEN", WSOL, sig="S%d" % i, router_accts=accts))
    samples = rd.fetch_router_samples(_FakeHelius(txs), router, limit=12)
    assert len(samples) == 12
    assert all(s["dir"] in ("buy", "sell") for s in samples)
    assert samples[0]["keys"][0] == "U" and samples[0]["accounts"] == [0, 1, 2]
    # a wsol-index rule should fit (WSOL at router account index 1 => buy)
    out = rd.study(samples, seed=5)
    assert out["n_accepted"] == 1 and out["accepted"][router]["kind"] == "wsol_index"


def test_load_samples_and_study_and_render(tmp_path):
    p = tmp_path / "router-samples.jsonl"
    lines = []
    for i in range(20):
        d = "buy" if i % 2 == 0 else "sell"
        disc = ("01" * 8) if d == "buy" else ("02" * 8)
        lines.append('{"router": "%s", "sig": "s%d", "pool": "P", "payer": "W%d", '
                     '"data": "%s", "accounts": [0,1], "keys": ["A","B"], "dir": "%s"}'
                     % (BOT, i, i, disc + "00" * 8, d))
    p.write_text("\n".join(lines) + "\n")
    samples = rd.load_samples(str(p))
    assert len(samples) == 20 and isinstance(samples[0]["data"], bytes)
    out = rd.study(samples, seed=1)
    assert out["n_accepted"] == 1 and BOT in out["accepted"]
    assert "ROUTER DIRECTION" in rd.render(out)


def test_study_reports_volume_weighted_coverage():
    # router A: clean disc split (accepted); router B: unseparable (not accepted)
    A, B = BOT, "DiabLokxisGR8P4Qqp2QL6PmBhhVsdzPstCE4WoLxc5z"
    samples = []
    for i in range(20):
        d = "buy" if i % 2 == 0 else "sell"
        disc = (b"\x01" * 8) if d == "buy" else (b"\x02" * 8)
        samples.append({"router": A, "data": disc, "accounts": [0], "keys": ["X"], "dir": d})
        samples.append({"router": B, "data": b"\x07" * 8, "accounts": [0], "keys": ["X"],
                        "dir": ("buy" if i % 2 else "sell")})
    counts = {A: 70, B: 30, next(iter(rd.DENYLIST)): 10}
    out = rd.study(samples, counts=counts, seed=1)
    assert A in out["accepted"] and B not in out["accepted"]
    assert out["total_volume"] == 110 and out["covered_volume"] == 70
    assert abs(out["coverage"] - 70 / 110) < 1e-9 and out["denylist_volume"] == 10
    txt = rd.render(out)
    assert "COVERAGE" in txt and "UNCOVERED" in txt


def test_load_rules_fails_safe(tmp_path):
    assert rd.load_rules(str(tmp_path / "missing.json")) == {}
    bad = tmp_path / "bad.json"; bad.write_text("not json")
    assert rd.load_rules(str(bad)) == {}
    good = tmp_path / "good.json"
    good.write_text('{"%s": {"kind": "disc", "map": {}}, "skip": {"no": "kind"}}' % JUP)
    loaded = rd.load_rules(str(good))
    assert JUP in loaded and "skip" not in loaded
