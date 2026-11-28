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


def test_apply_byte_flag_rule():
    rule = {"kind": "byte_flag", "offset": 8, "map": {"00": "buy", "01": "sell"}}
    assert rd.apply_rule(rule, b"\xaa" * 8 + b"\x00", [], []) == "buy"
    assert rd.apply_rule(rule, b"\xaa" * 8 + b"\x01", [], []) == "sell"
    assert rd.apply_rule(rule, b"\xaa" * 8 + b"\x02", [], []) is None   # unseen flag value -> abstain
    assert rd.apply_rule(rule, b"\xaa" * 8, [], []) is None             # offset past the data -> abstain


def test_find_rule_accepts_a_byte_flag_split():
    # same 8-byte discriminator for both directions; the side is byte[8] -- the disc rule can't see it
    out = []
    for i in range(20):
        d = "buy" if i % 2 == 0 else "sell"
        flag = b"\x00" if d == "buy" else b"\x01"
        out.append({"data": b"\x09" * 8 + flag + bytes(7), "accounts": [0], "keys": ["X"], "dir": d})
    r = rd.find_rule(out, seed=7)
    assert r["accepted"] and r["kind"] == "byte_flag" and r["rule"]["offset"] == 8 and r["precision"] == 1.0


def test_apply_mint_order_rule():
    rule = {"kind": "mint_order", "wsol_first": "buy", "other": "sell"}
    assert rd.apply_rule(rule, b"", [0, 1, 2], ["POOL", WSOL, "TOK"], token="TOK") == "buy"   # WSOL before token
    assert rd.apply_rule(rule, b"", [0, 1, 2], ["POOL", "TOK", WSOL], token="TOK") == "sell"  # token before WSOL
    assert rd.apply_rule(rule, b"", [0, 1, 2], ["POOL", WSOL, "TOK"], token=None) is None      # no token -> abstain
    assert rd.apply_rule(rule, b"", [0, 1, 2], ["POOL", "A", "B"], token="TOK") is None         # token absent -> abstain


def test_find_rule_accepts_a_mint_order_split():
    # one discriminator, WSOL at a VARYING index (so no fixed wsol_index works); direction = WSOL before token
    pats = {"buy": [["a", WSOL, "TOK"], ["a", "b", WSOL, "TOK"]],
            "sell": [["TOK", WSOL, "c"], ["a", "TOK", WSOL]]}
    out = []
    for i in range(24):
        d = "buy" if i % 2 == 0 else "sell"
        keys = pats[d][(i // 2) % 2]
        out.append({"data": b"\x05" * 8, "accounts": list(range(len(keys))), "keys": keys, "token": "TOK", "dir": d})
    r = rd.find_rule(out, seed=11)
    assert r["accepted"] and r["kind"] == "mint_order"


def _pumpswap_routed_tx(router, disc, base, quote, sig="S", router_accts=("U", "SRC", "DST")):
    # a getTransactionsForAddress(PUMPSWAP) tx: top-level router ix at index 1, inner pumpswap CPI under it
    return {"transaction": {"signatures": [sig], "message": {"instructions": [
        {"programId": "ComputeBudget111111111111111111111111111111", "accounts": [], "data": "11"},
        {"programId": router, "accounts": list(router_accts), "data": _b58encode(b"\xab" * 8 + b"\x01")}]}},
        "meta": {"innerInstructions": [{"index": 1, "instructions": [
            {"programId": PUMPSWAP, "accounts": ["POOL", "x", "x", base, quote, "y"],
             "data": _b58encode(disc + bytes(16))}]}]}}


def test_routed_sample_labels_a_routed_swap_and_skips_a_direct_one():
    s = rd._routed_sample(_pumpswap_routed_tx(BOT, BUY_EXACT_QUOTE_IN, "TOKEN", WSOL))
    assert s and s["router"] == BOT and s["dir"] == "buy" and s["payer"] == "U" and s["token"] == "TOKEN"
    assert s["keys"] == ["U", "SRC", "DST"] and s["accounts"] == [0, 1, 2]
    # a direct swap: pumpswap is itself the top-level program -> not routed -> None
    direct = {"transaction": {"signatures": ["D"], "message": {"instructions": [
        {"programId": PUMPSWAP, "accounts": ["POOL"], "data": "11"}]}},
        "meta": {"innerInstructions": [{"index": 0, "instructions": [
            {"programId": PUMPSWAP, "accounts": ["POOL", "x", "x", "TOKEN", WSOL, "y"],
             "data": _b58encode(SELL + bytes(16))}]}]}}
    assert rd._routed_sample(direct) is None


class _FakeRpc:
    def __init__(self, pages):
        self._pages = pages
        self.credits_used = 0

    def _rpc(self, method, params, credits):
        self.credits_used += credits
        return self._pages.pop(0) if self._pages else None


def test_fetch_pumpswap_routed_buckets_by_router_over_pages():
    p1 = {"data": [_pumpswap_routed_tx(BOT, BUY_EXACT_QUOTE_IN, "TOKEN", WSOL, sig="a"),
                   _pumpswap_routed_tx(JUP, SELL, "TOKEN", WSOL, sig="b")], "paginationToken": "t"}
    p2 = {"data": [_pumpswap_routed_tx(BOT, SELL, "TOKEN", WSOL, sig="c")], "paginationToken": None}
    rpc = _FakeRpc([p1, p2])
    out = rd.fetch_pumpswap_routed(rpc, max_pages=6, per_page=1000)
    assert len(out) == 3 and rpc.credits_used > 0
    assert {s["router"] for s in out} == {BOT, JUP}
    assert {s["dir"] for s in out if s["router"] == BOT} == {"buy", "sell"}   # balanced, both directions


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


def _history_tx(router, disc, base_mint, quote_mint, sig="SIG", router_accts=("U", "SRC", "DST")):
    # a getTransactionsForAddress(router) tx: top-level router ix at index 1 + inner pumpswap CPI (the label)
    return {"transaction": {"signatures": [sig], "message": {"instructions": [
        {"programId": "ComputeBudget111111111111111111111111111111", "accounts": [], "data": "11"},
        {"programId": router, "accounts": list(router_accts), "data": _b58encode(b"\xab" * 8 + b"\x00\x00")}]}},
        "meta": {"innerInstructions": [{"index": 1, "instructions": [
            {"programId": PUMPSWAP, "accounts": ["POOL", "x", "x", base_mint, quote_mint, "y"],
             "data": _b58encode(disc + bytes(16))}]}]}}


def test_fetch_router_samples_labels_recent_desc_over_pages():
    router = BOT
    txs = []
    for i in range(12):
        # same router disc both ways; WSOL at router account index 1 on a buy -> a wsol_index rule should fit
        d_buy = i % 2 == 0
        disc = BUY_EXACT_QUOTE_IN if d_buy else SELL
        accts = ("U", WSOL, "TOK") if d_buy else ("U", "TOK", WSOL)
        txs.append(_history_tx(router, disc, "TOKEN", WSOL, sig="S%d" % i, router_accts=accts))
    rpc = _FakeRpc([{"data": txs[:8], "paginationToken": "t"}, {"data": txs[8:], "paginationToken": None}])
    samples = rd.fetch_router_samples(rpc, router, pages=3, per_page=1000)
    assert len(samples) == 12 and rpc.credits_used > 0           # paged desc until the token ran out
    assert samples[0]["keys"][0] == "U" and samples[0]["accounts"] == [0, 1, 2]
    out = rd.study(samples, seed=5)
    assert out["n_accepted"] == 1 and out["accepted"][router]["kind"] == "wsol_index"


def test_decode_report_shows_buy_and_sell_discriminators():
    # router uses distinct discriminators for buy vs sell -> a disc rule, and the worksheet shows both
    samples = []
    for i in range(16):
        d = "buy" if i % 2 == 0 else "sell"
        disc = (b"\xaa" * 8) if d == "buy" else (b"\xbb" * 8)
        samples.append({"router": BOT, "data": disc + bytes(8), "accounts": [0], "keys": ["X"], "dir": d})
    txt = rd.decode_report(samples, counts={BOT: 123})
    assert "ROUTER DECODE" in txt and BOT in txt and "vol=123" in txt
    assert "aaaaaaaaaaaaaaaa(8)" in txt and "bbbbbbbbbbbbbbbb(8)" in txt and "ACCEPT disc" in txt


def test_dump_examples_shows_raw_buy_and_sell_for_hand_decoding():
    s = []
    for _ in range(3):
        s.append({"router": BOT, "data": b"\xc6\x2e\x15\x52\xb4\xd9\xe8\x70" + bytes(8),
                  "accounts": [0, 1, 2], "keys": ["U", WSOL, "TOK"], "token": "TOK", "dir": "buy"})
        s.append({"router": BOT, "data": b"\x33\xe6\x85\xa4\x01\x7f\x83\xad" + bytes(8),
                  "accounts": [0, 1, 2], "keys": ["U", "TOK", WSOL], "token": "TOK", "dir": "sell"})
    txt = rd.dump_examples(s, counts={BOT: 99}, per_dir=2)
    assert "ROUTER TX DUMP" in txt and "vol=99" in txt
    assert "BUY  c62e1552b4d9e870" in txt and "SELL 33e685a4017f83ad" in txt   # discriminators visible
    assert "[U WSOL TOK]" in txt and "[U TOK WSOL]" in txt                     # account layout marked


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


# --- chunk 1 (2026-11-28): per-instruction rules, data-blind guard, split stability ----------------------

SIDE_IX = b"\xaf\x05\x19\x81\xa0\xd8\x38\x9d"     # buy/sell ix with a side byte at [24] (6Vo3245-style)
SELL_IX = b"\xfd\x9c\xb9\xe4\xf0\x69\xdc\xa9"     # sell-only ix


def _side_samples(n=40):
    out = []
    for i in range(n):
        amt = (1000 + 37 * i).to_bytes(8, "little") + (5 + i).to_bytes(8, "little")
        if i % 4 == 3:
            out.append({"data": SELL_IX + amt + b"\x01" * 9, "accounts": [0], "keys": ["A"], "dir": "sell"})
        else:
            d = "buy" if i % 2 == 0 else "sell"
            out.append({"data": SIDE_IX + amt + (b"\x01" if d == "buy" else b"\x00") + b"\x00",
                        "accounts": [0], "keys": ["A"], "dir": d})
    return out


def test_apply_per_disc_nested_and_abstain():
    rule = {"kind": "per_disc", "prefix": 8, "map": {
        SIDE_IX.hex(): {"kind": "byte_flag", "offset": 24, "map": {"01": "buy", "00": "sell"}},
        SELL_IX.hex(): "sell"}}
    body = bytes(16)
    assert rd.apply_rule(rule, SIDE_IX + body + b"\x01\x00", [], []) == "buy"
    assert rd.apply_rule(rule, SIDE_IX + body + b"\x00\x00", [], []) == "sell"
    assert rd.apply_rule(rule, SIDE_IX + body + b"\x07\x00", [], []) is None     # unseen side value
    assert rd.apply_rule(rule, SELL_IX + body, [], []) == "sell"
    assert rd.apply_rule(rule, b"\x11" * 26, [], []) is None                      # unseen ix
    tag = {"kind": "per_disc", "prefix": 1, "map": {"06": "sell"}}
    assert rd.apply_rule(tag, b"\x06abc", [], []) == "sell"
    assert rd.apply_rule(tag, b"\x08abc", [], []) is None


def test_find_rule_stable_fits_a_side_byte_next_to_a_sell_only_ix():
    r = rd.find_rule_stable(_side_samples())
    assert r["accepted"] and r["kind"] == "per_disc"
    sub = r["rule"]["map"][SIDE_IX.hex()]
    assert sub["kind"] == "byte_flag" and sub["offset"] == 24
    assert r["rule"]["map"][SELL_IX.hex()] == "sell"
    assert rd.rule_fire_rate(r["rule"], _side_samples()) == 1.0


def test_data_blind_router_is_rejected():
    # identical instruction bytes labelled buy AND sell: the side is chosen at execution -> never decode
    out = []
    for i in range(20):
        out.append({"data": b"\x0c" * 8 + bytes([i % 3]) * 8, "accounts": [0], "keys": ["A"],
                    "dir": "buy" if i % 2 else "sell"})
    r = rd.find_rule_stable(out)
    assert not r["accepted"] and r["reason"].startswith("data-blind")


def test_disc_rule_refuses_amount_memorisation():
    # 30 distinct 8-byte prefixes (a native tag + amount bytes) is not an instruction set
    out = [{"data": b"\x02" + (i * 7919).to_bytes(7, "little"), "accounts": [0], "keys": ["A"],
            "dir": "buy" if i % 2 else "sell"} for i in range(30)]
    assert rd._disc_rule(out) is None


def test_study_reports_effective_coverage():
    ss = [{**s, "router": "R1"} for s in _side_samples()]
    rep = rd.study(ss, counts={"R1": 100, "R2": 300})
    assert rep["n_accepted"] == 1
    assert rep["coverage"] == 0.25 and rep["effective_coverage"] == 0.25
    assert "EFFECTIVE" in rd.render(rep)


def test_routers_to_fetch_skips_what_we_already_hold():
    counts = {"BIG": 1000, "HALF": 500, "DONE": 400, "ARB": 300, "NEW": 10}
    have = {"HALF": [112, 0], "DONE": [9, 5]}
    out = rd.routers_to_fetch(counts, have, denylist=frozenset({"ARB"}))
    assert out == ["BIG", "HALF", "NEW"]                 # DONE fit-able, ARB denylisted; volume order


def test_load_sampled_from_summary_and_jsonl(tmp_path):
    summ = tmp_path / "s.json"
    summ.write_text('{"R": [3, 1]}')
    assert rd.load_sampled(str(summ)) == {"R": [3, 1]}
    j = tmp_path / "a.jsonl"
    j.write_text('{"router":"R","sig":"x","data":"00","accounts":[],"keys":[],"dir":"sell"}\n')
    assert rd.load_sampled(str(j)) == {"R": [0, 1]}
    assert rd.load_sampled(str(tmp_path / "missing.json")) == {}


def test_shipped_sampled_summary_loads():
    have = rd.load_sampled(rd.SAMPLED_PATH)
    assert len(have) > 100 and all(len(v) == 2 for v in have.values())


def test_load_samples_merges_files_without_duplicate_sigs(tmp_path):
    line = '{"router":"R","sig":"%s","data":"00","accounts":[],"keys":[],"dir":"buy"}\n'
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(line % "s1" + line % "s2")
    b.write_text(line % "s2" + line % "s3")
    got = rd.load_samples("%s,%s" % (a, b))
    assert [s["sig"] for s in got] == ["s1", "s2", "s3"]


def test_why_dropped_names_the_reason():
    R = "RouterProg"
    base = _pumpswap_routed_tx(R, SELL, "TOKEN", WSOL)
    assert rd.why_dropped(base, R) == "kept"
    assert rd.why_dropped(base, "OtherRouter").startswith("no top-level router ix")
    no_ps = {"transaction": {"message": {"instructions": [{"programId": R, "accounts": ["U"], "data": "11"}]}},
             "meta": {"innerInstructions": []}}
    assert rd.why_dropped(no_ps, R) == "no inner pumpswap ix"
    odd = _pumpswap_routed_tx(R, b"\x09" * 8, "TOKEN", WSOL)
    assert rd.why_dropped(odd, R) == "pumpswap disc " + "09" * 8
    failed = {**base, "meta": {**base["meta"], "err": {"x": 1}}}
    assert rd.why_dropped(failed, R) == "failed tx"


def test_diagnose_router_tallies_one_page():
    R = "RouterProg"

    class H:
        def _rpc(self, method, params, credits):
            return {"data": [_pumpswap_routed_tx(R, SELL, "TOKEN", WSOL),
                             _pumpswap_routed_tx(R, b"\x09" * 8, "TOKEN", WSOL)]}
    out = rd.diagnose_router(H(), R)
    assert out["txs"] == 2 and out["reasons"]["kept"] == 1


def test_raw_record_keeps_every_tx_with_balances():
    tx = _pumpswap_routed_tx("R", b"\x09" * 8, "TOKEN", WSOL)       # an unlabelled tx is still recorded
    tx["meta"].update({"preTokenBalances": [{"owner": "U", "mint": "TOKEN", "uiTokenAmount": {"amount": "5"}}],
                       "postTokenBalances": [{"owner": "U", "mint": "TOKEN", "uiTokenAmount": {"amount": "9"}}],
                       "preBalances": [100], "postBalances": [60]})
    seen = []
    class H:
        def _rpc(self, method, params, credits):
            return {"data": [tx]}
    out = rd.fetch_router_samples(H(), "R", pages=1, per_page=10, raw_sink=seen.append)
    assert out == [] and len(seen) == 1
    r = seen[0]
    assert r["tok"] == [["U", "TOKEN", 4]] and r["payer_sol"] == -40
    assert r["top"][1]["p"] == "R" and r["inner"][0]["ix"][0]["p"] == PUMPSWAP


def test_v2_pumpswap_discs_label():
    assert rd._dir_from_pumpswap(rd.SELL_V2, "TOKEN", WSOL) == "sell"
    assert rd._dir_from_pumpswap(rd.BUY_V2, "TOKEN", WSOL) == "buy"
    assert rd._dir_from_pumpswap(rd.BUY_EXACT_QUOTE_IN_V2, "TOKEN", WSOL) == "buy"
    assert rd.label_from_tx(_inner_tx(rd.SELL_V2, "TOKEN", WSOL), "POOL") == "sell"


def _raw(disc, payer_delta, err=None, extra=None):
    ix = {"p": PUMPSWAP, "a": ["POOL", "U", "x", "TOKEN", WSOL], "d": _b58encode(disc + bytes(8))}
    return {"router": "R", "sig": "S", "err": err, "payer": "U",
            "top": [{"p": "R", "a": ["U", "POOL"], "d": _b58encode(b"\xaa" * 8)}],
            "inner": [{"i": 0, "ix": [ix] + (extra or [])}], "tok": [["U", "TOKEN", payer_delta]]}


def test_label_raw_checks_balances_and_washes():
    assert rd.label_raw(_raw(rd.BUY_V2, 50)) == ("buy", "TOKEN")
    assert rd.label_raw(_raw(rd.SELL_V2, -50)) == ("sell", "TOKEN")
    assert rd.label_raw(_raw(rd.BUY_V2, -50)) is None                      # balances contradict -> drop
    assert rd.label_raw(_raw(rd.BUY_V2, 0)) == ("buy", "TOKEN")            # payer unmoved: trust the ix
    assert rd.label_raw(_raw(rd.BUY_V2, 50, err={"x": 1})) is None         # failed tx
    wash = {"p": PUMPSWAP, "a": ["POOL", "U", "x", "TOKEN", WSOL], "d": _b58encode(rd.SELL_V2 + bytes(8))}
    assert rd.label_raw(_raw(rd.BUY_V2, 0, extra=[wash])) is None          # buy+sell in one tx


def test_samples_from_raw(tmp_path):
    import gzip
    import json as _j
    p = tmp_path / "raw.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write(_j.dumps(_raw(rd.BUY_V2, 5)) + "\n" + _j.dumps(_raw(rd.SELL_V2, 0, err={"e": 1})) + "\n")
    out = rd.samples_from_raw(str(p))
    assert len(out) == 1 and out[0]["dir"] == "buy" and out[0]["keys"] == ["U", "POOL"]
    assert out[0]["data"] == b"\xaa" * 8 and out[0]["token"] == "TOKEN"


def test_bit_flag_and_sentinel_rules():
    tr = [{"data": b"\x01" * 11 + bytes([0x10 | (i % 4)]), "dir": "buy"} for i in range(8)] + \
         [{"data": b"\x01" * 11 + bytes([0x20 | (i % 4)]), "dir": "sell"} for i in range(8)]
    r = rd._bit_flag_rule(tr)
    assert r["offset"] == 11 and rd.apply_rule(r, b"\x01" * 11 + b"\x13", [], []) == "buy"
    assert rd.apply_rule(r, b"\x01" * 11 + b"\x22", [], []) == "sell"
    mx = b"\xff" * 8
    tr2 = [{"data": b"\x00" * 8 + (i + 1).to_bytes(8, "little"), "dir": "buy"} for i in range(6)] + \
          [{"data": b"\x00" * 8 + mx, "dir": "sell"} for _ in range(6)] + \
          [{"data": b"\x00" * 8 + (99).to_bytes(8, "little"), "dir": "sell"}]   # one sell without the sentinel
    assert rd._const_field_rule(tr2) is None                                  # two-sided fails on the odd sell
    s = rd._sentinel_rule(tr2)
    assert s["is"] == "sell" and s["other"] is None
    assert rd.apply_rule(s, b"\x00" * 8 + mx, [], []) == "sell"
    assert rd.apply_rule(s, b"\x00" * 8 + (3).to_bytes(8, "little"), [], []) is None   # abstain, never 'buy'
