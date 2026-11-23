from tradingbot import symbol_flip as sfl
from tradingbot import precursor_gate as pg
from tradingbot.data import helius as he


# A gate that fires exactly when the slot symbol is "N" (eta 1.0 >= 0.5).
GATE = {"intercept": 0.0, "vis": "direct", "move": 0.2, "threshold": {"1": 0.5},
        "predictors": [{"coef": 1.0, "predicate": {"kind": "symbol", "symv": "N"}}]}


def _rec(sym):
    return {"sym": sym, "m2": "." + sym, "m3": ".." + sym, "f": {}}


def test_confuse_classifies_reproduce_miss_absent_flip_and_false_fire():
    # direct = the gate's own view (ground truth for the fire); box = the pre-LUT box.
    dir_bank = {1: _rec("N"), 2: _rec("N"), 3: _rec("N"), 4: _rec("S"), 5: _rec("b")}
    box_bank = {1: _rec("S"),             3: _rec("N"), 4: _rec("N"), 5: _rec("b")}  # slot 2 absent in the box
    r = sfl.confuse(box_bank, dir_bank, GATE, tier="1")
    c = r["counts"]
    assert c["direct_fires"] == 3           # slots 1,2,3 are sym N
    assert c["box_fires"] == 2              # slots 3,4 are sym N in the box
    assert c["reproduced"] == 1            # slot 3: both fire
    assert c["missed"] == 2                 # slots 1 (flip) and 2 (absent)
    assert c["miss_absent"] == 1            # slot 2
    assert c["miss_flip"] == 1             # slot 1: box saw S
    assert c["miss_flip_organic"] == 1     # the missed correct symbol was N
    assert c["false_fires"] == 1           # slot 4: box fires (N) but direct is S
    assert r["confusion"][("N", "S")] == 1
    assert r["confusion"][("N", "∅")] == 1   # the absent slot
    assert r["false_sym"]["N"] == 1


def test_confuse_perfect_when_box_matches_direct():
    bank = {1: _rec("N"), 2: _rec("S"), 3: _rec("N")}
    r = sfl.confuse(dict(bank), dict(bank), GATE)
    c = r["counts"]
    assert c["direct_fires"] == 2 and c["reproduced"] == 2 and c["missed"] == 0
    assert c["false_fires"] == 0 and c["miss_absent"] == 0


def _tx(slot, payer, lut_pool=False, dtok=-10.0, dsol=1.0, tok_after=5000.0, sol_after=50.0):
    """A minimal Helius buy tx shaped for launch_record (_vault / _keys / _top_level_pumpswap / _lut_sourced).
    lut_pool=True => the pool account is reached via a lookup table, so the box (vis_box_exact) drops it."""
    keys = [{"pubkey": payer, "source": "transaction"},
            {"pubkey": POOL, "source": "lookupTable" if lut_pool else "transaction"}]
    return {
        "slot": slot, "blockTime": 1000 + slot,
        "transaction": {"message": {"accountKeys": keys}},
        "meta": {
            "err": None,
            "logMessages": [f"Program {he.PUMPSWAP_PROGRAM} invoke [1]"],
            "preTokenBalances": [
                {"owner": POOL, "mint": MINT, "uiTokenAmount": {"uiAmount": tok_after - dtok}},
                {"owner": POOL, "mint": he.SOL_MINT, "uiTokenAmount": {"uiAmount": sol_after - dsol}},
            ],
            "postTokenBalances": [
                {"owner": POOL, "mint": MINT, "uiTokenAmount": {"uiAmount": tok_after}},
                {"owner": POOL, "mint": he.SOL_MINT, "uiTokenAmount": {"uiAmount": sol_after}},
            ],
        },
    }


POOL = "PoolAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
MINT = "MintBbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_bank_by_slot_distinguishes_box_from_direct_on_a_lut_dropped_new_wallet():
    # Slot 103 has a repeat wallet R (seen in slot 100) buying directly, plus a NEW wallet X buying via a LUT.
    # direct (LUT included): slot 103 has a new buyer (X) -> symbol "N".
    # box_visible (LUT dropped): slot 103 sees only the repeat wallet R, which (2 buys: slot 100 + 103) is the
    # most-frequent buyer -> dominant -> symbol "B". So the box misreads an organic-N slot as washbot-dominant B
    # -- exactly the live failure mode.
    txs = [
        _tx(100, "R"),
        _tx(101, "A"),
        _tx(102, "B"),
        _tx(103, "R"),                 # repeat wallet, static
        _tx(103, "X", lut_pool=True),  # new wallet, LUT-pooled (box can't see it)
        _tx(104, "C"),
    ]
    dir_bank = sfl.bank_by_slot(txs, POOL, MINT, box=False)
    box_bank = sfl.bank_by_slot(txs, POOL, MINT, box=True)
    assert 103 in dir_bank and 103 in box_bank
    assert dir_bank[103]["sym"] == "N"     # the LUT-included view sees the new wallet
    assert box_bank[103]["sym"] == "B"     # the LUT-blind box misses it -> misreads as dominant -> symbol flips

    # and the diagnostic reports that flip on a gate that fires on N
    r = sfl.confuse(box_bank, dir_bank, GATE)
    assert r["counts"]["direct_fires"] >= 1
    assert r["confusion"][("N", "B")] == 1
    assert r["counts"]["miss_flip_organic"] >= 1


def test_study_runs_and_aggregates(monkeypatch):
    txs = [_tx(100, "R"), _tx(101, "A"), _tx(102, "B"),
           _tx(103, "R"), _tx(103, "X", lut_pool=True), _tx(104, "C")]

    class _FakeHelius:
        def history_window(self, pool, from_ts, span_s, max_pages):
            return txs, True

    cap = [{"pool": POOL, "token_mint": MINT, "create_ts": 10_000,
            "slots": [[0.0, 100, 1, 0, ["R"]]]}]
    out = sfl.study(cap, _FakeHelius(), GATE, tier="1", max_credits=40_000, max_pools=10, now=10_100)
    s = out["summary"]
    assert s["compared"] == 1 and s["gate_vis"] == "direct"
    assert s["direct_fires"] >= 1 and s["missed"] >= 1
    assert "SYMBOL-FLIP CONFUSION" in sfl.render(out)
    assert "reproduction" in sfl.render(out).lower()


def test_study_skips_pump_graduations():
    cap = [{"pool": POOL, "token_mint": "somethingpump", "create_ts": 1,
            "slots": [[0.0, 100, 1, 0, ["R"]]]}]

    class _H:
        def history_window(self, *a):
            raise AssertionError("should not fetch a pump graduation")

    out = sfl.study(cap, _H(), GATE, max_pools=10, now=10)
    assert out["summary"]["hand_made_pools"] == 0 and out["summary"]["compared"] == 0
