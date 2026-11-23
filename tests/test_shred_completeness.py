import json

from tradingbot import shred_completeness as sc
from tradingbot.data import helius as he

POOL = "PoolAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
MINT = "MintBbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"   # not *pump: hand-made


def _tx(slot, payer, dtok, dsol, tok_after=5000.0, sol_after=50.0, top_level=True, lut_pool=False):
    """A minimal Helius tx dict shaped the way launch_record / _vault / _keys / _top_level_pumpswap /
    _lut_sourced read it. buy: dtok<0, dsol>0. top_level=False => aggregator CPI (invoke [2]).
    lut_pool=True => the pool account entered via a lookup table (box drops it today)."""
    keys = [{"pubkey": payer, "source": "transaction"},
            {"pubkey": POOL, "source": "lookupTable" if lut_pool else "transaction"}]
    depth = "1" if top_level else "2"
    return {
        "slot": slot, "blockTime": 1000 + slot,
        "transaction": {"message": {"accountKeys": keys}},
        "meta": {
            "err": None,
            "logMessages": [f"Program {he.PUMPSWAP_PROGRAM} invoke [{depth}]"],
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


def _buy(slot, payer, **kw):
    return _tx(slot, payer, dtok=-10.0, dsol=1.0, **kw)


# A capture that saw a buy by A in slot 100 and by E in slot 102, but NOT slot 101 (inside the watch window).
CAP_REC = {"pool": POOL, "token_mint": MINT, "create_ts": 10_000,
           "slots": [[0.0, 100, 1, 0, ["A"]], [4.0, 102, 1, 0, ["E"]]]}

# Chain truth over slots 100..102:
#  100: A, direct-static (box-visible, captured)            -> have
#  101: B direct-static (box-visible, NOT captured)         -> absent
#       C direct but LUT-pooled                             -> lut
#       D aggregator-routed (CPI)                           -> routed
#  102: E, direct-static (box-visible, captured)            -> have
CHAIN = [
    _buy(100, "A"),
    _buy(101, "B"),
    _buy(101, "C", lut_pool=True),
    _buy(101, "D", top_level=False),
    _buy(102, "E"),
]


def test_compare_pool_classifies_every_missing_trade():
    r = sc.compare_pool(CAP_REC, CHAIN)
    assert r is not None
    assert r["all"] == 5 and r["direct"] == 4 and r["box_visible"] == 3 and r["capture"] == 2
    assert r["routed"] == 1          # D
    assert r["lut"] == 1             # C
    assert r["absent"] == 1          # B (box-visible yet not captured = feed loss)
    assert r["all"] - r["capture"] == r["routed"] + r["lut"] + r["absent"]   # the missing split is exhaustive
    # buyer-set recall: cap {A,E}; chain {A,B,C,D,E}=5; box-visible {A,B,E}=3
    assert r["chain_buyers"] == 5 and r["box_buyers"] == 3
    assert r["cap_hit_chain"] == 2 and r["cap_hit_box"] == 2


def test_capture_cannot_exceed_chain_in_a_slot():
    # the capture over-counts slot 100 (3 buys) but the chain has 1 -> have clamps to 1, never 3.
    over = {"pool": POOL, "token_mint": MINT, "create_ts": 10_000,
            "slots": [[0.0, 100, 3, 0, ["A", "X", "Y"]]]}
    r = sc.compare_pool(over, [_buy(100, "A")])
    assert r["all"] == 1 and r["capture"] == 1


def test_no_chain_trades_returns_none():
    assert sc.compare_pool(CAP_REC, []) is None


class _FakeHelius:
    def __init__(self, by_pool):
        self.by_pool = by_pool
        self.calls = 0

    def history_window(self, pool, from_ts, span_s, max_pages):
        self.calls += 1
        return self.by_pool.get(pool, []), True


def test_study_aggregates_percentages_and_breakdown():
    h = _FakeHelius({POOL: CHAIN})
    out = sc.study([CAP_REC], h, max_credits=40_000, span_s=180.0, max_pools=10, now=10_100)
    s = out["summary"]
    assert s["compared"] == 1 and s["no_chain_in_window"] == 0
    assert s["chain_trades"] == 5 and s["captured_trades"] == 2
    assert s["have_pct"] == 40.0 and s["box_ceiling_pct"] == 60.0 and s["lut_ceiling_pct"] == 80.0
    assert s["missing_absent"] == 1 and s["missing_lut"] == 1 and s["missing_routed"] == 1
    assert s["buyer_recall_chain_pct"] == 40.0            # {A,E} of 5
    assert round(s["buyer_recall_box_pct"]) == 67         # 2 of 3
    assert h.calls == 1
    # freshness: capture create_ts 10_000, "now" 10_100 -> 100 s old
    assert out["freshness"]["newest_age_s"] == 100.0
    assert render_runs(out)


def render_runs(out):
    txt = sc.render(out)
    return "HAVE" in txt and "WHAT WE MISS" in txt and "BUYER-SET RECALL" in txt


def test_study_counts_pools_with_no_chain():
    h = _FakeHelius({})          # Helius returns nothing for the pool
    out = sc.study([CAP_REC], h, max_credits=40_000, max_pools=10, now=10_100)
    assert out["summary"]["compared"] == 0 and out["summary"]["no_chain_in_window"] == 1


def test_study_skips_pump_graduations():
    grad = {"pool": POOL, "token_mint": "somethingpump", "create_ts": 10_000,
            "slots": [[0.0, 100, 1, 0, ["A"]]]}
    h = _FakeHelius({POOL: CHAIN})
    out = sc.study([grad], h, max_credits=40_000, max_pools=10, now=10_100)
    assert out["summary"]["hand_made_pools"] == 0 and h.calls == 0


def test_load_capture_dedups_to_the_richest(tmp_path):
    f = tmp_path / "cap.jsonl"
    f.write_text("\n".join([
        json.dumps({"pool": POOL, "token_mint": MINT, "create_ts": 1, "slots": [[0, 100, 1, 0, ["A"]]]}),
        json.dumps({"pool": POOL, "token_mint": MINT, "create_ts": 2,
                    "slots": [[0, 100, 1, 0, ["A"]], [1, 101, 1, 0, ["B"]]]}),   # richer -> kept
        "",
    ]))
    rows = sc.load_capture(str(f))
    assert len(rows) == 1 and len(rows[0]["slots"]) == 2


def test_row_layout_matches_flurry_study():
    # the comparator inlines these to stay numpy-free for the box; pin them to flurry_study so a layout
    # change there (the launch_record row shape) is caught rather than silently diverging.
    from tradingbot import flurry_study as fs
    assert (sc._SLOT, sc._PAYER, sc._TOKD, sc._SOLD, sc._VIS) == (fs._SLOT, fs._PAYER, fs._TOKD, fs._SOLD, fs._VIS)
    assert sc.MAX_PAGES == fs.MAX_PAGES
    assert sc._classify(-10, 1) == fs.classify(-10, 1) == "buy"
    assert sc._classify(10, -1) == fs.classify(10, -1) == "sell"


def test_comparator_imports_no_numpy():
    import subprocess, sys
    # a fresh interpreter: importing the comparator must not pull numpy (the box venv is numpy-free).
    r = subprocess.run([sys.executable, "-c",
                        "import sys, tradingbot.shred_completeness; "
                        "sys.exit(1 if 'numpy' in sys.modules else 0)"])
    assert r.returncode == 0


def test_freshness_reports_age_and_span():
    rows = [{"pool": "p1", "create_ts": 1000, "slots": []},
            {"pool": "p2", "create_ts": 4600, "slots": []}]
    fr = sc.freshness(rows, now=5000)
    assert fr["pools"] == 2 and fr["newest_age_s"] == 400.0 and fr["span_hours"] == 1.0
