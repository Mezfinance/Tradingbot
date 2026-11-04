"""Filter study: alternative uses of a risk score on the history model's bottom 20%, judged against random."""
import json
import random

from tradingbot import launch_filter as lf
from tradingbot import launch_live as ll
from tradingbot.watcher import WatchStore
from tests.test_launch_live import _PathHelius


def _world(st, n=900, seed=3):
    """Operators (BAD creators, little early buying) pull at 200 s, after the live model's 90 s label; heavy early
    buying marks big winners, a third of which are early pulls at 40 s."""
    ll.ensure_schema(st)
    rnd = random.Random(seed)
    t0 = 1_800_000_000.0
    for i in range(n):
        create = t0 + i * 200
        kind = rnd.choices(["operator", "heavy", "plain"], [0.2, 0.25, 0.55])[0]
        early_pull = kind == "heavy" and rnd.random() < 0.3
        feat = {"log_buy_sol": (2.0 if kind == "heavy" else 0.0) + rnd.random(), "log_txs": rnd.random(),
                "creator_buys": rnd.random(), "hour_sin": 0.2}
        creator = f"BAD{rnd.randrange(15)}" if kind == "operator" else f"C{i}"
        pulled = create + 40 if early_pull else (create + 200 if kind == "operator" else None)
        ret = -1.0 if kind == "operator" or early_pull else (1.0 if kind == "heavy" else 0.02)
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, creator, create_ts, liquid, feat, score_h, "
                   "fill_px, fill_sol, ret, pulled_ts, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (f"P{i}", f"MP{i}", creator, create, 1, json.dumps(feat), rnd.random(), 100 / 1e6, 100.0, ret,
                    pulled, "done"))
    return t0 + n * 200


def test_filter_study_compares_each_variant_with_random_and_prices_early_exits():
    st = WatchStore(":memory:")
    now = _world(st)
    states = {f"P{i}": (lambda ts: (100.0, 1_000_000)) for i in range(900)}
    h = _PathHelius(states)
    out = lf.study(st, h, now=now, max_credits=100_000)
    assert out["tested"] > 300 and out["blocks"]
    names = [k for k in out if k.startswith(("now", "opt"))]
    assert len(names) == 5
    # operator signs catch the operators the < 90 s label never sees, and beat random skips
    op = out["opt 3: operator signs only"]["skip (every liquid pool)"]["skip riskiest 20%"]
    assert op["mean_pct"] > op["random_mean_pct"] and op["beats_random_pct"] >= 95
    # the live label skips heavy early buying, so it throws away more big winners than operator signs do
    now20 = out["now: pulled < 90 s (live model)"]["skip (every liquid pool)"]["skip riskiest 20%"]
    assert now20["big_win_pct"] < op["big_win_pct"]
    # option 5 never skips heavy early buying
    o5 = out["opt 5: live model, heavy early buying never skipped"]["skip (every liquid pool)"]["skip riskiest 20%"]
    assert int(o5["skipped: rugs / big winners"].split(" / ")[1]) == 0
    # option 1: D1-D2 pools priced at 60 / 90 s (flat pools: -6% after costs), cached
    d12 = out["history D1-D2 (test blocks)"]
    assert d12["sell every pool at 60 s"]["n"] > 0
    early = [k for k in out["opt 2: rugged before our exit"] if k.startswith("opt 1")]
    assert len(early) == 2 and "sell riskiest 20% early" in out["opt 2: rugged before our exit"][early[0]]
    assert "opt 1" not in " ".join(out["opt 5: live model, heavy early buying never skipped"])
    calls = h.calls
    lf.study(st, h, now=now, max_credits=100_000)
    assert h.calls == calls  # prices are never fetched twice
    assert lf.study(WatchStore(":memory:"))["pools"] == 0
