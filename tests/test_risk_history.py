"""risk-model --history: pools from the backfill + the scorer, walk-forward rug-vs-win replay, books, risk model."""
import json

import numpy as np

from tradingbot import launch_hist as lh
from tradingbot import launch_live as ll
from tradingbot import risk_history as rh
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore
from tests.test_launch_hist import FakeHelius, NOW, pool_history

D = 86400.0


class Varied(FakeHelius):
    """Pools whose second-1..3 buyers count tells the outcome: many early buyers -> pulled, few -> doubled."""

    def history_from_start(self, address, span_s, max_pages=6):
        self.history_calls += 1
        t0, up = self.pools[address]
        txs = pool_history(address, t0, up)
        if up < 0.1:  # rugs: two extra early buyers (more buyers in the first seconds)
            extra = [dict(txs[1], blockTime=t0 + 2) for _ in range(2)]
            txs = txs[:2] + extra + txs[2:]
        return txs, True, True


def _hist(n_days=30, per_day=40, seed=0):
    rng = np.random.default_rng(seed)
    st = WatchStore(":memory:")
    pools = {}
    for d in range(n_days):
        for k in range(per_day):
            pools[f"H{d}_{k}"] = (NOW - (n_days - d) * D + k * 600, 0.05 if rng.random() < 0.3 else 2.0)
    hel = Varied(pools)
    times = {p: v[0] for p, v in pools.items()}
    it = iter(times.values())
    hel.t_for = lambda lp: next(it)
    lh.backfill(st, hel, n_days + 2, 10**8, now=NOW)
    st.executemany("INSERT INTO watch_market(ts, sol_usd) VALUES (?, ?)",
                   [(NOW - (n_days + 2) * D + h * 3600, 100.0) for h in range((n_days + 3) * 24)])
    return st


def test_history_replay_runs_end_to_end_and_finds_the_planted_signal(cfg):
    hist = _hist()
    watch = WatchStore(":memory:")
    ll.ensure_schema(watch)
    cfg = cfg.with_overrides({"model_book_min_liq_usd": "1000"})
    rows, info = rh.load_pools(hist, watch, cfg)
    assert info["history rows"] == 1200 and len(rows) == 1200 and all("x" in r and "feat" not in r for r in rows)
    meta = rh.replay(rows, retrain_h=24, max_train=5000)
    assert meta["fits"] >= 20 and meta["scored"] == 22 * 40  # the first ~8 days train the first model (300 pools)
    scored = [r for r in rows if r.get("rw_pct") is not None]
    kept = [r["ret"] for r in scored if r["rw_pct"] < 50]
    skipped = [r["ret"] for r in scored if r["rw_pct"] >= 50]
    assert np.mean(kept) > np.mean(skipped) + 0.5          # the model learned that early crowding = pull
    out = rh.model(hist, watch, Journal(":memory:"), cfg, now=NOW, sims=20)
    assert {"every liquid pool", "launchrw50", "launchrw50_sz"} <= set(out["books"])
    assert out["books"]["launchrw50"]["stats"]["mean_trade_pct"] > out["books"]["every liquid pool"]["stats"]["mean_trade_pct"]
    assert out["gaps"]["launchrw50"]["source"].startswith("default")
    assert out["monthly"]["launchrw50"]
    text = rh.render(out)
    assert "SUPERMEME HISTORY REPLAY" in text and "BY MONTH" in text and "SUPERMEME RISK MODEL" in text


def test_scorer_rows_win_over_history_and_check_compares_rankings(cfg):
    hist = _hist(n_days=10, per_day=40)
    watch = WatchStore(":memory:")
    ll.ensure_schema(watch)
    cfg = cfg.with_overrides({"model_book_min_liq_usd": "1000"})
    rows, _ = rh.load_pools(hist, watch, cfg)
    last = rows[-1]
    watch.execute("INSERT INTO watch_launch_live(pool, creator, create_ts, liquid, status, ret, liq_usd, feat, rw_pct) "
                  "VALUES (?,?,?,?,?,?,?,?,?)", (last["pool"], "C", last["create_ts"], 1, "done", 0.42, 12_000.0,
                                                 json.dumps({"log_buyers": 1.0}), 30.0))
    rows2, info = rh.load_pools(hist, watch, cfg)
    r = next(x for x in rows2 if x["pool"] == last["pool"])
    assert r["src"] == "scorer" and r["ret"] == 0.42 and info["scorer rows"] == 1
    assert rh.check(rows2)["pools"] <= 1                     # too few ranked by both
    assert rh.model(None, WatchStore(":memory:"), None, cfg)["note"].startswith("fewer than 500")
