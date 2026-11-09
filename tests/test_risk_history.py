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


def test_a_sampled_backfill_widens_windows_and_rebuilds_full_size_days(cfg):
    hist = _hist(n_days=30, per_day=40)
    n = hist.query("SELECT COUNT(*) n FROM watch_launch_hist")[0]["n"]
    # the listing had 10x the pools the backfill fetched: a 10% random sample
    hist.executemany("INSERT INTO hist_candidates(pool, token_mint, lp_mint) VALUES (?,?,?)",
                     [(f"X{i}", "t", "l") for i in range(9 * n)])
    watch = WatchStore(":memory:")
    ll.ensure_schema(watch)
    cfg = cfg.with_overrides({"model_book_min_liq_usd": "1000"})
    rows, info = rh.load_pools(hist, watch, cfg)
    assert abs(info["sample share"] - 0.1) < 0.01 and all(abs(r["w"] - 10) < 1 for r in rows)
    out = rh.model(hist, watch, Journal(":memory:"), cfg, now=NOW, sims=20)
    assert out["sample"]["share"] == round(info["sample share"], 4)
    s = out["books"]["every liquid pool"]["stats"]
    assert s["sampled_trades"] < s["trades"] and abs(s["trades"] / s["sampled_trades"] - 10) < 1.5
    assert "sampled history" in rh.render(out)


def test_expand_keeps_real_day_swings_and_drops_sampling_noise():
    rng = np.random.default_rng(0)
    ts = []
    for d in range(60):   # every day drawn from one distribution: day-to-day differences are pure sampling noise
        for _ in range(20):
            ts.append({"ret": float(rng.normal(0.02, 0.5)), "cost": 100.0, "ts": d * D + 1000, "opened": d * D + 640,
                       "liq": 20_000.0, "w": 10.0})
    out, meta = rh.expand(ts)
    assert meta["k"] < 0.3 and len(out) == 60 * 200
    ts2 = [dict(t, ret=t["ret"] + (0.5 if (t["ts"] // D) % 2 else -0.5)) for t in ts]   # real swings, alternate days
    _, meta2 = rh.expand(ts2)
    assert meta2["k"] > 0.7
    full = [dict(t, w=1.0) for t in ts[:50]]
    assert rh.expand(full)[0] == full


def _tx(pool, mint, sol, tok_ui, tok_raw):
    return {"meta": {"err": None, "postTokenBalances": [
        {"owner": pool, "mint": "So11111111111111111111111111111111111111112", "uiTokenAmount": {"uiAmount": sol}},
        {"owner": pool, "mint": mint, "uiTokenAmount": {"uiAmount": tok_ui, "amount": str(tok_raw)}}]}, "blockTime": 1}


class _Hel:
    def __init__(self, states):
        self.states, self.credits_used, self.calls = states, 0, []

    def last_before(self, pool, ts):
        self.calls.append((pool, ts))
        self.credits_used += 10
        return self.states.get(pool)


def test_hold15_fetches_minute_15_states_and_prices_history_and_scorer_rows(cfg):
    hist = WatchStore(":memory:")
    t0 = NOW - 3 * D
    hist_row = {"pool": "PH", "token_mint": "MH", "create_ts": t0, "ret": 0.10, "liq_usd": 30_000.0, "rw_pct": 20.0,
                "entry_px": 0.001, "peak_sol": 100.0, "w": 1.0}                         # display units
    scorer_row = {"pool": "PS", "token_mint": "MS", "create_ts": t0 + 60, "ret": 0.05, "liq_usd": 35_000.0,
                  "rw_pct": 30.0, "entry_px": 1e-9, "peak_sol": 100.0, "raw": True, "w": 1.0}  # raw units
    pulled = {"pool": "PP", "token_mint": "MP", "create_ts": t0 + 120, "ret": 0.2, "liq_usd": 30_000.0,
              "rw_pct": 10.0, "entry_px": 0.001, "peak_sol": 100.0, "pulled_ts": t0 + 120 + 600, "w": 1.0}
    drained = dict(hist_row, pool="PD", token_mint="MD", create_ts=t0 + 180)
    skipped = dict(hist_row, pool="PK", rw_pct=80.0)                                    # not in the kept half
    thin = {"pool": "PT", "token_mint": "MT", "create_ts": t0 + 240, "ret": 0.30, "liq_usd": 10_000.0, "rw_pct": 5.0,
            "w": 1.0}
    rows = [hist_row, scorer_row, pulled, drained, skipped, thin]
    hel = _Hel({"PH": _tx("PH", "MH", 120.0, 100_000.0, 10**14),        # 0.0012 SOL/token: +20% - 6% = +14%
                "PS": _tx("PS", "MS", 110.0, 100.0, 10**11),            # 1.1e-9 per raw unit: +10% - 6% = +4%
                "PD": _tx("PD", "MD", 5.0, 100_000.0, 10**14)})         # SOL 5 < 10% of the 100 peak: rugged
    cands = rh.hold_candidates(rows)
    assert {r["pool"] for r in cands} == {"PH", "PS", "PP", "PD"}
    st = rh.fetch_holds(hist, hel, cands, max_credits=10**6)
    assert st["fetched"] == 3 and {p for p, _ in hel.calls} == {"PH", "PS", "PD"}   # the pulled pool needs no call
    assert all(ts == next(r for r in rows if r["pool"] == p)["create_ts"] + rh.HOLD_S for p, ts in hel.calls)
    have = rh._have_holds(hist)
    assert abs(rh.hold_ret(hist_row, have) - 0.14) < 1e-9
    assert abs(rh.hold_ret(scorer_row, have) - 0.04) < 1e-9
    assert rh.hold_ret(pulled, have) == -1.0 and rh.hold_ret(drained, have) == -1.0
    assert rh.fetch_holds(hist, hel, cands, max_credits=10**6)["fetched"] == 0      # nothing fetched twice
    trades, info = rh.hold_trades(rows, have, 0.0, 0.02, extra_impact=False)
    by_cost = sorted((t["cost"], round(t["ret"], 4)) for t in trades)
    assert info["band trades"] == 4 and info["minute-15 states missing (minute 6 used)"] == 0
    assert (500.0, 0.14) in by_cost and (500.0, 0.04) in by_cost and (500.0, -1.0) in by_cost
    assert (200.0, 0.3) in by_cost                       # thin pool: sz's $250 capped at 2% of $10k, minute 6
    worse, _ = rh.hold_trades(rows, have, 0.0, 0.02, extra_impact=True)
    assert sum(t["ret"] for t in worse) < sum(t["ret"] for t in trades)


def test_hold15_respects_the_credit_cap(cfg):
    hist = WatchStore(":memory:")
    rows = [{"pool": f"P{i}", "token_mint": "M", "create_ts": NOW - D + i, "ret": 0.1, "liq_usd": 30_000.0,
             "rw_pct": 10.0, "entry_px": 0.001, "peak_sol": 100.0} for i in range(10)]
    hel = _Hel({r["pool"]: _tx(r["pool"], "M", 100.0, 100_000.0, 10**14) for r in rows})
    st = rh.fetch_holds(hist, hel, rh.hold_candidates(rows), max_credits=35)
    assert st["fetched"] == 3 and st["credits_used"] == 30
    trades, info = rh.hold_trades(rows, rh._have_holds(hist), 0.0, 0.02, extra_impact=False)
    assert info["minute-15 states missing (minute 6 used)"] == 7


def test_history_replay_with_hold15_adds_the_two_books(cfg):
    hist = _hist(n_days=20, per_day=40)
    watch = WatchStore(":memory:")
    ll.ensure_schema(watch)
    cfg = cfg.with_overrides({"model_book_min_liq_usd": "1000"})
    out = rh.model(hist, watch, Journal(":memory:"), cfg, now=NOW, sims=10, hold15=True, helius=_Hel({}))
    assert out["hold15"] is not None and "candidates (kept $25-40k pools)" in out["hold15"]
    assert set(rh.HOLD_RULES) <= set(out["books"]) and set(rh.HOLD_RULES) <= set(out["monthly"])
    assert "sz + minute-15 hold" in rh.render(out)
