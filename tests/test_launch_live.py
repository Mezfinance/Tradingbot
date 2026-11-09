"""Second-4 launch scorer (shadow mode): training, LP-mint detection, decision, fill/exit reads, status."""
import json
import pytest
from tradingbot import launch_analysis as la
from tradingbot import launch_live as ll
from tradingbot.config import Config
from tradingbot.data import pumpswap as ps
from tradingbot.watcher import WatchStore
from tests.test_launch_analysis import _three_days
from tests.test_pumpswap import GRAD, HAND

ALPH = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
HAND_POOL = "5joVLztBfi7GryY3oWS8vjy3YfYUWggqQYAFP5DNpert"
HAND_LP = "3VrG6oUA8STG9gHUXVafni6V729q6suVR93pt1Xd2ygn"
GRAD_POOL, GRAD_LP = "6BNptAEKGETQFGQgDWiDhFqAgAmHusyqXqJnLLw74Exw", "4xapQw4x5o8ncfG7JXBWFg9GZ4grCiHpJZDx5kKXm2kr"


def b58decode(s: str) -> bytes:
    n = 0
    for c in s:
        n = n * 58 + ALPH.index(c)
    return n.to_bytes(32, "big")


def lp_raw(authority: str, supply: int) -> bytes:
    return (1).to_bytes(4, "little") + b58decode(authority) + supply.to_bytes(8, "little") + bytes([9]) + bytes(37)


def token_acct(amount: int) -> bytes:
    return bytes(64) + amount.to_bytes(8, "little") + bytes(93)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class FakeHelius:
    enabled, api_key = True, "k"

    def __init__(self, accounts):
        self.accts, self.credits_used, self.history_calls = accounts, 0, 0

    def accounts(self, addresses, commitment="confirmed"):
        self.credits_used += 1
        return {a: self.accts.get(a) for a in addresses}

    def history_until(self, address, until_ts, max_pages=4):
        self.history_calls += 1
        self.credits_used += 10
        return [{"blockTime": 0}], True, True


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = Config(db_path=str(tmp_path / "t.db"))
    st = WatchStore(":memory:")
    _three_days(st)
    assert ll.train_and_save(st, cfg) is not None
    p = ps.parse_pool(HAND)
    accts = {HAND_POOL: (ps.PROGRAM, HAND), GRAD_POOL: (ps.PROGRAM, GRAD),
             p["sol_vault"]: ("tok", token_acct(60_000_000_000)), p["token_vault"]: ("tok", token_acct(900))}
    hel = FakeHelius(accts)
    clock = Clock(2_000_000_000.0)
    create = clock.t - 1
    busy = [[0, 1, "OP", 1000, 50, 1000, 50]] + [[1 + j, 2 + j, f"W{j}", -1, 0.5, 1000 - j, 50 + 0.5 * j]
                                               for j in range(8)]
    monkeypatch.setattr(ll, "launch_record", lambda txs, pool, mint, window_s=60, exhausted=None: {
        "create_ts": create, "creator": "OP", "seed_sol": 50, "txs": busy})
    import queue
    st.execute("CREATE TABLE IF NOT EXISTS watch_market(ts REAL, sol_usd REAL)")
    st.execute("INSERT INTO watch_market(ts, sol_usd) VALUES(?, ?)", (clock.t - 60, 200.0))
    live = ll.LaunchLive(cfg, st, hel, clock=clock, workers=0, signals=queue.Queue())
    return cfg, st, hel, clock, live, p, create


def test_model_is_trained_without_name_features(setup):
    cfg, *_ = setup
    m = json.loads(ll.model_path(cfg).read_text())
    assert not set(m["names"]) & set(la.LIVE_DROP) and "log_buyers" in m["names"]
    assert m["meta"]["cut2"] >= m["meta"]["cut5"] and m["meta"]["rows"] == 360


def test_live_features_match_the_backtest_rows(setup):
    cfg, st, *_ = setup
    r = next(r for r in la.rows(st, cfg) if 4 in r.get("feat", {}))
    launch = json.loads(st.query("SELECT data FROM watch_enrich WHERE pool=?", (r["pool"],))[0]["data"])
    live = ll.live_features(launch, r["created_ts"], la._prior_pools(st, "OP", r["created_ts"]))
    ref = {k: v for k, v in r["feat"][4].items() if k not in la.LIVE_DROP}
    assert {k: live[k] for k in ref} == ref


def test_hand_made_pool_scored_filled_and_exited(setup):
    cfg, st, hel, clock, live, p, create = setup
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))  # seen once only
    live.step()
    assert st.query("SELECT status FROM watch_launch_live")[0]["status"] == "pending"
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    r = live._row(HAND_POOL)
    assert r["status"] == "scored" and r["liquid"] == 1 and r["score"] is not None and r["top5"] == 1
    assert r["create_ts"] == create and r["indexed_s"] == 8 and hel.history_calls == 1
    sig = live.signals.get_nowait()  # the pick goes to the launch books at once
    px = la.chain_price(json.loads(r["feat"]) and [[0, 1, "OP", 1000, 50, 1000, 50]] +
                        [[1 + j, 2 + j, f"W{j}", -1, 0.5, 1000 - j, 50 + 0.5 * j] for j in range(8)], 4)
    assert sig["kind"] == "launch" and sig["top5"] and sig["mint"] == p["token_mint"]
    assert sig["price"] == pytest.approx(px * 200) and sig["liq"] == pytest.approx(2 * r["sol_at_decide"] * 200)
    assert sig["created_ts"] == create
    clock.t = create + ll.FILL_S + 0.5
    live.step()
    r = live._row(HAND_POOL)
    assert r["fill_px"] == pytest.approx(60 / 900) and r["fill_sol"] == 60
    hel.accts[p["sol_vault"]] = ("tok", token_acct(75_000_000_000))
    clock.t = create + ll.EXIT_S
    live.step()
    r = live._row(HAND_POOL)
    assert r["status"] == "done" and r["ret"] == pytest.approx(75 / 60 - 1 - la.COST)
    out = ll.status(st, cfg, 24, now=clock.t)
    assert out["outcomes (fill at second ~6, out at minute 6, 6% cost)"]["top 5%"]["n"] == 1
    assert out["picks"]["top 5%"] == 1 and out["timing_s"]["decided_after_creation"] == 5.0


def test_pull_during_the_hold_is_minus_100(setup):
    cfg, st, hel, clock, live, p, create = setup
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    clock.t = create + 30
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 0))  # LP supply to zero: liquidity withdrawn
    clock.t = create + ll.EXIT_S
    live.step()
    r = live._row(HAND_POOL)
    assert r["pulled_ts"] == create + 30 and r["ret"] == -1.0


def test_graduation_and_late_pools_are_skipped(setup):
    cfg, st, hel, clock, live, p, create = setup
    live.on_lp_mint(GRAD_LP, lp_raw(GRAD_POOL, 100))
    live.step()
    live.save_state(force=True)
    counts = ll.status(st, cfg, 24, now=clock.t)["counts_today"]
    assert counts["skipped: graduation"] == 1 and not st.query("SELECT * FROM watch_launch_live")
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += 30  # decided far too late
    live.step()
    r = live._row(HAND_POOL)
    assert r["status"] == "skipped" and "too late" in r["reason"]
    live.save_state(force=True)  # why it was late: the decide started 30 s after its due time, not a slow call
    ws = json.loads(st.get_state("launch_live:ws"))
    assert ws["decide_start_lag_s"]["median"] == pytest.approx(30 - cfg.launch_live_decide_delay_s, abs=0.5)
    assert ws["history_call_s"]["n"] == 1 and "rate_limited" in ws["helius_process"]


def test_websocket_message_and_restart(setup):
    import base64
    cfg, st, hel, clock, live, p, create = setup
    msg = {"jsonrpc": "2.0", "method": "programNotification", "params": {"result": {"context": {"slot": 1}, "value": {
        "pubkey": HAND_LP, "account": {"data": [base64.b64encode(lp_raw(HAND_POOL, 100)).decode(), "base64"]}}}}}
    live.on_message(json.dumps(msg))
    live.on_message("not json")
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    again = ll.LaunchLive(cfg, st, hel, clock=clock, workers=0)  # restart while waiting for the outcome
    assert HAND_LP in again.seen and [j[2] for j in sorted(again.jobs)] == ["fill", "exit"]
    live.save_state(force=True)
    assert json.loads(st.get_state("launch_live:ws"))["messages"] == 2


def test_empty_history_is_retried(setup):
    cfg, st, hel, clock, live, p, create = setup
    calls = []
    real = hel.history_until
    hel.history_until = lambda *a, **k: (calls.append(1), ([], True, True) if len(calls) == 1 else real(*a, **k))[1]
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    assert live._row(HAND_POOL)["status"] == "pending"
    clock.t += ll.EMPTY_RETRY_S
    live.step()
    assert len(calls) == 2 and live._row(HAND_POOL)["status"] == "scored"


def test_real_vs_ideal_fills_pairs_book_trades_with_the_scorer(setup):
    from tradingbot.journal import Journal
    from tradingbot.models import Position
    cfg, st, hel, clock, live, p, create = setup
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    clock.t = create + ll.FILL_S + 0.5
    live.step()
    hel.accts[p["sol_vault"]] = ("tok", token_acct(75_000_000_000))
    clock.t = create + ll.EXIT_S
    live.step()  # ideal: 75/60 - 1 - 6% = +19%
    j = Journal(":memory:")
    pos = Position(None, p["token_mint"], "X", 1.0, 0.0, 100.0, 100.0, create + 7, 0.0, 1.0,
                   realized_usd=110.0, closed_at=create + 361, exit_reason="pool_age_exit", strategy="launch4_5")
    j.save_position(pos)
    out = ll.status(st, cfg, 24, now=clock.t, journal=j)["real vs ideal fills"]["launch4_5"]
    assert out["trades"] == 1 and out["filled at pool second (median)"] == 7
    assert out["gap per trade (real - ideal)"]["mean_pct"] == pytest.approx(10 - 19, abs=0.1)
    assert out["signalled but not bought (ideal)"]["n"] == 0  # launch4_5 bought the pool
    other = ll.status(st, cfg, 24, now=clock.t, journal=j)["real vs ideal fills"]["launch4_2"]
    if live._row(HAND_POOL)["top2"]:  # flagged for launch4_2 too, which didn't buy it
        assert other["signalled but not bought (ideal)"]["mean_pct"] == pytest.approx(19, abs=0.1)
    else:
        assert other["signalled but not bought (ideal)"]["n"] == 0


def test_liquidity_and_size_estimate(setup):
    cfg, st, hel, clock, live, p, create = setup
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    r = live._row(HAND_POOL)
    assert r["liq_usd"] == pytest.approx(2 * r["sol_at_decide"] * 200)
    out = ll.status(st, cfg, 24, now=clock.t)["liquidity at second 4 (USD) and trade size"]
    top5 = out["top 5%"]
    assert top5["n"] == 1 and top5["median"] == round(r["liq_usd"]) and top5["picks_per_day"] == 1.0
    full = top5["backtest edge"]
    assert full["best_size_usd"] == round(ll.BACKTEST_EDGE["top 5%"] * r["liq_usd"] / 8)
    assert full["profit_per_trade_usd"] == round(ll.BACKTEST_EDGE["top 5%"] ** 2 * r["liq_usd"] / 16)


def test_schema_migration_adds_liquidity_column():
    st = WatchStore(":memory:")
    st.execute(ll.SCHEMA.replace(", liq_usd REAL", ""))
    ll.ensure_schema(st)
    assert "liq_usd" in [c[1] for c in st.query("PRAGMA table_info(watch_launch_live)")]


def test_websocket_failure_is_recorded_not_silent(setup, monkeypatch):
    import asyncio, sys, threading
    cfg, st, hel, clock, live, p, create = setup
    monkeypatch.setitem(sys.modules, "websockets", None)  # as on an image without the package
    stop = threading.Event()

    async def run():
        try:
            await asyncio.wait_for(live._ws_loop(stop), timeout=0.3)
        except asyncio.TimeoutError:
            pass
    asyncio.run(run())
    assert live.ws["reconnects"] >= 1 and "websockets" in (live.ws["last_error"] or "")


def test_websockets_is_a_declared_dependency():
    from pathlib import Path
    assert '"websockets' in (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()


def test_drift_shows_pick_rate_and_shifted_features(setup):
    cfg, st, hel, clock, live, p, create = setup
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    d = ll.status(st, cfg, 24, now=clock.t)["live vs training (drift)"]
    assert d["pools"] == 1 and d["share_above_cut5_pct"] == 100.0
    assert set(d["training_score_quantiles"]) == {"10", "50", "90", "95", "98"}
    assert d["most_shifted_features"] and {"feature", "z", "present_pct"} <= set(d["most_shifted_features"][0])


def _done_rows(st, n, t0, rng):
    """Finished scorer rows: a feature that predicts the outcome, an hour feature that doesn't."""
    import random
    ll.ensure_schema(st)
    for i in range(n):
        good = rng.random() < 0.5
        feat = {"log_buyers": (2.0 if good else 0.0) + rng.random(), "hour_sin": rng.random(), "weekend": 1.0}
        st.execute("INSERT INTO watch_launch_live(pool, lp_mint, create_ts, decided_ts, liquid, feat, ret, status) "
                   "VALUES(?,?,?,?,?,?,?,?)", (f"D{t0}-{i}", f"L{t0}-{i}", t0 + i * 60, t0 + i * 60 + 4, 1,
                                               json.dumps(feat), 0.3 if good else -0.2, "done"))


def test_history_model_trains_on_scorer_rows_without_calendar_features(tmp_path):
    import random
    st = WatchStore(":memory:")
    cfg = Config(db_path=str(tmp_path / "bot.db"))
    _done_rows(st, 300, 2_000_000_000 - 3 * 86400, random.Random(1))
    m = ll.train_hist(st, cfg, now=2_000_000_000)
    assert m is not None and m.meta["rows"] == 300 and m.meta["clip_z"] == la.CLIP_Z
    assert "log_buyers" in m.names and not set(m.names) & set(ll.CALENDAR)
    assert m.score({"log_buyers": 2.5}) > m.score({"log_buyers": 0.5})
    # a wild input is capped, as in training: it can't move the score without limit
    assert abs(m.score({"log_buyers": 1e9}) - m.score({"log_buyers": 1e6})) < 1e-9


def test_history_flags_use_a_rolling_cut_off_and_reach_the_twin_books(setup, monkeypatch):
    import random
    from tradingbot import model_book as mb
    cfg, st, hel, clock, live, p, create = setup
    _done_rows(st, 300, clock.t - 3 * 86400, random.Random(2))
    assert ll.train_hist_and_save(st, cfg) is not None
    # under ROLLING_MIN past scores: the training cut-offs; after that, the past day's quantiles
    few = live._hist_verdict({"log_buyers": 2.9}, clock.t)
    assert few["top5h"] == int(few["score_h"] >= live.hist_model.meta["cut5"] - 1e-9)
    with live.lock:
        live.hist_scores.extend((clock.t - 10, 100.0) for _ in range(ll.ROLLING_MIN))  # a day of very high scores
    assert live._hist_verdict({"log_buyers": 2.9}, clock.t)["top5h"] == 0  # not in the top 5% of that day
    with live.lock:
        live.hist_scores.clear()
        live.hist_scores.extend((clock.t - 10, -100.0) for _ in range(ll.ROLLING_MIN))
    assert live._hist_verdict({"log_buyers": 0.0}, clock.t)["top5h"] == 1
    cfg.launch_book, cfg.launch_hist_book, cfg.launch_live = True, True, True
    try:
        assert mb.active_books(cfg)[-4:] == [mb.LAUNCH2, mb.LAUNCH5, mb.LAUNCH2H, mb.LAUNCH5H]
        sig = {"kind": "launch", "top2": False, "top5": False, "top2h": False, "top5h": True}
        assert mb.wants(mb.LAUNCH5H, sig) and not mb.wants(mb.LAUNCH2H, sig) and not mb.wants(mb.LAUNCH5, sig)
        assert mb.exit_deadline(cfg, mb.LAUNCH5H, {"created_ts": 100.0, "mint": "M"}) == 100.0 + 360
    finally:
        cfg.launch_book, cfg.launch_hist_book, cfg.launch_live = False, False, False


def test_decision_stores_the_history_verdict(setup):
    import random
    cfg, st, hel, clock, live, p, create = setup
    _done_rows(st, 300, clock.t - 3 * 86400, random.Random(3))
    ll.train_hist_and_save(st, cfg)
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    r = live._row(HAND_POOL)
    assert r["status"] == "scored" and r["score_h"] is not None and r["top5h"] in (0, 1)
    out = ll.status(st, cfg, 24, now=clock.t)
    assert "top 5% (history model)" in out["picks"]


def test_compare_ranks_both_models_on_the_same_pools_and_splits_rugs_by_pull_time():
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    t0 = 1_800_000_000.0
    for i in range(100):
        # old model: pool i scores i; history model: reversed. Pools 95-99 rug early (old's top 5%),
        # pools 0-2 rug late and 3-4 drain without a pull (history model's top 5%).
        ret, pulled = 0.1, None
        if i >= 95:
            ret, pulled = -1.0, t0 + i + 30
        elif i < 3:
            ret, pulled = -1.0, t0 + i + 300
        elif i < 5:
            ret = -1.0
        st.execute("INSERT INTO watch_launch_live(pool, create_ts, liquid, score, score_h, top2, top5, top2h, top5h, "
                   "ret, pulled_ts, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (f"P{i}", t0 + i, 1, float(i), float(-i), int(i >= 98), int(i >= 95), int(i < 2), int(i < 5),
                    ret, pulled, "done"))
    st.execute("INSERT INTO watch_launch_live(pool, create_ts, liquid, score, ret, status) VALUES "
               "('OLD', ?, 1, 5.0, 0.2, 'done')", (t0,))  # no history score: left out
    out = ll.compare(st, now=t0 + 200)
    assert out["pools"] == 100
    top5 = out["ranked in the window (same share of pools for both)"]["top 5%"]
    assert top5["picks_each"] == 5 and top5["picked by both"] == 0
    assert top5["old model"]["early_rug_pct"] == 100 and top5["old model"]["rugged_pct"] == 100
    hist = top5["history model"]
    assert hist["early_rug_pct"] == 0 and hist["later_rug_pct"] == 60 and hist["drained_no_pull_pct"] == 40
    assert out["live flags (as the books saw them)"]["old top2"]["n"] == 2
    assert out["every liquid pool"]["rugged_pct"] == 10
    dec = out["by score decile (ranked within the window)"]
    old_top, hist_top = dec["old model"]["D10 (highest scores)"], dec["history model"]["D10 (highest scores)"]
    assert old_top["n"] == 10 and old_top["rugged_pct"] == 50 and old_top["early_rug_pct"] == 50
    assert hist_top["rugged_pct"] == 50 and hist_top["later_rug_pct"] == 30
    assert dec["old model"]["D5"]["rugged_pct"] == 0
    assert ll.compare(WatchStore(":memory:"))["pools"] == 0


class _PathHelius:
    """Fake Helius for the exit study: pool state after its last tx by a time, from a table."""

    def __init__(self, states):
        self.states, self.credits_used, self.calls = states, 0, 0

    def last_before(self, pool, ts):
        self.calls += 1
        self.credits_used += 10
        sol, tok = self.states[pool](ts)
        return {"meta": {"err": None, "postTokenBalances": [
            {"owner": pool, "mint": ps.WSOL, "uiTokenAmount": {"uiAmount": sol, "amount": str(int(sol * 1e9))}},
            {"owner": pool, "mint": "M" + pool, "uiTokenAmount": {"uiAmount": tok / 1e6, "amount": str(tok)}}]},
            "blockTime": ts}


def test_exit_study_returns_by_exit_minute_rug_timing_and_liquidity():
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    t0 = 1_800_000_000.0
    states = {}
    for i in range(50):
        pool = f"P{i}"
        pulled = t0 + 150 if i >= 48 else None  # the two best-scored pools are pulled at 2.5 min
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, score, score_h, top2h, top5h, "
                   "fill_px, fill_sol, peak_sol, ret, pulled_ts, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (pool, "M" + pool, t0, 1, float(i), float(i), 0, int(i >= 48), 100.0 / 1_000_000, 100.0, 100.0,
                    -1.0 if pulled else 0.0, pulled, "done"))
        # price rises 10% a minute (SOL up, tokens down 0), so selling at minute 1 = +10% - 6% cost
        states[pool] = lambda ts: (100.0 * (1 + 0.1 * (ts - t0) / 60), 1_000_000)
    h = _PathHelius(states)
    out = ll.exit_study(st, h, now=t0 + 400, max_credits=100_000)
    g = out["groups"]["history model top 8%"]
    assert g["n"] == 4
    assert g["rug timing (LP pulls)"]["120-180s"].startswith("2 ")
    m1, m3 = g["returns by exit"]["sell at minute 1"], g["returns by exit"]["sell at minute 3"]
    assert m1["mean_pct"] == 4.0 and m1["rugged_pct"] == 0
    assert m3["rugged_pct"] == 50
    assert g["liquidity by minute"]["minute 3"]["LP pulled by now %"] == 50
    assert g["liquidity by minute"]["minute 1"]["median pool SOL vs fill (still live)"] == 1.1
    calls = h.calls
    ll.exit_study(st, h, now=t0 + 400, max_credits=100_000)
    assert h.calls == calls  # saved states are never fetched twice
    capped = ll.exit_study(WatchStore(":memory:"), h, now=t0)
    assert capped["pools"] == 0


def test_late_entry_study_enters_the_low_deciles_after_the_early_pulls():
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    t0 = 1_800_000_000.0
    states = {}
    for i in range(50):
        pool = f"P{i}"
        pulled = t0 + 60 if i == 0 else None  # pulled before the 90 s entry
        ret = 1.0 if i == 2 else 0.0  # P2 is a big winner of the scorer's second-6 trade
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, score, score_h, fill_px, "
                   "fill_sol, peak_sol, ret, pulled_ts, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (pool, "M" + pool, t0, 1, 0.0, float(i), 100 / 1e6, 100.0, 100.0, ret, pulled, "done"))

        def state(ts, i=i):
            sec = ts - t0
            if i == 1:
                return 5.0, 1_000_000  # drained before entry
            if i == 2 and sec >= 180:
                return 100.0, 500_000  # price doubles after entry
            if i == 3 and sec >= 360:
                return 5.0, 1_000_000  # drained after entry
            return 100.0, 1_000_000
        states[pool] = state
    h = _PathHelius(states)
    out = ll.late_entry_study(st, h, now=t0 + 2000, max_credits=100_000)
    d1 = out["groups"]["history D1 (lowest 10%)"]
    assert d1["n"] == 5 and d1["entered"] == 3
    assert d1["pulled before entry"] == 1 and d1["drained before entry"] == 1
    assert d1["big winners kept (second 6 -> minute 6 >= +50%, entered at 90 s)"] == 1
    m3, m6 = d1["enter 90 s, sell at 3 min"], d1["enter 90 s, sell at 6 min"]
    assert m3["mean_pct"] == 27.3 and m3["big_win_pct"] == 33 and m3["rugged_pct"] == 0
    assert m6["rugged_pct"] == 33 and d1["enter 90 s, sell at 20 min"]["rugged_pct"] == 33
    assert out["groups"]["random 60 of D3-D10"]["entered"] == 40
    calls = h.calls
    ll.late_entry_study(st, h, now=t0 + 2000, max_credits=100_000)
    assert h.calls == calls  # cached states are never fetched twice
    capped = ll.late_entry_study(st, _PathHelius(states), now=t0 + 2000, max_credits=0)
    assert capped["groups"]["history D1-D2"]["entered"] == 8  # all from the cache


def test_early_pull_study_finds_a_planted_early_puller_signal_and_beats_random_skips():
    import random as _random
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    rnd = _random.Random(3)
    t0 = 1_800_000_000.0
    for i in range(900):
        create = t0 + i * 200  # 50 hours
        puller = rnd.random() < 0.2
        creator = f"BAD{rnd.randrange(20)}" if puller else f"C{i}"
        feat = {"creator_buys": float(puller) * 2 + rnd.random(), "log_txs": rnd.random(), "hour_sin": 0.3,
                "name_len": 4.0}
        pulled = create + 40 if puller else (create + 500 if rnd.random() < 0.3 else None)
        ret = -1.0 if puller else rnd.choice([0.1, -0.1, 0.6])
        st.execute("INSERT INTO watch_launch_live(pool, creator, create_ts, liquid, feat, score_h, ret, pulled_ts, "
                   "status) VALUES (?,?,?,?,?,?,?,?,?)",
                   (f"P{i}", creator, create, 1, json.dumps(feat), rnd.random(), ret, pulled, "done"))
    out = ll.early_pull_study(st, now=t0 + 900 * 200)
    assert out["tested"] > 300 and out["blocks"]
    allp = out["every liquid pool (test blocks)"]
    assert allp["auc"] > 0.9
    s20 = allp["skip riskiest 20%"]
    assert s20["early_pulls_left_pct"] < allp["early_pull_pct"] / 2
    assert s20["mean_pct"] > s20["random_skip_mean_pct"] and s20["beats_random_share_pct"] == 100
    assert "history D1-D2 (test blocks)" in out
    assert all("name_len" not in b["top_weights"] and "hour_sin" not in b["top_weights"] for b in out["blocks"])
    assert ll.early_pull_study(WatchStore(":memory:"))["pools"] == 0


def _pull_done_rows(st, n, t0, rng):
    """Finished scorer rows where early pullers (LP pulled at 40 s) buy their own pool: creator_buys is high."""
    ll.ensure_schema(st)
    for i in range(n):
        puller = rng.random() < 0.2
        feat = {"creator_buys": float(puller) * 2 + rng.random(), "log_buyers": rng.random(), "hour_sin": 0.1}
        create = t0 + i * 60
        st.execute("INSERT INTO watch_launch_live(pool, lp_mint, creator, create_ts, decided_ts, liquid, feat, "
                   "score_h, ret, pulled_ts, status) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (f"Q{t0}-{i}", f"QL{t0}-{i}", "BAD" if puller else f"C{i}", create, create + 4, 1,
                    json.dumps(feat), rng.random(), -1.0 if puller else 0.2, create + 40 if puller else None, "done"))


def test_pull_model_trains_on_scorer_rows_and_keeps_risk_percentiles(tmp_path):
    import random
    st = WatchStore(":memory:")
    cfg = Config(db_path=str(tmp_path / "bot.db"))
    now = 2_000_000_000
    assert ll.train_pull(st, cfg, now=now) is None  # no rows yet
    _pull_done_rows(st, 600, now - 2 * 86400, random.Random(4))
    m = ll.train_pull(st, cfg, now=now)
    assert m is not None and m.meta["rows"] == 600 and m.meta["auc_in_sample"] > 0.9
    assert len(m.meta["risk_quantiles"]) == len(ll.RISK_LEVELS) and 100 <= m.meta["low_rows"] <= 140
    assert "hour_sin" not in m.names and "op_early_pulls" in m.names
    assert m.score({"creator_buys": 2.5}) > m.score({"creator_buys": 0.5})
    assert ll.train_pull_and_save(st, cfg) == ll.pull_model_path(cfg)


def test_low_flags_follow_the_risk_percentile():
    assert ll.low_flags(0, 5.0) == {"d12": False, "d12s10": False, "d12s20": False, "d12s30": False, "d12s50": False}
    assert ll.low_flags(1, None) == {"d12": True, "d12s10": False, "d12s20": False, "d12s30": False, "d12s50": False}
    f = ll.low_flags(1, 75.0)  # riskier than 75% of the past day's bottom-20% pools
    assert f["d12"] and f["d12s10"] and f["d12s20"] and not f["d12s30"] and not f["d12s50"]


def test_low_decile_books_get_their_flags_from_the_live_scorer(setup, monkeypatch):
    import random
    from tradingbot import model_book as mb
    cfg, st, hel, clock, live, p, create = setup
    _done_rows(st, 300, clock.t - 3 * 86400, random.Random(5))
    _pull_done_rows(st, 600, clock.t - 2 * 86400, random.Random(6))
    assert ll.train_hist_and_save(st, cfg) is not None and ll.train_pull_and_save(st, cfg) is not None
    # loading a model rebuilds its rolling history from the table, so load both before planting scores
    assert live._load_hist_model() is not None and live._load_pull_model() is not None
    # bottom 20%: the training quantile until ROLLING_MIN past scores, then the past day's
    with live.lock:
        live.hist_scores.clear()
        live.hist_scores.extend((clock.t - 10, 100.0) for _ in range(ll.ROLLING_MIN))
    assert live._hist_verdict({"log_buyers": 0.0}, clock.t)["low_h"] == 1
    with live.lock:
        live.hist_scores.clear()
        live.hist_scores.extend((clock.t - 10, -100.0) for _ in range(ll.ROLLING_MIN))
    assert live._hist_verdict({"log_buyers": 0.0}, clock.t)["low_h"] == 0
    # risk percentile: training percentiles first, then the past day's bottom-20% risks
    few = live._risk_verdict({"creator_buys": 0.2}, "NEW", clock.t, clock.t)
    assert few["risk"] is not None and 0 <= few["risk_pct"] <= 100
    with live.lock:
        live.low_risks.clear()
        live.low_risks.extend((clock.t - 10, -100.0) for _ in range(ll.ROLLING_MIN))
    assert live._risk_verdict({"creator_buys": 0.2}, "NEW", clock.t, clock.t)["risk_pct"] == 100.0
    # the creator's record from the table counts: BAD pulled early many times before
    risky = live._risk_verdict({"creator_buys": 0.2}, "BAD", clock.t, clock.t)["risk"]
    safe = live._risk_verdict({"creator_buys": 0.2}, "NEW", clock.t, clock.t)["risk"]
    assert risky != safe
    # a decision on a bottom-20% pool sends a signal with every level's flag and stores the verdict
    with live.lock:
        live.low_risks.clear()
        live.low_risks.extend((clock.t - 10, 100.0) for _ in range(ll.ROLLING_MIN))  # this pool is the safest
    monkeypatch.setattr(live, "_hist_verdict", lambda feat, now: {"score_h": -5.0, "top2h": 0, "top5h": 0, "low_h": 1})
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    r = live._row(HAND_POOL)
    assert r["low_h"] == 1 and r["risk"] is not None and r["risk_pct"] == 0.0
    sig = live.signals.get_nowait()
    assert sig["d12"] and sig["d12s10"] and sig["d12s30"] and sig["d12s50"]
    assert any(k.startswith("history bottom 20%") for k in
               ll.status(st, cfg, 24, now=clock.t)["outcomes (fill at second ~6, out at minute 6, 6% cost)"])
    cfg.launch_low_book, cfg.launch_live = True, True
    try:
        assert mb.active_books(cfg)[-5:] == list(mb.LAUNCH_LOW_CUT)
        low_only = dict(sig, top2=False, top5=False)  # the fixture's pool is also an old-model pick
        assert all(mb.wants(b, low_only) for b in mb.LAUNCH_LOW_CUT) and not mb.wants(mb.LAUNCH2, low_only)
        assert not mb.wants("launchd12_s30", dict(sig, d12s30=False))
        assert mb.exit_deadline(cfg, "launchd12_s30", {"created_ts": 100.0, "mint": "M"}) == 100.0 + 360
    finally:
        cfg.launch_low_book, cfg.launch_live = False, False


class _AgeHelius:
    def __init__(self, first):
        self.first, self.credits_used, self.calls = first, 0, 0

    def first_tx_time(self, address):
        self.calls += 1
        self.credits_used += 10
        return self.first.get(address)


def test_token_age_study_splits_new_tokens_from_new_pools_of_old_tokens():
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    t0 = 1_800_000_000.0
    first = {}
    for i in range(100):
        old = i < 10 and i % 2 == 0  # half of D1 (the lowest scores) are pools of tokens a week old
        first[f"M{i}"] = t0 - 7 * 86400 if old else t0 + i * 60 - 30
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, score_h, ret, status) "
                   "VALUES (?,?,?,?,?,?,?)", (f"P{i}", f"M{i}", t0 + i * 60, 1, float(i), 0.0 if old else 0.5, "done"))
    h = _AgeHelius(first)
    out = ll.token_age_study(st, h, now=t0 + 100 * 60)
    low = out["history D1-D2"]
    assert low["all"]["n"] == 20 and low["share by token age (%)"]["over 1 day old"] == 25.0
    assert low["over 1 day old"]["mean_pct"] == 0.0 and low["fresh (minted < 10 min before the pool)"]["mean_pct"] == 50.0
    assert out["random 150 of D3-D10"]["all"]["n"] == 80 and out["fetch"]["fetched"] == 100
    calls = h.calls
    ll.token_age_study(st, h, now=t0 + 100 * 60)
    assert h.calls == calls  # cached


def test_low_decile_trades_lists_every_bottom_20pct_trade_with_win_rate_and_return_bands():
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    t0 = 1_800_000_000.0
    rets = [-1.0, -0.7, -0.2, 0.1, 0.8]  # one per band
    for i in range(25):
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, score_h, ret, pulled_ts, "
                   "risk_pct, status) VALUES (?,?,?,?,?,?,?,?,?)",
                   (f"P{i}", f"M{i}", t0 + i, 1, float(i), rets[i % 5], t0 + i + 40 if i % 5 == 0 else None,
                    10.0 * i if i < 3 else None, "done"))
    out = ll.low_decile_trades(st, now=t0 + 100)
    assert out["pools"] == 25 and out["d1_d2_trades"] == 5 and len(out["trades"]) == 5
    s = out["summary"]
    assert s["win_pct"] == 40 and all(v == 20 for v in s["returns (% of trades)"].values())
    first = out["trades"][0]
    assert first["return_pct"] == -100.0 and first["pulled_after_s"] == 40 and first["risk_pct"] == 0


def _rw_done_rows(st, n, t0, rng):
    """Finished scorer rows: early pullers buy their own pool (creator_buys high, -100%), big winners draw many
    buyers (log_buyers high, +80%), the rest drift."""
    ll.ensure_schema(st)
    for i in range(n):
        kind = rng.choices(["rug", "win", "flat"], [0.2, 0.2, 0.6])[0]
        feat = {"creator_buys": (2.0 if kind == "rug" else 0.0) + rng.random(),
                "log_buyers": (2.0 if kind == "win" else 0.0) + rng.random(), "hour_sin": 0.1}
        create = t0 + i * 60
        st.execute("INSERT INTO watch_launch_live(pool, lp_mint, creator, create_ts, decided_ts, liquid, feat, "
                   "score_h, ret, pulled_ts, status) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (f"R{t0}-{i}", f"RL{t0}-{i}", f"C{i}", create, create + 4, 1, json.dumps(feat), rng.random(),
                    {"rug": -1.0, "win": 0.8, "flat": 0.05}[kind], create + 40 if kind == "rug" else None, "done"))


def test_rw_flags_keep_the_less_risky_half():
    assert ll.rw_flags(0, 10.0, None) == {"rw50": True, "d12rw50": False}
    assert ll.rw_flags(1, 60.0, 30.0) == {"rw50": False, "d12rw50": True}
    assert ll.rw_flags(1, None, None) == {"rw50": False, "d12rw50": False}


def test_rug_vs_win_model_trains_scores_and_feeds_its_books(setup, monkeypatch, tmp_path):
    import random
    from tradingbot import model_book as mb
    cfg, st, hel, clock, live, p, create = setup
    assert ll.train_rw(st, cfg, now=clock.t) is None  # no rows yet
    _done_rows(st, 300, clock.t - 3 * 86400, random.Random(5))
    _rw_done_rows(st, 600, clock.t - 2 * 86400, random.Random(8))
    m = ll.train_rw(st, cfg, now=clock.t)
    assert m is not None and len(m.meta["quantiles"]) == len(ll.RISK_LEVELS)
    rug_like = {"creator_buys": 2.5, "log_buyers": 0.5}
    win_like = {"creator_buys": 0.5, "log_buyers": 2.5}
    assert m.score(rug_like) > m.score({"creator_buys": 0.5, "log_buyers": 0.5}) > m.score(win_like)
    assert ll.RugWinModel.from_json(json.loads(json.dumps(m.to_json()))).score(win_like) == pytest.approx(
        m.score(win_like))
    assert ll.train_rw_and_save(st, cfg) == ll.rw_model_path(cfg)
    assert ll.train_hist_and_save(st, cfg) is not None
    assert live._load_hist_model() is not None and live._load_rw_model() is not None
    # percentile: the past day's liquid pools once there are enough of them
    with live.lock:
        live.rw_all.clear()
        live.rw_all.extend((clock.t - 10, 5.0) for _ in range(ll.ROLLING_MIN))
    v = live._rw_verdict(win_like, "NEW", clock.t, clock.t, low=False)
    assert v["rw_pct"] == 0.0 and "rw_low_pct" not in v
    # a decision sends the rug-vs-win flags; a bottom-20% pool also gets its rank among those
    with live.lock:
        live.rw_low.clear()
        live.rw_low.extend((clock.t - 10, 5.0) for _ in range(ll.ROLLING_MIN))
    monkeypatch.setattr(live, "_hist_verdict", lambda feat, now: {"score_h": -5.0, "top2h": 0, "top5h": 0, "low_h": 1, "rw_low6": 1})
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    r = live._row(HAND_POOL)
    assert r["rw"] is not None and r["rw_pct"] < 5 and r["rw_low_pct"] == 0.0  # safer than every planted score
    sig = live.signals.get_nowait()
    assert sig["rw50"] and sig["d12rw50"]
    outcomes = ll.status(st, cfg, 24, now=clock.t)["outcomes (fill at second ~6, out at minute 6, 6% cost)"]
    assert any(k.startswith("rug-vs-win") for k in outcomes)
    cfg.launch_rw_book, cfg.launch_live = True, True
    try:
        assert mb.active_books(cfg)[-2:] == list(mb.LAUNCH_RW_CUT)
        rw_only = dict(sig, top2=False, top5=False, d12=False)
        assert mb.wants("launchrw50", rw_only) and mb.wants("launchd12rw50", rw_only)
        assert not mb.wants("launchd12rw50", dict(rw_only, d12rw50=False))
        assert not mb.wants(mb.LAUNCH2, rw_only)
        assert mb.exit_deadline(cfg, "launchrw50", {"created_ts": 100.0, "mint": "M"}) == 100.0 + 360
    finally:
        cfg.launch_rw_book, cfg.launch_live = False, False


def test_rug_vs_win_follows_the_study_rule(setup, monkeypatch):
    """11-05: as launch-filter-study option 4: trained on every finished pool (not the last 14 days), each pool
    ranked against the past 6 h (the study's block) and the bottom 20% taken over the same 6 h."""
    import random
    cfg, st, hel, clock, live, p, create = setup
    cfg.launch_train_days = 1.0
    _rw_done_rows(st, 600, clock.t - 30 * 86400, random.Random(8))  # a month old: outside launch_train_days
    m = ll.train_rw(st, cfg, now=clock.t)
    assert m is not None and m.meta["rows"] >= 600 and m.meta["days"] >= 29
    assert ll.train_rw_and_save(st, cfg) is not None and live._load_rw_model() is not None
    win_like = {"creator_buys": 0.5, "log_buyers": 2.5}
    with live.lock:  # riskier scores, but 7 h old: outside the 6 h window, so the training quantiles rank it
        live.rw_all.clear()
        live.rw_all.extend((clock.t - 7 * 3600, 5.0) for _ in range(ll.ROLLING_MIN))
    old = live._rw_verdict(win_like, "NEW", clock.t, clock.t, low=False)["rw_pct"]
    with live.lock:
        live.rw_all.clear()
        live.rw_all.extend((clock.t - 5 * 3600, 5.0) for _ in range(ll.ROLLING_MIN))
    assert live._rw_verdict(win_like, "NEW", clock.t, clock.t, low=False)["rw_pct"] == 0.0 != old
    # bottom 20%: the past day's scores set low_h (the d12 books), the past 6 h set rw_low6 (the rw books)
    assert ll.train_hist_and_save(st, cfg) is not None and live._load_hist_model() is not None
    score = live._load_hist_model().score({"log_buyers": 1.0})
    with live.lock:
        live.hist_scores.clear()
        live.hist_scores.extend((clock.t - 20 * 3600, score + 1.0) for _ in range(6 * ll.ROLLING_MIN))  # day
        live.hist_scores.extend((clock.t - 3600, score - 1.0) for _ in range(ll.ROLLING_MIN))  # last 6 h: lower
    v = live._hist_verdict({"log_buyers": 1.0}, clock.t)
    assert v["low_h"] == 1 and v["rw_low6"] == 0
    # rows scored before the change have no rw_low6: the flag falls back to low_h
    assert ll.rw_low({"low_h": 1, "rw_low6": None}) and not ll.rw_low({"low_h": 1, "rw_low6": 0})
    assert ll.rw_low({"low_h": 1})


def test_rug_vs_win_model_is_retrained_every_6_hours(tmp_path, monkeypatch):
    import subprocess
    import threading
    cfg = Config(db_path=str(tmp_path / "t.db"), launch_rw_book=True)
    live = ll.LaunchLive(cfg, WatchStore(":memory:"), FakeHelius({}), clock=Clock(0.0), workers=0)
    runs, waits = [], []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: (runs.append(cmd), ll.rw_model_path(cfg).parent.mkdir(
        parents=True, exist_ok=True), ll.rw_model_path(cfg).write_text("{}"),
        subprocess.CompletedProcess(cmd, 0, b"", b""))[-1])

    class Stop(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) >= 2:
                self.set()
            return self.is_set()
    live._rw_retrain_loop(Stop())  # no model: trains at once, then waits for the file to age 6 h
    assert len(runs) == 1 and runs[0][-1] == "launch-rw-train"
    assert waits[0] == ll.RW_RETRY_S and waits[1] <= ll.RW_RETRY_S


def test_earlier_rule_twin_books_score_rank_and_retrain_on_their_own(setup, monkeypatch):
    """11-05: launchrw50_24h / launchd12rw50_24h keep the earlier live rule (14-day model, 24 h ranking, low_h)
    beside the study-rule books."""
    import os
    import random
    import subprocess
    import time
    import threading
    from tradingbot import model_book as mb
    cfg, st, hel, clock, live, p, create = setup
    cfg.launch_train_days = 1.0
    _rw_done_rows(st, 600, clock.t - 30 * 86400, random.Random(8))  # older than the 24 h rule's 1-day window
    assert ll.train_rw24_and_save(st, cfg, now=clock.t) is None  # the earlier rule sees no rows inside its window
    _rw_done_rows(st, 600, clock.t - 12 * 3600, random.Random(9))
    assert ll.train_rw24_and_save(st, cfg, now=clock.t) == ll.rw24_model_path(cfg)
    assert ll.train_rw_and_save(st, cfg) == ll.rw_model_path(cfg)
    m24 = ll.RugWinModel.from_json(json.loads(ll.rw24_model_path(cfg).read_text()))
    m6 = ll.RugWinModel.from_json(json.loads(ll.rw_model_path(cfg).read_text()))
    assert m24.meta["rows"] < m6.meta["rows"]  # 14-day window vs every finished pool
    win_like = {"creator_buys": 0.5, "log_buyers": 2.5}
    assert live._load_rw_model("rw24") is not None  # loading rebuilds the window from the table: load first
    with live.lock:  # 20 h old: inside the 24 h rule's window, outside the 6 h one
        live.rw24_all.clear()
        live.rw24_low.clear()
        live.rw24_all.extend((clock.t - 20 * 3600, 5.0) for _ in range(ll.ROLLING_MIN))
        live.rw24_low.extend((clock.t - 20 * 3600, 5.0) for _ in range(ll.ROLLING_MIN))
    v = live._rw_verdict(win_like, "NEW", clock.t, clock.t, low=True, track="rw24")
    assert v["rw24_pct"] == 0.0 and v["rw24_low_pct"] == 0.0 and "rw_pct" not in v
    assert ll.rw24_flags(1, 10.0, 10.0) == {"rw50_24": True, "d12rw50_24": True}
    assert ll.rw24_flags(0, 10.0, None) == {"rw50_24": True, "d12rw50_24": False}
    cfg.launch_rw24_book = cfg.launch_live = True
    try:
        assert mb.active_books(cfg)[-2:] == list(mb.LAUNCH_RW24_CUT)
        assert mb.wants("launchrw50_24h", {"rw50_24": True}) and not mb.wants("launchrw50", {"rw50_24": True})
        assert mb.exit_deadline(cfg, "launchd12rw50_24h", {"created_ts": 100.0, "mint": "M"}) == 100.0 + 360
        # a decision stores both rules' ranks and sends the twin flags
        monkeypatch.setattr(live, "_hist_verdict", lambda feat, now: {"score_h": -5.0, "top2h": 0, "top5h": 0,
                                                                      "low_h": 1, "rw_low6": 0})
        live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
        live.step()
        clock.t += cfg.launch_live_decide_delay_s
        live.step()
        r = live._row(HAND_POOL)
        assert r["rw24"] is not None and r["rw24_pct"] is not None and r["rw24_low_pct"] is not None
        sig = live.signals.get_nowait()
        assert "rw50_24" in sig and "d12rw50_24" in sig and not sig["d12rw50"]  # rw_low6 0: not in the 6 h bottom
        # retrain: the 24 h rule's model once a day, the study rule's every 6 h
        cfg.launch_rw_book = True
        runs = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: (runs.append(cmd[-1]),
                                                                  subprocess.CompletedProcess(cmd, 0, b"", b""))[-1])
        old = time.time() - 7 * 3600
        os.utime(ll.rw_model_path(cfg), (old, old))  # 7 h old: due under the 6 h rule, not the 24 h one
        stop = threading.Event()
        monkeypatch.setattr(stop, "wait", lambda t=None: stop.set() or True)
        live._rw_retrain_loop(stop)
        assert runs == ["launch-rw-train"]
    finally:
        cfg.launch_rw24_book = cfg.launch_live = cfg.launch_rw_book = False


def test_rw_check_tells_a_working_rule_from_chance():
    """A rule whose kept half wins (riskier scores rug) beats skipped, random keeps, and has a rug AUC > 0.5;
    the 24 h columns are random, so that rule shows no gap."""
    import random
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    rng = random.Random(1)
    t0 = 1_700_000_000.0
    for i in range(240):
        risky = i % 2 == 0
        ret = -1.0 if risky and rng.random() < 0.5 else rng.uniform(-0.2, 0.6)
        st.execute("INSERT INTO watch_launch_live(pool, create_ts, liquid, rw, rw_pct, rw24, rw24_pct, low_h, ret, "
                   "status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (f"P{i}", t0 + i * 300, 1, 0.5 if risky else -0.5, 80.0 if risky else 20.0, rng.random(),
                    rng.uniform(0, 100), 0, ret, "done"))
    out = ll.rw_check(st, hours=6, now=t0 + 240 * 300)
    life = out["whole life"]
    study = life["study rule (6 h)"]["every liquid pool minus the riskiest half"]
    assert study["kept minus skipped (pts)"]["mean"] > 20
    assert study["random keeps of the same size"]["share worse than kept"] > 0.99
    assert study["AUC (score = riskiness)"]["rugs (want > 0.5)"] > 0.7
    assert study["blocks where kept beat skipped"] == "5 of 5"
    other = life["24 h rule"]["every liquid pool minus the riskiest half"]
    assert abs(other["kept minus skipped (pts)"]["mean"]) < study["kept minus skipped (pts)"]["mean"]
    assert life["study rule (6 h)"]["history bottom 20% minus the riskiest half"]["pools"] == 0
    assert out["last 6 h"]["pools"] == 72


def test_early_rw_check_splits_early_pools_and_trades_by_the_rw_score(tmp_path):
    """Early pools that rose carry a low rug-vs-win percentile and those that faded a high one: the 6 h rule's kept
    half wins; an early-book trade is matched by token, a stuck one counts as -100%."""
    import random
    from tradingbot.journal import Journal
    from tests.test_model_book import snap, T0
    rng = random.Random(2)
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    for i in range(120):
        created = T0 + i * 600
        up = rng.random() < 0.4
        st.execute("INSERT INTO watch_pools(pool, mint, dex, name, created_ts, seen_ts, first) VALUES(?,?,?,?,?,?,?)",
                   (f"P{i}", f"M{i}", "pumpswap", f"P{i} / SOL", created, created + 120,
                    json.dumps(snap(1.0, liq=20_000.0))))
        st.execute("INSERT INTO watch_snaps(pool, age_min, ts, data) VALUES(?,?,?,?)",
                   (f"P{i}", 5, created + 360, json.dumps(snap(1.5 if up else 0.8, liq=20_000.0))))
        if i % 10:  # every tenth pool unscored (not liquid at second 4)
            st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, rw, rw_pct, rw24, rw24_pct, "
                       "status) VALUES (?,?,?,?,?,?,?,?,?)",
                       (f"P{i}", f"M{i}", created, 1, -0.5 if up else 0.5, 20.0 if up else 80.0, rng.random(),
                        rng.uniform(0, 100), "done"))
    j = Journal(":memory:")
    j.execute("INSERT INTO positions(token, cost_usd, opened_at, closed_at, realized_usd, strategy) VALUES (?,?,?,?,?,?)",
              ("M1", 100.0, T0 + 600 + 130, T0 + 600 + 400, 130.0, "early6"))
    j.execute("INSERT INTO positions(token, cost_usd, opened_at, strategy) VALUES (?,?,?,?)",
              ("M2", 100.0, T0 + 1200 + 130, "early6"))
    cfg = Config(db_path=str(tmp_path / "bot.db"))
    out = ll.early_rw_check(st, j, cfg, now=T0 + 121 * 600)
    pools = out["every eligible early pool (snapshot yardstick)"]
    assert pools["n"] == 120 and pools["with a rug-vs-win score"] == 108
    study = pools["study rule (6 h)"]
    assert study["kept minus skipped (pts)"]["mean"] > 40
    assert study["random keeps of the same size"]["share worse than kept"] > 0.99
    trades = out["early books' real trades"]["early6"]
    assert trades["n"] == 2 and trades["with a rug-vs-win score"] == 2
    assert trades["all"]["mean_pct"] == -35.0  # +30% and a stuck -100%


def test_fill_gap_study_finds_where_real_fills_fall_short():
    """Two trades fill like the scorer; one was sold after the LP pull (-100% vs ideal +20%); one never sold."""
    from tradingbot.journal import Journal
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    j = Journal(":memory:")
    t0 = 1_700_000_000.0
    cases = [  # token, ideal, real, buy px, sell px, sold at pool second, pulled at, closed
        ("A", 0.10, 0.09, 1.0, 1.15, 362, None, True),
        ("B", -0.20, -0.21, 1.0, 0.85, 362, None, True),
        ("C", 0.20, -1.0, 1.0, None, 430, 400, True),
        ("D", 0.05, None, 1.0, None, None, None, False),
    ]
    for i, (tok, ideal, real, bpx, spx, out_s, pulled, closed) in enumerate(cases):
        c = t0 + i * 1000
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, ret, status, liq_usd, pulled_ts) "
                   "VALUES (?,?,?,?,?,?,?,?)", (f"P{tok}", tok, c, 1, ideal, "done", 20_000.0,
                                                 c + pulled if pulled else None))
        pid = j.execute("INSERT INTO positions(token, symbol, cost_usd, opened_at, closed_at, realized_usd, "
                        "exit_reason, strategy) VALUES (?,?,?,?,?,?,?,?)",
                        (tok, tok, 100.0, c + 9, c + out_s if closed else None,
                         100 * (1 + real) if closed else 0, "time_stop" if closed else None, "launchrw50"))
        j.execute("INSERT INTO fills(position_id, side, price, usd, fee_usd, ts) VALUES (?,?,?,?,?,?)",
                  (pid, "buy", bpx, -100.0, 0.2, c + 9))
        if spx:
            j.execute("INSERT INTO fills(position_id, side, price, usd, fee_usd, ts) VALUES (?,?,?,?,?,?)",
                      (pid, "sell", spx, 100 * (1 + real), 0.2, c + out_s))
    out = ll.fill_gap_study(st, j, books=("launchrw50",), now=t0 + 10_000)["launchrw50"]
    assert out["trades"] == 4
    assert out["sold after the LP was pulled"] == 1
    worst = out["worst trades"]
    assert worst[0]["symbol"] == "C" and worst[1]["symbol"] == "D" and worst[1]["exit"] == "never sold"
    bands = {b["band"]: b for b in out["by sale second (pool age at sale)"]}
    assert bands["420 s+"]["n"] == 1 and bands["420 s+"]["share_of_total_gap_pct"] > 40
    assert out["price-only gap (swap move vs scorer's gross move), mean / median"][2] == 2


def test_size_study_keeps_real_results_at_real_size_and_charges_impact_on_bigger_trades():
    from tradingbot.journal import Journal
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    j = Journal(":memory:")
    t0 = 1_700_000_000.0
    # token, liquidity at second 4 (USD), SOL in pool at fill / exit, price move, real $ back on $100
    cases = [("THIN", 10_000.0, 25.0, 40.0, 0.6, 150.0), ("DEEP", 100_000.0, 250.0, 300.0, 0.2, 117.0),
             ("RUG", 20_000.0, 50.0, 1.0, -1.0, 0.0)]
    for i, (tok, liq, fsol, esol, move, back) in enumerate(cases):
        c = t0 + i * 1000
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, liquid, ret, status, liq_usd, "
                   "sol_at_decide, fill_sol, exit_sol, fill_px, exit_px) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (f"P{tok}", tok, c, 1, max(-1.0, move - 0.06), "done", liq, fsol, fsol, esol, 1.0, 1.0 + move))
        pid = j.execute("INSERT INTO positions(token, symbol, cost_usd, opened_at, closed_at, realized_usd, strategy) "
                        "VALUES (?,?,?,?,?,?,?)", (tok, tok, 100.0, c + 9, c + 365, back, "launchrw50"))
        j.execute("INSERT INTO fills(position_id, side, price, usd, fee_usd, ts) VALUES (?,?,?,?,?,?)",
                  (pid, "buy", 1.0, -100.0, 0.3, c + 9))
    out = ll.size_study(st, j, now=t0 + 10_000)
    assert out["trades"] == 3 and out["rugs"] == 1
    p = out["policies"]
    assert p["fixed $100"]["total_usd"] == round((150 - 100) + (117 - 100) - 100)  # the real result, unchanged
    assert p["fixed $100"]["rugs_usd"] == -100 and p["fixed $1000"]["rugs_usd"] == -1000
    # $1000 costs more per $ in the thin pool than in the deep one
    thin10 = p["fixed $1000"]["by liquidity"]["< 15k"] / 1000
    assert thin10 < p["fixed $100"]["by liquidity"]["< 15k"] / 100
    deep_drop = p["fixed $100"]["by liquidity"]["50k+"] / 100 - p["fixed $1000"]["by liquidity"]["50k+"] / 1000
    thin_drop = p["fixed $100"]["by liquidity"]["< 15k"] / 100 - thin10
    assert thin_drop > deep_drop > 0
    assert out["calibration by liquidity"][0]["band"] == "< 15k"


def test_block_compare_scores_each_book_per_block_against_the_market(cfg):
    """Two 6 h blocks: the market loses in both, the sized book wins in both, launch4_5 loses more than the market
    in the second; the sized book's mean is weighted by money traded and a never-sold trade counts as -100%."""
    from tradingbot.journal import Journal
    cfg.launch_book = cfg.launch_live = cfg.launch_sized_book = True
    st = WatchStore(":memory:")
    ll.ensure_schema(st)
    j = Journal(":memory:")
    t0 = 1_700_006_400.0  # a UTC 6 h boundary (00:00 + 6 h multiple)
    t0 -= t0 % 21600
    for blk in range(2):
        for i in range(10):
            st.execute("INSERT INTO watch_launch_live(pool, create_ts, liquid, rw_pct, low_h, ret, status, liq_usd) "
                       "VALUES (?,?,?,?,?,?,?,?)", (f"P{blk}{i}", t0 + blk * 21600 + i * 60, 1, 20.0 if i < 5 else 80.0,
                                                    0, 0.4 if i < 5 else -0.6, "done", 10_000.0))

    def pos(book, ts, cost, realized, closed=True):
        j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
                  "VALUES ('T','T',?,?,0,?,?,?)", (cost, realized, ts, ts + 360 if closed else None, book))
    for blk in range(2):
        base = t0 + blk * 21600
        for i in range(5):
            pos("launchrw50_sz", base + i, 250.0, 300.0)  # +20% on $250
            pos("launchrw50_sz", base + 10 + i, 25.0, 20.0)  # -20% on $25
            pos("launch4_5", base + i, 100.0, 90.0 if blk == 0 else 20.0)
    pos("launchrw50_sz", t0 + 21600 + 100, 25.0, 0.0, closed=False)  # never sold: -$25
    out = ll.block_compare(st, j, cfg, now=t0 + 2 * 21600)
    b0, b1 = out["blocks"]
    assert b0["market ideal %"] == -10.0 and b0["sz picks ideal %"] == 40.0
    assert b0["books"]["launchrw50_sz"]["mean %"] == round(100 * (250 - 25) / 1375, 1)
    assert b1["books"]["launchrw50_sz"]["n"] == 11
    s = out["summary"]["launchrw50_sz"]
    assert s["vs market ideal"]["won"] == 2 and s["vs market ideal"]["blocks"] == 2
    head = out["launchrw50_sz vs each book (blocks where both traded 5+)"]["launch4_5"]
    assert head["won"] == 2
    text = ll.render_block_compare(out)
    assert "launchrw50_sz" in text and "BLOCK COMPARE" in text


def test_pools_under_the_liquidity_minimum_are_measured_but_never_scored_or_sent(setup):
    """11-08: pools under $5k at second 4 keep their fill and minute-6 reads (status thin -> thin_done) for the
    pool-size bands; they get no flags, no signal and stay out of the creator record."""
    cfg, st, hel, clock, live, p, create = setup
    st.execute("UPDATE watch_market SET sol_usd=40")  # 2 x ~51.5 SOL x $40 = ~$4.1k, under the $5k minimum
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    r = live._row(HAND_POOL)
    assert r["status"] == "thin" and r["liquid"] == 0 and r["liq_usd"] < 5000 and not r["top5"]
    assert r["rw_pct"] is None and live.signals.empty()
    clock.t = create + ll.FILL_S + 0.5
    live.step()
    hel.accts[p["sol_vault"]] = ("tok", token_acct(75_000_000_000))
    clock.t = create + ll.EXIT_S
    live.step()
    r = live._row(HAND_POOL)
    assert r["status"] == "thin_done" and r["ret"] == pytest.approx(75 / 60 - 1 - la.COST)
    assert st.query("SELECT COUNT(*) FROM watch_launch_live WHERE status IN ('scored','done')")[0][0] == 0


def test_thin_tracking_can_be_switched_off(setup):
    cfg, st, hel, clock, live, p, create = setup
    live.cfg = cfg.with_overrides({"launch_live_track_thin": "0"})
    st.execute("UPDATE watch_market SET sol_usd=40")
    live.on_lp_mint(HAND_LP, lp_raw(HAND_POOL, 100))
    live.step()
    clock.t += cfg.launch_live_decide_delay_s
    live.step()
    assert live._row(HAND_POOL)["status"] == "skipped"


def test_band_flags_follow_each_bands_rule():
    """11-08 band-rw-study: thin pools skip the riskiest 20% of their band, $15-25k the riskiest 70%, $25-40k keep
    the live rule (all bands ranked together), nothing above $40k or below $5k."""
    f = lambda liq, rw, rwb: ll.band_flags(liq, rw, rwb)["rwband"]
    assert f(10_000, 90.0, 79.0) and not f(10_000, 10.0, 81.0)     # thin: band rank < 80, the live rank ignored
    assert f(20_000, 90.0, 29.0) and not f(20_000, 10.0, 31.0)     # $15-25k: band rank < 30
    assert f(30_000, 49.0, 99.0) and not f(30_000, 51.0, 1.0)      # $25-40k: live rank < 50
    assert not f(50_000, 1.0, 1.0) and not f(4_000, 1.0, 1.0) and not f(None, 1.0, 1.0)
    assert not f(10_000, 1.0, None)                                # no band rank yet: no flag
    assert ll.band_of(14_999) == "$5-15k" and ll.band_of(40_000) is None


def test_rug_vs_win_ranks_each_pool_within_its_band_and_feeds_the_band_books(setup):
    import random
    from tradingbot import model_book as mb
    cfg, st, hel, clock, live, p, create = setup
    _rw_done_rows(st, 600, clock.t - 2 * 86400, random.Random(8))
    assert ll.train_rw_and_save(st, cfg) is not None and live._load_rw_model() is not None
    win_like = {"creator_buys": 0.5, "log_buyers": 2.5}
    rug_like = {"creator_buys": 2.5, "log_buyers": 0.5}
    model = live._load_rw_model()
    with live.lock:  # the thin band's past day: every pool safer than rug_like, riskier than win_like
        live.rw_band["$5-15k"].clear()
        mid = (model.score(win_like) + model.score(rug_like)) / 2
        live.rw_band["$5-15k"].extend((clock.t - 3600, mid) for _ in range(ll.ROLLING_MIN))
        live.rw_band["$15-25k"].clear()
    thin_safe = live._rw_verdict(win_like, "N", clock.t, clock.t, low=False, liq=10_000)
    thin_risky = live._rw_verdict(rug_like, "N", clock.t, clock.t, low=False, liq=10_000)
    assert thin_safe["rwb_pct"] == 0.0 and thin_risky["rwb_pct"] == 100.0
    assert live._rw_verdict(win_like, "N", clock.t, clock.t, low=False, liq=20_000)["rwb_pct"] is None  # too few yet
    assert "rwb_pct" not in live._rw_verdict(win_like, "N", clock.t, clock.t, low=False, liq=60_000)
    assert len(live.rw_band["$5-15k"]) == ll.ROLLING_MIN + 2
    # a newly loaded model rebuilds the band windows from the stored liquid pools of the past day
    st.execute("UPDATE watch_launch_live SET decided_ts=?, liq_usd=12000", (clock.t - 3600,))
    live.rw_mtime, live.rw_checked = None, 0.0
    live._load_rw_model()
    assert len(live.rw_band["$5-15k"]) >= 600 and not live.rw_band["$15-25k"]
    # the twin books: same signals, thin size $100 vs $250, $100 elsewhere, minute-6 exit
    cfg.launch_band_book, cfg.launch_live = True, True
    try:
        assert set(mb.LAUNCH_BAND_CUT) <= set(mb.active_books(cfg))
        sig = {"rwband": True, "rw50": False, "mint": "M", "created_ts": 100.0}
        assert mb.wants(mb.LAUNCH_BAND, sig) and mb.wants(mb.LAUNCH_BAND_SZ, sig)
        assert not mb.wants(mb.LAUNCH_BAND, dict(sig, rwband=False)) and not mb.wants(mb.LAUNCH_SIZED, sig)
        assert mb.band_usd(mb.LAUNCH_BAND, 10_000) == 100.0 and mb.band_usd(mb.LAUNCH_BAND_SZ, 10_000) == 250.0
        assert mb.band_usd(mb.LAUNCH_BAND, 20_000) == 100.0 == mb.band_usd(mb.LAUNCH_BAND_SZ, 30_000)
        assert mb.exit_deadline(cfg, mb.LAUNCH_BAND_SZ, sig) == 100.0 + 360
    finally:
        cfg.launch_band_book, cfg.launch_live = False, False


def test_hold_books_take_kept_25_40k_pools_and_sell_at_minute_10_and_15(cfg):
    from tradingbot import model_book as mb
    assert ll.band_flags(30_000, 40.0, None)["rw50big"] and not ll.band_flags(30_000, 60.0, None)["rw50big"]
    assert not ll.band_flags(20_000, 10.0, None)["rw50big"] and not ll.band_flags(45_000, 10.0, None)["rw50big"]
    assert not ll.band_flags(None, 10.0, None)["rw50big"] and not ll.band_flags(30_000, None, None)["rw50big"]
    assert ll.band_flags(100_000, 10.0, None) == {"rwband": False, "rw50big": False}
    cfg.launch_hold_book, cfg.launch_live = True, True
    try:
        assert set(mb.LAUNCH_HOLD_CUT) <= set(mb.active_books(cfg))
        sig = {"rw50big": True, "rw50": True, "mint": "M", "created_ts": 100.0}
        assert mb.wants(mb.LAUNCH_HOLD10, sig) and mb.wants(mb.LAUNCH_HOLD15, sig)
        assert not mb.wants(mb.LAUNCH_HOLD10, dict(sig, rw50big=False))
        assert mb.exit_deadline(cfg, mb.LAUNCH_HOLD10, sig) == 100.0 + 600
        assert mb.exit_deadline(cfg, mb.LAUNCH_HOLD15, sig) == 100.0 + 900
        assert mb.band_usd(mb.LAUNCH_HOLD10, 30_000) == 250.0 == mb.band_usd(mb.LAUNCH_HOLD15, 39_000)
        assert mb.band_usd(mb.LAUNCH_HOLD15_MAX, 30_000) == 500.0 and mb.wants(mb.LAUNCH_HOLD15_MAX, sig)
        assert mb.exit_deadline(cfg, mb.LAUNCH_HOLD15_MAX, sig) == 100.0 + 900
        assert mb.LAUNCH_HOLD15_MAX in mb.active_books(cfg)
        assert mb.hold_min(cfg, mb.LAUNCH_HOLD15) == 15.0
        assert mb.signal_kind(mb.LAUNCH_HOLD10) == "launch"
        from tradingbot import dashboard
        assert dashboard.family_of(mb.LAUNCH_HOLD10) == "Rug-vs-win"
    finally:
        cfg.launch_hold_book, cfg.launch_live = False, False


def test_big_book_ranks_40_80k_pools_by_the_plus50_model_and_holds_while_rising(setup):
    import random
    from tradingbot import model_book as mb
    cfg, st, hel, clock, live, p, create = setup
    assert ll.big_flags(60_000, 75.0)["big30"] and not ll.big_flags(60_000, 65.0)["big30"]
    assert not ll.big_flags(30_000, 99.0)["big30"] and not ll.big_flags(90_000, 99.0)["big30"]
    assert not ll.big_flags(None, 99.0)["big30"] and not ll.big_flags(60_000, None)["big30"]
    assert ll.train_big(st, cfg, now=clock.t) is None  # no rows yet
    _rw_done_rows(st, 600, clock.t - 2 * 86400, random.Random(9))
    st.execute("UPDATE watch_launch_live SET liq_usd=60000")
    cfg.launch_big_book = True
    try:
        assert ll.train_big_and_save(st, cfg) is not None
        model = live._load_big_model()
        assert model.meta["band_rows"] == 600 and model.meta["band_quantiles"]
        win_like = {"creator_buys": 0.5, "log_buyers": 2.5}
        flat_like = {"creator_buys": 0.5, "log_buyers": 0.5}
        assert model.score(win_like) > model.score(flat_like)
        with live.lock:  # the band's past day: every pool between the two
            live.big_band.clear()
            mid = (model.score(win_like) + model.score(flat_like)) / 2
            live.big_band.extend((clock.t - 3600, mid) for _ in range(ll.ROLLING_MIN))
        assert live._big_verdict(win_like, "N", clock.t, clock.t, 60_000)["big_pct"] == 100.0
        assert live._big_verdict(flat_like, "N", clock.t, clock.t, 60_000)["big_pct"] == 0.0
        assert live._big_verdict(win_like, "N", clock.t, clock.t, 30_000) == {}  # other bands: not scored
        cfg.launch_live = True
        assert mb.LAUNCH_BIG in mb.active_books(cfg)
        sig = {"big30": True, "mint": "M", "created_ts": 100.0}
        assert mb.wants(mb.LAUNCH_BIG, sig) and not mb.wants(mb.LAUNCH_BIG, dict(sig, big30=False))
        assert mb.exit_deadline(cfg, mb.LAUNCH_BIG, sig) == 100.0 + 900
        assert mb.hold_checks(mb.LAUNCH_BIG, sig) == [100.0 + 360, 100.0 + 600]
        assert mb.hold_checks(mb.LAUNCH_HOLD15, sig) is None
        assert mb.band_usd(mb.LAUNCH_BIG, 60_000) == 250.0
        assert not mb.hold_check_fails(0.05, None) and mb.hold_check_fails(-0.01, None)
        assert mb.hold_check_fails(0.10, 0.20) and not mb.hold_check_fails(0.25, 0.20)
        from tradingbot import dashboard
        assert dashboard.family_of(mb.LAUNCH_BIG) == "Rug-vs-win"
    finally:
        cfg.launch_big_book = cfg.launch_live = False


class _QuoteJupiter:
    """Sell quotes from a list (price per token), one per check."""

    def __init__(self, quotes):
        self.quotes = list(quotes)

    def prices(self, mints):
        return {}

    def quote_price(self, mint, usd=20.0):
        return None

    def sell_price(self, mint, qty):
        return self.quotes.pop(0) if self.quotes else None


@pytest.mark.parametrize("quotes,sold_at,reason", [
    ([1.2e-5, 1.3e-5], 900, "pool_age_exit"),  # up and rising at 6 and 10: held to 15
    ([0.9e-5], 360, "hold_check"),            # down at 6: out at once
    ([1.3e-5, 1.1e-5], 600, "hold_check"),    # still up at 10 but lower than at 6: out at 10
])
def test_engine_runs_the_conditional_hold_checks(monkeypatch, quotes, sold_at, reason):
    from types import SimpleNamespace
    from tradingbot import engine as eng_mod
    from tradingbot import model_book as mb
    from tradingbot.engine import Engine
    from tradingbot.execution.paper import PaperExecutor
    from tradingbot.journal import Journal
    from tests.test_model_book import FakeDex
    cfg = Config(db_path=":memory:", launch_big_book=True, launch_live=True)
    j = Journal(":memory:")
    paper = PaperExecutor(cfg)
    jup = _QuoteJupiter(quotes)
    eng = Engine(cfg, FakeDex(None), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j,
                 jupiter=jup, model_executor=paper)
    book = eng.books[mb.LAUNCH_BIG]
    created = 1_000_000.0
    clock = [created + 10]
    monkeypatch.setattr(eng_mod.time, "time", lambda: clock[0])
    fill = paper.buy("BIGMINT", 250.0, 1e-5)
    pos = eng._open_position(book, "BIGMINT", "BIG", fill, 1e-5, None)
    eng._deadlines[pos.id] = created + 900
    eng._holds[pos.id] = {"at": mb.hold_checks(mb.LAUNCH_BIG, {"created_ts": created}), "last": None}
    eng.current_prices = lambda tokens, held=None: {}
    sold = None
    while clock[0] < created + 1000 and sold is None:
        eng.manage_positions()
        if not j.open_positions(book.name):
            sold = clock[0]
        clock[0] += 10
    assert sold is not None and created + sold_at <= sold < created + sold_at + 10
    [(r,)] = j.query("SELECT reason FROM fills WHERE side='sell'")
    assert r == reason
