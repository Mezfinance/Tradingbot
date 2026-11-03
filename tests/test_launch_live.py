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
