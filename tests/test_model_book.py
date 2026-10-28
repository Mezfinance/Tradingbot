import json
import random
import time
from types import SimpleNamespace

from tradingbot import model_book as mb
from tradingbot import risk
from tradingbot.config import Config
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore
from tests.test_analyst_engine import FakeDex
from tests.test_watch_analysis import T0, add_pool, candle_path, snap

LAST = T0 + 600 * 900


def store_with_signal(n=600, seed=2):
    """Pools every 15 min; busy ones (many buyers at 5 min) mostly double, quiet ones mostly fade."""
    rng = random.Random(seed)
    store = WatchStore(":memory:")
    for i in range(n):
        created = T0 + i * 900
        busy = rng.random() < 0.3
        pumps = rng.random() < (0.6 if busy else 0.03)
        path = [1.3, 2.4, 1.9] if pumps else [0.9, 0.7, 0.6]
        add_pool(store, f"P{i}", f"M{i}", created, dex="pumpswap",
                 snaps={5: snap(1.0, buyers=rng.randint(60, 120) if busy else rng.randint(2, 15), sellers=10)},
                 candles=candle_path(created, 5, path))
    return store


def test_train_save_load_and_score(tmp_path):
    cfg = Config(db_path=str(tmp_path / "bot.db"))
    store = store_with_signal()
    model = mb.train(store, cfg, now=LAST + 2 * 86400)
    assert model is not None and model.meta["rows"] >= 200
    assert model.meta["top_pump_rate_in_sample"] > model.meta["pump_rate"]
    path = tmp_path / "model5.json"
    path.write_text(json.dumps(model.to_json()))
    again = mb.Model.from_json(json.loads(path.read_text()))
    pool = {"pool": "X", "mint": "MX", "dex": "pumpswap", "name": "X / SOL", "created_ts": T0, "first": json.dumps(snap(1.0))}
    busy, quiet = snap(1.0, buyers=300, sellers=10), snap(1.0, buyers=3, sellers=10)
    assert again.score(mb.features(pool, busy)) > again.threshold > again.score(mb.features(pool, quiet))


def test_book_queues_only_top_scores_with_enough_liquidity(tmp_path):
    cfg = Config(db_path=str(tmp_path / "bot.db"))
    model = mb.train(store_with_signal(), cfg, now=LAST + 2 * 86400)
    mb.model_path(cfg).parent.mkdir(parents=True)
    mb.model_path(cfg).write_text(json.dumps(model.to_json()))
    book = mb.ModelBook(cfg)
    pool = {"pool": "X", "mint": "MX", "dex": "pumpswap", "name": "X / SOL", "created_ts": T0, "first": json.dumps(snap(1.0))}
    assert book.consider(pool, 15, snap(1.0, buyers=300)) is None  # only the 5-minute snapshot counts
    assert book.consider(pool, 5, snap(1.0, buyers=3)) is None  # below the cut-off
    assert book.consider(pool, 5, snap(1.0, buyers=300, liq=1000.0)) is None  # too thin to trade
    assert book.consider(pool, 5, snap(1.0, buyers=300, liq=20_000.0)) is not None
    sig = book.signals.get_nowait()
    assert sig["mint"] == "MX" and sig["liq"] == 20_000.0


def test_engine_buys_signals_with_the_books_own_exits(cfg, good_pair, tmp_path):
    cfg.db_path = str(tmp_path / "bot.db")
    model = mb.train(store_with_signal(), cfg, now=LAST + 2 * 86400)
    mb.model_path(cfg).parent.mkdir(parents=True)
    mb.model_path(cfg).write_text(json.dumps(model.to_json()))
    j = Journal(":memory:")
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j,
                 model_executor=PaperExecutor(cfg))
    book = eng.books[mb.NAME]
    assert book.cash == cfg.model_book_cash_usd and book.exit_cfg.take_profit_pcts == (100.0,)
    pool = {"pool": "X", "mint": "MX", "dex": "pumpswap", "name": "WIN / SOL", "created_ts": T0, "first": json.dumps(snap(1.0))}
    eng.model_book.consider(pool, 5, snap(1.0, buyers=300, liq=4_000.0))  # below the liquidity floor
    eng.model_book.consider(pool, 5, snap(1.0, buyers=300, liq=6_000.0))  # 2% of $6k = $120, capped at $100
    eng.model_book.consider(pool, 5, snap(1.0, buyers=300, liq=6_000.0))  # same token again: skipped
    assert eng.buy_model_signals() == 3  # one buy per book: same signal, separate ledgers
    [climb] = j.open_positions(mb.CLIMB)
    assert eng.books[mb.CLIMB].exit_cfg.max_hold_hours * 60 == 10 and climb.id in eng._climbs
    [pos] = j.open_positions(mb.NAME)
    [fast] = j.open_positions(mb.FAST)
    assert fast.token_address == pos.token_address and eng.books[mb.FAST].cash < cfg.model_book_cash_usd
    assert eng.books[mb.NAME].exit_cfg.max_hold_hours == 24 and eng.books[mb.FAST].exit_cfg.max_hold_hours * 60 == 10
    assert 95 <= pos.cost_usd <= 101 and pos.stop_price == pos.entry_price * 0.5
    assert j.query("SELECT detail FROM events WHERE kind='entry_skipped' AND strategy=?", (mb.NAME,))[0][0] == "already held"
    # +100% sells everything at once (the Claude book's ladder would sell only part at +50%).
    assert risk.plan_exits(pos, pos.entry_price * 1.6, pos.opened_at + 60, book.exit_cfg) == []
    assert risk.plan_exits(pos, pos.entry_price * 2.05, pos.opened_at + 60, book.exit_cfg) == [(pos.qty, "take_profit_1")]


def test_watcher_hands_five_minute_snapshots_to_the_listener():
    from tests.test_watcher import FakeGecko, make, pool
    g = FakeGecko({1: [pool("A", "MA")]}, live={"A": pool("A", "MA", vol24="900")})
    w, _ = make(g, watch_ages_min="5,15")
    seen = []
    w.on_snapshot = lambda p, age, d: seen.append((p["pool"], p["mint"], age, "price" in d))
    w.discover(T0)
    w.snapshot(T0 + 5 * 60 + 5)
    assert seen == [("A", "MA", 5, True)]


def test_a_position_with_no_route_out_backs_off_and_is_valued_at_zero(cfg, good_pair, monkeypatch):
    from tradingbot import engine as eng_mod
    from tradingbot.execution.base import failed
    from tradingbot.report import build, render
    j = Journal(":memory:")
    paper = PaperExecutor(cfg)
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j,
                 model_executor=paper)
    book = eng.books[mb.NAME]
    fill = paper.buy(good_pair.token_address, 100.0, good_pair.price_usd)
    pos = eng._open_position(book, good_pair.token_address, "RUG", fill, good_pair.price_usd, None)
    sells = []
    paper.sell = lambda token, qty, price: sells.append(1) or failed(0.0, "no route")
    crash = {good_pair.token_address: good_pair.price_usd * 0.3}  # through the -50% stop
    clock = [pos.opened_at + 60]
    monkeypatch.setattr(eng_mod.time, "time", lambda: clock[0])
    eng.current_prices = lambda tokens, held=None: crash
    for _ in range(6):  # a minute of 10-second monitor passes
        eng.manage_positions()
        clock[0] += 10
    assert len(sells) == 1  # one attempt, then a 2-minute back-off
    assert j.query("SELECT COUNT(*) FROM events WHERE kind='tx_failed'")[0][0] == 0
    assert j.query("SELECT COUNT(*) FROM events WHERE kind='exit_blocked'")[0][0] == 1
    assert eng.equity(book, crash) > book.cash  # not yet written off
    clock[0] += eng_mod.STUCK_AFTER_S
    assert eng.equity(book, crash) == book.cash  # can't be sold: worth nothing
    import time as real_time
    from tradingbot import report as report_mod
    later = real_time.time() + eng_mod.STUCK_AFTER_S + 60  # events carry wall-clock time
    monkeypatch.setattr(report_mod.time, "time", lambda: later)
    rep = build(j, cfg)
    m5 = next(s for s in rep["strategies"] if s["strategy"] == mb.NAME)
    assert m5["closed_trades"] == 1 and m5["stuck_trades"] == 1 and m5["win_rate"] == 0
    assert m5["mean_trade_return_pct"] == -100.0 and m5["exit_reasons"] == {"stuck": 1}
    text = render(rep)
    assert "EXIT BLOCKED" in text and "of which stuck" in text


class RuggedJupiter:
    """No feed lists the token; a $20 buy quote on the drained pool implies a huge price, selling returns ~0."""

    def __init__(self, sell=1e-9):
        self.sell = sell
        self.sell_checks = 0

    def prices(self, mints):
        return {}

    def quote_price(self, mint, usd=20.0):
        return 0.5227

    def sell_price(self, mint, qty):
        self.sell_checks += 1
        return self.sell


def _held_position(cfg, good_pair, jupiter):
    j = Journal(":memory:")
    paper = PaperExecutor(cfg)
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j,
                 jupiter=jupiter, model_executor=paper)
    book = eng.books[mb.NAME]
    fill = paper.buy("RUGMINT", 100.0, 2.5e-5)
    pos = eng._open_position(book, "RUGMINT", "**", fill, 2.5e-5, None)
    return eng, j, pos


def test_a_token_no_feed_lists_is_priced_by_selling_it_not_by_a_buy_quote(cfg, good_pair):
    eng, j, pos = _held_position(cfg, good_pair, RuggedJupiter())
    eng.manage_positions()
    [(reason,)] = j.query("SELECT reason FROM fills WHERE side='sell'")
    assert reason == "stop_loss"  # was take_profit_1 at the buy-implied 0.5227 (21,000x entry)


def test_a_take_profit_needs_a_sell_quote_that_clears_the_target(cfg, good_pair):
    jup = RuggedJupiter()
    eng, j, pos = _held_position(cfg, good_pair, jup)
    eng.current_prices = lambda tokens, held=None: {"RUGMINT": 0.5227}  # bogus feed price
    eng.manage_positions()
    assert jup.sell_checks == 1
    assert [r for (r,) in j.query("SELECT reason FROM fills WHERE side='sell'")] == ["stop_loss"]
    # A real pump: the sell quote agrees, so the take-profit goes through.
    jup2 = RuggedJupiter(sell=2.5e-5 * 2.2)
    eng2, j2, _ = _held_position(cfg, good_pair, jup2)
    eng2.current_prices = lambda tokens, held=None: {"RUGMINT": 2.5e-5 * 2.3}
    eng2.manage_positions()
    assert [r for (r,) in j2.query("SELECT reason FROM fills WHERE side='sell'")] == ["take_profit_1"]


class FakeExtras:
    def __init__(self, liquidity):
        self.liquidity = liquidity

    def jupiter_token(self, mint):
        return {"holderCount": 420, "liquidity": self.liquidity, "organicScore": 12.0,
                "audit": {"topHoldersPercentage": 31.0, "devMints": 2}}


def test_rug_logging_records_the_entry_audit_and_why_a_trade_lost(cfg, good_pair):
    eng, j, pos = _held_position(cfg, good_pair, RuggedJupiter())
    eng.extras = FakeExtras(liquidity=40_000.0)
    eng._log_model_audit(eng.books[mb.NAME], {"mint": "RUGMINT", "score": 1.2, "liq": 30_000.0})
    [audit] = [json.loads(r["detail"]) for r in j.query("SELECT detail FROM events WHERE kind='model_entry_audit'")]
    assert audit["holders"] == 420 and audit["top_holders_pct"] == 31.0 and audit["snapshot_liq"] == 30_000.0
    eng.extras.liquidity = 1_000.0  # pulled: under 25% of the 40k at entry
    eng.manage_positions()  # sells at ~0 through the stop
    [cause] = [json.loads(r["detail"]) for r in j.query("SELECT detail FROM events WHERE kind='model_exit_cause'")]
    assert cause["cause"] == "liquidity_pulled" and cause["return_pct"] <= -80
    s = mb.rug_summary(j)
    assert s["finished_trades"] == 1 and s["rugs"] == 1 and s["rug_rate_pct"] == 100.0
    assert s["losing_exit_causes"] == {"liquidity_pulled": 1}
    assert s["entry_audit_rugs"]["median"]["holders"] == 420


def test_exit_cause_classes():
    ex = FakeExtras(liquidity=50_000.0)
    assert mb.exit_cause(ex, "M", 50_000.0, -0.9, blocked=True)["cause"] == "no_route"
    assert mb.exit_cause(ex, "M", 50_000.0, -0.9, blocked=False)["cause"] == "dumped"
    assert mb.exit_cause(ex, "M", 50_000.0, -0.55, blocked=False)["cause"] == "stopped"
    assert mb.exit_cause(None, "M", None, -0.9, blocked=False)["cause"] == "dumped"  # no audit source: price only


def test_a_blocked_exit_records_its_cause_once(cfg, good_pair, monkeypatch):
    from tradingbot.execution.base import failed
    eng, j, pos = _held_position(cfg, good_pair, None)
    eng.extras = FakeExtras(liquidity=None)
    eng.books[mb.NAME].executor.sell = lambda token, qty, price: failed(0.0, "no route")
    eng.current_prices = lambda tokens, held=None: {"RUGMINT": 1e-6}
    eng.manage_positions()
    eng._blocked.clear()  # as after a restart
    eng.manage_positions()
    causes = [json.loads(r["detail"])["cause"] for r in j.query("SELECT detail FROM events WHERE kind='model_exit_cause'")]
    assert causes == ["no_route"]


def test_an_emptied_pool_s_frozen_price_is_not_believed(cfg, good_pair):
    """Live 10-27: three rugged positions kept DexScreener's last trade price (liquidity $0, no route out),
    so the book showed ~$428 of value nobody could realise and their stops never fired."""
    from dataclasses import replace as dc_replace
    from tradingbot.execution.base import failed
    jup = RuggedJupiter(sell=None)  # no route
    eng, j, pos = _held_position(cfg, good_pair, jup)
    frozen = dc_replace(good_pair, token_address="RUGMINT", price_usd=6e-5, liquidity_usd=0.0)
    eng.dex = FakeDex(frozen)
    assert eng.current_prices({"RUGMINT"}, {"RUGMINT": pos.qty}) == {"RUGMINT": 0.0}
    eng.books[mb.NAME].executor.sell = lambda token, qty, price: failed(0.0, "no route")
    eng.manage_positions()
    assert j.query("SELECT COUNT(*) FROM events WHERE kind='exit_blocked'")[0][0] == 1  # the stop fired
    # A pool that still has liquidity keeps its feed price.
    eng.dex = FakeDex(dc_replace(frozen, liquidity_usd=50_000.0))
    assert eng.current_prices({"RUGMINT"}, {"RUGMINT": pos.qty}) == {"RUGMINT": 6e-5}


def test_a_rate_limited_exit_is_retried_not_marked_blocked(cfg, good_pair):
    """A 429 from Jupiter used to look like 'no route': the exit was marked blocked and an emptied pool
    priced at 0. Now a refused quote only postpones the sell to the next monitor pass."""
    from dataclasses import replace as dc_replace
    from tradingbot.data.jupiter import RATE_LIMITED
    from tradingbot.execution.base import failed

    class LimitedJupiter(RuggedJupiter):
        def last_failure(self):
            return RATE_LIMITED

    eng, j, pos = _held_position(cfg, good_pair, LimitedJupiter(sell=None))
    eng.dex = FakeDex(dc_replace(good_pair, token_address="RUGMINT", price_usd=6e-5, liquidity_usd=0.0))
    assert eng.current_prices({"RUGMINT"}, {"RUGMINT": pos.qty}) == {}  # unknown, not worth $0
    eng.current_prices = lambda tokens, held=None: {"RUGMINT": 1e-5}  # below the stop
    eng.books[mb.NAME].executor.sell = lambda token, qty, price: failed(0.0, RATE_LIMITED)
    eng.manage_positions()
    assert pos.id not in eng._blocked
    assert j.query("SELECT COUNT(*) FROM events WHERE kind IN ('exit_blocked', 'tx_failed')")[0][0] == 0


def test_the_report_shows_only_the_model5_book_with_its_rugs(cfg, good_pair):
    from tradingbot.report import build_model, render_model
    eng, j, pos = _held_position(cfg, good_pair, RuggedJupiter())
    eng.extras = FakeExtras(liquidity=40_000.0)
    eng._log_model_audit(eng.books[mb.NAME], {"mint": "RUGMINT", "score": 1.2, "liq": 30_000.0})
    eng.extras.liquidity = 1_000.0
    eng.manage_positions()  # a rug: sold through the stop for ~0
    rep = build_model(j, cfg)
    [m5, fast, climb] = rep["books"]
    assert m5["name"] == mb.NAME and m5["rugs"]["rugs"] == 1 and fast["name"] == mb.FAST and fast["buys"] == 0
    assert m5["started"] and fast["started"]  # both books had equity snapshots from the running engine
    text = render_model(rep)
    assert "running since (UTC)" in text and "running for" in text
    assert "MODEL PAPER BOOKS" in text and "model5_10m" in text and "liquidity_pulled" in text and "1 (100%)" in text
    assert climb["name"] == mb.CLIMB and "model5_climb" in text and "out at 3 min unless +5%" in text
    assert "filters_only" not in text and "Claude API" not in text and "not trained yet" in text


def test_short_hold_exits_after_ten_minutes_and_old_pools_are_skipped(cfg, good_pair):
    """10-28: model5 holds 10 minutes (most rugs come after the 15-minute mark) and only buys pools young
    enough for the tested window."""
    import time as real_time
    book_cfg = mb.exit_config(cfg, mb.FAST)
    assert book_cfg.max_hold_hours * 60 == 10 and mb.exit_config(cfg).max_hold_hours == 24
    eng, j, pos = _held_position(cfg, good_pair, None)
    eng.current_prices = lambda tokens, held=None: {"RUGMINT": pos.entry_price * 1.2}
    assert risk.plan_exits(pos, pos.entry_price * 1.2, pos.opened_at + 9 * 60, book_cfg) == []
    assert risk.plan_exits(pos, pos.entry_price * 1.2, pos.opened_at + 10 * 60 + 1, book_cfg)[0][1] == "time_stop"
    sig = {"mint": "NEW", "name": "N / SOL", "score": 1.0, "price": 1.0, "liq": 9000.0, "ts": real_time.time()}
    fast = eng.books[mb.FAST]
    assert eng._model_skip_reason(fast, dict(sig, created_ts=real_time.time() - 6 * 60)) is None
    assert eng._model_skip_reason(fast, dict(sig, created_ts=real_time.time() - 11 * 60)) == "pool too old for the short hold"
    assert eng._model_skip_reason(eng.books[mb.NAME], dict(sig, created_ts=real_time.time() - 11 * 60)) is None



def test_an_absurd_live_price_is_reported_as_no_liquidity_not_a_price_run(cfg, good_pair):
    """10-28: buy quotes on empty pools implied 268,000x and 122,000,000x the snapshot and were logged as
    'price ran +26860426%'."""
    import time as real_time

    class EmptyPoolJupiter(RuggedJupiter):
        def __init__(self, implied):
            super().__init__()
            self.implied = implied

        def quote_price(self, mint, usd=20.0):
            return self.implied

    sig = {"mint": "NEW", "name": "N / SOL", "score": 1.0, "price": 1e-5, "liq": 9000.0, "ts": real_time.time()}
    eng, j, _ = _held_position(cfg, good_pair, EmptyPoolJupiter(2.7))  # 270,000x the snapshot
    assert eng._buy_model(eng.books[mb.FAST], sig).startswith("no real liquidity (live price 270,000x")
    eng.jupiter = EmptyPoolJupiter(1.3e-5)  # +30%: a genuine run-up
    assert eng._buy_model(eng.books[mb.FAST], sig) == "price ran +30% since the snapshot"


def test_hand_made_pumpswap_pools_get_a_pool_age_exit_in_the_10_minute_book(cfg, good_pair):
    sig = {"mint": "Scam111", "dex": "pumpswap", "created_ts": 1000.0, "name": "SCAM / SOL",
           "price": 2.5e-5, "liq": 30_000.0, "score": 1.0}
    assert mb.is_handmade("pumpswap", "Scam111") and not mb.is_handmade("pumpswap", "Abcpump")
    assert not mb.is_handmade("meteora-damm-v2", "Scam111")
    assert mb.exit_deadline(cfg, mb.FAST, sig) == 1000.0 + 15 * 60
    assert mb.exit_deadline(cfg, mb.NAME, sig) is None  # the 24h book keeps its own rules
    assert mb.exit_deadline(cfg, mb.FAST, {**sig, "mint": "Abcpump"}) is None  # pump.fun graduation
    cfg.model_book_handmade_exit_age_min = 0
    assert mb.exit_deadline(cfg, mb.FAST, sig) is None
    cfg.model_book_handmade_exit_age_min = 15


def test_the_pool_age_exit_sells_and_survives_a_restart(cfg, good_pair):
    j = Journal(":memory:")
    paper = PaperExecutor(cfg)
    make = lambda: Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg),
                          j, model_executor=paper)
    eng = make()
    created = time.time() - 14 * 60  # pool is 14 min old: deadline in 1 min
    sig = {"mint": "Scam111", "dex": "pumpswap", "created_ts": created, "name": "SCAM / SOL",
           "price": 2.5e-5, "liq": 30_000.0, "score": 1.0}
    assert eng._buy_model(eng.books[mb.FAST], sig) is None
    [pos] = j.open_positions(mb.FAST)
    eng = make()  # restart: the deadline comes back from the journal
    assert eng._deadlines == {pos.id: created + 15 * 60}
    eng.current_prices = lambda tokens, held=None: {"Scam111": 2.6e-5}
    eng.manage_positions()
    assert [p.id for p in j.open_positions(mb.FAST)] == [pos.id]  # 14 min old, inside the 10-min hold: still held
    eng._deadlines[pos.id] = time.time() - 1
    eng.manage_positions()
    assert j.open_positions(mb.FAST) == []
    assert [r for (r,) in j.query("SELECT reason FROM fills WHERE side='sell'")] == ["pool_age_exit"]


def test_the_climb_book_sells_positions_that_have_not_climbed(cfg, good_pair):
    j = Journal(":memory:")
    make = lambda: Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg),
                          j, model_executor=PaperExecutor(cfg))
    eng = make()
    assert mb.active_books(cfg) == [mb.NAME, mb.FAST, mb.CLIMB] and mb.CLIMB in eng.books
    assert mb.climb_check(cfg, mb.FAST, 0.0, 1.0) is None
    sig = lambda mint: {"mint": mint, "dex": "pumpswap", "created_ts": time.time() - 6 * 60, "name": mint,
                        "price": 1e-5, "liq": 30_000.0, "score": 1.0}
    book = eng.books[mb.CLIMB]
    assert eng._buy_model(book, sig("FLAT")) is None and eng._buy_model(book, sig("UP")) is None
    flat, up = sorted(j.open_positions(mb.CLIMB), key=lambda p: p.token_address)
    assert eng._climbs[up.id][1] == 1e-5 * 1.05  # +5% over the market price at the buy, not over the fill
    prices = {"FLAT": 1.02e-5, "UP": 1.06e-5}
    eng.current_prices = lambda tokens, held=None: {t: prices[t] for t in tokens}
    eng.manage_positions()  # before the 3-minute check: both held
    assert {p.token_address for p in j.open_positions(mb.CLIMB)} == {"FLAT", "UP"}
    eng = make()  # restart: checks come back from the journal
    assert set(eng._climbs) == {flat.id, up.id}
    eng.current_prices = lambda tokens, held=None: {t: prices[t] for t in tokens}
    for c in eng._climbs.values():
        c[2] = 1.02e-5 if c is eng._climbs[flat.id] else 1.06e-5
        c[0] = time.time() - 1  # check is due
    prices["UP"] = 1.01e-5  # it climbed earlier, so a later dip doesn't fail it
    eng.manage_positions()
    assert [p.token_address for p in j.open_positions(mb.CLIMB)] == ["UP"]
    assert [r for (r,) in j.query("SELECT reason FROM fills WHERE side='sell'")] == ["no_climb"]
    assert up.id not in eng._climbs  # passed: no more checks
    cfg.model_book_climb = False
    assert mb.CLIMB not in make().books
    cfg.model_book_climb = True


def test_a_switched_off_book_is_left_alone(cfg, good_pair):
    eng, j, pos = _held_position(cfg, good_pair, RuggedJupiter())  # an open model5 position
    cfg.model_book_slow = False
    try:
        eng2 = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j,
                      jupiter=RuggedJupiter(), model_executor=PaperExecutor(cfg))
        assert mb.NAME not in eng2.books and mb.active_books(cfg) == [mb.FAST, mb.CLIMB]
        seen = []
        eng2.current_prices = lambda tokens, held=None: seen.append(set(tokens)) or {}
        eng2.manage_positions()
        assert seen == [set()]  # no price lookups (or sell quotes) for the switched-off book
        assert [p.id for p in j.open_positions(mb.NAME)] == [pos.id]  # left in the journal as it was
    finally:
        cfg.model_book_slow = True
