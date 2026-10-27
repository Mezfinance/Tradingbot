import json
import random
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
    assert eng.buy_model_signals() == 1
    [pos] = j.open_positions(mb.NAME)
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


def test_the_report_shows_only_the_model5_book_with_its_rugs(cfg, good_pair):
    from tradingbot.report import build_model, render_model
    eng, j, pos = _held_position(cfg, good_pair, RuggedJupiter())
    eng.extras = FakeExtras(liquidity=40_000.0)
    eng._log_model_audit(eng.books[mb.NAME], {"mint": "RUGMINT", "score": 1.2, "liq": 30_000.0})
    eng.extras.liquidity = 1_000.0
    eng.manage_positions()  # a rug: sold through the stop for ~0
    rep = build_model(j, cfg)
    assert rep["book"]["strategy"] == mb.NAME and rep["rugs"]["rugs"] == 1
    text = render_model(rep)
    assert "MODEL5 PAPER BOOK" in text and "liquidity_pulled" in text and "1 of 1 finished trades" in text
    assert "filters_only" not in text and "Claude API" not in text and "not trained yet" in text


def test_short_hold_exits_after_ten_minutes_and_old_pools_are_skipped(cfg, good_pair):
    """10-28: model5 holds 10 minutes (most rugs come after the 15-minute mark) and only buys pools young
    enough for the tested window."""
    import time as real_time
    book_cfg = mb.exit_config(cfg)
    assert book_cfg.max_hold_hours * 60 == 10
    eng, j, pos = _held_position(cfg, good_pair, None)
    eng.current_prices = lambda tokens, held=None: {"RUGMINT": pos.entry_price * 1.2}
    assert risk.plan_exits(pos, pos.entry_price * 1.2, pos.opened_at + 9 * 60, book_cfg) == []
    assert risk.plan_exits(pos, pos.entry_price * 1.2, pos.opened_at + 10 * 60 + 1, book_cfg)[0][1] == "time_stop"
    sig = {"mint": "NEW", "name": "N / SOL", "score": 1.0, "price": 1.0, "liq": 9000.0, "ts": real_time.time()}
    book = eng.books[mb.NAME]
    assert eng._model_skip_reason(book, dict(sig, created_ts=real_time.time() - 6 * 60)) is None
    assert eng._model_skip_reason(book, dict(sig, created_ts=real_time.time() - 11 * 60)) == "pool too old for the short hold"
