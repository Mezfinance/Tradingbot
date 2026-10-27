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
    eng.current_prices = lambda tokens: crash
    for _ in range(6):  # a minute of 10-second monitor passes
        eng.manage_positions()
        clock[0] += 10
    assert len(sells) == 1  # one attempt, then a 2-minute back-off
    assert j.query("SELECT COUNT(*) FROM events WHERE kind='tx_failed'")[0][0] == 0
    assert j.query("SELECT COUNT(*) FROM events WHERE kind='exit_blocked'")[0][0] == 1
    assert eng.equity(book, crash) > book.cash  # not yet written off
    clock[0] += eng_mod.STUCK_AFTER_S
    assert eng.equity(book, crash) == book.cash  # can't be sold: worth nothing
    text = render(build(j, cfg))
    assert "EXIT BLOCKED" in text
