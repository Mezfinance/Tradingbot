"""The trading engine.

Two loops share one journal:
  * monitor loop (every `monitor_interval_s`): price every open position and run exits;
  * scan loop (every `scan_interval_s`): discover, screen, research with Claude, enter.
Keeping them apart means Claude research (which can take minutes) never delays a stop.

Each strategy trades its own "book" (cash + positions) with the same risk rules.
The "claude" book is the real strategy; other books are baselines (see shadows.py).
"""

from __future__ import annotations

import copy
import json
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from tradingbot import factors, risk, shadows
from tradingbot.analyst import Analyst
from tradingbot.config import Config
from tradingbot.data.dexscreener import DexScreener
from tradingbot.data.jupiter import RATE_LIMITED, Jupiter
from tradingbot.data.rugcheck import RugCheck
from tradingbot.execution.base import Executor, Fill
from tradingbot.journal import Journal
from tradingbot.models import FactorResult, PairSnapshot, Position, SafetyReport
from tradingbot.tracker import Tracker
from tradingbot.backups import DailyBackup

log = logging.getLogger(__name__)

MAIN = "claude"
CANARY = "canary"
CANARY_HALT = "launch_canary:halted"  # state key: set = the real-money launch canary buys nothing (kill switch)
CANARY_ALLOW = "launch_canary:allow"  # state key: buys left before the canary waits (canary-one sets 1); absent = no limit
PROBE_PULLED_SHARE = 0.10  # the probe survival watch calls a pool rugged when its SOL falls below this share of probe-time
BOOK_PRIORITY = {MAIN: 0, CANARY: 1, "canary_rw50_sz": 1}  # real-money books are exited before simulated baselines
REANALYSE_AFTER_S = 6 * 3600
RECHECK_REJECTED_S = 30 * 60
DUST_QTY = 1e-12
EQUITY_SNAPSHOT_S = 60
EXIT_RETRY_S = 120  # a sell with no route is retried this often
SELL_REVERT_RETRIES = 2  # a reverted pool-minute exit is tried again at once this many times
STUCK_AFTER_S = 0  # a position with no route out is valued at zero from its first blocked sale (11-05, was 1800);
# a later sale that finds a route closes it and brings the money back
BUY_WORKERS = 16  # every book that wants a signal starts its buy at once (up to 17 books run)
SELL_WORKERS = 16  # exits due in the same monitor pass are sold in parallel
MODEL_MIN_USD = 5.0  # paper model books: no trade below this
WALLET_SWEEP_S = 3600  # how often the canary's wallet closes empty token accounts a sale's own close missed
CANARY_MIN_USD = 0.01  # the real-money canary may go down to a cent (Jupiter routes $0.001 USDC swaps)
NO_LIQUIDITY_MULT = 10.0  # model entry: a live price this many times the snapshot means an empty pool
DEAD_LIQ_USD = 100.0  # a DexScreener pool with less liquidity than this is treated as emptied (rugged)
QUOTE_PRICE_MAX = 10  # positions per monitor pass priced from a swap quote when no feed lists the token
MODEL_POLL_S = 0.5  # how often queued model signals are picked up (early books lose ~6-8 pts per minute of delay)
MODEL_SIGNAL_MAX_AGE_S = 180  # a signal older than this is stale (the edge is in the first minutes)

Survivor = tuple[PairSnapshot, FactorResult, SafetyReport | None]


@dataclass
class Book:
    name: str
    executor: Executor
    cash: float
    fixed_usd: float | None = None  # fixed trade size instead of risk-based sizing (canary)
    exit_cfg: Config | None = None  # the book's own stop / take-profit / hold rules (model5); None = cfg


class Engine:
    def __init__(self, cfg: Config, dex: DexScreener, rug: RugCheck, analyst: Analyst | None,
                 executor: Executor, journal: Journal, jupiter: Jupiter | None = None,
                 shadow_executors: dict[str, Executor] | None = None,
                 canary_executor: Executor | None = None, gecko=None, triage=None,
                 model_executor: Executor | None = None, extras=None,
                 launch_canary_executor: Executor | None = None):
        self.cfg, self.dex, self.rug, self.analyst = cfg, dex, rug, analyst
        self.extras = extras  # optional free-source client (data/extras.py): model5's holder audits
        self.triage = triage  # optional cheap first pass before the analyst (tradingbot/triage.py)
        self.gecko = gecko  # optional GeckoTerminal discovery (trending by trading activity)
        self.journal = journal
        self.jupiter = jupiter  # None: no round-trip preflight, prices from DexScreener (tests)
        self._reset_books()
        self._last_sweep, self._sweeper = 0.0, None
        self.books: dict[str, Book] = {MAIN: self._open_book(MAIN, executor)}
        for name, ex in (shadow_executors or {}).items():
            self.books[name] = self._open_book(name, ex)
        if canary_executor is not None and cfg.canary_every_n > 0:
            self.books[CANARY] = self._open_book(CANARY, canary_executor, cfg.canary_budget_usd, cfg.canary_usd)
        self.model_book = None
        self._pool_minute_books: set[str] = set()
        if model_executor is not None and cfg.model_book:
            from tradingbot import model_book as mb
            for name in mb.active_books(cfg):  # same signals, separate cash and ledgers
                if name in mb.CANARY_BOOKS:  # real money: only with the live executor, at its own size and budget
                    if launch_canary_executor is None:
                        continue
                    book = self._open_book(name, launch_canary_executor, cfg.launch_canary_budget_usd,
                                           cfg.launch_canary_usd)
                    book.exit_cfg = mb.exit_config(cfg, name)
                    self.books[name] = book
                    continue
                ex = model_executor if name == mb.NAME else copy.copy(model_executor)
                if name in mb.LAUNCH_CUT and hasattr(ex, "sell_slippage_bps"):
                    ex.sell_slippage_bps = cfg.launch_sell_slippage_bps  # a minute-6 exit must go through
                book = self._open_book(name, ex, cfg.model_book_cash_usd, cfg.model_book_usd)
                book.exit_cfg = mb.exit_config(cfg, name)
                self.books[name] = book
            self.model_book = mb.ModelBook(cfg)
            self._pool_minute_books = set(mb.POOL_MINUTE_EXIT)
        self._rng = random.Random(cfg.shadow_seed)
        self.lock = threading.RLock()  # guards book cash and position state
        self.tracker = Tracker(journal, dex, jupiter)
        self.backups = DailyBackup(journal, cfg)
        self._cooldown: dict[str, float] = journal.load_cooldowns()  # token -> time it may be re-screened
        self._last_prices: dict[str, float] = {}
        self._quote_cursor = 0  # rotates which unlisted tokens get a sell-quote price each pass
        self._blocked: dict[int, dict] = {}  # position id -> exit blocked (no route): since, next try, note
        self._deadlines = self._load_deadlines()  # position id -> sell by this time (pool-age exit)
        self._climbs = self._load_climbs()  # position id -> [check time, price to reach, best price seen]
        self._holds = self._load_holds()  # position id -> {"at": [check times left], "last": return at the last check}
        self._probe_watches = self._load_probe_watches()  # pool -> a 1-slot probe awaiting its survival read
        self._canary_climbs = self._load_canary_climbs()  # position id -> the two-stage entry's reserve-priced climb check
        self._canary_trails = self._load_canary_trails()  # position id -> the two-stage entry's reserve-priced trailing stop
        self._canary_helius = None  # lazy live Helius client for the entry volatility gate (main key)
        self._model_pool = None  # threads that let every book buy a signal at the same time
        self._sell_pool = None  # threads that sell every due exit of a monitor pass at once
        self._last_snapshot = 0.0
        self._stop = threading.Event()
        self.restart_reason: str | None = None
        if cfg.learning:
            from tradingbot.learning import versions
            versions.apply_active(journal, cfg)
        if cfg.learning and analyst is not None and hasattr(analyst, "playbook") and analyst.playbook is None:
            from tradingbot.learning.playbook import Playbook
            analyst.playbook = Playbook(journal, cfg)
        if analyst is not None and hasattr(analyst, "seed_history"):
            analyst.seed_history([r["ts"] for r in journal.query(
                "SELECT ts FROM costs WHERE kind='claude_api' AND ts > ?", (time.time() - 3600,))])
        self._record_downtime()

    def _record_downtime(self) -> None:
        """If the bot was off, record for how long and how many positions went unmanaged."""
        last = self.journal.get_state("heartbeat")
        if last is None:
            return
        gap = time.time() - float(last)
        if gap > max(60.0, 3 * self.cfg.monitor_interval_s):
            n_open = len(self.journal.open_positions(strategy=None))
            self.journal.execute("INSERT INTO downtime(start_ts, end_ts, open_positions) VALUES(?,?,?)",
                                 (float(last), time.time(), n_open))
            log.warning("bot was offline for %.0f min with %d open positions unmanaged", gap / 60, n_open)

    def _cool(self, token: str) -> None:
        until = time.time() + RECHECK_REJECTED_S
        self._cooldown[token] = until
        self.journal.set_cooldown(token, until)

    def _reset_books(self) -> None:
        """One-off fresh start for the books in `reset_books` (comma list), keyed by `reset_books_id` so a
        later restart doesn't repeat it; runs before any book reads its cash."""
        names = [n.strip() for n in (self.cfg.reset_books or "").split(",") if n.strip()]
        tag = (self.cfg.reset_books_id or "").strip()
        if not names or not tag or self.journal.get_state(f"books_reset:{tag}"):
            return
        for name in names:
            moved = self.journal.archive_strategy(name, tag)
            log.info("book %s reset (%d positions kept as %s@%s)", name, moved, name, tag)
        self.journal.set_state(f"books_reset:{tag}", ",".join(names))

    def _open_book(self, name: str, executor: Executor, start_cash: float | None = None,
                   fixed_usd: float | None = None) -> Book:
        start = self.cfg.starting_cash_usd if start_cash is None else start_cash
        return Book(name, executor, self.journal.get_cash(start, name), fixed_usd)

    # Convenience accessors for the main book.
    @property
    def executor(self) -> Executor:
        return self.books[MAIN].executor

    @executor.setter
    def executor(self, ex: Executor) -> None:
        self.books[MAIN].executor = ex

    @property
    def cash(self) -> float:
        return self.books[MAIN].cash

    # --- loops ----------------------------------------------------------------
    def tick(self) -> None:
        """One synchronous pass of everything (used by `once` and by tests)."""
        self.manage_positions()
        self.scan_and_enter()
        self.update_outcomes()

    def run_forever(self) -> None:
        from tradingbot import health

        monitor = threading.Thread(target=self._loop, name="monitor", daemon=True,
                                   args=(self.cfg.monitor_interval_s, self.manage_positions))
        monitor.start()
        health.start_watchdog(self.journal, self.cfg, self._stop)
        if self.cfg.learning and self.analyst is not None:
            from tradingbot.learning.learner import Learner
            learner = Learner(self.journal, self.cfg, restart=self.request_restart)
            threading.Thread(target=self._loop, name="learner", daemon=True,
                             args=(self.cfg.learning_interval_s, learner.step)).start()
        if self.cfg.watch and self.cfg.db_path != ":memory:":
            from tradingbot.watcher import Watcher  # research only: records new pools, never trades
            watcher = Watcher.from_config(self.cfg)
            if self.model_book is not None:
                watcher.on_snapshot = self.model_book.consider
                watcher.on_discover = self.model_book.consider_first
                self.model_book.store = watcher.store
                import httpx
                from tradingbot.data.helius import Helius
                self.model_book.helius = Helius(client=httpx.Client(timeout=5))  # runs on the watcher thread
                threading.Thread(target=self._loop, name="model", daemon=True,
                                 args=(MODEL_POLL_S, self.buy_model_signals)).start()
            threading.Thread(target=self._loop, name="watcher", daemon=True,
                             args=(self.cfg.watch_poll_s, watcher.step)).start()
            if self.cfg.launch_live:  # second-4 launch scorer; its picks feed the launch books
                import httpx
                from tradingbot.data.helius import Helius
                from tradingbot.launch_live import LaunchLive
                signals = self.model_book.signals if self.model_book is not None and self.cfg.launch_book else None
                LaunchLive(self.cfg, watcher.store, Helius(client=httpx.Client(timeout=10), role="live"),
                           signals=signals).start()  # the scorer runs on the live (main) key
            if self.cfg.enrich:
                from tradingbot.enrich import Enricher
                enricher = Enricher.from_config(self.cfg, watcher.store, watcher)
                threading.Thread(target=self._loop, name="enricher", daemon=True,
                                 args=(self.cfg.enrich_poll_s, enricher.step)).start()
            if self.cfg.market_monitor and self.cfg.launch_live:  # hourly meme-market measures, read only
                from tradingbot import market_monitor as mm
                from tradingbot.watcher import WatchStore, watch_db_path
                monitor = mm.MarketMonitor(WatchStore(watch_db_path(self.cfg)), self.journal, self.cfg)
                threading.Thread(target=self._loop, name="market-monitor", daemon=True,
                                 args=(mm.MONITOR_POLL_S, monitor.step)).start()
        # Own thread: sampling after each scan cycle left ~18-minute gaps while Claude was busy.
        threading.Thread(target=self._loop, name="tracker", daemon=True,
                         args=(self.cfg.track_interval_s, self.update_outcomes)).start()
        if self.cfg.health_port > 0:
            from tradingbot import dashboard as dash
            board = None
            if dash.token() and self.cfg.db_path != ":memory:":
                store = None
                if self.cfg.launch_live:
                    from tradingbot.watcher import WatchStore, watch_db_path
                    store = WatchStore(watch_db_path(self.cfg))  # its own connection for the dashboard thread
                board = dash.Dashboard(self.journal, self.cfg, store).start()
            health.start_health_server(self.journal, self.cfg, board)
        try:
            self._loop(self.cfg.scan_interval_s, self._scan_cycle)
        finally:
            self._stop.set()

    def _scan_cycle(self) -> None:
        if self.analyst is not None or any(n in self.books for n in shadows.KNOWN):
            self.scan_and_enter()  # nothing uses the screened candidates when Claude and the baselines are off
        self.backups.run_if_due()
        self.sweep_canary_wallet_if_due()
        self.journal.set_state("scan_heartbeat", repr(time.time()))

    def sweep_canary_wallet_if_due(self, wait: bool = False) -> None:
        """Every WALLET_SWEEP_S, close the real-money wallet's empty token accounts (rent ~0.0015 SOL each) that a
        sale's own close missed, on a thread of its own (a close can take a minute to confirm)."""
        from tradingbot import model_book as mb
        books = [b for n, b in self.books.items() if n in mb.CANARY_BOOKS]
        ex = books[0].executor if books else None
        if ex is None or not hasattr(ex, "sweep") or time.time() - self._last_sweep < WALLET_SWEEP_S:
            return
        if self._sweeper is not None and self._sweeper.is_alive():
            return
        self._last_sweep = time.time()
        keep = {p.token_address for b in books for p in self.journal.open_positions(b.name)}

        def run():
            try:
                res = ex.sweep(keep)
                if res.get("closed") or res.get("dust_sold"):
                    log.info("wallet sweep: closed %d empty accounts ($%.2f back), sold %d dust ($%.2f)",
                             res.get("closed", 0), res.get("usd_back", 0.0),
                             res.get("dust_sold", 0), res.get("dust_usd", 0.0))
                    self.journal.log_event("wallet_sweep", None, json.dumps(res, default=str)[:2000],
                                           strategy=books[0].name)
            except Exception:  # noqa: BLE001 (the sweep must never disturb trading)
                log.warning("wallet sweep failed", exc_info=True)

        self._sweeper = threading.Thread(target=run, name="wallet-sweep", daemon=True)
        self._sweeper.start()
        if wait:
            self._sweeper.join()

    def _loop(self, interval: float, fn) -> None:
        while not self._stop.is_set():
            started = time.time()
            try:
                fn()
            except Exception as exc:  # keep the loop alive; state is persisted
                name = getattr(fn, "__name__", str(fn))
                log.exception("%s failed", name)
                try:  # counted during code probation (learning/engineer.py)
                    self.journal.log_event("loop_error", None, f"{name}: {type(exc).__name__}: {exc}"[:500])
                except Exception:  # noqa: BLE001 (never let error logging break the loop)
                    pass
            self._stop.wait(max(0.0, interval - (time.time() - started)))

    def request_restart(self, reason: str) -> None:
        """Stop all loops so the process exits and the supervisor starts it again (new code)."""
        log.warning("restart requested: %s", reason)
        self.restart_reason = reason
        self._stop.set()

    # --- prices ---------------------------------------------------------------
    def current_prices(self, tokens: set[str], held: dict[str, float] | None = None) -> dict[str, float]:
        """`held` (token -> quantity) prices tokens neither feed lists from a quote to sell that quantity."""
        prices: dict[str, float] = {}
        if not tokens:
            return prices
        if self.jupiter is not None:
            prices = {m: p.usd for m, p in self.jupiter.prices(sorted(tokens)).items()}
        missing = tokens - prices.keys()
        dead: set[str] = set()
        if missing:
            for snap in self.dex.pairs(missing):
                if snap.liquidity_usd is not None and snap.liquidity_usd < DEAD_LIQ_USD:
                    dead.add(snap.token_address)  # an emptied pool's last trade price: nobody can sell at it
                else:
                    prices[snap.token_address] = snap.price_usd
        missing = tokens - prices.keys()
        if missing and self.jupiter is not None and hasattr(self.jupiter, "quote_price"):
            # Brand-new (or rugged) tokens neither feed lists, QUOTE_PRICE_MAX per pass in turn: always taking the
            # first few alphabetically left later ones (early6t's Green Cat, ARMCHIP) unpriced for a day (11-06).
            order = sorted(missing)
            start = self._quote_cursor % len(order)
            self._quote_cursor += QUOTE_PRICE_MAX
            for mint in (order[start:] + order[:start])[:QUOTE_PRICE_MAX]:
                if held and mint in held and hasattr(self.jupiter, "sell_price"):
                    p = self.jupiter.sell_price(mint, held[mint])
                    limited = getattr(self.jupiter, "last_failure", lambda: "")() == RATE_LIMITED
                    if p is None and mint in dead and not limited:
                        p = 0.0  # no route out of an emptied pool: worth nothing (fires the stop)
                else:
                    p = self.jupiter.quote_price(mint)
                if p or (p == 0.0 and mint in dead):
                    prices[mint] = p
        self._last_prices.update(prices)
        return prices

    # --- positions ----------------------------------------------------------
    def manage_positions(self) -> dict[str, float]:
        # The real book is handled first so simulated latency on baselines never delays it.
        # Only books that are running: a switched-off book's positions stay in the journal as they are.
        positions = sorted((p for p in self.journal.open_positions(strategy=None) if p.strategy in self.books),
                           key=lambda p: BOOK_PRIORITY.get(p.strategy, 2))
        held: dict[str, float] = {}
        for p in positions:
            held[p.token_address] = max(held.get(p.token_address, 0.0), p.qty)
        prices = self.current_prices(set(held), held)
        now = time.time()
        self.journal.set_state("heartbeat", repr(now))
        self._run_probe_watches()  # record rugger/survivor for probes whose watch window is up (read only)
        from tradingbot import model_book as mb
        jobs: list[tuple] = []
        for pos in positions:
            price = prices.get(pos.token_address)
            book = self.books.get(pos.strategy)
            hold_only = pos.strategy in mb.LAUNCH_CUT
            if hold_only and not price:
                # A seconds-old token often has no feed price and no sell route yet, which used to read as $0
                # and fire the stop (a launch4_5 trade was sold at pool second 17). These books only ever sell
                # at their pool minute, so plan with the entry price; a real rug shows as no route at the exit.
                price = pos.entry_price
            if book is None:
                continue
            no_price = price is None
            if no_price:
                # No feed and no quote priced it this pass. It used to be skipped outright, so its deadline or hold
                # never sold it, it was never marked exit-blocked ($0) and equity kept it at cost: early6t held two
                # such trades for a day (11-06). Plan with the entry price for the time exits only; the sale itself
                # quotes for real and either fills or marks the exit blocked.
                price = pos.entry_price
            blocked = self._blocked.get(pos.id)
            if blocked and now < blocked["next"]:
                continue  # no route out last time: retry every EXIT_RETRY_S, not every pass
            before = (pos.tp_hits, pos.stop_price)
            exit_cfg = book.exit_cfg or self.cfg
            exits = risk.plan_exits(pos, price, now, exit_cfg)
            if hold_only or no_price:  # no stop on a made-up price (and none for launch books, as in their backtest)
                exits = [(q, r) for q, r in exits if r not in ("stop_loss", "breakeven_stop")
                         and not (no_price and r.startswith("take_profit"))]
            if not exits and now >= self._deadlines.get(pos.id, float("inf")):
                exits = [(pos.qty, "pool_age_exit")]  # hand-made pool: out before its creator usually pulls it
            if not exits and self._climb_failed(pos.id, price, now):
                exits = [(pos.qty, "no_climb")]  # flat after the first minutes: pulled pools mostly look like this
            if not exits and self._canary_climb_failed(book, pos, now):
                exits = [(pos.qty, "no_climb")]  # two-stage entry: not up by the check (reserve-priced), bail
            if not exits:
                reason = self._canary_hold_exit(book, pos, now)  # two-stage entry: trailing stop (fade) or stall (flat)
                if reason:
                    exits = [(pos.qty, reason)]
            if not exits and self._hold_failed(pos, now):
                exits = [(pos.qty, "hold_check")]  # not up (or falling) at a pool-minute check: out now
            if any(r.startswith("take_profit") for _, r in exits):
                # A feed price can be stale or bogus (a rugged pool once showed 21,000x entry); take profit
                # only if a quote to sell the position clears the target too.
                confirmed = self._confirm_sell_price(pos)
                if confirmed is not None and confirmed < price:
                    pos.tp_hits, pos.stop_price = before
                    log.info("[%s] %s take-profit not confirmed: feed %.10g, sell quote %.10g",
                             book.name, pos.symbol, price, confirmed)
                    price = confirmed
                    exits = risk.plan_exits(pos, price, now, exit_cfg)
            if exits:
                jobs.append((book, pos, exits, price, before))
            else:
                self.journal.save_position(pos)
        # 11-05: sells used to run one after another (~3.5 s each with the simulated landing), so books holding
        # the same token sold up to ~30 s apart and every other exit waited. Now they all go out at once.
        if len(jobs) > 1:
            if self._sell_pool is None:
                from concurrent.futures import ThreadPoolExecutor
                self._sell_pool = ThreadPoolExecutor(max_workers=SELL_WORKERS, thread_name_prefix="sell")
            list(self._sell_pool.map(lambda job: self._run_exits(*job), jobs))
        else:
            for job in jobs:
                self._run_exits(*job)
        self.snapshot_equity(prices)
        return prices

    def _run_exits(self, book: Book, pos: Position, exits: list, price: float, before: tuple) -> None:
        try:
            for qty, reason in exits:
                if not self._sell(book, pos, qty, price, reason):
                    # Roll the ladder back so the missed target is retried next pass.
                    pos.tp_hits, pos.stop_price = before
                    break
        finally:
            self.journal.save_position(pos)

    def _confirm_sell_price(self, pos: Position) -> float | None:
        if self.jupiter is None or not hasattr(self.jupiter, "sell_price"):
            return None
        try:
            return self.jupiter.sell_price(pos.token_address, pos.qty)
        except Exception:  # noqa: BLE001 (a failed check falls back to the feed price, as before)
            log.warning("sell-quote check failed for %s", pos.symbol, exc_info=True)
            return None

    def snapshot_equity(self, prices: dict[str, float], force: bool = False) -> None:
        if not force and time.time() - self._last_snapshot < EQUITY_SNAPSHOT_S:
            return
        self._last_snapshot = time.time()
        for book in self.books.values():
            held = sum(self._position_value(p, prices) for p in self.journal.open_positions(book.name))
            self.journal.snapshot_equity(book.name, book.cash, held)

    def _position_value(self, pos: Position, prices: dict[str, float]) -> float:
        """Market value; a position nobody can sell (no route for STUCK_AFTER_S) is worth nothing."""
        blocked = self._blocked.get(pos.id)
        if blocked and time.time() - blocked["since"] >= STUCK_AFTER_S:
            return 0.0
        from tradingbot import model_book as mb
        if pos.strategy in mb.LAUNCH_CUT:
            # A seconds-old token has no reliable feed price: no price at all, $0, or a Jupiter /price/v3 figure
            # up to ~10^7 below what the pool trades at (11-04: launch books showed every open trade as a full
            # loss in equity and drawdown). These books hold for minutes and sell at a pool minute whatever the
            # price, so an open trade counts at its cost until then; a real rug shows at that sale as no route,
            # and STUCK_AFTER_S zeroes it above.
            return pos.qty * pos.entry_price
        return pos.qty * prices.get(pos.token_address, pos.entry_price)

    def equity(self, book: Book | None = None, prices: dict[str, float] | None = None) -> float:
        book = book or self.books[MAIN]
        prices = self._last_prices if prices is None else prices
        held = sum(self._position_value(p, prices) for p in self.journal.open_positions(book.name))
        return book.cash + held

    def entries_allowed(self, book: Book | None = None) -> bool:
        book = book or self.books[MAIN]
        if len(self.journal.open_positions(book.name)) >= self.cfg.max_open_positions:
            return False
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        loss_today = -self.journal.realized_since(day_start, book.name)
        if loss_today >= self.equity(book) * self.cfg.daily_loss_limit_pct / 100:
            log.info("[%s] daily loss limit hit (%.2f USD); no new entries today", book.name, loss_today)
            return False
        return True

    # --- entries ------------------------------------------------------------
    def discover(self) -> tuple[dict[str, list[str]], set[str]]:
        """Token addresses from every enabled feed -> which feeds found them; plus paid-boosted set."""
        feeds = {f.strip() for f in self.cfg.discovery.split(",") if f.strip()}
        found: dict[str, list[str]] = {}
        boosted: set[str] = set()
        if "dexscreener" in feeds:
            dex_addrs, boosted = self.dex.discover()
            for a in dex_addrs:
                found.setdefault(a, []).append("dexscreener_listings")
        if "geckoterminal" in feeds and self.gecko is not None:
            for a in self.gecko.discover():
                found.setdefault(a, []).append("geckoterminal_trending")
        return found, boosted

    def screen(self) -> list[Survivor]:
        """Discover tokens and return those that clear every deterministic gate, best first."""
        addrs, boosted = self.discover()
        survivors: list[Survivor] = []
        now = time.time()
        for snap in self.dex.pairs(list(addrs), boosted):
            token = snap.token_address
            snap.sources = addrs.get(token, [])
            if self._cooldown.get(token, 0) > now:
                continue
            if (self.journal.recently_seen(token, REANALYSE_AFTER_S)
                    or self.journal.recently_seen(token, REANALYSE_AFTER_S, stage="triaged_out")):
                continue
            if factors.market_filters(snap, self.cfg):
                continue
            safety = self.rug.report(token)
            result = factors.evaluate(snap, safety, self.cfg)
            if not result.passed:
                self._cool(token)
                self.journal.log_decision(snap, result, safety=safety)
                log.info("screened out %s: %s", snap.symbol, "; ".join(result.rejections))
                continue
            survivors.append((snap, result, safety))
        survivors.sort(key=lambda t: t[1].score, reverse=True)
        return survivors

    def scan_and_enter(self) -> None:
        survivors = self.screen()
        analysed: list[tuple[Survivor, str]] = []  # (survivor, claude action)
        main = self.books[MAIN]
        if self.analyst is not None and self.entries_allowed(main):
            held = {p.token_address for p in self.journal.open_positions(MAIN)}
            slots = self.cfg.max_open_positions - len(held)
            for snap, result, safety in survivors:
                if slots <= 0 or not self.analyst.budget_ok():
                    break
                if snap.token_address in held:
                    continue
                tri = None
                if self.triage is not None:
                    tri = self.triage.check(snap, safety, result)
                    if self.triage.last_cost_usd:
                        self.journal.log_cost("triage", self.triage.last_cost_usd, snap.token_address, MAIN)
                    if self.triage.last_outcome == "unavailable":
                        break  # API down: leave this and later candidates queued
                    if not self.triage.goes_on(tri):
                        decision_id = self.journal.log_decision(snap, result, stage="triaged_out", safety=safety,
                                                                triage=tri)
                        self.tracker.track(decision_id, snap.token_address, snap.price_usd)
                        analysed.append(((snap, result, safety), "pass"))  # same pool for the random baseline
                        continue
                verdict = self.analyst.analyse(snap, safety, result)
                cost = getattr(self.analyst, "last_cost_usd", 0.0)
                if cost:
                    self.journal.log_cost("claude_api", cost, snap.token_address, MAIN)
                if verdict is None:
                    if getattr(self.analyst, "last_outcome", "failed") == "unavailable":
                        break  # API down or paused: leave this and later candidates queued
                    self._cool(snap.token_address)
                    continue
                decision_id = self.journal.log_decision(snap, result, verdict, analysed=True, safety=safety,
                                                        playbook_arm=getattr(self.analyst, "last_arm", None),
                                                        triage=tri)
                self.tracker.track(decision_id, snap.token_address, snap.price_usd)
                analysed.append(((snap, result, safety), verdict.action))
                log.info("%s -> %s (conviction %.2f): %s",
                         snap.symbol, verdict.action, verdict.conviction, verdict.thesis)
                if (verdict.action == "buy" and verdict.conviction >= self.cfg.min_conviction
                        and verdict.social_score >= self.cfg.min_social_score):
                    if self.buy(main, snap, verdict.conviction, decision_id, self._size_factor(result)):
                        slots -= 1
                        self._maybe_canary(snap, verdict.conviction, decision_id)
        self._track_unanalysed(survivors, analysed)
        self.after_screen(survivors, analysed)

    def _maybe_canary(self, snap: PairSnapshot, conviction: float, decision_id: int) -> None:
        """Mirror every Nth Claude buy with a small real trade (same decision, same exit rules)."""
        book = self.books.get(CANARY)
        if book is None:
            return
        count = int(self.journal.get_state("canary_count") or 0) + 1
        self.journal.set_state("canary_count", str(count))
        if count % self.cfg.canary_every_n == 0:
            self.buy(book, snap, conviction, decision_id)

    def _track_unanalysed(self, survivors: list[Survivor], analysed: list[tuple[Survivor, str]]) -> None:
        """Candidates Claude didn't get to (budget, slots) still get a price path, once per 6h."""
        done = {s[0].token_address for s, _ in analysed}
        for snap, result, safety in survivors:
            if snap.token_address in done or self.journal.recently_seen(snap.token_address, REANALYSE_AFTER_S,
                                                                         stage="survivor"):
                continue
            decision_id = self.journal.log_decision(snap, result, stage="survivor", safety=safety)
            self.tracker.track(decision_id, snap.token_address, snap.price_usd)

    def after_screen(self, survivors: list[Survivor], analysed: list[tuple[Survivor, str]]) -> None:
        """Let each baseline book pick from the same candidates Claude saw."""
        baselines = [b for b in self.books.values() if b.name not in BOOK_PRIORITY and b.exit_cfg is None]
        if not baselines:
            return
        buy_rate = shadows.claude_buy_rate(self.journal, self.cfg)
        for book in baselines:
            for snap, result, _safety in shadows.picks(book.name, survivors, analysed, buy_rate, self._rng):
                if not self.entries_allowed(book):
                    break
                self.buy(book, snap, self.cfg.shadow_conviction, None, self._size_factor(result))

    def _size_factor(self, result: FactorResult | None) -> float:
        """Partial-LP-lock exceptions trade at reduced size (same rule for every book)."""
        return self.cfg.partial_lock_size_factor if getattr(result, "partial_lock", False) else 1.0

    def preflight(self, snap: PairSnapshot, usd: float) -> tuple[PairSnapshot | None, str]:
        """Re-validate a token right before buying. Research can take minutes.

        Returns (fresh snapshot, "") or (None, reason to skip).
        """
        fresh = self.dex.pairs([snap.token_address])
        if not fresh:
            return None, "pair disappeared"
        now = fresh[0]
        if self.jupiter is not None:
            # DexScreener can lag a fast mover by a minute; Jupiter's price is what a swap
            # would see now, and is the same source the monitor uses for exits.
            live = self.jupiter.prices([snap.token_address]).get(snap.token_address)
            if live is not None:
                now = replace(now, price_usd=live.usd)
        if rejections := factors.market_filters(now, self.cfg):
            return None, "no longer passes: " + "; ".join(rejections)
        drift = (now.price_usd / snap.price_usd - 1) * 100
        if drift > self.cfg.max_entry_drift_pct:
            return None, f"price ran {drift:+.1f}% during research"
        if self.jupiter is not None:
            rt = self.jupiter.round_trip(snap.token_address, usd)
            if rt is None:
                return None, "no swap route"
            loss, impact = rt
            if loss > self.cfg.max_round_trip_loss_pct:
                return None, f"round-trip loss {loss:.1f}% (honeypot/tax/thin pool)"
            if impact > self.cfg.max_price_impact_pct:
                return None, f"price impact {impact:.2f}%"
        return now, ""

    def buy(self, book: Book, snap: PairSnapshot, conviction: float, decision_id: int | None,
            size_factor: float = 1.0) -> bool:
        if any(p.token_address == snap.token_address for p in self.journal.open_positions(book.name)):
            return False
        if book.fixed_usd is not None:
            usd = min(book.fixed_usd, book.cash)
        else:
            usd = risk.position_size_usd(self.equity(book), book.cash, conviction, self.cfg)
        usd *= max(0.0, min(1.0, size_factor))  # can only shrink a position, never grow it
        if usd < 1:
            log.info("[%s] position size too small for %s", book.name, snap.symbol)
            return False
        fresh, reason = self.preflight(snap, usd)
        if fresh is None:
            log.info("[%s] skip buy %s: %s", book.name, snap.symbol, reason)
            self.journal.log_event("entry_skipped", snap.token_address, reason, strategy=book.name)
            return False
        fill = book.executor.buy(snap.token_address, usd, fresh.price_usd)
        if fill is None:
            why = getattr(book.executor, "last_note", "") or ""
            self.journal.log_event("entry_skipped", snap.token_address,
                                   f"executor declined: {why}" if why else "executor declined", strategy=book.name)
            return False
        if not fill.ok:
            self._charge_failed(book, snap.token_address, fill)
            return False
        self._open_position(book, snap.token_address, snap.symbol, fill, fresh.price_usd, decision_id)
        return True

    def _open_position(self, book: Book, token: str, symbol: str, fill: Fill, trigger: float,
                       decision_id: int | None) -> Position:
        with self.lock:
            book.cash -= fill.usd
            self.journal.set_cash(book.cash, book.name)
            pos = Position(
                id=None, token_address=token, symbol=symbol, entry_price=fill.price,
                qty=fill.qty, initial_qty=fill.qty, cost_usd=fill.usd, opened_at=time.time(),
                stop_price=risk.initial_stop(fill.price, book.exit_cfg or self.cfg), peak_price=fill.price,
                strategy=book.name,
            )
            self.journal.save_position(pos, decision_id)
            self.journal.log_fill(pos.id, "buy", fill.qty, fill.price, fill.usd, fill.fee_usd, "entry", fill.tx,
                                  trigger_price=trigger)
        log.info("[%s] BUY %s $%.2f @ %.10g (stop %.10g)", book.name, symbol, fill.usd, fill.price, pos.stop_price)
        return pos

    # --- model5: the 5-minute pump model's paper book ----------------------------------
    def buy_model_signals(self) -> int:
        """Buy the pools the model queued (from the watcher thread). Own checks: the Claude book's age and
        liquidity filters are tuned for older tokens and would reject every 5-minute pool."""
        from tradingbot import model_book as mb
        bought = 0
        while self.model_book is not None:
            try:
                sig = self.model_book.signals.get_nowait()
            except queue.Empty:
                break
            if sig.get("rug_check"):  # blind test: the prediction is recorded before any trade or outcome
                self.journal.log_event("rug_prediction", sig["mint"],
                                       json.dumps({"pool": sig.get("pool"), **sig["rug_check"]}), strategy="model")
            sig.setdefault("picked_ts", time.time())
            targets = []
            for name in mb.BOOKS:
                book = self.books.get(name)
                if book is None or mb.signal_kind(name) != sig.get("kind"):
                    continue  # early signals to the early books, launch signals to the launch books, model5 to the rest
                if not mb.wants(name, sig):
                    continue  # e.g. a fixed-only signal for a rolling book, a top-5% pick for the top-2% book
                targets.append(book)
            if len(targets) > 1:  # all books buy at once: one after another cost each later book ~3.5 s
                if self._model_pool is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self._model_pool = ThreadPoolExecutor(max_workers=BUY_WORKERS, thread_name_prefix="model-buy")
                bought += sum(self._model_pool.map(lambda b: self._try_model(b, sig), targets))
            else:
                bought += sum(self._try_model(b, sig) for b in targets)
        return bought

    def _try_model(self, book: Book, sig: dict) -> bool:
        try:
            reason = self._model_skip_reason(book, sig)
            if reason is None:
                reason = self._buy_model(book, sig)
        except Exception as exc:  # noqa: BLE001 (one book's failure must not stop the others)
            log.exception("[%s] model buy failed", book.name)
            reason = f"error: {type(exc).__name__}"
        if reason:
            self.journal.log_event("entry_skipped", sig["mint"], reason, strategy=book.name)
            log.info("[%s] skip %s: %s", book.name, sig["name"][:24], reason)
            return False
        return True

    def _model_skip_reason(self, book: Book, sig: dict) -> str | None:
        c = self.cfg
        if time.time() - sig["ts"] > MODEL_SIGNAL_MAX_AGE_S:
            return "signal too old"
        from tradingbot import model_book as mb
        skip = mb.verdict_skip(book.name, sig)
        if skip:
            return skip
        if book.name in mb.POOL_MINUTE_EXIT and sig.get("created_ts") and \
                time.time() > sig["created_ts"] + (mb.POOL_MINUTE_EXIT[book.name] - 0.5) * 60:
            return f"pool too old to sell by minute {mb.POOL_MINUTE_EXIT[book.name]:g}"
        if (book.name in mb.SHORT and sig.get("created_ts")
                and (time.time() - sig["created_ts"]) / 60 > c.model_book_max_entry_age_min):
            return "pool too old for the short hold"
        open_ = self.journal.open_positions(book.name)
        if any(p.token_address == sig["mint"] for p in open_):
            return "already held"
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        if book.name in mb.CANARY_BOOKS:  # real money: hard limits of its own
            if self.journal.get_state(CANARY_HALT):
                return "canary halted (kill switch)"
            allow = self.journal.get_state(CANARY_ALLOW)
            if allow is not None and int(allow) <= 0:  # one trade at a time: canary-one allows the next
                return "canary waiting for canary-one (one trade at a time)"
            if c.launch_canary_max_trades > 0:  # a fixed-size test run: stop buying after N trades in this book
                done = self.journal.query("SELECT COUNT(*) n FROM positions WHERE strategy=?", (book.name,))[0]["n"]
                if done >= c.launch_canary_max_trades:
                    return f"canary test run done ({done} of {c.launch_canary_max_trades} trades)"
            dd = mb.DD_LIMITS.get(mb.LAUNCH_SIZED_DD)
            if c.launch_canary_follow_dd and dd:  # the full-size paper twin's drawdown decides, not the canary's fees
                lost = self._window_loss(mb.LAUNCH_SIZED_DD, self.journal.open_positions(mb.LAUNCH_SIZED_DD),
                                         time.time() - dd[0] * 3600)
                if lost >= dd[1]:
                    return f"twin {mb.LAUNCH_SIZED_DD} paused: past {dd[0]:g}h drawdown limit (down ${lost:,.0f})"
            if c.launch_canary_max_liq_usd > 0 and (sig.get("liq") or 0) >= c.launch_canary_max_liq_usd:
                return f"canary trades thin pools only (< ${c.launch_canary_max_liq_usd:,.0f})"
            if c.launch_canary_every_n > 1:  # real fees: a sample of the twin's ~700 signals a day is enough to calibrate
                n = int(self.journal.get_state("launch_canary:count") or 0) + 1
                self.journal.set_state("launch_canary:count", str(n))
                if n % c.launch_canary_every_n:
                    return f"canary samples 1 in {c.launch_canary_every_n} signals"
            if len(open_) >= c.launch_canary_max_open:
                return "max open positions"
            if (c.launch_canary_daily_loss_usd > 0  # 0 = no daily stop (user, 11-10: an exact replica of sz)
                    and -self.journal.realized_since(day_start, book.name) >= c.launch_canary_daily_loss_usd):
                return "daily loss limit"
            return None
        if len(open_) >= c.model_book_max_open:
            return "max open positions"
        dd = mb.DD_LIMITS.get(book.name)
        if dd:
            lost = self._window_loss(book.name, open_, time.time() - dd[0] * 3600)
            if lost >= dd[1]:
                return f"past {dd[0]:g}h drawdown limit (down ${lost:,.0f})"
        if c.model_book_daily_loss_pct > 0 and \
                -self.journal.realized_since(day_start, book.name) >= self.equity(book) * c.model_book_daily_loss_pct / 100:
            return "daily loss limit"
        return None

    def _window_loss(self, name: str, open_: list, since: float) -> float:
        """Realized loss since `since` (sales in the window), plus open positions whose exit has been blocked since
        then (no route out: written off at $0, as equity and the dd-limits study count them)."""
        lost = -self.journal.realized_since(since, name)
        return lost + sum(p.cost_usd for p in open_ if self._blocked.get(p.id, {}).get("since", 0) >= since)

    def _buy_model(self, book: Book, sig: dict) -> str | None:
        c = self.cfg
        started = time.time()
        from tradingbot import model_book as mb
        canary = book.name in mb.CANARY_BOOKS
        if canary and c.launch_canary_probe and not sig.get("entry") and hasattr(book.executor, "probe_1slot"):
            # 1-slot probe measurement (11-12): land committed money (slot N) then pull it out next slot, so a
            # size-reacting rugger is tripped while our sell escapes, and we NEVER commit. Runs before the Jupiter
            # price/route block, as a seconds-old pool has no Jupiter price. The survival watch scheduled here reads the
            # pool at probe_ts + the watch window and records rugger (pulled, we dodged) / survivor (a pool to enter).
            if sig.get("pool") and hasattr(book.executor, "remember_pool"):
                book.executor.remember_pool(sig["mint"], sig["pool"])
            res = book.executor.probe_1slot(sig["mint"], c.launch_canary_probe_usd)
            res["liq"] = sig.get("liq")
            # Money LANDED only on ok/trapped; a honeypot (sim-blocked, nothing sent) or a skip committed nothing, so a
            # survival watch on it would wrongly mark a pool we never touched "survivor". Only watch committed probes.
            res["committed"] = res.get("verdict") in ("ok", "trapped")
            self.journal.log_event("canary_probe", sig["mint"], json.dumps(res), strategy=book.name)
            if res["committed"]:
                self._schedule_probe_watch(book, sig, res)
            allow = self.journal.get_state(CANARY_ALLOW)
            if allow is not None:  # one probe at a time (canary-one / Execute one probe): this used the allowance
                self.journal.set_state(CANARY_ALLOW, str(max(0, int(allow) - 1)))
            return f"probe: {res.get('verdict')}"
        if canary and c.launch_canary_feint and c.launch_canary_feint_only and hasattr(book.executor, "feint"):
            # Measurement run: probe every routed pool and never commit, so run it before the Jupiter price/route
            # block -- a seconds-old pool Jupiter can't price yet must still be feinted, not dropped as 'no live price'.
            if sig.get("pool") and hasattr(book.executor, "remember_pool"):
                book.executor.remember_pool(sig["mint"], sig["pool"])
            verdict = book.executor.feint(sig["mint"], c.launch_canary_feint_usd, commit=False)
            self.journal.log_event("canary_feint", sig["mint"], json.dumps(
                {"verdict": verdict, "note": getattr(book.executor, "last_note", "") or "",
                 **(getattr(book.executor, "last_feint", None) or {})}), strategy=book.name)
            return f"feint only: pool {verdict}"
        if canary and sig.get("entry"):
            base = c.launch_canary_entry_usd  # two-stage entry: the real commit on a confirmed double-survivor
        elif canary and c.launch_canary_flat_usd > 0:
            base = c.launch_canary_flat_usd  # one tiny size for every pick: does any real buy set the operators off?
        elif canary and c.launch_canary_band_scale > 0 and sig.get("liq") is not None:
            # the twin's own sizes scaled down (11-07: 1/25 = $10 thin / $1 at $15-25k / $4 above), same ratios
            base = mb.band_usd(mb.CANARY_TWIN, sig["liq"]) * c.launch_canary_band_scale
        else:
            base = mb.band_usd(book.name, sig.get("liq")) or book.fixed_usd or c.model_book_usd
        liq = sig.get("liq")  # a fast signal has none yet (emitted at detection, before the scorer reads the pool)
        usd = min(base, book.cash) if liq is None else min(base, c.model_book_liq_share * liq, book.cash)
        if usd < (CANARY_MIN_USD if canary else MODEL_MIN_USD):
            return "size too small (cash or liquidity)"
        trigger = sig.get("price") or 0.0
        if sig.get("pool") and hasattr(book.executor, "remember_pool"):  # the direct route swaps on this pool
            book.executor.remember_pool(sig["mint"], sig["pool"])
        # A seconds-old pool has no Jupiter price; the fast canary's direct executor reads the pool reserves itself.
        rt_fn = getattr(book.executor, "round_trip", None)  # a direct executor simulates a buy+sell on the pool itself
        if sig.get("kind") != "fast" and rt_fn is not None:
            # Direct route: gate the honeypot / sell-route on a simulated round trip on the ACTUAL pumpswap pool (free,
            # nothing sent), so it never depends on Jupiter having indexed a seconds-old pool. The executor fills from
            # live reserves, so the signal's chain price stands as the trigger.
            rt = rt_fn(sig["mint"], usd)
            if rt is None:
                return "no direct route (pool unreadable)"
            if rt[0] > c.max_round_trip_loss_pct:
                return f"round-trip loss {rt[0]:.1f}% ({rt[1]})"
        elif sig.get("kind") != "fast" and self.jupiter is not None:
            live = self.jupiter.prices([sig["mint"]]).get(sig["mint"])
            live_usd = live.usd if live is not None else self.jupiter.quote_price(sig["mint"])  # too new to list
            if live_usd is None:
                return self._jup_failure("no live price")
            if live_usd > NO_LIQUIDITY_MULT * sig["price"]:
                # A buy quote on an empty or drained pool implies an absurd price (10-28: 268,000x and 122,000,000x
                # the snapshot): there is nothing real to buy, which is not the price running up.
                return f"no real liquidity (live price {live_usd / sig['price']:,.0f}x the snapshot)"
            # Launch picks are bought whatever the move since second 4, as in their backtest (fill at second 6).
            if sig.get("kind") != "launch" and (live_usd / sig["price"] - 1) * 100 > c.model_book_max_drift_pct:
                return f"price ran {(live_usd / sig['price'] - 1) * 100:+.0f}% since the snapshot"
            trigger = live_usd
            rt = self.jupiter.round_trip(sig["mint"], usd)
            if rt is None:
                return self._jup_failure("no swap route")
            if rt[0] > c.max_round_trip_loss_pct:
                return f"round-trip loss {rt[0]:.1f}% (honeypot/tax/thin pool)"
        if canary and c.launch_canary_feint and not c.launch_canary_feint_only and hasattr(book.executor, "feint"):
            # Commit path: poke the pool, commit only if it wasn't pulled in reaction (feint-only is handled above).
            verdict = book.executor.feint(sig["mint"], c.launch_canary_feint_usd, commit=True)
            self.journal.log_event("canary_feint", sig["mint"], json.dumps(
                {"verdict": verdict, "note": getattr(book.executor, "last_note", "") or "",
                 **(getattr(book.executor, "last_feint", None) or {})}), strategy=book.name)
            if verdict != "alive":
                return f"feint: pool {verdict}"
        fill = book.executor.buy(sig["mint"], usd, trigger)
        if fill is None:
            return f"executor declined: {getattr(book.executor, 'last_note', '') or 'no reason given'}"
        if not fill.ok:
            self._charge_failed(book, sig["mint"], fill)
            return "transaction failed"
        symbol = (sig["name"].split(" / ")[0] or sig["mint"][:6])[:16]
        pos = self._open_position(book, sig["mint"], symbol, fill, trigger or fill.price, None)
        if canary and self.journal.get_state(CANARY_ALLOW) is not None:  # used up the trade canary-one allowed
            self.journal.set_state(CANARY_ALLOW, str(max(0, int(self.journal.get_state(CANARY_ALLOW)) - 1)))
        deadline = mb.exit_deadline(self.cfg, book.name, sig)
        if deadline is not None:
            self._deadlines[pos.id] = deadline
            self.journal.log_event("model_exit_deadline", sig["mint"],
                                   json.dumps({"position_id": pos.id, "deadline": deadline}), strategy=book.name)
        check = mb.climb_check(self.cfg, book.name, pos.opened_at, trigger)
        if check is not None:
            self._climbs[pos.id] = [check[0], check[1], 0.0]
            self.journal.log_event("model_climb_check", sig["mint"],
                                   json.dumps({"position_id": pos.id, "at": check[0], "price": check[1]}),
                                   strategy=book.name)
        if canary and sig.get("entry") and (c.launch_canary_climb_check_s > 0 or c.launch_canary_trail_pct > 0
                                             or c.launch_canary_stall_window_s > 0):
            # Two-stage entry exits (11-13), all priced from the pool's own reserves via the paid Helius RPC (Jupiter
            # can't quote a ~2-min-old pool), so the raw entry price from the fill is the baseline. One read here.
            entry_px = book.executor.pool_price(sig["mint"]) if hasattr(book.executor, "pool_price") else None
            if entry_px and c.launch_canary_climb_check_s > 0:
                # Climb check: at entry + climb_check_s, bail unless up by climb_min_gain_pct, else ride to the exit.
                rec = {"position_id": pos.id, "at": pos.opened_at + c.launch_canary_climb_check_s,
                       "entry_px": entry_px, "min_gain": c.launch_canary_climb_min_gain_pct / 100.0}
                self._canary_climbs[pos.id] = rec
                self.journal.log_event("canary_climb_check", sig["mint"], json.dumps(rec), strategy=book.name)
            if entry_px and (c.launch_canary_trail_pct > 0 or c.launch_canary_stall_window_s > 0):
                # Polled hold exits on one reserve read per pass: a trailing stop (sell on a trail_pct drop off the
                # running peak -- the downward fade) and a stall exit (sell when the rolling peak-to-trough range over
                # stall_window_s falls below the vol gate -- the wash bot going flat before an atomic pull, 11-13).
                rec = {"position_id": pos.id, "peak": entry_px, "trail": c.launch_canary_trail_pct / 100.0,
                       "window_s": c.launch_canary_stall_window_s, "thr": c.launch_canary_vol_min_pct / 100.0}
                self._canary_trails[pos.id] = rec
                self.journal.log_event("canary_trail_arm", sig["mint"], json.dumps(rec), strategy=book.name)
        checks = mb.hold_checks(book.name, sig)
        if checks:
            self._holds[pos.id] = {"at": list(checks), "last": None}
            self.journal.log_event("model_hold_check", sig["mint"],
                                   json.dumps({"position_id": pos.id, "at": checks}), strategy=book.name)
        self.journal.log_event("model_buy", sig["mint"], f"score {sig['score']:.3f} liq ${sig['liq']:,.0f}",
                               strategy=book.name)
        if sig.get("kind") in ("early", "launch"):  # where the seconds between the pool's creation and our fill go
            self.journal.log_event("early_timing", sig["mint"], json.dumps(
                {k: round(v, 2) for k, v in (("created", sig.get("created_ts")), ("seen", sig.get("seen_ts")),
                                             ("found", sig.get("ts")), ("decided", sig.get("decided_ts")),
                                             ("picked", sig.get("picked_ts")),
                                             ("started", started), ("filled", pos.opened_at)) if v}),
                strategy=book.name)
        self._log_model_audit(book, sig)
        return None

    def _load_deadlines(self) -> dict[int, float]:
        """Pool-age exits of open positions, so they survive a restart."""
        open_ids = {p.id for p in self.journal.open_positions(strategy=None)}
        out = {}
        for r in self.journal.query("SELECT detail FROM events WHERE kind='model_exit_deadline'"):
            d = json.loads(r["detail"])
            if d["position_id"] in open_ids:
                out[d["position_id"]] = d["deadline"]
        return out

    def _load_climbs(self) -> dict[int, list]:
        """Pending climb checks of open positions. The best price seen restarts at 0 after a restart."""
        open_ids = {p.id for p in self.journal.open_positions(strategy=None)}
        out = {}
        for r in self.journal.query("SELECT detail FROM events WHERE kind='model_climb_check'"):
            d = json.loads(r["detail"])
            if d["position_id"] in open_ids:
                out[d["position_id"]] = [d["at"], d["price"], 0.0]
        return out

    def _load_holds(self) -> dict[int, dict]:
        """Pending conditional hold checks of open positions (the last check's return restarts unknown)."""
        open_ids = {p.id for p in self.journal.open_positions(strategy=None)}
        out = {}
        now = time.time()
        for r in self.journal.query("SELECT detail FROM events WHERE kind='model_hold_check'"):
            d = json.loads(r["detail"])
            if d["position_id"] in open_ids:
                # a check missed while down: run the latest one due now, skip the older ones
                due = [t for t in d["at"] if t <= now]
                out[d["position_id"]] = {"at": due[-1:] + [t for t in d["at"] if t > now], "last": None}
        return out

    def _load_probe_watches(self) -> dict:
        """Pending 1-slot-probe survival watches, re-armed after a restart: canary_probe_watch events whose pool has no
        canary_probe_verdict yet (an overdue one is read on the next monitor pass)."""
        done = {r["token"] for r in self.journal.query(
            "SELECT DISTINCT token FROM events WHERE kind='canary_probe_verdict'")}
        out: dict = {}
        for r in self.journal.query("SELECT token, detail FROM events WHERE kind='canary_probe_watch'"):
            d = json.loads(r["detail"] or "{}")
            pool = d.get("pool")
            if pool and r["token"] not in done:
                out[pool] = {"mint": r["token"], "pool": pool, "due": d["due"], "probe_ts": d["probe_ts"],
                             "sol_before": d.get("sol_before"), "liq": d.get("liq"), "book": d.get("book"),
                             "created_ts": d.get("created_ts")}
        return out

    def _load_canary_climbs(self) -> dict[int, dict]:
        """Pending two-stage-entry climb checks of open positions, re-armed after a restart."""
        open_ids = {p.id for p in self.journal.open_positions(strategy=None)}
        out: dict[int, dict] = {}
        for r in self.journal.query("SELECT detail FROM events WHERE kind='canary_climb_check'"):
            d = json.loads(r["detail"])
            if d.get("position_id") in open_ids and d.get("entry_px"):
                out[d["position_id"]] = d
        return out

    def _load_canary_trails(self) -> dict[int, dict]:
        """Pending two-stage-entry trailing stops of open positions, re-armed after a restart. The peak resets to the
        last armed peak (the real high between arm and restart is lost), but the pool-minute deadline backstops."""
        open_ids = {p.id for p in self.journal.open_positions(strategy=None)}
        out: dict[int, dict] = {}
        for r in self.journal.query("SELECT detail FROM events WHERE kind='canary_trail_arm'"):
            d = json.loads(r["detail"])
            if d.get("position_id") in open_ids and d.get("peak"):
                out[d["position_id"]] = d
        return out

    def _schedule_probe_watch(self, book: Book, sig: dict, res: dict) -> None:
        """After a probe, schedule a pool read at probe_ts + watch window + margin to decide rugger vs survivor."""
        pool = res.get("pool") or sig.get("pool")
        if not pool:
            return
        now = time.time()
        rec = {"mint": sig["mint"], "pool": pool, "probe_ts": now, "sol_before": res.get("sol_before"),
               "liq": sig.get("liq"), "book": book.name, "created_ts": sig.get("created_ts"),
               "due": now + self.cfg.launch_canary_probe_watch_s + self.cfg.launch_canary_probe_margin_s}
        self._probe_watches[pool] = rec
        self.journal.log_event("canary_probe_watch", sig["mint"], json.dumps(rec), strategy=book.name)

    def _run_probe_watches(self) -> None:
        """Read each probe's pool once its watch window is up and record the verdict: the pool's SOL is gone / far below
        the probe-time level -> `rugger` (we tripped it and dodged); still liquid -> `survivor` (a pool to enter for
        real). Read only (one reserve read per due watch); never commits."""
        if not self._probe_watches:
            return
        from tradingbot import model_book as mb
        now = time.time()
        ex = next((b.executor for b in self.books.values() if b.name in mb.CANARY_BOOKS), None)
        for pool in [p for p, w in self._probe_watches.items() if now >= w["due"]]:
            w = self._probe_watches.pop(pool)
            sol_after, verdict = None, "unknown"
            if ex is not None and hasattr(ex, "pool_sol"):
                try:
                    sol_after = ex.pool_sol(w["mint"])
                except Exception:  # noqa: BLE001 (a read failure must never stop the monitor loop)
                    sol_after = None
                before = w.get("sol_before") or 0.0
                if sol_after is None:
                    verdict = "rugger"  # pool unreadable / gone
                elif before and sol_after < PROBE_PULLED_SHARE * before:
                    verdict = "rugger"
                else:
                    verdict = "survivor"
            self.journal.log_event("canary_probe_verdict", w["mint"], json.dumps(
                {"pool": pool, "verdict": verdict, "sol_before": w.get("sol_before"), "sol_after": sol_after,
                 "liq": w.get("liq"), "watched_s": round(now - w["probe_ts"], 1)}), strategy=w.get("book"))
            if verdict == "survivor" and self.cfg.launch_canary_enter:
                try:
                    self._canary_enter(w)
                except Exception:  # noqa: BLE001 (an entry failure must never stop the monitor loop)
                    log.exception("canary two-stage entry failed")

    def _canary_helius_client(self):
        """Lazy live (main-key) Helius client for the two-stage entry's volatility gate — the paid tool, since the
        gate is part of committing real money."""
        if self._canary_helius is None:
            import httpx
            from tradingbot.data.helius import Helius
            self._canary_helius = Helius(client=httpx.Client(timeout=10), role="live")
        return self._canary_helius

    def _canary_enter(self, w: dict) -> None:
        """A probe survivor (passed the ~48 s watch): run the confirmation-window volatility gate (live Helius), then a
        SECOND 1-slot probe ("one more hit" — some ruggers fire only on the second visible commit); if it survives too,
        enter for real at the entry size. Logs each stage; opens nothing unless both gates pass."""
        from tradingbot import launch_canary as lc
        from tradingbot import model_book as mb
        c = self.cfg
        cbook = next((b for b in self.books.values() if b.name in mb.CANARY_BOOKS), None)
        if cbook is None or self.journal.get_state(CANARY_HALT):
            return
        mint, pool = w["mint"], w.get("pool")
        open_ = self.journal.open_positions(cbook.name)
        if len(open_) >= c.launch_canary_max_open or any(p.token_address == mint for p in open_):
            return
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        if c.launch_canary_daily_loss_usd > 0 \
                and -self.journal.realized_since(day_start, cbook.name) >= c.launch_canary_daily_loss_usd:
            return
        if c.launch_canary_vol_gate:  # enter only if the pool moved during the confirmation window (not dead-flat)
            conf_ts = w["probe_ts"] + lc.CONF_AFTER_S
            ok, rng, n = lc.volatility_ok(self._canary_helius_client(), pool, mint, conf_ts, c.launch_canary_vol_min_pct)
            self.journal.log_event("canary_entry_gate", mint, json.dumps(
                {"pool": pool, "stage": "volatility", "ok": ok, "range": rng, "samples": n,
                 "min_pct": c.launch_canary_vol_min_pct}), strategy=cbook.name)
            if not ok:
                return
        ex = cbook.executor
        if not hasattr(ex, "probe_1slot"):
            return
        if pool and hasattr(ex, "remember_pool"):
            ex.remember_pool(mint, pool)
        res2 = ex.probe_1slot(mint, c.launch_canary_probe2_usd)
        self.journal.log_event("canary_probe2", mint, json.dumps({**res2, "pool": pool}), strategy=cbook.name)
        if res2.get("verdict") != "ok":  # trapped / honeypot / buy_failed: it insta-rugged on the second hit, skip
            return
        entry_sig = {"mint": mint, "pool": pool, "name": (mint or "")[:10], "score": 0.0,
                     "liq": w.get("liq") or 0.0, "price": 0.0, "created_ts": w.get("created_ts"),
                     "kind": "canary_entry", "entry": True, "rw50": True}
        note = self._buy_model(cbook, entry_sig)
        self.journal.log_event("canary_entry", mint, json.dumps(
            {"pool": pool, "result": note or "bought"}), strategy=cbook.name)

    def _hold_failed(self, pos: Position, now: float) -> bool:
        """At each due check (model_book.HOLD_CHECK_MIN) quote a sale of the position: True (sell) when it would
        return less than model_book.HOLD_CHECK_GAIN of the cost, or less than at the previous check. A failed quote
        retries next pass (the pool-minute deadline still sells it)."""
        h = self._holds.get(pos.id)
        if h is not None and h.get("sell"):
            return True  # a failed check whose sale didn't go through: keep trying (exit-blocked rules apply)
        if h is None or not h["at"] or now < h["at"][0]:
            return False
        from tradingbot import model_book as mb
        px = self._confirm_sell_price(pos)
        if not px or not pos.initial_qty or not pos.cost_usd:
            return False
        ret = px * pos.qty / (pos.cost_usd * pos.qty / pos.initial_qty) - 1
        fails = mb.hold_check_fails(ret, h["last"])
        self.journal.log_event("hold_check_result", pos.token_address,
                               json.dumps({"position_id": pos.id, "ret": round(ret, 4),
                                           "last": h["last"], "sell": fails}), strategy=pos.strategy)
        h["at"].pop(0)
        h["last"] = ret
        if fails:
            h["sell"] = True
        elif not h["at"]:
            self._holds.pop(pos.id, None)
        return fails

    def _climb_failed(self, pos_id: int, price: float, now: float) -> bool:
        """True once the check time passes without the price having reached its target; a pass ends the check."""
        c = self._climbs.get(pos_id)
        if c is None:
            return False
        c[2] = max(c[2], price)
        if now < c[0]:
            return False
        if c[2] >= c[1]:
            del self._climbs[pos_id]
            return False
        return True

    def _canary_climb_failed(self, book: Book, pos: Position, now: float) -> bool:
        """Two-stage entry climb check: at entry + climb_check_s, read the pool's current price from its reserves
        (the paid Helius RPC, since Jupiter can't quote a ~2-min-old pool) and compare to the entry price. Up by
        min_gain -> keep (ride to the pool-minute exit); not up -> bail (no_climb). Runs once; an unreadable price
        retries next pass (the pool-minute deadline still exits it)."""
        rec = self._canary_climbs.get(pos.id)
        if rec is None or now < rec["at"]:
            return False
        px = book.executor.pool_price(pos.token_address) if hasattr(book.executor, "pool_price") else None
        if not px:
            return False  # can't read the pool now; try again next pass
        self._canary_climbs.pop(pos.id, None)  # the check runs once
        gain = (px / rec["entry_px"] - 1.0) if rec.get("entry_px") else None
        kept = gain is not None and gain >= rec.get("min_gain", 0.0)
        self.journal.log_event("canary_climb_result", pos.token_address, json.dumps(
            {"position_id": pos.id, "entry_px": rec.get("entry_px"), "price": px, "gain": gain, "kept": kept}),
            strategy=book.name)
        return not kept

    def _canary_hold_exit(self, book: Book, pos: Position, now: float) -> str | None:
        """Two-stage entry hold exits on one pool-reserve read per pass (paid Helius RPC; Jupiter can't quote a
        ~2-min-old pool). Two complementary rules: a trailing stop that sells on a `trail` drop off the running peak
        (the downward fade), and a stall exit that sells when the rolling peak-to-trough range over the last `window_s`
        falls below `thr` -- the wash bot going flat before an atomic pull, the same volatility measure the entry gate
        uses, applied rolling. Returns the exit reason or None; an unreadable price is skipped (deadline backstops)."""
        rec = self._canary_trails.get(pos.id)
        if rec is None:
            return None
        px = book.executor.pool_price(pos.token_address) if hasattr(book.executor, "pool_price") else None
        if not px:
            return None  # can't read the pool now; try again next pass
        if px > rec["peak"]:
            rec["peak"] = px
        trail = rec.get("trail", 0.0)
        if trail > 0 and px <= rec["peak"] * (1.0 - trail):  # the downward fade
            self._canary_trails.pop(pos.id, None)
            self.journal.log_event("canary_trail_result", pos.token_address, json.dumps(
                {"position_id": pos.id, "peak": rec["peak"], "price": px, "drawdown": 1.0 - px / rec["peak"]}),
                strategy=book.name)
            return "trail_stop"
        win = rec.get("window_s", 0.0)
        if win > 0:  # the flat stall: rolling peak-to-trough range over the window below the vol gate
            rec.setdefault("start_t", now)
            samples = rec.setdefault("samples", [])
            samples.append((now, px))
            while len(samples) > 1 and samples[0][0] < now - win:
                samples.pop(0)
            if now - rec["start_t"] >= win:  # a full window has elapsed since entry
                lo = min(m for _, m in samples)
                hi = max(m for _, m in samples)
                if lo > 0 and (hi - lo) / lo < rec.get("thr", 0.0):
                    self._canary_trails.pop(pos.id, None)
                    self.journal.log_event("canary_stall_result", pos.token_address, json.dumps(
                        {"position_id": pos.id, "price": px, "range": (hi - lo) / lo, "window_s": win}),
                        strategy=book.name)
                    return "stall_exit"
        return None

    def _jup_failure(self, default: str) -> str:
        """Skip reason for a failed Jupiter lookup: a refused request is not a missing route."""
        return RATE_LIMITED if getattr(self.jupiter, "last_failure", lambda: "")() == RATE_LIMITED else default

    def _log_model_audit(self, book: Book, sig: dict) -> None:
        """Holder audit at the moment of buying (model_entry_audit), for the live rug analysis."""
        from tradingbot import model_book as mb
        try:
            audit = mb.entry_audit(self.extras, sig["mint"])
        except Exception:  # noqa: BLE001 (logging must never stop trading)
            log.warning("entry audit failed for %s", sig["mint"], exc_info=True)
            audit = {}
        audit.update({"score": round(sig["score"], 4), "snapshot_liq": sig["liq"]})
        self.journal.log_event("model_entry_audit", sig["mint"], json.dumps(audit), strategy=book.name)

    def _log_exit_cause(self, book: Book, pos: Position, blocked: bool) -> None:
        """Why a model5 trade lost (model_exit_cause): no route, liquidity pulled, dumped or an ordinary stop."""
        from tradingbot import model_book as mb
        if book.name not in mb.BOOKS or not pos.cost_usd:
            return
        if self.journal.query("SELECT 1 FROM events WHERE strategy=? AND kind='model_exit_cause' AND token=? "
                              "AND ts >= ?", (book.name, pos.token_address, pos.opened_at or 0)):
            return  # already recorded (e.g. blocked before a restart)
        entry = self.journal.query("SELECT detail FROM events WHERE strategy=? AND kind='model_entry_audit' "
                                   "AND token=? ORDER BY ts DESC LIMIT 1", (book.name, pos.token_address))
        audit = json.loads(entry[0]["detail"]) if entry else {}
        entry_liq = audit.get("liquidity") or audit.get("snapshot_liq")
        try:
            cause = mb.exit_cause(self.extras, pos.token_address, entry_liq,
                                  pos.realized_usd / pos.cost_usd - 1, blocked)
        except Exception:  # noqa: BLE001
            log.warning("exit-cause check failed for %s", pos.symbol, exc_info=True)
            cause = {"cause": "no_route" if blocked else "unknown"}
        cause["position_id"] = pos.id
        self.journal.log_event("model_exit_cause", pos.token_address, json.dumps(cause), strategy=book.name)
        log.info("[%s] %s lost: %s", book.name, pos.symbol, cause["cause"])

    def _sell(self, book: Book, pos: Position, qty: float, price: float, reason: str) -> bool:
        fill = book.executor.sell(pos.token_address, min(qty, pos.qty), price)
        # a pool-minute exit that reverts is retried at once (each try costs its fee), not at the next monitor pass:
        # waiting 10-40 s on a fast pool cost the launch books ~1 pt/trade (11-06 fill-gap study)
        tries = SELL_REVERT_RETRIES if book.name in self._pool_minute_books else 0
        while tries and fill is not None and not fill.ok and "reverted" in (fill.note or ""):
            self._charge_failed(book, pos.token_address, fill)
            log.info("[%s] sell %s (%s) reverted; retrying now", book.name, pos.symbol, reason)
            fill = book.executor.sell(pos.token_address, min(qty, pos.qty), price)
            tries -= 1
        if fill is not None and not fill.ok and (RATE_LIMITED in (fill.note or "")
                                                 or (fill.note or "").startswith("not sent:")):
            log.info("[%s] sell %s (%s) postponed: %s", book.name, pos.symbol, reason, fill.note)
            return False  # Jupiter refused the quote: says nothing about the route, retry next pass
        if fill is not None and not fill.ok and fill.usd == 0:
            self._exit_blocked(book, pos, reason, fill.note)  # nothing was sent: no route out
            return False
        if fill is None or not fill.ok:
            if fill is not None:
                self._charge_failed(book, pos.token_address, fill)
            log.error("[%s] sell failed for %s (%s); will retry", book.name, pos.symbol, reason)
            return False
        self._blocked.pop(pos.id, None)
        with self.lock:
            pos.qty -= fill.qty
            pos.realized_usd += fill.usd
            book.cash += fill.usd
            self.journal.set_cash(book.cash, book.name)
            self.journal.log_fill(pos.id, "sell", fill.qty, fill.price, fill.usd, fill.fee_usd, reason, fill.tx,
                                  trigger_price=price)
            if pos.qty <= DUST_QTY * pos.initial_qty:
                pos.qty = 0.0
                pos.closed_at = time.time()
                pos.exit_reason = reason
                self._canary_climbs.pop(pos.id, None)  # drop the two-stage climb record on close
                self._canary_trails.pop(pos.id, None)  # drop the two-stage trailing stop on close
        log.info("[%s] SELL %s %s qty=%.6g @ %.10g -> $%.2f",
                 book.name, pos.symbol, reason, fill.qty, fill.price, fill.usd)
        if pos.closed_at is not None and pos.realized_usd < pos.cost_usd:
            self._log_exit_cause(book, pos, blocked=False)
        return True

    def _exit_blocked(self, book: Book, pos: Position, reason: str, note: str) -> None:
        """No route to sell (often a rug: liquidity pulled). Retry every EXIT_RETRY_S; record it once an
        hour instead of a failed transaction every monitor pass (nothing is sent, nothing is paid)."""
        now = time.time()
        if pos.id not in self._blocked:  # after a restart, keep counting from the first recorded block
            first = self.journal.query(
                "SELECT MIN(ts) AS t FROM events WHERE strategy=? AND token=? AND ts >= ? AND (kind='exit_blocked' "
                "OR (kind='tx_failed' AND usd = 0 AND detail LIKE '%route%'))",
                (book.name, pos.token_address, pos.opened_at or 0))[0]["t"]
            self._blocked[pos.id] = {"since": first or now, "logged": 0.0, "note": note}
            self._log_exit_cause(book, pos, blocked=True)
        b = self._blocked[pos.id]
        b["next"], b["note"] = now + EXIT_RETRY_S, note or "no route"
        if now - b["logged"] >= 3600:
            b["logged"] = now
            self.journal.log_event("exit_blocked", pos.token_address,
                                   f"{reason}: {b['note']} (blocked {(now - b['since']) / 60:.0f} min)", strategy=book.name)
            log.warning("[%s] cannot sell %s (%s): %s; retrying every %ds", book.name, pos.symbol, reason,
                        b["note"], EXIT_RETRY_S)

    def _charge_failed(self, book: Book, token: str, fill: Fill) -> None:
        """A reverted transaction still costs its network fee."""
        with self.lock:
            book.cash -= fill.usd
            self.journal.set_cash(book.cash, book.name)
        self.journal.log_event("tx_failed", token, fill.note, fill.usd, strategy=book.name)
        log.warning("[%s] transaction failed for %s: %s (fee $%.4f)", book.name, token, fill.note, fill.usd)

    # --- measurement ----------------------------------------------------------
    def update_outcomes(self) -> None:
        self.tracker.poll()
