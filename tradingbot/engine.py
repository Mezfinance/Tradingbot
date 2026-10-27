"""The trading engine.

Two loops share one journal:
  * monitor loop (every `monitor_interval_s`): price every open position and run exits;
  * scan loop (every `scan_interval_s`): discover, screen, research with Claude, enter.
Keeping them apart means Claude research (which can take minutes) never delays a stop.

Each strategy trades its own "book" (cash + positions) with the same risk rules.
The "claude" book is the real strategy; other books are baselines (see shadows.py).
"""

from __future__ import annotations

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
from tradingbot.data.jupiter import Jupiter
from tradingbot.data.rugcheck import RugCheck
from tradingbot.execution.base import Executor, Fill
from tradingbot.journal import Journal
from tradingbot.models import FactorResult, PairSnapshot, Position, SafetyReport
from tradingbot.tracker import Tracker
from tradingbot.backups import DailyBackup

log = logging.getLogger(__name__)

MAIN = "claude"
CANARY = "canary"
BOOK_PRIORITY = {MAIN: 0, CANARY: 1}  # real-money books are exited before simulated baselines
REANALYSE_AFTER_S = 6 * 3600
RECHECK_REJECTED_S = 30 * 60
DUST_QTY = 1e-12
EQUITY_SNAPSHOT_S = 60
EXIT_RETRY_S = 120  # a sell with no route is retried this often
STUCK_AFTER_S = 1800  # ...and after this long the position is valued at zero (it cannot be sold)
QUOTE_PRICE_MAX = 10  # positions per monitor pass priced from a swap quote when no feed lists the token
MODEL_POLL_S = 5  # how often queued model5 signals are bought
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
                 model_executor: Executor | None = None):
        self.cfg, self.dex, self.rug, self.analyst = cfg, dex, rug, analyst
        self.triage = triage  # optional cheap first pass before the analyst (tradingbot/triage.py)
        self.gecko = gecko  # optional GeckoTerminal discovery (trending by trading activity)
        self.journal = journal
        self.jupiter = jupiter  # None: no round-trip preflight, prices from DexScreener (tests)
        self.books: dict[str, Book] = {MAIN: self._open_book(MAIN, executor)}
        for name, ex in (shadow_executors or {}).items():
            self.books[name] = self._open_book(name, ex)
        if canary_executor is not None and cfg.canary_every_n > 0:
            self.books[CANARY] = self._open_book(CANARY, canary_executor, cfg.canary_budget_usd, cfg.canary_usd)
        self.model_book = None
        if model_executor is not None and cfg.model_book:
            from tradingbot import model_book as mb
            book = self._open_book(mb.NAME, model_executor, cfg.model_book_cash_usd, cfg.model_book_usd)
            book.exit_cfg = mb.exit_config(cfg)
            self.books[mb.NAME] = book
            self.model_book = mb.ModelBook(cfg)
        self._rng = random.Random(cfg.shadow_seed)
        self.lock = threading.RLock()  # guards book cash and position state
        self.tracker = Tracker(journal, dex, jupiter)
        self.backups = DailyBackup(journal, cfg)
        self._cooldown: dict[str, float] = journal.load_cooldowns()  # token -> time it may be re-screened
        self._last_prices: dict[str, float] = {}
        self._blocked: dict[int, dict] = {}  # position id -> exit blocked (no route): since, next try, note
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
                threading.Thread(target=self._loop, name="model", daemon=True,
                                 args=(MODEL_POLL_S, self.buy_model_signals)).start()
            threading.Thread(target=self._loop, name="watcher", daemon=True,
                             args=(self.cfg.watch_poll_s, watcher.step)).start()
            if self.cfg.enrich:
                from tradingbot.enrich import Enricher
                enricher = Enricher.from_config(self.cfg, watcher.store, watcher)
                threading.Thread(target=self._loop, name="enricher", daemon=True,
                                 args=(self.cfg.enrich_poll_s, enricher.step)).start()
        # Own thread: sampling after each scan cycle left ~18-minute gaps while Claude was busy.
        threading.Thread(target=self._loop, name="tracker", daemon=True,
                         args=(self.cfg.track_interval_s, self.update_outcomes)).start()
        if self.cfg.health_port > 0:
            health.start_health_server(self.journal, self.cfg)
        try:
            self._loop(self.cfg.scan_interval_s, self._scan_cycle)
        finally:
            self._stop.set()

    def _scan_cycle(self) -> None:
        self.scan_and_enter()
        self.backups.run_if_due()
        self.journal.set_state("scan_heartbeat", repr(time.time()))

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
        if missing:
            prices.update({s.token_address: s.price_usd for s in self.dex.pairs(missing)})
        missing = tokens - prices.keys()
        if missing and self.jupiter is not None and hasattr(self.jupiter, "quote_price"):
            for mint in sorted(missing)[:QUOTE_PRICE_MAX]:  # brand-new (or rugged) tokens neither feed lists
                if held and mint in held and hasattr(self.jupiter, "sell_price"):
                    p = self.jupiter.sell_price(mint, held[mint])
                else:
                    p = self.jupiter.quote_price(mint)
                if p:
                    prices[mint] = p
        self._last_prices.update(prices)
        return prices

    # --- positions ----------------------------------------------------------
    def manage_positions(self) -> dict[str, float]:
        # The real book is handled first so simulated latency on baselines never delays it.
        positions = sorted(self.journal.open_positions(strategy=None),
                           key=lambda p: BOOK_PRIORITY.get(p.strategy, 2))
        held: dict[str, float] = {}
        for p in positions:
            held[p.token_address] = max(held.get(p.token_address, 0.0), p.qty)
        prices = self.current_prices(set(held), held)
        now = time.time()
        self.journal.set_state("heartbeat", repr(now))
        for pos in positions:
            price = prices.get(pos.token_address)
            book = self.books.get(pos.strategy)
            if price is None or book is None:
                if price is None:
                    log.warning("no price for open position %s (%s)", pos.symbol, pos.strategy)
                continue
            blocked = self._blocked.get(pos.id)
            if blocked and now < blocked["next"]:
                continue  # no route out last time: retry every EXIT_RETRY_S, not every pass
            before = (pos.tp_hits, pos.stop_price)
            exit_cfg = book.exit_cfg or self.cfg
            exits = risk.plan_exits(pos, price, now, exit_cfg)
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
            for qty, reason in exits:
                if not self._sell(book, pos, qty, price, reason):
                    # Roll the ladder back so the missed target is retried next pass.
                    pos.tp_hits, pos.stop_price = before
                    break
            self.journal.save_position(pos)
        self.snapshot_equity(prices)
        return prices

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
            book = self.books[mb.NAME]
            reason = self._model_skip_reason(book, sig)
            if reason is None:
                reason = self._buy_model(book, sig)
            if reason:
                self.journal.log_event("entry_skipped", sig["mint"], reason, strategy=book.name)
                log.info("[%s] skip %s: %s", book.name, sig["name"][:24], reason)
            else:
                bought += 1
        return bought

    def _model_skip_reason(self, book: Book, sig: dict) -> str | None:
        c = self.cfg
        if time.time() - sig["ts"] > MODEL_SIGNAL_MAX_AGE_S:
            return "signal too old"
        open_ = self.journal.open_positions(book.name)
        if any(p.token_address == sig["mint"] for p in open_):
            return "already held"
        if len(open_) >= c.model_book_max_open:
            return "max open positions"
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        if -self.journal.realized_since(day_start, book.name) >= self.equity(book) * c.model_book_daily_loss_pct / 100:
            return "daily loss limit"
        return None

    def _buy_model(self, book: Book, sig: dict) -> str | None:
        c = self.cfg
        usd = min(c.model_book_usd, c.model_book_liq_share * sig["liq"], book.cash)
        if usd < 5:
            return "size too small (cash or liquidity)"
        trigger = sig["price"]
        if self.jupiter is not None:
            live = self.jupiter.prices([sig["mint"]]).get(sig["mint"])
            live_usd = live.usd if live is not None else self.jupiter.quote_price(sig["mint"])  # too new to list
            if live_usd is None:
                return "no live price"
            if (live_usd / sig["price"] - 1) * 100 > c.model_book_max_drift_pct:
                return f"price ran {(live_usd / sig['price'] - 1) * 100:+.0f}% since the snapshot"
            trigger = live_usd
            rt = self.jupiter.round_trip(sig["mint"], usd)
            if rt is None:
                return "no swap route"
            if rt[0] > c.max_round_trip_loss_pct:
                return f"round-trip loss {rt[0]:.1f}% (honeypot/tax/thin pool)"
        fill = book.executor.buy(sig["mint"], usd, trigger)
        if fill is None:
            return f"executor declined: {getattr(book.executor, 'last_note', '') or 'no reason given'}"
        if not fill.ok:
            self._charge_failed(book, sig["mint"], fill)
            return "transaction failed"
        symbol = (sig["name"].split(" / ")[0] or sig["mint"][:6])[:16]
        self._open_position(book, sig["mint"], symbol, fill, trigger, None)
        self.journal.log_event("model_buy", sig["mint"], f"score {sig['score']:.3f} liq ${sig['liq']:,.0f}",
                               strategy=book.name)
        return None

    def _sell(self, book: Book, pos: Position, qty: float, price: float, reason: str) -> bool:
        fill = book.executor.sell(pos.token_address, min(qty, pos.qty), price)
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
        log.info("[%s] SELL %s %s qty=%.6g @ %.10g -> $%.2f",
                 book.name, pos.symbol, reason, fill.qty, fill.price, fill.usd)
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
