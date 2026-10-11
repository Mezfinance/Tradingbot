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
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from tradingbot import factors, risk
from tradingbot.analyst import Analyst
from tradingbot.config import Config
from tradingbot.data.dexscreener import DexScreener
from tradingbot.data.jupiter import Jupiter
from tradingbot.data.rugcheck import RugCheck
from tradingbot.execution.base import Executor, Fill
from tradingbot.journal import Journal
from tradingbot.models import FactorResult, PairSnapshot, Position, SafetyReport

log = logging.getLogger(__name__)

MAIN = "claude"
REANALYSE_AFTER_S = 6 * 3600
RECHECK_REJECTED_S = 30 * 60
OUTCOME_HORIZON_S = 6 * 3600
DUST_QTY = 1e-12
EQUITY_SNAPSHOT_S = 60

Survivor = tuple[PairSnapshot, FactorResult, SafetyReport | None]


@dataclass
class Book:
    name: str
    executor: Executor
    cash: float


class Engine:
    def __init__(self, cfg: Config, dex: DexScreener, rug: RugCheck, analyst: Analyst | None,
                 executor: Executor, journal: Journal, jupiter: Jupiter | None = None):
        self.cfg, self.dex, self.rug, self.analyst = cfg, dex, rug, analyst
        self.journal = journal
        self.jupiter = jupiter  # None: no round-trip preflight, prices from DexScreener (tests)
        self.books: dict[str, Book] = {MAIN: self._open_book(MAIN, executor)}
        self.lock = threading.RLock()  # guards book cash and position state
        self._cooldown: dict[str, float] = {}  # token -> time it may be re-screened
        self._last_prices: dict[str, float] = {}
        self._last_snapshot = 0.0
        self._stop = threading.Event()

    def _open_book(self, name: str, executor: Executor) -> Book:
        return Book(name, executor, self.journal.get_cash(self.cfg.starting_cash_usd, name))

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
        monitor = threading.Thread(target=self._loop, name="monitor", daemon=True,
                                   args=(self.cfg.monitor_interval_s, self.manage_positions))
        monitor.start()
        try:
            self._loop(self.cfg.scan_interval_s, self._scan_cycle)
        finally:
            self._stop.set()

    def _scan_cycle(self) -> None:
        self.scan_and_enter()
        self.update_outcomes()

    def _loop(self, interval: float, fn) -> None:
        while not self._stop.is_set():
            started = time.time()
            try:
                fn()
            except Exception:  # keep the loop alive; state is persisted
                log.exception("%s failed", getattr(fn, "__name__", fn))
            self._stop.wait(max(0.0, interval - (time.time() - started)))

    # --- prices ---------------------------------------------------------------
    def current_prices(self, tokens: set[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        if not tokens:
            return prices
        if self.jupiter is not None:
            prices = {m: p.usd for m, p in self.jupiter.prices(sorted(tokens)).items()}
        missing = tokens - prices.keys()
        if missing:
            prices.update({s.token_address: s.price_usd for s in self.dex.pairs(missing)})
        self._last_prices.update(prices)
        return prices

    # --- positions ----------------------------------------------------------
    def manage_positions(self) -> dict[str, float]:
        positions = self.journal.open_positions(strategy=None)
        prices = self.current_prices({p.token_address for p in positions})
        now = time.time()
        for pos in positions:
            price = prices.get(pos.token_address)
            book = self.books.get(pos.strategy)
            if price is None or book is None:
                if price is None:
                    log.warning("no price for open position %s (%s)", pos.symbol, pos.strategy)
                continue
            before = (pos.tp_hits, pos.stop_price)
            for qty, reason in risk.plan_exits(pos, price, now, self.cfg):
                if not self._sell(book, pos, qty, price, reason):
                    # Roll the ladder back so the missed target is retried next pass.
                    pos.tp_hits, pos.stop_price = before
                    break
            self.journal.save_position(pos)
        self.snapshot_equity(prices)
        return prices

    def snapshot_equity(self, prices: dict[str, float], force: bool = False) -> None:
        if not force and time.time() - self._last_snapshot < EQUITY_SNAPSHOT_S:
            return
        self._last_snapshot = time.time()
        for book in self.books.values():
            held = sum(p.qty * prices.get(p.token_address, p.entry_price)
                       for p in self.journal.open_positions(book.name))
            self.journal.snapshot_equity(book.name, book.cash, held)

    def equity(self, book: Book | None = None, prices: dict[str, float] | None = None) -> float:
        book = book or self.books[MAIN]
        prices = self._last_prices if prices is None else prices
        held = sum(p.qty * prices.get(p.token_address, p.entry_price)
                   for p in self.journal.open_positions(book.name))
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
    def screen(self) -> list[Survivor]:
        """Discover tokens and return those that clear every deterministic gate, best first."""
        addrs, boosted = self.dex.discover()
        held = {p.token_address for p in self.journal.open_positions(MAIN)}
        survivors: list[Survivor] = []
        now = time.time()
        for snap in self.dex.pairs(addrs, boosted):
            token = snap.token_address
            if token in held or self._cooldown.get(token, 0) > now:
                continue
            if self.journal.recently_seen(token, REANALYSE_AFTER_S):
                continue
            if factors.market_filters(snap, self.cfg):
                continue
            safety = self.rug.report(token)
            result = factors.evaluate(snap, safety, self.cfg)
            if not result.passed:
                self._cooldown[token] = now + RECHECK_REJECTED_S
                self.journal.log_decision(snap, result)
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
            slots = self.cfg.max_open_positions - len(self.journal.open_positions(MAIN))
            for snap, result, safety in survivors:
                if slots <= 0 or not self.analyst.budget_ok():
                    break
                verdict = self.analyst.analyse(snap, safety, result)
                cost = getattr(self.analyst, "last_cost_usd", 0.0)
                if cost:
                    self.journal.log_cost("claude_api", cost, snap.token_address, MAIN)
                if verdict is None:
                    self._cooldown[snap.token_address] = time.time() + RECHECK_REJECTED_S
                    continue
                decision_id = self.journal.log_decision(snap, result, verdict, analysed=True)
                analysed.append(((snap, result, safety), verdict.action))
                log.info("%s -> %s (conviction %.2f): %s",
                         snap.symbol, verdict.action, verdict.conviction, verdict.thesis)
                if verdict.action == "buy" and verdict.conviction >= self.cfg.min_conviction:
                    if self.buy(main, snap, verdict.conviction, decision_id):
                        slots -= 1
        self.after_screen(survivors, analysed)

    def after_screen(self, survivors: list[Survivor], analysed: list[tuple[Survivor, str]]) -> None:
        """Hook for baseline books (see shadows.py). The main engine does nothing here."""

    def preflight(self, snap: PairSnapshot, usd: float) -> tuple[PairSnapshot | None, str]:
        """Re-validate a token right before buying. Research can take minutes.

        Returns (fresh snapshot, "") or (None, reason to skip).
        """
        fresh = self.dex.pairs([snap.token_address])
        if not fresh:
            return None, "pair disappeared"
        now = fresh[0]
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

    def buy(self, book: Book, snap: PairSnapshot, conviction: float, decision_id: int | None) -> bool:
        if any(p.token_address == snap.token_address for p in self.journal.open_positions(book.name)):
            return False
        usd = risk.position_size_usd(self.equity(book), book.cash, conviction, self.cfg)
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
            self.journal.log_event("entry_skipped", snap.token_address, "executor declined", strategy=book.name)
            return False
        if not fill.ok:
            self._charge_failed(book, snap.token_address, fill)
            return False
        with self.lock:
            book.cash -= fill.usd
            self.journal.set_cash(book.cash, book.name)
            pos = Position(
                id=None, token_address=snap.token_address, symbol=snap.symbol, entry_price=fill.price,
                qty=fill.qty, initial_qty=fill.qty, cost_usd=fill.usd, opened_at=time.time(),
                stop_price=risk.initial_stop(fill.price, self.cfg), peak_price=fill.price, strategy=book.name,
            )
            self.journal.save_position(pos, decision_id)
            self.journal.log_fill(pos.id, "buy", fill.qty, fill.price, fill.usd, fill.fee_usd, "entry", fill.tx,
                                  trigger_price=fresh.price_usd)
        log.info("[%s] BUY %s $%.2f @ %.10g (stop %.10g)", book.name, snap.symbol, fill.usd, fill.price, pos.stop_price)
        return True

    def _sell(self, book: Book, pos: Position, qty: float, price: float, reason: str) -> bool:
        fill = book.executor.sell(pos.token_address, min(qty, pos.qty), price)
        if fill is None or not fill.ok:
            if fill is not None:
                self._charge_failed(book, pos.token_address, fill)
            log.error("[%s] sell failed for %s (%s); will retry", book.name, pos.symbol, reason)
            return False
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

    def _charge_failed(self, book: Book, token: str, fill: Fill) -> None:
        """A reverted transaction still costs its network fee."""
        with self.lock:
            book.cash -= fill.usd
            self.journal.set_cash(book.cash, book.name)
        self.journal.log_event("tx_failed", token, fill.note, fill.usd, strategy=book.name)
        log.warning("[%s] transaction failed for %s: %s (fee $%.4f)", book.name, token, fill.note, fill.usd)

    # --- measurement ----------------------------------------------------------
    def update_outcomes(self) -> None:
        pending = self.journal.pending_outcomes(OUTCOME_HORIZON_S)
        if not pending:
            return
        prices = {s.token_address: s.price_usd for s in self.dex.pairs({r["token"] for r in pending})}
        for row in pending:
            # A token that vanished from DexScreener is treated as a total loss.
            self.journal.set_outcome(row["id"], prices.get(row["token"]))
