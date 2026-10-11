"""The trading loop: manage open positions, then scan, screen, research and enter."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from tradingbot import factors, risk
from tradingbot.analyst import Analyst
from tradingbot.config import Config
from tradingbot.data.dexscreener import DexScreener
from tradingbot.data.rugcheck import RugCheck
from tradingbot.execution.base import Executor
from tradingbot.journal import Journal
from tradingbot.models import FactorResult, PairSnapshot, Position

log = logging.getLogger(__name__)

REANALYSE_AFTER_S = 6 * 3600
RECHECK_REJECTED_S = 30 * 60
OUTCOME_HORIZON_S = 6 * 3600
DUST_QTY = 1e-12


class Engine:
    def __init__(self, cfg: Config, dex: DexScreener, rug: RugCheck, analyst: Analyst | None,
                 executor: Executor, journal: Journal):
        self.cfg, self.dex, self.rug, self.analyst = cfg, dex, rug, analyst
        self.executor, self.journal = executor, journal
        self.cash = journal.get_cash(cfg.starting_cash_usd)
        self._cooldown: dict[str, float] = {}  # token -> time it may be re-screened

    # --- top level ----------------------------------------------------------
    def tick(self) -> None:
        prices = self.manage_positions()
        if self.entries_allowed(prices):
            self.scan_and_enter()
        self.update_outcomes()

    def run_forever(self) -> None:
        while True:
            started = time.time()
            try:
                self.tick()
            except Exception:  # keep the loop alive; positions are persisted
                log.exception("tick failed")
            time.sleep(max(0.0, self.cfg.scan_interval_s - (time.time() - started)))

    # --- positions ----------------------------------------------------------
    def manage_positions(self) -> dict[str, float]:
        positions = self.journal.open_positions()
        if not positions:
            return {}
        prices = {s.token_address: s.price_usd for s in self.dex.pairs(p.token_address for p in positions)}
        now = time.time()
        for pos in positions:
            price = prices.get(pos.token_address)
            if price is None:
                log.warning("no price for open position %s", pos.symbol)
                continue
            before = (pos.tp_hits, pos.stop_price)
            for qty, reason in risk.plan_exits(pos, price, now, self.cfg):
                if not self._sell(pos, qty, price, reason):
                    # Roll the ladder back so the missed target is retried next tick.
                    pos.tp_hits, pos.stop_price = before
                    break
            self.journal.save_position(pos)
        return prices

    def equity(self, prices: dict[str, float]) -> float:
        held = sum(p.qty * prices.get(p.token_address, p.entry_price) for p in self.journal.open_positions())
        return self.cash + held

    def entries_allowed(self, prices: dict[str, float]) -> bool:
        if len(self.journal.open_positions()) >= self.cfg.max_open_positions:
            return False
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        loss_today = -self.journal.realized_since(day_start)
        if loss_today >= self.equity(prices) * self.cfg.daily_loss_limit_pct / 100:
            log.warning("daily loss limit hit (%.2f USD); no new entries today", loss_today)
            return False
        return True

    # --- entries ------------------------------------------------------------
    def screen(self) -> list[tuple[PairSnapshot, FactorResult, object]]:
        """Discover tokens and return those that clear every deterministic gate, best first."""
        addrs, boosted = self.dex.discover()
        held = {p.token_address for p in self.journal.open_positions()}
        survivors = []
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
        if self.analyst is None:
            return
        slots = self.cfg.max_open_positions - len(self.journal.open_positions())
        for snap, result, safety in self.screen():
            if slots <= 0 or not self.analyst.budget_ok():
                break
            verdict = self.analyst.analyse(snap, safety, result)
            if verdict is None:
                self._cooldown[snap.token_address] = time.time() + RECHECK_REJECTED_S
                continue
            decision_id = self.journal.log_decision(snap, result, verdict, analysed=True)
            log.info("%s -> %s (conviction %.2f): %s", snap.symbol, verdict.action, verdict.conviction, verdict.thesis)
            if verdict.action == "buy" and verdict.conviction >= self.cfg.min_conviction:
                if self._buy(snap, verdict.conviction, decision_id):
                    slots -= 1

    def _buy(self, snap: PairSnapshot, conviction: float, decision_id: int) -> bool:
        prices = {p.token_address: p.entry_price for p in self.journal.open_positions()}
        usd = risk.position_size_usd(self.equity(prices), self.cash, conviction, self.cfg)
        if usd < 1:
            log.info("position size too small for %s", snap.symbol)
            return False
        fill = self.executor.buy(snap.token_address, usd, snap.price_usd)
        if fill is None:
            return False
        self.cash -= fill.usd
        self.journal.set_cash(self.cash)
        pos = Position(
            id=None, token_address=snap.token_address, symbol=snap.symbol, entry_price=fill.price,
            qty=fill.qty, initial_qty=fill.qty, cost_usd=fill.usd, opened_at=time.time(),
            stop_price=risk.initial_stop(fill.price, self.cfg), peak_price=fill.price,
        )
        self.journal.save_position(pos, decision_id)
        self.journal.log_fill(pos.id, "buy", fill.qty, fill.price, fill.usd, fill.fee_usd, "entry", fill.tx)
        log.info("BUY %s $%.2f @ %.10g (stop %.10g)", snap.symbol, fill.usd, fill.price, pos.stop_price)
        return True

    def _sell(self, pos: Position, qty: float, price: float, reason: str) -> bool:
        fill = self.executor.sell(pos.token_address, min(qty, pos.qty), price)
        if fill is None:
            log.error("sell failed for %s (%s); will retry next tick", pos.symbol, reason)
            return False
        pos.qty -= fill.qty
        pos.realized_usd += fill.usd
        self.cash += fill.usd
        self.journal.set_cash(self.cash)
        self.journal.log_fill(pos.id, "sell", fill.qty, fill.price, fill.usd, fill.fee_usd, reason, fill.tx)
        if pos.qty <= DUST_QTY * pos.initial_qty:
            pos.qty = 0.0
            pos.closed_at = time.time()
            pos.exit_reason = reason
        log.info("SELL %s %s qty=%.6g @ %.10g -> $%.2f", pos.symbol, reason, fill.qty, fill.price, fill.usd)
        return True

    # --- measurement ----------------------------------------------------------
    def update_outcomes(self) -> None:
        pending = self.journal.pending_outcomes(OUTCOME_HORIZON_S)
        if not pending:
            return
        prices = {s.token_address: s.price_usd for s in self.dex.pairs({r["token"] for r in pending})}
        for row in pending:
            # A token that vanished from DexScreener is treated as a total loss.
            self.journal.set_outcome(row["id"], prices.get(row["token"]))
