"""Position sizing and exit rules. Pure and deterministic, no I/O."""

from __future__ import annotations

from tradingbot.config import Config
from tradingbot.factors import ramp
from tradingbot.models import Position


def position_size_usd(equity: float, cash: float, conviction: float, cfg: Config) -> float:
    """Size so that hitting the stop loses about risk_per_trade_pct of equity.

    Conviction scales the size between 50% and 100% of that amount; it can never
    push past the per-position cap or available cash.
    """
    risk_usd = equity * cfg.risk_per_trade_pct / 100
    full = risk_usd / (cfg.stop_loss_pct / 100)
    scale = 0.5 + 0.5 * ramp(conviction, cfg.min_conviction, 1.0)
    cap = equity * cfg.max_position_pct / 100
    return max(0.0, min(full * scale, cap, cash))


def initial_stop(entry_price: float, cfg: Config) -> float:
    return entry_price * (1 - cfg.stop_loss_pct / 100)


def plan_exits(pos: Position, price: float, now: float, cfg: Config) -> list[tuple[float, str]]:
    """Return sell orders as (token_qty, reason). Updates pos.peak_price, tp_hits and stop_price.

    Order of precedence: hard stop, time stop, trailing stop, then take-profit ladder.
    """
    if pos.qty <= 0:
        return []
    pos.peak_price = max(pos.peak_price, price)

    if price <= pos.stop_price:
        return [(pos.qty, "stop_loss" if pos.tp_hits == 0 else "breakeven_stop")]
    if now - pos.opened_at >= cfg.max_hold_hours * 3600:
        return [(pos.qty, "time_stop")]
    if pos.tp_hits > 0 and price <= pos.peak_price * (1 - cfg.trailing_stop_pct / 100):
        return [(pos.qty, "trailing_stop")]

    orders: list[tuple[float, str]] = []
    remaining = pos.qty
    while pos.tp_hits < len(cfg.take_profit_pcts):
        target = pos.entry_price * (1 + cfg.take_profit_pcts[pos.tp_hits] / 100)
        if price < target:
            break
        last = pos.tp_hits == len(cfg.take_profit_pcts) - 1
        qty = remaining if last else min(remaining, pos.initial_qty * cfg.take_profit_fractions[pos.tp_hits])
        pos.tp_hits += 1
        orders.append((qty, f"take_profit_{pos.tp_hits}"))
        remaining -= qty
        # After the first target is banked, the rest of the trade cannot lose money.
        pos.stop_price = max(pos.stop_price, pos.entry_price)
        if remaining <= 0:
            break
    return orders
