import pytest

from tradingbot import risk
from tradingbot.execution.paper import PaperExecutor
from tradingbot.models import Position


def make_pos(cfg, entry=1.0, qty=100.0, opened_at=0.0):
    return Position(id=1, token_address="M", symbol="M", entry_price=entry, qty=qty, initial_qty=qty,
                    cost_usd=entry * qty, opened_at=opened_at, stop_price=risk.initial_stop(entry, cfg),
                    peak_price=entry)


def test_size_respects_risk_budget(cfg):
    # 2% of 1000 at a 25% stop = $80 at full conviction; capped by 10% = $100
    assert risk.position_size_usd(1000, 1000, 1.0, cfg) == pytest.approx(80)
    assert risk.position_size_usd(1000, 1000, cfg.min_conviction, cfg) == pytest.approx(40)


def test_size_capped_by_cash_and_max_position(cfg):
    assert risk.position_size_usd(1000, 15, 1.0, cfg) == 15
    cfg.risk_per_trade_pct = 50
    assert risk.position_size_usd(1000, 1000, 1.0, cfg) == pytest.approx(100)


def test_stop_loss(cfg):
    pos = make_pos(cfg)
    assert risk.plan_exits(pos, 0.74, 10, cfg) == [(100.0, "stop_loss")]


def test_take_profit_ladder_and_breakeven(cfg):
    pos = make_pos(cfg)
    orders = risk.plan_exits(pos, 1.6, 10, cfg)
    assert orders == [(40.0, "take_profit_1")]
    assert pos.stop_price == 1.0  # moved to breakeven
    pos.qty -= 40


def test_gap_through_all_targets_sells_everything(cfg):
    pos = make_pos(cfg)
    orders = risk.plan_exits(pos, 6.0, 10, cfg)
    assert [r for _, r in orders] == ["take_profit_1", "take_profit_2", "take_profit_3"]
    assert sum(q for q, _ in orders) == pytest.approx(100)


def test_trailing_stop_after_first_target(cfg):
    pos = make_pos(cfg)
    risk.plan_exits(pos, 1.6, 10, cfg)
    pos.qty = 60
    risk.plan_exits(pos, 2.2, 20, cfg)  # new peak, below TP2 (2.5)
    assert risk.plan_exits(pos, 1.5, 30, cfg) == [(60, "trailing_stop")]


def test_time_stop(cfg):
    pos = make_pos(cfg)
    assert risk.plan_exits(pos, 1.1, cfg.max_hold_hours * 3600 + 1, cfg) == [(100.0, "time_stop")]


def test_paper_round_trip_loses_costs(cfg):
    ex = PaperExecutor(cfg)
    buy = ex.buy("M", 100, 1.0)
    sell = ex.sell("M", buy.qty, 1.0)
    assert sell.usd < 100  # slippage + fees on both legs
    assert sell.usd == pytest.approx(100 * 0.99 / 1.03 * 0.97 * 0.99)
