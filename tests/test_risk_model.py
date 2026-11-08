"""risk-model: risk-adjusted stats per book and the trade-size simulation."""
import json

import numpy as np
import pytest

from tradingbot import launch_live as ll
from tradingbot import risk_model as rm
from tradingbot.journal import Journal
from tradingbot.watcher import WatchStore

D = 86400.0
T0 = 1_700_006_400.0


def _book(j, st, name, rets, liq=20_000.0, cost=100.0, per_day=10):
    for i, r in enumerate(rets):
        ts = T0 + (i // per_day) * D + (i % per_day) * 600
        tok = f"{name}{i}"
        j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
                  "VALUES (?,?,?,?,0,?,?,?)", (tok, tok, cost, cost * (1 + r), ts, ts + 360, name))
        if st is not None:
            st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, status, liq_usd, feat) "
                       "VALUES (?,?,?,?,?,?)", (tok, tok, ts - 10, "done", liq, json.dumps({})))


def _cfg(cfg):
    return cfg.with_overrides({"model_book_cash_usd": "10000", "model_book_liq_share": "0.02"})


def test_stats_and_sizing_on_a_steady_winner(cfg):
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    rng = np.random.default_rng(3)
    _book(j, st, "launchrw50_sz", list(rng.normal(0.05, 0.3, 200).clip(-1, 3)))
    out = rm.model(j, st, _cfg(cfg), books=["launchrw50_sz"], now=T0 + 30 * D, sims=60)
    s = out["books"]["launchrw50_sz"]["stats"]
    assert s["trades"] == 200 and s["days"] == 20 and s["with_liquidity_pct"] == 100
    assert s["sharpe"] is not None and s["max_drawdown_pct"] <= 0
    rows = out["books"]["launchrw50_sz"]["sizing"]
    small = next(r for r in rows if r["capital"] == 10_000 and r["pct_per_trade"] == 0.25)
    big = next(r for r in rows if r["capital"] == 1_000_000 and r["pct_per_trade"] == 10.0)
    # $1M x 10% = $100k a trade against a $400 cap (2% of $20k): every trade is capped
    assert big["capped_pct"] == 100 and small["capped_pct"] == 0
    assert "SUPERMEME RISK MODEL" in rm.render(out)


def test_bigger_trades_pay_impact_and_stuck_trades_count(cfg):
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    _book(j, st, "launchrw50", [0.10] * 40)
    ts = rm.trades(j, st, "launchrw50", 0, T0 + 30 * D, True)
    pnl, capped = rm.day_tables(ts, 0.02)
    i100 = int(np.argmin(abs(rm.GRID - 100))); i400 = int(np.searchsorted(rm.GRID, 400)) - 1   # just under the $400 cap (2% of $20k)
    per_usd_100 = pnl[0, i100] / (10 * rm.GRID[i100]); per_usd_400 = pnl[0, i400] / (10 * rm.GRID[i400])
    assert per_usd_100 == pytest.approx(0.10, abs=0.002)        # the size really traded: unchanged
    assert per_usd_400 < per_usd_100                              # bigger: extra impact
    assert per_usd_400 == pytest.approx(0.10 - 4 * (rm.GRID[i400] - 100) / 20_000, abs=0.002)
    j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
              "VALUES ('S','S',100,0,0,?,NULL,'launchrw50')", (T0 + 5 * D,))
    assert rm.trades(j, st, "launchrw50", 0, T0 + 30 * D, True)[-1]["ret"] == -1.0
    assert len(rm.trades(j, st, "launchrw50", 0, T0 + 5 * D + 60, True)) == 40   # just bought: not counted yet


def test_correlation_portfolio_and_too_few_trades(cfg):
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    rng = np.random.default_rng(1)
    a = list(rng.normal(0.03, 0.2, 100).clip(-1, 3))
    _book(j, st, "launchrw50", a)
    _book(j, st, "launchrw50_sz", a)                              # identical trades: correlation 1
    _book(j, None, "early6", [0.1] * 5)
    out = rm.model(j, st, _cfg(cfg), books=["launchrw50", "launchrw50_sz", "early6"], now=T0 + 30 * D, sims=20)
    assert out["too few trades"] == {"early6": 5}
    assert out["correlation"]["launchrw50"]["launchrw50_sz"] == pytest.approx(1.0)
    assert out["equal_weight_portfolio"]["books"] == 2


def test_sizing_keeps_a_sized_books_own_size_split(cfg):
    """A book that trades $250 in thin winning pools and $25 in losing mid pools makes money at its own sizes;
    giving every trade one size (the old simulation) turned it into a loser."""
    st, j = WatchStore(":memory:"), Journal(":memory:")
    ll.ensure_schema(st)
    _book(j, st, "launchrw50_sz", [0.30] * 20, liq=12_000.0, cost=250.0, per_day=2)     # thin: 2 a day
    for i in range(80):                                                                # mid: 8 a day, losing
        ts = T0 + (i // 8) * D + 300 + i * 30
        tok = f"mid{i}"
        j.execute("INSERT INTO positions(token, symbol, cost_usd, realized_usd, qty, opened_at, closed_at, strategy) "
                  "VALUES (?,?,25,?,0,?,?,'launchrw50_sz')", (tok, tok, 25 * 0.9, ts, ts + 360))
        st.execute("INSERT INTO watch_launch_live(pool, token_mint, create_ts, status, liq_usd, feat) "
                   "VALUES (?,?,?,?,?,?)", (tok, tok, ts - 10, "done", 20_000.0, json.dumps({})))
    ts = rm.trades(j, st, "launchrw50_sz", 0, T0 + 30 * D, True)
    pnl, _ = rm.day_tables(ts, 0.02)
    mean_cost = (2 * 250 + 8 * 25) / 10
    i = int(np.argmin(abs(rm.GRID - mean_cost)))
    real_day = 2 * 250 * 0.30 - 8 * 25 * 0.10                                          # +130 $ a day
    assert pnl[0, i] == pytest.approx(real_day * rm.GRID[i] / mean_cost, rel=0.05)
    assert pnl[0, i] > 0
    out = rm.model(j, st, _cfg(cfg), books=["launchrw50_sz"], now=T0 + 30 * D, sims=40)
    row = next(r for r in out["books"]["launchrw50_sz"]["sizing"] if r["capital"] == 10_000 and r["pct_per_trade"] == 0.5)
    assert row["ret_median"] > 0            # $50 average trade, the same split: a winner, as the book really was
    assert out["books"]["launchrw50_sz"]["stats"]["actual_trade_usd"] == pytest.approx(mean_cost)
