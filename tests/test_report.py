from dataclasses import replace
from types import SimpleNamespace

import pytest

from tradingbot import report
from tradingbot.costs import usage_cost_usd
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tests.test_analyst_engine import BUY_JSON, FakeClient, FakeDex, resp
from tradingbot.analyst import Analyst


def test_usage_cost_includes_web_searches():
    usage = SimpleNamespace(input_tokens=100_000, output_tokens=10_000, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0,
                            server_tool_use=SimpleNamespace(web_search_requests=5))
    # 0.1M * $4 + 0.01M * $20 + 5 * $0.01
    assert usage_cost_usd("claude-opus-5-5", usage) == pytest.approx(0.4 + 0.2 + 0.05)


def test_max_drawdown():
    assert report.max_drawdown_pct([100, 120, 90, 130, 117]) == pytest.approx(25.0)


def test_bootstrap_detects_clear_gap_and_not_noise():
    lo, hi = report.bootstrap_diff_ci([10, 12, 11, 9, 13, 10] * 5, [0, 1, -1, 0, 2, -2] * 5)
    assert lo > 0
    lo, hi = report.bootstrap_diff_ci([1, -1, 2, -2] * 5, [1, -1, 2, -2] * 5)
    assert lo < 0 < hi


def test_report_end_to_end(cfg, good_pair, safe):
    journal = Journal(":memory:")
    client = FakeClient([resp("end_turn", BUY_JSON)])
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, client), PaperExecutor(cfg), journal)
    eng.tick()
    eng.dex.pair = replace(good_pair, price_usd=good_pair.price_usd * 0.5)
    eng.manage_positions()
    eng.snapshot_equity({}, force=True)

    rep = report.build(journal, cfg)
    [claude] = rep["strategies"]
    assert claude["closed_trades"] == 1 and claude["win_rate"] == 0
    assert claude["exit_reasons"] == {"stop_loss": 1}
    assert claude["claude_api_usd"] > 0
    assert claude["net_after_api_usd"] < claude["equity"] - cfg.starting_cash_usd + 1e-9
    assert claude["median_exit_slippage_pct"] == pytest.approx(3.0)  # PaperExecutor slippage
    text = report.render(rep)
    assert "STRATEGY COMPARISON" in text and "need 30+" in text


def test_model_book_exit_slippage_leaves_out_rugs_and_bad_trigger_prices(tmp_path):
    import time
    from tradingbot.config import Config
    from tradingbot.models import Position
    cfg = Config(db_path=str(tmp_path / "bot.db"))
    j = Journal(cfg.db_path)
    # sells: a normal 2% slip, a rug (fill 1% of trigger), a bad trigger (fill 10^7 x the trigger)
    for k, (price, trigger) in enumerate([(0.98, 1.0), (0.01, 1.0), (1.0, 1e-7)]):
        pos = Position(id=None, token_address=f"T{k}", symbol=f"T{k}", entry_price=1.0, qty=1.0, initial_qty=1.0,
                       cost_usd=100.0, opened_at=time.time(), stop_price=0.5, peak_price=1.0, strategy="launchd12")
        j.save_position(pos, None)
        j.log_fill(pos.id, "buy", 1.0, 1.0, 100.0, 0.0, "entry", "sim", trigger_price=1.0)
        j.log_fill(pos.id, "sell", 1.0, price, 100.0 * price, 0.0, "pool_age_exit", "sim", trigger_price=trigger)
    out = report._model_book_report(j, cfg, "launchd12")
    assert out["book"]["median_exit_slippage_excl_rugs_pct"] == pytest.approx(2.0)
    assert out["book"]["exits_with_bad_trigger"] == 1
