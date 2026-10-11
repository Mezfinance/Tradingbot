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
