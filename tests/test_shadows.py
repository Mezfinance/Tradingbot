import random
from types import SimpleNamespace

import pytest

from tradingbot import report, shadows
from tradingbot.analyst import Analyst
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tests.test_analyst_engine import BUY_JSON, FakeClient, FakeDex, resp

PASS_JSON = BUY_JSON.replace('"buy"', '"pass"')


def test_parse_shadows():
    assert shadows.parse_shadows("filters_only, random") == ["filters_only", "random"]
    assert shadows.parse_shadows("") == []
    with pytest.raises(ValueError):
        shadows.parse_shadows("moonshot")


def test_buy_rate_prior_then_data(cfg):
    j = Journal(":memory:")
    assert shadows.claude_buy_rate(j, cfg) == pytest.approx(0.25)
    for i in range(80):
        j.db.execute("INSERT INTO decisions(ts, token, stage, action, conviction) VALUES(0, ?, 'analysed', ?, 0.9)",
                     (f"T{i}", "buy" if i < 40 else "pass"))
    assert shadows.claude_buy_rate(j, cfg) == pytest.approx((40 + 5) / 100)


def test_random_picks_respect_rate():
    analysed = [((f"s{i}", None, None), "pass") for i in range(2000)]
    got = shadows.picks("random", [], analysed, 0.3, random.Random(1))
    assert 500 < len(got) < 700


def make_engine(cfg, pair, safe, client):
    j = Journal(":memory:")
    eng = Engine(cfg, FakeDex(pair), SimpleNamespace(report=lambda m: safe), Analyst(cfg, client),
                 PaperExecutor(cfg), j,
                 shadow_executors={"filters_only": PaperExecutor(cfg), "random": PaperExecutor(cfg)})
    return eng, j


def test_filters_only_buys_even_when_claude_passes(cfg, good_pair, safe):
    eng, j = make_engine(cfg, good_pair, safe, FakeClient([resp("end_turn", PASS_JSON)]))
    eng.tick()
    assert j.open_positions("claude") == []
    assert len(j.open_positions("filters_only")) == 1


def test_books_hold_same_token_independently(cfg, good_pair, safe):
    eng, j = make_engine(cfg, good_pair, safe, FakeClient([resp("end_turn", BUY_JSON)]))
    eng.tick()
    assert len(j.open_positions("claude")) == 1
    assert len(j.open_positions("filters_only")) == 1
    assert eng.books["claude"].cash < cfg.starting_cash_usd
    assert eng.books["filters_only"].cash < cfg.starting_cash_usd


def test_report_compares_books(cfg, good_pair, safe):
    eng, j = make_engine(cfg, good_pair, safe, FakeClient([resp("end_turn", PASS_JSON)]))
    eng.tick()
    eng.snapshot_equity({}, force=True)
    rep = report.build(j, cfg)
    assert [s["strategy"] for s in rep["strategies"]][0] == "claude"
    assert {s["strategy"] for s in rep["strategies"]} == {"claude", "filters_only", "random"}
    assert "claude vs filters_only" in report.render(rep)
