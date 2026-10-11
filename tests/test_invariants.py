"""Safety invariants. The self-writing engineer cannot edit existing tests, and every code
change must pass the full suite, so these hold for every version the bot ships itself."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tradingbot import factors, risk
from tradingbot.analyst import SYSTEM_PROMPT, Analyst
from tradingbot.config import LIVE_CONFIRM_PHRASE, Config
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tradingbot.learning import engineer
from tradingbot.learning.mission import MISSION
from tradingbot.models import Position, SafetyReport
from tests.test_analyst_engine import BUY_JSON, FakeClient, FakeDex, resp


def test_analyst_keeps_mission_and_untrusted_content_rule():
    assert SYSTEM_PROMPT.startswith(MISSION)
    assert "untrusted" in SYSTEM_PROMPT.lower()


@pytest.mark.parametrize("safety", [
    None,
    SafetyReport(score=1, lp_locked_pct=100, risks=["danger: Freeze authority"], danger=True),
    SafetyReport(score=1, lp_locked_pct=10, risks=[], danger=False),
    SafetyReport(score=999_999, lp_locked_pct=100, risks=[], danger=False),
])
def test_unsafe_tokens_are_always_rejected(cfg, good_pair, safety):
    assert not factors.evaluate(good_pair, safety, cfg).passed


@pytest.mark.parametrize("safety", [
    None,
    SafetyReport(score=1, lp_locked_pct=100, risks=["danger: Mint authority"], danger=True),
    SafetyReport(score=1, lp_locked_pct=10, risks=[], danger=False),  # below even the established floor
    SafetyReport(score=999_999, lp_locked_pct=100, risks=[], danger=False),
])
def test_unsafe_established_tokens_are_always_rejected(cfg, good_pair, safety):
    old_deep = replace(good_pair, liquidity_usd=500_000,
                       created_at_ms=int((good_pair.fetched_at - 10 * 86400) * 1000))
    assert factors.is_established(old_deep, cfg)
    assert not factors.evaluate(old_deep, safety, cfg).passed


def test_established_lp_floor_never_below_30_pct():
    assert Config().established_min_lp_locked_pct >= 30
    from tradingbot.learning.params import TUNABLES
    assert TUNABLES["established_min_lp_locked_pct"].lo >= 30


def test_no_liquidity_is_always_rejected(cfg, good_pair, safe):
    assert not factors.evaluate(replace(good_pair, liquidity_usd=None), safe, cfg).passed


@pytest.mark.parametrize("conviction", [0.0, 0.65, 1.0, 5.0])
def test_position_size_never_exceeds_caps(cfg, conviction):
    for equity, cash in ((1000, 1000), (1000, 20), (50, 50)):
        usd = risk.position_size_usd(equity, cash, conviction, cfg)
        assert 0 <= usd <= min(cash, equity * cfg.max_position_pct / 100)


def test_stop_always_fires(cfg):
    pos = Position(id=1, token_address="M", symbol="M", entry_price=1.0, qty=10, initial_qty=10, cost_usd=10,
                   opened_at=0, stop_price=risk.initial_stop(1.0, cfg), peak_price=1.0)
    assert pos.stop_price < 1.0
    assert risk.plan_exits(pos, 0.01, 1, cfg) == [(10, "stop_loss")]


def test_engine_never_buys_a_honeypot(cfg, good_pair, safe):
    honeypot = SimpleNamespace(round_trip=lambda token, usd: (60.0, 0.5),
                               prices=lambda mints: {})
    j = Journal(":memory:")
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)])), PaperExecutor(cfg), j, jupiter=honeypot)
    eng.tick()
    assert j.open_positions() == []


def test_engine_never_buys_without_a_safety_report(cfg, good_pair):
    j = Journal(":memory:")
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None),
                 Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)])), PaperExecutor(cfg), j)
    eng.tick()
    assert j.open_positions() == []


def test_live_trading_stays_locked():
    with pytest.raises(ValueError):
        Config(mode="live").validate()
    with pytest.raises(ValueError):
        Config(canary_every_n=1).validate()
    Config(mode="live", live_confirm=LIVE_CONFIRM_PHRASE).validate()


def test_engineer_cannot_unprotect_its_guardrails():
    for path in ("tradingbot/risk.py", "tradingbot/config.py", "tradingbot/execution/", "tradingbot/learning/",
                 "tradingbot/journal.py", "tradingbot/cli.py"):
        assert path in engineer.PROTECTED
    assert engineer.is_protected("tests/test_invariants.py", {"tests/test_invariants.py"})
