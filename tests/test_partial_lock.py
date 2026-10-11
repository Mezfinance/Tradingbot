import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tradingbot import factors
from tradingbot.analyst import Analyst, build_brief
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tradingbot.learning import optimizer
from tradingbot.models import SafetyReport
from tests.test_analyst_engine import BUY_JSON, FakeClient, FakeDex, resp


def lock(pct, score=300, danger=False):
    return SafetyReport(score=score, lp_locked_pct=pct, risks=["danger: x"] if danger else [], danger=danger)


@pytest.fixture
def weak_pair(good_pair):
    """Fresh, passes the market filters, but only middling factors (like Altai when it was checked)."""
    return replace(good_pair, buys={"h1": 520}, sells={"h1": 480}, price_change={"m5": 1, "h1": 6},
                   volume={"m5": 2_000, "h1": 30_000})


def test_strong_fresh_token_passes_with_partial_lock(cfg, good_pair):
    r = factors.evaluate(good_pair, lock(66), cfg)
    assert r.score >= cfg.partial_lock_min_score
    assert r.passed and r.partial_lock


def test_full_lock_is_not_flagged_as_exception(cfg, good_pair):
    r = factors.evaluate(good_pair, lock(100), cfg)
    assert r.passed and not r.partial_lock


@pytest.mark.parametrize("safety, why", [
    (lock(45), "below the 50% floor"),
    (lock(66, score=1_500), "RugCheck not clean enough for an exception"),
    (lock(66, danger=True), "danger flag"),
])
def test_exception_needs_everything_else_clean(cfg, good_pair, safety, why):
    r = factors.evaluate(good_pair, safety, cfg)
    assert not r.passed and not r.partial_lock, why


def test_weak_token_gets_no_exception(cfg, weak_pair):
    r = factors.evaluate(weak_pair, lock(66), cfg)
    assert r.score < cfg.partial_lock_min_score
    assert not r.passed and any("LP locked 66% < 80%" in x for x in r.rejections)


def test_established_tokens_keep_their_own_rule(cfg, good_pair):
    old_deep = replace(good_pair, liquidity_usd=300_000,
                       created_at_ms=int((good_pair.fetched_at - 10 * 86400) * 1000))
    r = factors.evaluate(old_deep, lock(40), cfg)
    assert r.profile == "established" and not r.partial_lock


def test_exception_trades_at_half_size_and_claude_is_told(cfg, good_pair):
    safety = lock(66)
    full = Journal(":memory:")
    Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: lock(100)),
           Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)])), PaperExecutor(cfg), full).tick()
    client = FakeClient([resp("end_turn", BUY_JSON)])
    half = Journal(":memory:")
    Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safety),
           Analyst(cfg, client), PaperExecutor(cfg), half).tick()
    [a], [b] = full.open_positions(), half.open_positions()
    assert b.cost_usd == pytest.approx(a.cost_usd * cfg.partial_lock_size_factor)
    assert '"partial_lp_lock_exception": true' in client.calls[0]["messages"][0]["content"]
    snap = json.loads(half.query("SELECT snapshot FROM decisions WHERE stage='analysed'")[0]["snapshot"])
    assert snap["partial_lock"] is True


def test_replay_applies_the_same_rule(cfg, good_pair):
    j = Journal(":memory:")
    r = factors.evaluate(good_pair, lock(66), cfg)
    j.log_decision(good_pair, r, safety=lock(66))
    row = {**dict(j.query("SELECT * FROM decisions")[0]), "action": "buy", "conviction": 0.9}
    assert optimizer.passes(row, cfg)
    assert not optimizer.passes(row, optimizer.with_changes(cfg, {"partial_lock_min_pct": 70}))
    assert not optimizer.passes({**row, "factor_score": 0.6}, cfg)
