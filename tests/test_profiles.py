import json
from dataclasses import replace

import pytest

from tradingbot import factors
from tradingbot.journal import Journal
from tradingbot.learning import optimizer
from tradingbot.models import SafetyReport

DAY_MS = 86_400_000


@pytest.fixture
def established(good_pair):
    """A 10-day-old, deep pool in a steady, broad uptrend (like the trending tokens we saw live)."""
    return replace(
        good_pair, liquidity_usd=370_000, fdv_usd=4_000_000,
        created_at_ms=int(good_pair.fetched_at * 1000) - 10 * DAY_MS,
        volume={"m5": 20_000, "h1": 330_000, "h6": 1_500_000, "h24": 2_400_000},
        price_change={"m5": 0.5, "h1": 3.0, "h6": 18.0, "h24": 45.0},
        buys={"h1": 900, "h6": 5_200, "h24": 11_000}, sells={"h1": 800, "h6": 4_100, "h24": 9_500},
    )


def partial_lock(pct):
    return SafetyReport(score=500, lp_locked_pct=pct, risks=[], danger=False)


def test_profile_needs_both_age_and_depth(cfg, good_pair, established):
    assert factors.profile(established, cfg) == "established"
    assert factors.profile(replace(established, liquidity_usd=60_000), cfg) == "fresh"
    two_days = replace(established, created_at_ms=int(established.fetched_at * 1000) - 2 * DAY_MS)
    assert factors.profile(two_days, cfg) == "fresh"


def test_partial_lock_ok_only_for_established(cfg, established):
    result = factors.evaluate(established, partial_lock(33), cfg)
    assert result.passed, result.rejections
    assert result.profile == "established" and "trend" in result.components

    two_days = replace(established, created_at_ms=int(established.fetched_at * 1000) - 2 * DAY_MS)
    assert any("LP locked 33% < 80%" in r for r in factors.evaluate(two_days, partial_lock(33), cfg).rejections)
    assert any("LP locked 25% < 30%" in r for r in factors.evaluate(established, partial_lock(25), cfg).rejections)


def test_trend_score_prefers_steady_climbs(established):
    steady = factors.trend_components(established)
    blow_off = factors.trend_components(replace(established, price_change={**established.price_change, "h24": 550}))
    sliding = factors.trend_components(replace(established, price_change={**established.price_change, "h1": -9}))
    falling = factors.trend_components(replace(established, price_change={**established.price_change, "h24": -20}))
    assert steady["trend"] == 1.0
    assert blow_off["trend"] < steady["trend"]
    assert sliding["holding"] < steady["holding"]
    assert falling["trend"] == 0.0


def test_fresh_tokens_keep_spike_scoring(cfg, good_pair, safe):
    r = factors.evaluate(good_pair, safe, cfg)
    assert r.profile == "fresh" and "momentum" in r.components


def test_journal_and_replay_know_the_profile(cfg, established):
    j = Journal(":memory:")
    result = factors.evaluate(established, partial_lock(33), cfg)
    j.log_decision(established, result, safety=partial_lock(33))
    row = j.query("SELECT * FROM decisions")[0]
    assert json.loads(row["snapshot"])["profile"] == "established"

    replay_row = {**dict(row), "action": "buy", "conviction": 0.9}
    assert optimizer.passes(replay_row, cfg)  # 33% lock is fine for an established token
    stricter = optimizer.with_changes(cfg, {"established_min_lp_locked_pct": 50})
    assert not optimizer.passes(replay_row, stricter)  # the optimiser may only tighten it
