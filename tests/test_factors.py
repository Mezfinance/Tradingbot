from dataclasses import replace

from tradingbot import factors
from tradingbot.data.dexscreener import parse_pair
from tradingbot.data.rugcheck import parse_summary
from tradingbot.models import SafetyReport


def test_good_pair_passes(good_pair, safe, cfg):
    r = factors.evaluate(good_pair, safe, cfg)
    assert r.passed, r.rejections
    assert 0.55 <= r.score <= 1.0


def test_bonding_curve_rejected(good_pair, safe, cfg):
    r = factors.evaluate(replace(good_pair, liquidity_usd=None), safe, cfg)
    assert not r.passed
    assert any("bonding curve" in x for x in r.rejections)


def test_missing_safety_is_unsafe(good_pair, cfg):
    r = factors.evaluate(good_pair, None, cfg)
    assert "safety check unavailable" in r.rejections


def test_rugcheck_danger_rejected(good_pair, cfg):
    bad = SafetyReport(score=100, lp_locked_pct=100, risks=["danger: Mint authority enabled"], danger=True)
    r = factors.evaluate(good_pair, bad, cfg)
    assert any("danger" in x for x in r.rejections)


def test_unlocked_lp_rejected(good_pair, cfg):
    r = factors.evaluate(good_pair, SafetyReport(score=1, lp_locked_pct=10, risks=[], danger=False), cfg)
    assert any("LP locked" in x for x in r.rejections)


def test_parabolic_move_scores_lower_than_healthy(good_pair):
    healthy = factors.components(good_pair)["momentum"]
    parabolic = factors.components(replace(good_pair, price_change={**good_pair.price_change, "h1": 600}))["momentum"]
    assert parabolic < healthy


def test_sell_pressure_lowers_score(good_pair, safe, cfg):
    dumped = replace(good_pair, buys={"h1": 300}, sells={"h1": 900})
    assert factors.evaluate(dumped, safe, cfg).score < factors.evaluate(good_pair, safe, cfg).score


def test_parse_pair_handles_missing_liquidity():
    raw = {
        "chainId": "solana", "dexId": "pumpfun", "pairAddress": "P",
        "baseToken": {"address": "M", "symbol": "X", "name": "X"},
        "priceUsd": "0.0000065", "txns": {"h1": {"buys": 3, "sells": 2}},
        "volume": {"h1": 100}, "priceChange": {"h1": 5}, "fdv": 6500, "pairCreatedAt": 1,
    }
    snap = parse_pair(raw)
    assert snap.liquidity_usd is None and snap.buys["h1"] == 3


def test_parse_rugcheck_summary():
    r = parse_summary({"score": 1, "lpLockedPct": 100, "risks": [{"name": "Freeze", "level": "danger"}]})
    assert r.danger and r.risks == ["danger: Freeze"]
