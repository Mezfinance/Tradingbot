"""Deterministic screening: hard safety filters, then a 0..1 composite factor score.

Nothing here calls an LLM. Claude only sees tokens that clear this gate, which
keeps cost bounded and stops a persuasive narrative from overriding a bad contract.

Two profiles:
  fresh        young or thin pools: strict LP-lock rule, scored on short-term momentum.
  established  at least `established_min_age_hours` old with at least
               `established_min_liquidity_usd` liquidity: a lower LP-lock rule (shared pools
               show partial locks from ordinary liquidity providers) and a trend score
               (steady buying, sustained gains, real volume) instead of spike-chasing.
Every other safety rule (RugCheck danger flags and score, missing safety data, the
buy-then-sell honeypot check in the engine) applies to both.
"""

from __future__ import annotations

from tradingbot.config import Config
from tradingbot.models import FactorResult, PairSnapshot, SafetyReport

FRESH, ESTABLISHED = "fresh", "established"

WEIGHTS = {
    "buy_pressure": 0.25,
    "momentum": 0.20,
    "acceleration": 0.20,
    "turnover": 0.15,
    "breadth": 0.15,
    "socials": 0.05,
}

TREND_WEIGHTS = {
    "buy_pressure": 0.25,
    "trend": 0.25,
    "holding": 0.15,
    "turnover": 0.15,
    "breadth": 0.15,
    "socials": 0.05,
}


def is_established(p: PairSnapshot, cfg: Config) -> bool:
    age = p.age_minutes
    return (age is not None and age >= cfg.established_min_age_hours * 60
            and (p.liquidity_usd or 0) >= cfg.established_min_liquidity_usd)


def profile(p: PairSnapshot, cfg: Config) -> str:
    return ESTABLISHED if is_established(p, cfg) else FRESH


def ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0..1 between lo and hi, clamped."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def market_filters(p: PairSnapshot, cfg: Config) -> list[str]:
    """Rejections from market data alone. Cheap, so run first."""
    out: list[str] = []
    if p.liquidity_usd is None:
        out.append("no AMM liquidity (bonding curve or unknown)")
    elif p.liquidity_usd < cfg.min_liquidity_usd:
        out.append(f"liquidity ${p.liquidity_usd:,.0f} < ${cfg.min_liquidity_usd:,.0f}")
    age = p.age_minutes
    if age is None:
        out.append("unknown pair age")
    elif age < cfg.min_age_minutes:
        out.append(f"too new ({age:.0f}m)")
    elif age > cfg.max_age_hours * 60:
        out.append(f"too old ({age / 60:.0f}h)")
    if p.volume.get("h1", 0) < cfg.min_volume_h1_usd:
        out.append(f"h1 volume ${p.volume.get('h1', 0):,.0f} < ${cfg.min_volume_h1_usd:,.0f}")
    if p.fdv_usd and p.fdv_usd > cfg.max_fdv_usd:
        out.append(f"fdv ${p.fdv_usd:,.0f} above cap")
    if p.price_change.get("m5", 0) < -20:
        out.append("dumping in last 5m")
    return out


def safety_filters(safety: SafetyReport | None, cfg: Config, established: bool = False,
                   lp_floor: float | None = None) -> list[str]:
    out: list[str] = []
    if safety is None:
        out.append("safety check unavailable")
    else:
        if safety.danger:
            out.append("rugcheck danger: " + "; ".join(r for r in safety.risks if r.startswith("danger")))
        if safety.score is not None and safety.score > cfg.max_rugcheck_score:
            out.append(f"rugcheck score {safety.score} > {cfg.max_rugcheck_score}")
        need = lp_floor if lp_floor is not None else (
            cfg.established_min_lp_locked_pct if established else cfg.min_lp_locked_pct)
        if safety.lp_locked_pct is not None and safety.lp_locked_pct < need:
            out.append(f"LP locked {safety.lp_locked_pct:.0f}% < {need:.0f}%")
    return out


def hard_filters(p: PairSnapshot, safety: SafetyReport | None, cfg: Config) -> list[str]:
    """Reasons to reject outright, with the standard LP-lock rule for the token's profile."""
    return market_filters(p, cfg) + safety_filters(safety, cfg, is_established(p, cfg))


def partial_lock_exception(kind: str, score: float, safety: SafetyReport | None, cfg: Config) -> bool:
    """A fresh token whose LP lock is between `partial_lock_min_pct` and `min_lp_locked_pct` may
    still pass, but only when everything else is exceptionally strong: a factor score of at least
    `partial_lock_min_score` and a clean RugCheck (no danger, score <= `partial_lock_max_rugcheck_score`).
    Such trades are bought at `partial_lock_size_factor` of the normal size."""
    if kind != FRESH or safety is None or safety.danger or safety.lp_locked_pct is None:
        return False
    return (cfg.partial_lock_min_pct <= safety.lp_locked_pct < cfg.min_lp_locked_pct
            and score >= cfg.partial_lock_min_score
            and (safety.score or 0) <= cfg.partial_lock_max_rugcheck_score)


def components(p: PairSnapshot) -> dict[str, float]:
    buys, sells = p.buys.get("h1", 0), p.sells.get("h1", 0)
    trades = buys + sells
    buy_ratio = buys / trades if trades else 0.0

    # Momentum rewards a healthy climb and fades parabolic moves (late entries).
    h1 = p.price_change.get("h1", 0.0)
    if h1 <= 0:
        momentum = 0.0
    elif h1 <= 80:
        momentum = ramp(h1, 0, 30)
    else:
        momentum = 1.0 - ramp(h1, 80, 400) * 0.8

    # Acceleration: is the last 5m running hotter than the hourly average?
    vol_h1 = p.volume.get("h1", 0.0)
    vol_m5 = p.volume.get("m5", 0.0)
    accel = (vol_m5 * 12 / vol_h1) if vol_h1 else 0.0

    liq = p.liquidity_usd or 0.0
    turnover = vol_h1 / liq if liq else 0.0

    return {
        "buy_pressure": ramp(buy_ratio, 0.45, 0.70),
        "momentum": momentum,
        "acceleration": ramp(accel, 0.5, 2.0),
        # Very high turnover on thin liquidity is usually wash trading.
        "turnover": ramp(turnover, 0.2, 2.0) * (0.5 if turnover > 8 else 1.0),
        "breadth": ramp(trades, 100, 1500),
        "socials": 1.0 if p.socials else 0.3,
    }


def trend_components(p: PairSnapshot) -> dict[str, float]:
    """Scoring for established tokens: a sustained, broad-based uptrend rather than a spike."""
    buys, sells = p.buys.get("h6", 0), p.sells.get("h6", 0)
    buy_ratio = buys / (buys + sells) if buys + sells else 0.0

    # 24h trend: reward a healthy climb, fade a parabolic one (late entry into a blow-off).
    h24 = p.price_change.get("h24", 0.0)
    if h24 <= 0:
        trend = 0.0
    elif h24 <= 150:
        trend = ramp(h24, 0, 40)
    else:
        trend = 1.0 - ramp(h24, 150, 600) * 0.8

    liq = p.liquidity_usd or 0.0
    vol_24 = p.volume.get("h24", 0.0)
    turnover = vol_24 / liq if liq else 0.0
    trades_24 = p.buys.get("h24", 0) + p.sells.get("h24", 0)
    return {
        "buy_pressure": ramp(buy_ratio, 0.48, 0.62),
        "trend": trend,
        # Holding its gains right now, not sliding: last hour roughly flat or up.
        "holding": ramp(p.price_change.get("h1", 0.0), -10, 2),
        # Very high daily turnover relative to liquidity is usually wash trading.
        "turnover": ramp(turnover, 0.5, 4.0) * (0.5 if turnover > 30 else 1.0),
        "breadth": ramp(trades_24, 1_000, 15_000),
        "socials": 1.0 if p.socials else 0.3,
    }


def evaluate(p: PairSnapshot, safety: SafetyReport | None, cfg: Config) -> FactorResult:
    kind = profile(p, cfg)
    comps, weights = (trend_components(p), TREND_WEIGHTS) if kind == ESTABLISHED else (components(p), WEIGHTS)
    score = sum(weights[k] * v for k, v in comps.items())
    exception = partial_lock_exception(kind, score, safety, cfg)
    rejections = market_filters(p, cfg) + safety_filters(
        safety, cfg, kind == ESTABLISHED, lp_floor=cfg.partial_lock_min_pct if exception else None)
    if not rejections and score < cfg.min_factor_score:
        rejections.append(f"factor score {score:.2f} < {cfg.min_factor_score:.2f}")
    return FactorResult(passed=not rejections, score=round(score, 4), components=comps, rejections=rejections,
                        profile=kind, partial_lock=exception and not rejections)
