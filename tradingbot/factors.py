"""Deterministic screening: hard safety filters, then a 0..1 composite factor score.

Nothing here calls an LLM. Claude only sees tokens that clear this gate, which
keeps cost bounded and stops a persuasive narrative from overriding a bad contract.
"""

from __future__ import annotations

from tradingbot.config import Config
from tradingbot.models import FactorResult, PairSnapshot, SafetyReport

WEIGHTS = {
    "buy_pressure": 0.25,
    "momentum": 0.20,
    "acceleration": 0.20,
    "turnover": 0.15,
    "breadth": 0.15,
    "socials": 0.05,
}


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


def safety_filters(safety: SafetyReport | None, cfg: Config) -> list[str]:
    out: list[str] = []
    if safety is None:
        out.append("safety check unavailable")
    else:
        if safety.danger:
            out.append("rugcheck danger: " + "; ".join(r for r in safety.risks if r.startswith("danger")))
        if safety.score is not None and safety.score > cfg.max_rugcheck_score:
            out.append(f"rugcheck score {safety.score} > {cfg.max_rugcheck_score}")
        if safety.lp_locked_pct is not None and safety.lp_locked_pct < cfg.min_lp_locked_pct:
            out.append(f"LP locked {safety.lp_locked_pct:.0f}% < {cfg.min_lp_locked_pct:.0f}%")
    return out


def hard_filters(p: PairSnapshot, safety: SafetyReport | None, cfg: Config) -> list[str]:
    """Reasons to reject outright. Empty list means the token may be scored."""
    return market_filters(p, cfg) + safety_filters(safety, cfg)


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


def evaluate(p: PairSnapshot, safety: SafetyReport | None, cfg: Config) -> FactorResult:
    rejections = hard_filters(p, safety, cfg)
    comps = components(p)
    score = sum(WEIGHTS[k] * v for k, v in comps.items())
    if not rejections and score < cfg.min_factor_score:
        rejections.append(f"factor score {score:.2f} < {cfg.min_factor_score:.2f}")
    return FactorResult(passed=not rejections, score=round(score, 4), components=comps, rejections=rejections)
