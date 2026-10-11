"""Settings the learning system may change by itself, and the hard limits on each.

Filter thresholds may only move in the *stricter* direction from their defaults:
loosening a filter would let in tokens the bot never analysed, which no recorded
data can validate. Exit and sizing settings may move both ways within bounds.
Risk per trade can never exceed 3% of equity.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingbot.config import Config

DEFAULTS = Config()


@dataclass(frozen=True)
class Tunable:
    name: str
    lo: float
    hi: float
    kind: str  # "exit", "filter", "selection", "sizing"
    why: str


TUNABLES: dict[str, Tunable] = {t.name: t for t in [
    # exits: replayable on recorded price paths
    Tunable("stop_loss_pct", 10, 40, "exit", "hard stop below entry"),
    Tunable("trailing_stop_pct", 10, 50, "exit", "trail from peak after first target"),
    Tunable("max_hold_hours", 2, 24, "exit", "time stop"),
    Tunable("tp1_pct", 20, 150, "exit", "first take-profit target (sells 40%)"),
    Tunable("tp2_pct", 60, 400, "exit", "second take-profit target (sells 30%)"),
    Tunable("tp3_pct", 150, 1500, "exit", "third take-profit target (sells the rest)"),
    # selection: replayable on analysed candidates
    Tunable("min_conviction", 0.5, 0.95, "selection", "Claude conviction needed to buy"),
    Tunable("min_factor_score", DEFAULTS.min_factor_score, 0.9, "filter", "composite score gate"),
    Tunable("min_liquidity_usd", DEFAULTS.min_liquidity_usd, 250_000, "filter", "liquidity floor"),
    Tunable("min_volume_h1_usd", DEFAULTS.min_volume_h1_usd, 500_000, "filter", "1h volume floor"),
    Tunable("min_age_minutes", DEFAULTS.min_age_minutes, 360, "filter", "skip pairs younger than this"),
    Tunable("max_age_hours", 6, DEFAULTS.max_age_hours, "filter", "skip pairs older than this"),
    Tunable("max_fdv_usd", 250_000, DEFAULTS.max_fdv_usd, "filter", "skip tokens valued above this"),
    Tunable("max_entry_drift_pct", 3, DEFAULTS.max_entry_drift_pct, "filter", "skip if price ran during research"),
    # sizing
    Tunable("risk_per_trade_pct", 0.5, 3.0, "sizing", "equity lost if the stop is hit"),
    Tunable("max_open_positions", 1, 8, "sizing", "concurrent positions"),
]}

TP_KEYS = ("tp1_pct", "tp2_pct", "tp3_pct")


def current_values(cfg: Config) -> dict[str, float]:
    out = {name: getattr(cfg, name) for name in TUNABLES if name not in TP_KEYS}
    for key, value in zip(TP_KEYS, cfg.take_profit_pcts):
        out[key] = value
    return out


def validate(changes: dict[str, float], cfg: Config) -> tuple[dict[str, float], list[str]]:
    """Clamp-free validation: returns (accepted changes, reasons for any rejected ones)."""
    ok: dict[str, float] = {}
    problems: list[str] = []
    for name, value in changes.items():
        t = TUNABLES.get(name)
        if t is None:
            problems.append(f"{name}: not a tunable setting")
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            problems.append(f"{name}: not a number")
            continue
        if not t.lo <= value <= t.hi:
            problems.append(f"{name}={value}: outside allowed range [{t.lo}, {t.hi}]")
            continue
        ok[name] = int(value) if name == "max_open_positions" else value
    merged = {**current_values(cfg), **ok}
    if not merged["tp1_pct"] < merged["tp2_pct"] < merged["tp3_pct"]:
        problems.append("take-profit targets must increase: tp1 < tp2 < tp3")
        for key in TP_KEYS:
            ok.pop(key, None)
    return ok, problems


def to_overrides(changes: dict[str, float], cfg: Config) -> dict[str, str]:
    """Convert tunable values into Config.with_overrides strings."""
    out: dict[str, str] = {}
    tps = list(cfg.take_profit_pcts)
    for name, value in changes.items():
        if name in TP_KEYS:
            tps[TP_KEYS.index(name)] = float(value)
        else:
            out[name] = str(value)
    if any(k in changes for k in TP_KEYS):
        out["take_profit_pcts"] = ",".join(str(x) for x in tps)
    return out
