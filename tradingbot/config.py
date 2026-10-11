"""Runtime configuration. Every value can be overridden with a TB_* env var."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

LIVE_CONFIRM_PHRASE = "I_ACCEPT_TOTAL_LOSS_RISK"


@dataclass
class Config:
    # --- mode -----------------------------------------------------------
    mode: str = "paper"  # "paper" or "live"
    starting_cash_usd: float = 1000.0
    db_path: str = "tradingbot.db"
    scan_interval_s: int = 60

    # --- hard filters (deterministic, applied before Claude is called) --
    min_liquidity_usd: float = 15_000.0
    min_volume_h1_usd: float = 25_000.0
    min_age_minutes: float = 10.0
    max_age_hours: float = 72.0
    max_fdv_usd: float = 25_000_000.0
    max_rugcheck_score: int = 2_000  # rugcheck raw score; lower is safer
    min_lp_locked_pct: float = 80.0
    min_factor_score: float = 0.55  # 0..1 composite needed to reach Claude

    # --- Claude analyst -------------------------------------------------
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_web_searches: int = 5
    max_analyses_per_hour: int = 20  # cost cap
    min_conviction: float = 0.65  # Claude conviction needed to buy

    # --- risk -----------------------------------------------------------
    risk_per_trade_pct: float = 2.0  # % of equity lost if stop is hit
    max_position_pct: float = 10.0  # % of equity in any single token
    max_open_positions: int = 5
    stop_loss_pct: float = 25.0
    take_profit_pcts: list[float] = field(default_factory=lambda: [50.0, 150.0, 400.0])
    take_profit_fractions: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    trailing_stop_pct: float = 30.0  # armed after first take-profit
    max_hold_hours: float = 24.0
    daily_loss_limit_pct: float = 10.0  # halts new entries for the UTC day
    slippage_bps: int = 300
    fee_bps: int = 100  # swap fee + priority fee estimate

    # --- live -----------------------------------------------------------
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    keypair_path: str = "./keypair.json"
    live_confirm: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in fields(cls):
            raw = os.environ.get(f"TB_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            current = getattr(cfg, f.name)
            if isinstance(current, bool):
                value = raw.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            elif isinstance(current, list):
                value = [float(x) for x in raw.split(",")]
            else:
                value = raw
            setattr(cfg, f.name, value)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ValueError(f"TB_MODE must be paper or live, got {self.mode!r}")
        if self.mode == "live" and self.live_confirm != LIVE_CONFIRM_PHRASE:
            raise ValueError(
                f"Live mode requires TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}. "
                "Run in paper mode until the journal shows a real edge."
            )
        if len(self.take_profit_pcts) != len(self.take_profit_fractions):
            raise ValueError("take_profit_pcts and take_profit_fractions must align")
        if abs(sum(self.take_profit_fractions) - 1.0) > 1e-6:
            raise ValueError("take_profit_fractions must sum to 1.0")
