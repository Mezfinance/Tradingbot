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
    discovery: str = "dexscreener,geckoterminal"  # token discovery feeds (comma list)
    monitor_interval_s: int = 10  # open positions are priced and exited on this cadence
    track_interval_s: int = 60  # candidate price paths (tracker) are sampled on this cadence
    watchdog_s: int = 300  # exit (for a supervisor restart) if the monitor loop stalls this long
    scan_watchdog_s: int = 1800  # same for the scan loop; Claude research makes cycles long
    health_port: int = 0  # serve GET /health on this port when > 0

    # --- daily journal backups (see backups.py) --------------------------
    backup_hour_utc: int = 0  # back up on the first scan cycle after this UTC hour each day
    backup_keep_local: int = 7  # compressed copies kept on the volume
    backup_dir: str = ""  # default: a "backups" folder next to the database
    backup_prefix: str = "journal/"  # key prefix in the bucket
    # --- pump watcher (tradingbot/watcher.py): records new pools and their early numbers for research
    watch: bool = True
    watch_db_path: str = ""  # default: watch.db next to the journal
    watch_poll_s: int = 30
    watch_launch_sample_pct: float = 20.0  # share of bonding-curve launches kept (graduations: all)
    watch_ages_min: str = "5,15,30,60,180,360,1440"  # snapshot each pool at these ages
    watch_info_age_min: float = 15.0  # token info (socials, developer, holders) taken at this age
    watch_ohlcv_min_volume_usd: float = 5000.0  # fetch 5-min candles at 24h only above this 24h volume
    watch_backup_keep_local: int = 2

    # --- hard filters (deterministic, applied before Claude is called) --
    min_liquidity_usd: float = 15_000.0
    min_volume_h1_usd: float = 25_000.0
    min_age_minutes: float = 10.0
    max_age_hours: float = 720.0  # 30 days: trending established tokens are often better than fresh launches
    max_fdv_usd: float = 25_000_000.0
    max_rugcheck_score: int = 2_000  # rugcheck raw score; lower is safer
    min_lp_locked_pct: float = 80.0
    # Established tokens (old enough and deep enough) use a lower LP-lock rule; see factors.py.
    established_min_age_hours: float = 72.0
    established_min_liquidity_usd: float = 100_000.0
    established_min_lp_locked_pct: float = 30.0
    # Fresh tokens with a partial LP lock may pass only if everything else is exceptionally strong.
    partial_lock_min_pct: float = 50.0
    partial_lock_min_score: float = 0.75  # vs the normal min_factor_score
    partial_lock_max_rugcheck_score: int = 1_000  # vs the normal max_rugcheck_score
    partial_lock_size_factor: float = 0.5  # fraction of the normal position size
    min_factor_score: float = 0.55  # 0..1 composite needed to reach Claude

    # --- Claude analyst -------------------------------------------------
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_web_searches: int = 8  # 5 was too few to also check the contract and deployer
    max_analyses_per_hour: int = 20  # cost cap
    # cheap first pass: a small model drops obvious junk before the full analysis (tradingbot/triage.py)
    triage: bool = True
    triage_model: str = "claude-haiku-5-5"
    triage_effort: str = "low"
    triage_audit_pct: float = 10.0  # share of triage skips sent to the full analysis anyway, to measure triage
    api_account_pause_s: int = 600  # pause Claude calls after a billing/auth error
    api_transient_pause_s: int = 60  # pause after rate limits, overload or network errors
    min_conviction: float = 0.65  # Claude conviction needed to buy
    min_social_score: float = 0.0  # Claude's organic social score needed to buy (0 = no gate; optimiser-tunable)

    # --- self-improvement (see tradingbot/learning) ----------------------
    learning: bool = True  # run the learning thread inside `run`
    learning_interval_s: int = 300
    learning_budget_usd_per_day: float = 10.0  # cap on Claude spend for reviews/optimiser/engineer
    learning_max_per_cycle: int = 3
    postmortem_delay_h: float = 6.0  # wait so the review can see what happened after the exit
    missed_winner_multiple: float = 2.0  # review passes that later reached this multiple
    missed_reviews_per_day: int = 5
    lesson_min_support: int = 3  # trades supporting a lesson before the analyst sees it
    playbook_holdout: float = 0.25  # share of analyses run without the playbook (control arm)
    optimizer_every_trades: int = 20  # closed Claude trades between optimiser rounds
    optimizer_min_trades: int = 20  # candidates with price paths needed before any change
    optimizer_min_holdout: int = 8
    optimizer_holdout_frac: float = 0.3  # most recent share of candidates used only for validation
    optimizer_min_gain_pct: float = 1.0  # required holdout gain, % of trade size per candidate
    optimizer_budget_usd: float = 3.0  # Claude spend cap per optimiser round
    rollback_min_trades: int = 10  # trades under a new version before it is re-checked
    engineer_enabled: bool = True  # let the bot write, test and ship its own code changes
    engineer_interval_h: float = 24.0  # at most one code attempt per this many hours
    engineer_budget_usd: float = 6.0  # Claude spend cap per code attempt
    code_probation_trades: int = 10  # trades on new code before it is kept or rolled back
    code_rollback_margin_pct: float = 5.0  # allowed drop in edge vs filters_only, % per trade
    code_max_errors: int = 20  # loop errors on new code that force a rollback
    code_dir: str = ""  # default: a "code" folder next to the database
    tests_dir: str = ""  # tests for the image code (TB_TESTS_DIR=/app/tests in Docker)

    # --- baselines (simulated books traded alongside, see shadows.py) ---
    shadows: str = "filters_only,random"  # "" disables
    shadow_conviction: float = 0.825  # fixed sizing input for baseline buys
    shadow_seed: int = 7

    # --- canary: small real trades mirroring 1 in N Claude buys (calibrates the sim) ---
    canary_every_n: int = 0  # 0 disables; needs the live keypair and confirmation phrase
    canary_usd: float = 5.0  # fixed size per canary trade
    canary_budget_usd: float = 100.0  # canary book's starting cash (keep the wallet at about this)

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
    slippage_bps: int = 300  # swap reverts if output falls further than this
    fee_bps: int = 100  # used only by the offline PaperExecutor

    # --- execution realism (simulated and live) -------------------------
    max_price_impact_pct: float = 3.0
    max_round_trip_loss_pct: float = 10.0  # buy-then-sell quote; catches honeypots/taxes
    max_entry_drift_pct: float = 15.0  # skip if price ran this much during research
    priority_fee_sol: float = 0.001  # base + priority fee per transaction
    ata_rent_sol: float = 0.00203928  # new token account; live executor does not reclaim it
    sim_latency_s: float = 3.0  # decision -> landed transaction
    sim_fill_mode: str = "midpoint"  # expected | midpoint | pessimistic (see SimExecutor)

    # --- live -----------------------------------------------------------
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    keypair_path: str = "./keypair.json"
    live_confirm: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        overrides = {}
        for f in fields(cls):
            raw = os.environ.get(f"TB_{f.name.upper()}")
            if raw is not None and raw != "":
                overrides[f.name] = raw
        return cls().with_overrides(overrides)

    def with_overrides(self, raw: dict[str, str]) -> "Config":
        """Apply string overrides (from env vars or --set key=value), coerced to each field's type."""
        names = {f.name for f in fields(self)}
        for name, text in raw.items():
            if name not in names:
                raise ValueError(f"unknown setting {name!r}")
            current = getattr(self, name)
            if isinstance(current, bool):
                value = text.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                value = int(text)
            elif isinstance(current, float):
                value = float(text)
            elif isinstance(current, list):
                value = [float(x) for x in text.split(",")]
            else:
                value = text
            setattr(self, name, value)
        self.validate()
        return self

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ValueError(f"TB_MODE must be paper or live, got {self.mode!r}")
        if self.mode == "live" and self.live_confirm != LIVE_CONFIRM_PHRASE:
            raise ValueError(
                f"Live mode requires TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}. "
                "Run in paper mode until the journal shows a real edge."
            )
        if self.canary_every_n > 0 and self.live_confirm != LIVE_CONFIRM_PHRASE:
            raise ValueError(f"Canary trades spend real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
        if self.canary_every_n > 0 and self.mode == "live":
            raise ValueError("Canary trades are for paper mode; live mode already trades for real")
        if self.sim_fill_mode not in ("expected", "midpoint", "pessimistic"):
            raise ValueError("sim_fill_mode must be expected, midpoint or pessimistic")
        if len(self.take_profit_pcts) != len(self.take_profit_fractions):
            raise ValueError("take_profit_pcts and take_profit_fractions must align")
        if abs(sum(self.take_profit_fractions) - 1.0) > 1e-6:
            raise ValueError("take_profit_fractions must sum to 1.0")
