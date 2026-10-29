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
    # --- watcher enrichment (tradingbot/enrich.py): extra pump factors, each budgeted
    enrich: bool = True
    enrich_poll_s: int = 20
    enrich_lists_every_s: int = 120  # DexScreener boosts/profiles + GeckoTerminal trending
    enrich_market_every_s: int = 300  # SOL price + new-pool counts
    tg_channels: str = ""  # public Telegram calling channels to scan (comma list of t.me names)
    tg_every_s: int = 180
    enrich_min_buyers_m5: int = 20  # launches/other pools need this many buyers at 5 min for Helius/X (graduations always)
    enrich_control_share: float = 0.2  # share of the Helius/X budget spent on random eligible tokens (control)
    enrich_rank_window_h: float = 6.0  # recent eligible tokens the rank threshold is set from
    enrich_rank_min_history: int = 30  # below this many scored tokens, every eligible token is looked up
    # Helius credits: 30k/day in total (~0.9M/month). Since 10-29 two thirds go to the rug catcher's operator
    # lookups (every pool with >= operator_min_liq_usd at 5 min), a third to the early-buyer enrichment.
    helius_daily_credits: int = 10_000
    operator_daily_credits: int = 20_000
    operator_min_liq_usd: float = 5000.0
    enrich_wallets_per_token: int = 8  # early buyers whose wallet age/funder is looked up
    x_daily_budget_usd: float = 3.0
    x_cost_per_post: float = 0.005  # X pay-per-use estimate; adjust to your bill
    x_cost_per_count_call: float = 0.005
    narrative_daily_usd: float = 2.0  # Haiku narrative tags for every recorded pool (~$0.0003 each)
    enrich_macro_every_s: int = 600  # BTC/ETH, fear & greed, SOL funding, stablecoins, network load, meme index
    enrich_biz_every_s: int = 600  # 4chan /biz/ threads with contract addresses
    x_watch_accounts: str = "elonmusk"  # mega-accounts whose latest posts are matched against token themes
    x_watch_every_s: int = 3600
    x_paid_callers: str = ""  # known paid promoter X handles (comma list), flagged when they post a token
    kol_wallets: str = ""  # known influencer / copy-traded wallets (comma list), flagged when they buy early
    cex_wallets: str = ""  # known exchange hot wallets (comma list), flagged when they fund early buyers

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

    # --- model5: the 5-minute pump model's paper book (tradingbot/model_book.py) ---
    model_book: bool = True
    model_book_cash_usd: float = 10_000.0
    model_book_usd: float = 100.0  # per trade, never more than model_book_liq_share of the pool
    model_book_liq_share: float = 0.02
    model_book_min_liq_usd: float = 5_000.0
    model_book_top_pct: float = 2.0  # buy pools scoring in the top this-% of recent pools
    model_book_max_open: int = 120
    model_book_daily_loss_pct: float = 15.0  # of the book's equity; halts its entries for the UTC day
    model_book_max_drift_pct: float = 25.0  # skip if the live price ran this far above the snapshot
    model_book_max_impact_pct: float = 5.0  # its buys are up to 2% of a pool, so 2-4% impact is normal
    model_book_tp_pct: float = 100.0
    model_book_sl_pct: float = 50.0
    # Two paper books trade the same signals with separate cash and ledgers (10-28):
    #   model5     - the original exits: out after `model_book_hold_min` (24h)
    #   model5_10m - out after `model_book_fast_hold_min` (10 min; 10-28 backtest on 1-min candles: picks held
    #                10 min +17%/trade, held 24h -26%, as most rugs come after the 15-minute mark). Not yet proven.
    model_book_hold_min: float = 1440.0
    model_book_slow: bool = True  # the 24h `model5` book (off on Fly since 10-28: 61% rugged, no use to the test)
    model_book_fast: bool = True
    model_book_fast_hold_min: float = 10.0
    model_book_max_entry_age_min: float = 10.0  # model5_10m skips pools older than this (the tested window)
    # model5_10m sells hand-made pumpswap pools (mint not ending in 'pump': their creator holds the LP and can
    # pull it) by this pool age, whatever the hold says; 0 = off. 15 min was the best cut-off on the 10-26 backup.
    model_book_handmade_exit_age_min: float = 15.0
    # model5_climb: the 10-minute book's rules, plus an early exit for positions that aren't climbing: sell
    # `model_book_climb_after_min` after the buy unless the price has been >= `model_book_climb_min_gain_pct` above
    # the buy's market price. Pulled pools mostly sit flat first (10-28 backtest: rugs 20% -> 7% of picks).
    model_book_climb: bool = True
    model_book_climb_after_min: float = 3.0
    model_book_climb_min_gain_pct: float = 5.0
    # model5_ops: trades only signals the rug catcher calls `repeat operator` (creator launched a pool before),
    # holds `model_book_ops_hold_min`, no pool-age exit (it sold the repeat operators' pools during the climb:
    # 10-28 backtest +6.8% -> +3.6%/trade at 10 min). Backtest +9.0%/trade at 12 min, 90% CI -1..+20, n=63.
    model_book_ops: bool = True
    model_book_ops_hold_min: float = 12.0
    model_train_days: float = 4.0

    # --- Claude analyst -------------------------------------------------
    # Off = no Opus analyst, no triage, no learning thread (they only serve Claude's own trades); the
    # watcher's cheap Haiku narrative tags still use ANTHROPIC_API_KEY. Turned off on Fly 2026-10-27
    # after two weeks of zero Claude buys.
    claude_trading: bool = True
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
