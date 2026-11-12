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
    watch_backup_keep_local: int = 1  # every copy is also in the bucket
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
    # Helius credits (free plan: 1M/month, ~33k/day). Since 11-01: 20k/day in budgets, leaving room for the
    # unbudgeted calls (pool creators, on-the-spot creator reads, token decimals). Launch data gets the most:
    # it feeds the first-seconds backtest; the early-buyer and operator lookups added little so far.
    helius_daily_credits: int = 5_000
    operator_daily_credits: int = 10_000
    operator_min_liq_usd: float = 5000.0
    # Launch data (research): a stable random share of hand-made pumpswap pools gets its first 3 minutes of
    # transactions from Helius (~10 credits each), to test whether buying in the first seconds would pay.
    launch_sample_pct: float = 20.0  # busy pools need up to 4 pages of 100 txs: ~30 credits per pool
    launch_daily_credits: int = 6_000
    # Second-4 launch scorer (shadow mode, never trades): an LP-mint websocket sees each new pumpswap pool,
    # its first 4 seconds come from Helius, the launch4 model scores it and the would-be fill / minute-6 exit
    # are read from the pool's vaults (table watch_launch_live, CLI launch-status).
    launch_live: bool = False
    market_monitor: bool = False  # hourly meme-market measures in watch.db (market_monitor.py), read only
    market_monitor_since: float = 0.0  # unix time: the monitor's baseline starts no earlier (hours before it are kept, not compared)
    launch_live_daily_credits: int = 20_000
    launch_live_decide_delay_s: float = 4.0  # after the websocket notification (~1 s after creation)
    launch_live_ws_commitment: str = "confirmed"  # LP-mint websocket commitment; 'processed' detects ~0.5 s earlier (fast canary)
    launch_live_track_thin: bool = True  # pools under model_book_min_liq_usd at second 4: still read the fill and
    # minute-6 exit (2 credits) for the pool-size bands; status 'thin' -> 'thin_done', never scored or traded
    launch_train_days: float = 14.0
    # Paper books launch4_2 / launch4_5 on the scorer's top 2% / top 5% picks (needs launch_live and model_book):
    # bought through Jupiter with the model books' simulated executor, sold at pool minute 6, no stop/take-profit.
    launch_book: bool = False
    # twins of the launch books on the history-trained model with a rolling 24 h cut-off (launch4h_2 / launch4h_5)
    launch_hist_book: bool = False
    # launch4_2m5 (11-04): launch4_2's picks (the old model's top2 flag), sold at pool minute 5 instead of 6
    launch_m5_book: bool = False
    # launchd12 + launchd12_s10/_s20/_s30/_s50 (11-04): the history model's bottom 20% (rolling 24 h), minus the
    # riskiest 0/10/20/30/50% by the early-pull model (model/pull.json), sold at pool minute 6
    launch_low_book: bool = False
    # launchrw50 / launchd12rw50 (11-04): every liquid pool / the history model's bottom 20%, minus the riskiest
    # half by the rug-vs-win model (model/rw.json: P(rug before the exit) - P(+50% win)), sold at pool minute 6
    launch_rw_book: bool = False
    launch_big_book: bool = False  # 11-08: launchbig40k, the +50% model's best 30% of $40-80k pools held to minute 15 while rising
    launch_big_mid_book: bool = False  # 11-09: launchbig15k, the +50% model's best 30% of $15-25k pools, $100, out at minute 6
    launch_thin_book: bool = False  # 11-09: nine thin-band books (pools under $15k): every pool / rw50 / band rule x $100 / $250 / 2% of pool
    launch_hold_book: bool = False  # 11-08: launchrw50_25k10 / _25k15, rug-vs-win kept $25-40k pools held to pool minute 10 / 15
    launch_band_book: bool = False  # 11-08: launchrwband / launchrwband_sz, rug-vs-win ranked within each liquidity band
    launch_sized_book: bool = False  # 11-06: launchrw50_sz, launchrw50's signals sized by liquidity band
    launch_sized_dd_book: bool = False  # 11-09: launchrw50_sz_dd24, sz that stops buying while down $1,000 over 24 h (was $500)
    launch_rw24_book: bool = False  # 11-05: twin rw books on the earlier rule (24 h ranking, nightly 14-day model)
    # canary_rw50_sz (11-06): REAL MONEY. launchrw50_sz's thin-pool picks bought and sold for real through Jupiter
    # (live executor), to measure how real fills differ from the simulated ones. Needs the keypair and TB_LIVE_CONFIRM.
    launch_canary: bool = False
    launch_canary_usd: float = 10.0  # per trade
    launch_canary_flat_usd: float = 0.0  # > 0: every canary trade at this size (e.g. 0.01, the smallest useful swap), not the bands
    launch_canary_band_scale: float = 0.0  # > 0: the twin's own band size x this (0.04 = 1/25: $10 / $1 / $4), not CANARY_SIZE_BANDS
    launch_canary_budget_usd: float = 50.0  # the book's starting cash; keep about this much USDC in the wallet
    launch_canary_daily_loss_usd: float = 50.0  # no new buys once today's realized loss reaches this
    launch_canary_follow_dd: bool = True  # 11-10: no new buys while the paper twin launchrw50_sz_dd24 is paused by its
                                          # past-24h drawdown limit (the canary's own P&L is mostly fees)
    launch_canary_max_open: int = 5
    launch_canary_max_trades: int = 0  # > 0: no new buys once the book has made this many trades (a test run, 11-10)
    launch_canary_max_liq_usd: float = 15_000.0  # only pools below this liquidity at second 4 (the $250 band); 0 = all
    launch_canary_every_n: int = 1  # buy 1 in N of the twin's signals (1 = all; each round trip costs ~0.001 SOL)
    launch_canary_priority_fee_sol: float = 0.0005  # cap on the priority fee per transaction
    launch_canary_sell_slippage_bps: int = 1000  # sells must go through on a fast pool
    launch_canary_clips: int = 1  # REAL MONEY: split each canary buy into this many equal back-to-back direct swaps into
    # the same pool (5 x $1 vs 1 x $5): does the running total of our capital trigger the pull, or is each small buy judged on its own?
    launch_canary_route: str = "jupiter"  # 'direct' = swap on the pumpswap pool; 'lp' = enter as a liquidity provider (11-10); both trade SOL
    # Feint probe (11-12): before committing, poke the pool with a $launch_canary_feint_usd buy and sell it straight
    # back, then re-read the pool and commit only if its liquidity wasn't pulled in reaction. $25 is big enough to
    # trigger the size-sensitive operators a penny probe misses, so surviving it tests the regime the real buy runs in.
    launch_canary_feint: bool = False  # REAL MONEY (direct route): gate every canary buy behind a feint probe
    launch_canary_feint_usd: float = 25.0  # the feint (bait) size
    launch_canary_feint_wait_s: float = 4.0  # wait after the feint before re-reading the pool (the $25 reaction is ~1s)
    launch_canary_feint_drop_pct: float = 50.0  # SOL-vault drop vs before the feint that counts as 'pulled'
    launch_canary_feint_hold_s: float = 0.0  # 0 = atomic buy+sell in one tx (never hold); > 0 = buy, hold, then sell
    launch_canary_feint_slippage_bps: int = 100  # the feint's sell leg sells the buy's guaranteed minimum, so a tight
    # slippage here means less dust left behind (cheaper feint); too tight and the buy reverts (a free inconclusive)
    launch_canary_feint_only: bool = False  # run the feint but never commit: measure the feint's pull-trigger rate only
    launch_canary_feint_all: bool = False  # feint EVERY liquid pool (>= model_book_min_liq_usd), not just the scorer's
    # rug-vs-win picks: the feint itself is the signal, so the scorer's ranking no longer decides which pools we probe
    # 1-slot commit probe (11-12, measurement only, REAL MONEY): probe every liquid candidate -- buy $usd so committed
    # money LANDS (slot N, which may SET a size-reacting rugger), then sell the buy's guaranteed floor in the next slot
    # (N+1), which escapes every measured pull (100% land >=2 slots after the buy). Records only, never commits; the
    # engine watches the pool for watch_s (the measured max pull age) and logs rugger (tripped and dodged) / survivor.
    launch_canary_probe: bool = False  # on: run the 1-slot probe on every liquid candidate (direct route), no entry
    launch_canary_probe_usd: float = 25.0  # the probe size; must be big enough to trip the size-reacting operators
    launch_canary_probe_watch_s: float = 48.0  # watch the pool this long after the probe (the measured max $25 pull age)
    launch_canary_probe_margin_s: float = 12.0  # extra margin on the watch before calling a pool a survivor
    launch_canary_probe_blind: bool = True  # fire the probe's SELL blind (not waiting for the buy to be visible), after
    # a short `launch_canary_probe_delay_s` pause. Firing INSTANTLY lands buy+sell in the same block (gap 0), which is
    # net-zero at the slot boundary -- a rugger polling the pool's reserves reads it like the atomic feint, no committed
    # money persists. The pause (half a slot) pushes the sell into N+1 (gap 1): the buy commits and is readable for a
    # full slot before the pull -- so it trips a state-watcher AND still escapes the ~18s reactive pull -- while the
    # pause also lets the buy land first, cutting the revert rate. A confirmed fallback sell still recovers a revert.
    # False = the observe-lag path (poll until the buy is visible, then sell; gap 2, never reverts).
    launch_canary_probe_delay_s: float = 0.2  # blind only: wait this long after sending the buy before firing the
    # pre-signed sell, so it lands one slot after the buy (gap 1) instead of the same block (gap 0). ~half a Solana slot.
    launch_canary_probe_priority_sol: float = 0.0  # the probe's SELL can pay this much priority outright (max out, not
    # the capped estimate) to try to land in the next slot. OFF by default (0 = normal priority): a live test (11-12)
    # showed the gap stays 2-3 whether or not it is paid -- the gap is set by the observe-lag (we wait for the buy to be
    # visible before firing the sell), not by fee priority on a lone dependent sell, so the extra fee buys no speed.
    # Fast canary (11-11): buy every new hand-made pool at the earliest instant (on the LP-mint websocket + 1 pool read,
    # ~pool second 2-3), no filter, direct route, to test whether an invisible-early entry dodges the operators' reaction.
    launch_fast_canary: bool = False  # REAL MONEY: buy on detection, before the scorer's second-4 decide
    launch_fast_canary_exit_s: float = 60.0  # sell each fast-canary position this many seconds after pool creation
    # Wallet rotation (11-11): spread canary trades across up to 10 wallets, one open trade per wallet, so each buy
    # enters on a fresh/rested wallet (a brand-new wallet's first trade survives; the operators seem to pull once they
    # recognise our wallet). Direct route only. Wallets come from the TB_CANARY_WALLETS secret.
    launch_canary_rotate: bool = False  # REAL MONEY: rotate wallets, one open trade per wallet
    launch_canary_wallets_path: str = ""  # file of rotation keys (JSON array); default: the entrypoint's secret copy
    # Fresh wallet per trade (11-11): every trade on a brand-new wallet, funded through rotating single-use temp
    # treasuries so no two trade wallets share a funder (defeats per-wallet AND funder flagging). Direct route only.
    launch_canary_fresh: bool = False  # REAL MONEY: a fresh wallet per trade via the rotating-treasury chain
    launch_canary_treasury_path: str = ""  # main treasury key file; default: the entrypoint's secret copy
    launch_canary_fresh_slots: int = 10  # parallel trade slots (each cycles its own fresh wallets)
    launch_canary_fresh_jitter_usd: float = 2.0  # +-$ of SOL randomised onto each wallet's funding (no uniform balances)
    launch_canary_fresh_float_sol: float = 0.02  # extra SOL funded per wallet beyond the trade (MIN_SOL + fee headroom)
    shortbook_feed_url: str = ""  # MZF Short Only on the dashboard: the shortbook app's /api/feed (Fly private network)
    # One-off fresh start for some books (comma list): their history is kept as '<book>@<reset_books_id>' and
    # their cash goes back to the starting amount. Runs once per reset_books_id, at engine start.
    reset_books: str = ""
    # Books switched off one by one (comma list), whatever their family switch says; their history stays in the journal
    disabled_books: str = ""
    reset_books_id: str = ""
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
    model_book_daily_loss_pct: float = 15.0  # of the book's equity; halts its entries for the UTC day (0 = no limit)
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
    # model5_ops9 / model5_ops10: trade only signals the rug catcher calls `repeat operator` (creator launched a
    # pool before), hold 9 / 10 minutes, no pool-age exit. 10-28 backtest with exact on-chain pull times (60 picks):
    # +8.1% / +7.6% per trade (90% CI about -3..+18), 13% / 15% rugged; 12 min +3.6%, 15 min -9.3%.
    model_book_ops: bool = True
    # Early books (10-31) early5/early6/early7: hand-made pumpswap pools scored at first sight (~pool minute 2) by
    # their own model (`model/early.json`, top `early_top_pct` of recent scores), sold by pool minute 5/6/7.
    # 10-28 backup walk-forward, exact on-chain pulls (51 trades): +8.8 / +9.9 / +9.0%/trade, 90% CI about -2..+20;
    # minute 8 -1.8%. Designed after looking at the data, so the live run is the test. Off unless switched on.
    early_book: bool = False
    early_top_pct: float = 5.0
    # early5r/early6r/early7r: the same model and exits, but buy the top `early_top_pct` of the past day's eligible
    # pools (rolling cut-off) instead of the fixed one set at training. Walk-forward on the 10-28 backup (snapshot
    # yardstick): -4.9%/trade, 90% CI -16..+6, vs -6.5% fixed and -4.1% for every eligible pool. Off unless switched on.
    early_rolling: bool = False
    # early6w (11-03): early6's model and minute-6 exit, but a wider fixed cut-off: the top `early_wide_top_pct`
    # of the recent training scores instead of the top 5% (`early-walk-forward` backtests it). Off unless switched on.
    early_wide: bool = False
    early_wide_top_pct: float = 16.0
    # early6t (11-03): early6's model and minute-6 exit, top `early_tight_top_pct` of the recent training scores.
    early_tight: bool = False
    early_tight_top_pct: float = 2.0
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
    # launch books sell like the real canary (launch_canary_sell_slippage_bps): with 3% the simulator reverted 139
    # minute-6 sales on fast pools (11-06), retried them 10-40 s late and gave up ~1 pt/trade, mostly on big winners
    launch_sell_slippage_bps: int = 1000
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
        if self.launch_canary and self.live_confirm != LIVE_CONFIRM_PHRASE:
            raise ValueError(f"The launch canary spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
        if self.launch_canary and self.mode == "live":
            raise ValueError("The launch canary is for paper mode; live mode already trades for real")
        if self.launch_fast_canary and self.live_confirm != LIVE_CONFIRM_PHRASE:
            raise ValueError(f"The fast launch canary spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
        if self.launch_fast_canary and self.mode == "live":
            raise ValueError("The fast launch canary is for paper mode; live mode already trades for real")
        if self.launch_live_ws_commitment not in ("processed", "confirmed", "finalized"):
            raise ValueError("launch_live_ws_commitment must be processed, confirmed or finalized")
        if self.launch_canary_rotate:
            if self.live_confirm != LIVE_CONFIRM_PHRASE:
                raise ValueError(f"Wallet rotation spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
            if self.launch_canary_route != "direct":
                raise ValueError("Wallet rotation is built for the direct route; set TB_LAUNCH_CANARY_ROUTE=direct")
        if self.launch_canary_fresh:
            if self.live_confirm != LIVE_CONFIRM_PHRASE:
                raise ValueError(f"The fresh-wallet chain spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
            if self.launch_canary_route != "direct":
                raise ValueError("The fresh-wallet chain is built for the direct route; set TB_LAUNCH_CANARY_ROUTE=direct")
            if self.launch_canary_fresh_slots < 1:
                raise ValueError("launch_canary_fresh_slots must be at least 1")
        if self.launch_canary_clips > 1:
            if self.live_confirm != LIVE_CONFIRM_PHRASE:
                raise ValueError(f"Clip buys spend real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
            if self.launch_canary_route != "direct":
                raise ValueError("Clip buys are built for the direct route; set TB_LAUNCH_CANARY_ROUTE=direct")
        if self.launch_canary_feint:
            if self.live_confirm != LIVE_CONFIRM_PHRASE:
                raise ValueError(f"The feint probe spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
            if self.launch_canary_route != "direct":
                raise ValueError("The feint probe is built for the direct route; set TB_LAUNCH_CANARY_ROUTE=direct")
        if self.launch_canary_feint_all and not self.launch_canary_feint:
            raise ValueError("TB_LAUNCH_CANARY_FEINT_ALL needs the feint on; set TB_LAUNCH_CANARY_FEINT=1")
        if self.launch_canary_probe:
            if self.live_confirm != LIVE_CONFIRM_PHRASE:
                raise ValueError(f"The 1-slot probe spends real money; set TB_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}")
            if self.launch_canary_route != "direct":
                raise ValueError("The 1-slot probe is built for the direct route; set TB_LAUNCH_CANARY_ROUTE=direct")
        if self.canary_every_n > 0 and self.mode == "live":
            raise ValueError("Canary trades are for paper mode; live mode already trades for real")
        if self.sim_fill_mode not in ("expected", "midpoint", "pessimistic"):
            raise ValueError("sim_fill_mode must be expected, midpoint or pessimistic")
        if len(self.take_profit_pcts) != len(self.take_profit_fractions):
            raise ValueError("take_profit_pcts and take_profit_fractions must align")
        if abs(sum(self.take_profit_fractions) - 1.0) > 1e-6:
            raise ValueError("take_profit_fractions must sum to 1.0")
