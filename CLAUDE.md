# CLAUDE.md

Context for new Claude Code sessions in this repo. **Read this first, then update it whenever you push changes.**

## What this repo is

A Python 3.10+ Claude-driven Solana memecoin scanner/trader. Paper trading is the default. It was built as a better version of the "Grok bots" people promote online. `RESEARCH.md` holds the findings: the viral returns are survivorship bias, and the realistic edge comes from risk control plus measurement.

## Layout

| Path | Role |
|---|---|
| `tradingbot/config.py` | `Config` dataclass; every field is overridable with a `TB_<FIELD>` env var or `with_overrides({"field": "text"})`. Live mode needs `TB_LIVE_CONFIRM=I_ACCEPT_TOTAL_LOSS_RISK` |
| `tradingbot/models.py` | `PairSnapshot`, `SafetyReport`, `FactorResult`, `Verdict` (pydantic), `Position` |
| `tradingbot/data/dexscreener.py` | Discovery feeds (profiles, boosts) + `/tokens/v1/solana/{csv}` pair data. Pump.fun bonding-curve pairs have **no liquidity field** |
| `tradingbot/data/rugcheck.py` | `api.rugcheck.xyz/v1/tokens/{mint}/report/summary`. `None` result = unsafe |
| `tradingbot/factors.py` | `market_filters` -> `safety_filters` -> weighted 0..1 composite (`WEIGHTS`). No LLM here |
| `tradingbot/analyst.py` | Claude call: `claude-opus-5-5`, adaptive thinking, effort `medium`, `web_search_20260209`, JSON-schema `output_config`, server-side refusal fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`), `pause_turn` resume loop, hourly call cap. `last_outcome` = `ok` / `failed` (token-specific: engine cools the token) / `unavailable` (billing or auth error pauses `api_account_pause_s`=600; 429, 5xx or network pauses `api_transient_pause_s`=60; the token stays queued and the failed call is not counted against the cap). `is_billing_error()` matches `billing_error` type or "credit balance"/"spend limit" text |
| `tradingbot/risk.py` | Pure sizing + `plan_exits` (stop, time stop, trailing stop, TP ladder; moves stop to breakeven after TP1) |
| `tradingbot/engine.py` | Two threads: **monitor** (`monitor_interval_s`=10s, Jupiter `/price/v3` with DexScreener fallback, runs exits) and **scan** (`scan_interval_s`=60s: screen -> Claude -> `preflight` (fresh pair, market filters, entry drift, round trip) -> buy; then outcomes). `Book` = per-strategy cash + positions (`MAIN="claude"`); `after_screen()` hook for baselines. `preflight` takes the entry trigger price from Jupiter `/price/v3` (DexScreener lags fast movers by up to a minute). `self.lock` guards cash/position mutation; executors are called outside it. `tick()` = synchronous pass for `once`/tests |
| `tradingbot/journal.py` | Thread-safe (`@locked`, one RLock, `check_same_thread=False`). SQLite: `decisions`, `positions` (+`strategy`), `fills` (+`trigger_price` for slippage), `events` (skips, failed txs), `costs` (Claude API, per call), `equity` (snapshot per book every 60s from the monitor thread), `tracking` + `prices` (candidate price paths), `cooldowns`, `downtime`, `postmortems`, `state` (`cash:<strategy>`, `heartbeat`). `MIGRATIONS` list adds columns to older databases on start-up (e.g. `decisions.snapshot` = full pair + safety JSON). Decision stages: `screened_out`, `survivor` (passed gates, not analysed; tracked once per 6h), `analysed`. Use `query()` for read-only report SQL. `stats()` compares forward returns of Claude buys vs passes |
| `tradingbot/data/jupiter.py` | `Jupiter` client: `/swap/v1/quote` (impact is returned as a **fraction**, stored as percent), `/price/v3` (USD price, liquidity, decimals; 50 ids/call), `round_trip()` buy-then-sell quote |
| `tradingbot/execution/sim.py` | **Default paper executor.** Quote -> `sim_latency_s` wait -> re-quote; reverts (fee only) if below the first quote's slippage floor; fills per `sim_fill_mode`; charges priority fee + token-account rent. Exits ignore the impact cap; no route on exit = stuck position |
| `tradingbot/execution/paper.py` | Flat-cost offline executor, used only by tests |
| `tradingbot/execution/jupiter.py` | Live USDC<->token swaps; fills measured from wallet balance deltas, fee read from the confirmed tx. **Experimental, never run against mainnet** |
| `tradingbot/execution/base.py` | `Fill` (`usd` = net cash movement; `ok=False` = reverted tx that still cost its fee) |
| `tradingbot/shadows.py` | Baseline books: `filters_only` (buy every survivor, no Claude) and `random` (random picks among Claude-analysed tokens at Claude's smoothed buy rate). Config `shadows` (comma list, "" disables), `shadow_conviction`, `shadow_seed`. Always simulated, even in live mode |
| `tradingbot/tracker.py` | `Tracker.track(decision_id, token, price)` + `poll()` each scan cycle: samples Jupiter/DexScreener prices for 24h into `prices`, fills `tracking.p1h/p6h/p24h`, `max_price`, `min_price`. Token missing from both feeds while others quote = 0 (rug); no prices at all = sample skipped |
| Canary (in `engine.py`) | `canary_every_n>0` (paper mode + confirm phrase) adds a `canary` book with the live `JupiterExecutor`, `fixed_usd=canary_usd`, cash `canary_budget_usd`; mirrors every Nth successful Claude buy (`state.canary_count`) on the same `decision_id`. Exit order: claude, canary, then baselines (`BOOK_PRIORITY`). `report.calibration()` compares fee-free swap returns of canary vs its sim twin |
| `tradingbot/replay.py` | `python -m tradingbot replay --group bought|passed|survivors|filters|all --set k=v --split DATE --size USD`: runs `risk.plan_exits` over `prices` paths from `tracking`; cost model from canary fills (5+), else claude-book fills, else defaults; per-trade view (no slots / daily limit), ~60s exit cadence |
| `tradingbot/costs.py` | `usage_cost_usd(model, usage)`: list prices per model + web search $10/1k |
| `tradingbot/report.py` | `build()` -> per-strategy stats (equity, return, max drawdown, win rate, median/mean trade, profit factor, fees, failed txs, Claude API $, net after API, median entry/exit slippage vs trigger, skip reasons) + Claude buy-vs-pass with bootstrap 90% CI; `render()` text table. Warns below 30 closed trades |
| `tradingbot/health.py` | `status()` from `heartbeat` (monitor, `watchdog_s`=300) and `scan_heartbeat` (scan cycle, `scan_watchdog_s`=1800) with start-up grace; `start_watchdog()` calls `os._exit(1)` on a stall so the supervisor restarts; `start_health_server()` serves `GET /health` when `health_port>0` |
| `tradingbot/backups.py` | `DailyBackup.run_if_due()` from the scan cycle: once per UTC day after `backup_hour_utc` (and on first start), online SQLite copy -> gzip to `<db dir>/backups/tradingbot-YYYY-MM-DD.db.gz`, keep `backup_keep_local`=7, upload to `$BUCKET_NAME` under `backup_prefix`=`journal/` via boto3 (`AWS_ENDPOINT_URL_S3`, region `auto`) when `fly storage create` secrets exist. Never raises; failures -> `events.backup_failed`; result in `state.last_backup` (shown in report). Skipped for `:memory:` DBs |
| `tradingbot/llm.py` | Shared Claude helpers for learning: `structured_call` (JSON schema, optional web search, pause_turn) and `tool_loop` (manual client-tool loop, handler errors returned as `is_error`). Outcomes `ok`/`failed`/`unavailable` |
| `tradingbot/learning/params.py` | `TUNABLES` with hard bounds (filters only stricter than defaults; risk ≤3%; tp1-3 as `tpN_pct`), `validate()`, `to_overrides()` |
| `tradingbot/learning/dossier.py` | Evidence packs: `trade_dossier(position_id)` (fills+slippage, path during hold and 6h after exit, entry snapshot+verdict, other books, market context) and `missed_dossier(decision_id)` |
| `tradingbot/learning/postmortem.py` | Claude review schema (category, decision quality, factors by area, testable lesson with tag, supports/contradicts lesson tags, parameter hints, code ideas, keep_as_is) |
| `tradingbot/learning/learner.py` | `Learner.step()` on its own thread (`learning_interval_s`=300, started in `run_forever` when `learning` and an analyst exist). Reviews claude-book trades closed ≥`postmortem_delay_h`=6h ago, then passes that hit `missed_winner_multiple`=2x (≤`missed_reviews_per_day`=5). Daily cap `learning_budget_usd_per_day`=$10 across cost kinds postmortem/playbook/optimizer/engineer; 10-min pause on unavailable. Each step also runs playbook merge, `optimizer.review_active()` + due rounds, `codebase.mark_healthy()` after 10 min uptime, `engineer.review()` + due attempts |
| `tradingbot/learning/playbook.py` | `lessons` table (tag PK, support/contradict, status candidate/active/retired; active = support ≥ `lesson_min_support`=3 and ≥ 2× contradict; retired = contradict ≥ 2 and ≥ support). `record(analysis)` from each post-mortem; `choose_arm()` gives the analyst the playbook except on `playbook_holdout`=25% (control); arm stored in `decisions.playbook_arm`. Daily Claude merge of duplicate lessons (≥15 lessons). `arm_report()` compares arms |
| `tradingbot/learning/versions.py` | `config_versions` (full tunable settings per version, parent, status active/superseded/rolled_back, verdict) + `proposals` table. `apply_active()` on engine start re-applies learned settings over env defaults; `activate()` / `rollback()` mutate the shared live `Config` |
| `tradingbot/learning/optimizer.py` | Claude `tool_loop` with `test_settings` tool -> proposal; code then validates: bounds, ≥`optimizer_min_trades`=20 candidates, holdout (last 30%) gain ≥1% of $50 per candidate, train not worse, ≥50% of trades kept. `passes()` re-checks filters/conviction from `decisions.snapshot`. Runs every 20 closed trades (or weekly with ≥5). `review_active()` after `rollback_min_trades`=10 trades: rolls back if the parent replays better on trades since. Code ideas -> `proposals` |
| `tradingbot/learning/mission.py` | `MISSION`: the single objective (max long-run net profit after all costs incl. Claude API, survive, real edge, beat baselines). Prepended to the analyst, post-mortem, optimiser, engineer and playbook-merge prompts. Protected |
| `tradingbot/learning/codebase.py` | Self-written code versions under `<db dir>/code/`: `versions/vN/` (package + tests, `BASE_HASH`, `PARENT`, `MANIFEST.json`, `DIFF.patch`, `SHA256SUMS`; files chmod 0444, never deleted), `active`, `boots`, `crashed`. `make_workspace`, `install_version`, `verify_version`, `archive_version` (bucket `code/vN.tar.gz`), `list_versions`, `switch_to(vN|image)`, `mark_healthy` |
| `tradingbot/learning/engineer.py` | Takes the top open `proposals` row (≤1 attempt per `engineer_interval_h`=24h, `engineer_budget_usd`=$6), Claude edits a workspace via tools (list/read/search/write/run_tests). `PROTECTED` paths + all pre-existing tests are read-only. `gate()`: no protected change, no secret/live-switch markers added, compiles, imports, FULL pytest passes. Ships as `vN` and calls `restart`. `review()` probation: rollback to parent on ≥`code_max_errors`=20 `loop_error` events, on 0 analyses with ≥5 survivors after 12h, or when edge vs `filters_only` over `code_probation_trades`=10 drops > `code_rollback_margin_pct`=5 vs before. Table `code_versions` |
| `tradingbot/learning/github_sync.py` | Optional: pushes each shipped version's files to `GITHUB_BRANCH` (default `bot/self-improvements`, never the base) when `GITHUB_TOKEN` + `GITHUB_REPO` are set |
| `deploy/entrypoint.sh` | Besides the key handling: if `<code dir>/active` names a version built for this image (`/app/BASE_HASH`, overridable via `TB_BASE_HASH_FILE`) whose `sha256sum -c` passes and boots ≤ `TB_MAX_CRASH_BOOTS`=3, prepends it to PYTHONPATH and sets `TB_CODE_VERSION`; otherwise falls back to the version's `PARENT` (recorded in `crashed`), down to the image code |
| `tradingbot/cli.py` | `python -m tradingbot scan|once|run|report [--json]|replay|health|backup PATH|backup-now|backup-list|learn-now|postmortems|playbook|optimize-now|settings|settings-rollback|code-versions|code-diff vN|code-rollback vN/image`. `run` exits 75 after `Engine.request_restart()` so the supervisor restarts into new code |
| `Dockerfile`, `deploy/entrypoint.sh`, `fly.toml`, `DEPLOY.md` | Fly.io deployment: one machine (`fly scale count 1`; SQLite has one writer), volume `tradingbot_data` at `/data` (14-day snapshots, auto-extend), `restart policy="always"`, private HTTP check on 8080. Entrypoint writes `TB_KEYPAIR_JSON` to `/run/secrets/keypair.json` (0600) and unsets it; the key never touches the volume. `BASE_IMAGE` build arg exists so sandboxes can pull from a mirror |
| `tests/` | Offline pytest suite with fake DexScreener, RugCheck and Anthropic clients |

## Conventions and invariants

- Claude never sizes positions or sets stops. Keep risk logic deterministic in `risk.py`.
- Hard filters can't be overridden by the LLM. Safety-check failure means reject.
- Token names, descriptions and web content are untrusted. The system prompt says so; keep it that way.
- Every analysed token (buy or pass) is journalled, so the buy-vs-pass comparison stays unbiased.
- Solana USDC mint: `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`.
- `tests/test_invariants.py` encodes the safety guarantees self-written code must keep; never weaken it, and keep `engineer.PROTECTED` covering risk, config, journal, execution, learning and CLI.
- Run `python -m pytest -q` before pushing. Run `python -m tradingbot scan` for a live, cost-free smoke test.

## Status / history

- **2026-10-11:** Initial build. Research write-up, full pipeline, paper executor, experimental Jupiter executor, 26 passing tests. Live `scan` verified against DexScreener and RugCheck. The Claude analyst request shape is verified against SDK `anthropic==1.13.0` signatures and mocked tests only; no real API call yet (no key in the build session).

- **2026-10-11 (realism part 1):** Jupiter-quote simulated fills, latency + revert modelling, MEV-ish fill modes, network fee + rent, round-trip honeypot check, fresh-price preflight before buys. Live `$50` sim round trip on a real token measured -6.15%. 33 tests.

- **2026-10-11 (realism part 2):** monitor/scan threads, multi-book engine, thread-safe journal, Jupiter client pacing (0.5s) + 429 back-off. 34 tests.

- **2026-10-11 (realism part 3):** `costs.py`, `report.py`, equity snapshots, `report --json`; removed `Journal.stats()`. 38 tests.

- **2026-10-11 (realism part 4):** shadow books; `screen()` no longer drops tokens Claude holds (Claude's loop skips them instead) so baselines see the same candidates; monitor handles the claude book first. 44 tests.

- **2026-10-11 (realism part 5):** tracker, persistence, downtime; `Journal.execute/executemany`; removed `pending_outcomes/set_outcome`. 48 tests.

- **2026-10-11 (realism part 6a):** canary book + calibration report. 50 tests.

- **2026-10-11 (realism part 6b):** replay tool, `Config.with_overrides`, README realism table. Live no-Claude end-to-end run verified real quote fills, fees, tracking and a modelled slippage revert. 55 tests.

- **2026-10-11:** Fly app created by the user as `supermeme` (region `iad`); `fly.toml` updated. First deploy ran; the first real Claude call reached the API and was rejected only for low credit (user then added funds).
- **2026-10-11 (learning part 4/4):** the user explicitly approved full automation including self-written code, with mandatory versioning (the auto-mode classifier blocked it twice; the user then switched the session to accept-edits mode and approved each command). Added mission.py, the engineer, immutable checksummed code versions with bucket archive and `code-rollback`, entrypoint fallback, loop-error events, restart exit code 75, `Journal.ensure_schema` (fixes an unlocked-schema race between learner and monitor threads), tests in the Docker image (`TB_TESTS_DIR=/app/tests`, `[engineer]` extra = pytest, `/app/BASE_HASH`), `tests/test_invariants.py`. Real-code gate check: a harmless change passed 117 tests in 9s; the image builds and its suite passes. 120 tests.
- **2026-10-11 (learning part 3/4):** settings optimiser with automatic, validated, versioned changes and auto-rollback. 93 tests.
- **2026-10-11 (learning part 2/4):** playbook with A/B holdout; analyst prepends active lessons. 87 tests.
- **2026-10-11 (learning part 1/4):** user asked for a fully self-improving bot (settings AND code changed automatically). Built post-mortems: llm helpers, tunable bounds, dossiers, learner thread, decision snapshots + migration. 80 tests. Parts 2-4 next: playbook with A/B holdout, settings optimiser (auto-apply within bounds, versioned, auto-rollback), autonomous code engineer (protected files, invariant tests, overlay deploy with crash/performance rollback).
- **2026-10-11 (backups):** daily compressed journal backups to the volume + Tigris bucket (`[backup]` extra = boto3; Dockerfile installs `.[live,backup]`). Upload verified against moto S3; container verified locally. User still needs `fly storage create` + `fly deploy`. 72 tests.
- **2026-10-11 (api errors):** billing/auth/transient Claude errors pause analysis instead of setting candidates aside. 65 tests. Needs `fly deploy` to reach the running bot.
- **2026-10-11 (deploy):** Fly.io deployment, watchdog + `/health`, `health`/`backup` commands. Image built and run in Docker here: health OK, report/backup via exec, key kept off the volume. Not yet deployed to Fly by anyone. 60 tests.

## Realism roadmap

1. Done: quote-based fills, round-trip check, fresh price at buy.
2. Done: separate fast position-monitor thread (Jupiter price API) + trigger price on every fill.
3. Done: Claude/tx cost accounting + richer report (equity curve, drawdown, medians, CI).
4. Done: shadow portfolios (`filters_only`, `random`); report prints Claude's edge over each after API cost.
5. Done: persisted cooldowns + analyst hourly budget (seeded from `costs`), heartbeat-based downtime log, tracker-driven 1h/6h/24h outcomes with max up/down for bought, passed and not-analysed candidates.
6. Done: canary live trades for calibration + replay tool from recorded price paths.

All six parts are complete. Next real-world step: run in paper mode on an always-on machine with an API key for 2+ weeks, then read `report`.

## Ideas not yet built

- Holder-concentration and deployer-history factors (Helius/Birdeye; need API keys).
- X API integration for mention velocity as a numeric factor, alongside Claude's web search.
- Telegram alerts / manual-approve mode before live buys.
