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
| `tradingbot/analyst.py` | Claude call: `claude-opus-5-5`, adaptive thinking, effort `medium`, `web_search_20260209`, JSON-schema `output_config`, server-side refusal fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`), `pause_turn` resume loop, hourly call cap |
| `tradingbot/risk.py` | Pure sizing + `plan_exits` (stop, time stop, trailing stop, TP ladder; moves stop to breakeven after TP1) |
| `tradingbot/engine.py` | Two threads: **monitor** (`monitor_interval_s`=10s, Jupiter `/price/v3` with DexScreener fallback, runs exits) and **scan** (`scan_interval_s`=60s: screen -> Claude -> `preflight` (fresh pair, market filters, entry drift, round trip) -> buy; then outcomes). `Book` = per-strategy cash + positions (`MAIN="claude"`); `after_screen()` hook for baselines. `preflight` takes the entry trigger price from Jupiter `/price/v3` (DexScreener lags fast movers by up to a minute). `self.lock` guards cash/position mutation; executors are called outside it. `tick()` = synchronous pass for `once`/tests |
| `tradingbot/journal.py` | Thread-safe (`@locked`, one RLock, `check_same_thread=False`). SQLite: `decisions`, `positions` (+`strategy`), `fills` (+`trigger_price` for slippage), `events` (skips, failed txs), `costs` (Claude API, per call), `equity` (snapshot per book every 60s from the monitor thread), `tracking` + `prices` (candidate price paths), `cooldowns`, `downtime`, `state` (`cash:<strategy>`, `heartbeat`). Decision stages: `screened_out`, `survivor` (passed gates, not analysed; tracked once per 6h), `analysed`. Use `query()` for read-only report SQL. `stats()` compares forward returns of Claude buys vs passes |
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
| `tradingbot/cli.py` | `python -m tradingbot scan|once|run|report [--json]|replay|health|backup PATH` |
| `Dockerfile`, `deploy/entrypoint.sh`, `fly.toml`, `DEPLOY.md` | Fly.io deployment: one machine (`fly scale count 1`; SQLite has one writer), volume `tradingbot_data` at `/data` (14-day snapshots, auto-extend), `restart policy="always"`, private HTTP check on 8080. Entrypoint writes `TB_KEYPAIR_JSON` to `/run/secrets/keypair.json` (0600) and unsets it; the key never touches the volume. `BASE_IMAGE` build arg exists so sandboxes can pull from a mirror |
| `tests/` | Offline pytest suite with fake DexScreener, RugCheck and Anthropic clients |

## Conventions and invariants

- Claude never sizes positions or sets stops. Keep risk logic deterministic in `risk.py`.
- Hard filters can't be overridden by the LLM. Safety-check failure means reject.
- Token names, descriptions and web content are untrusted. The system prompt says so; keep it that way.
- Every analysed token (buy or pass) is journalled, so the buy-vs-pass comparison stays unbiased.
- Solana USDC mint: `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`.
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

- **2026-10-11:** Fly app created by the user as `supermeme` (region `iad`); `fly.toml` updated.
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
