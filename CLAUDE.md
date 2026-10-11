# CLAUDE.md

Context for new Claude Code sessions in this repo. **Read this first, then update it whenever you push changes.**

## What this repo is

A Python 3.10+ Claude-driven Solana memecoin scanner/trader. Paper trading is the default. It was built as a better version of the "Grok bots" people promote online. `RESEARCH.md` holds the findings: the viral returns are survivorship bias, and the realistic edge comes from risk control plus measurement.

## Layout

| Path | Role |
|---|---|
| `tradingbot/config.py` | `Config` dataclass; every field is overridable with a `TB_<FIELD>` env var. Live mode needs `TB_LIVE_CONFIRM=I_ACCEPT_TOTAL_LOSS_RISK` |
| `tradingbot/models.py` | `PairSnapshot`, `SafetyReport`, `FactorResult`, `Verdict` (pydantic), `Position` |
| `tradingbot/data/dexscreener.py` | Discovery feeds (profiles, boosts) + `/tokens/v1/solana/{csv}` pair data. Pump.fun bonding-curve pairs have **no liquidity field** |
| `tradingbot/data/rugcheck.py` | `api.rugcheck.xyz/v1/tokens/{mint}/report/summary`. `None` result = unsafe |
| `tradingbot/factors.py` | `market_filters` -> `safety_filters` -> weighted 0..1 composite (`WEIGHTS`). No LLM here |
| `tradingbot/analyst.py` | Claude call: `claude-opus-5-5`, adaptive thinking, effort `medium`, `web_search_20260209`, JSON-schema `output_config`, server-side refusal fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`), `pause_turn` resume loop, hourly call cap |
| `tradingbot/risk.py` | Pure sizing + `plan_exits` (stop, time stop, trailing stop, TP ladder; moves stop to breakeven after TP1) |
| `tradingbot/engine.py` | Tick = manage positions -> daily-loss / slot check -> screen -> analyse -> buy; then back-fill 6h outcomes. In-memory 30-min cooldown for rejected/failed tokens |
| `tradingbot/journal.py` | SQLite: `decisions`, `positions`, `fills`, `state(cash)`. `stats()` compares forward returns of Claude buys vs passes |
| `tradingbot/execution/paper.py` | Pessimistic fills (slippage + fee both legs) |
| `tradingbot/execution/jupiter.py` | Live USDC<->token via `lite-api.jup.ag/swap/v1`, solders signing, RPC confirm. **Experimental, never run against mainnet** |
| `tradingbot/cli.py` | `python -m tradingbot scan|once|run|report` |
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

## Ideas not yet built

- Holder-concentration and deployer-history factors (Helius/Birdeye; need API keys).
- X API integration for mention velocity as a numeric factor, alongside Claude's web search.
- Historical backtester driven by recorded `decisions` snapshots.
- Telegram alerts / manual-approve mode before live buys.
