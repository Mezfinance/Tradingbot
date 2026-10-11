# Tradingbot

A Claude-driven Solana memecoin scanner and trader. It works through the same pipeline as the viral "Grok bots", adds deterministic safety and risk controls, and journals every decision so you can measure whether it has an edge before risking money.

Read [RESEARCH.md](RESEARCH.md) first. It covers why the "$100 to $50k" claims don't hold up and what that means for the design.

## How it works

```
DexScreener feeds ──► market filters ──► RugCheck safety ──► factor score ──► Claude research ──► risk engine ──► executor
   (discovery)        liquidity, age,    mint/freeze, LP     buy pressure,    web search, buy/   sizing, stops,   paper or
                      volume, fdv        lock, danger flags  momentum, accel  pass + conviction  TP ladder        Jupiter (live)
                                                                     │                                 │
                                                                     └──────────► SQLite journal ◄─────┘
```

- **Deterministic gates first.** Claude only sees tokens with real AMM liquidity, a clean RugCheck and a factor score of at least 0.55.
- **Claude decides buy or pass.** It uses live web search and returns a schema-enforced verdict with thesis, red flags, catalysts, a social-signal classification and sources.
- **Risk is not the LLM's job.** Each trade risks 2% of equity, with a 25% stop, TP ladder at +50/+150/+400%, breakeven after TP1, a 30% trailing stop, a 24h time stop, a 10% daily loss limit and at most 5 positions.
- **Measurement.** Every candidate's price is tracked for 24 hours, whether Claude bought it, passed on it or never got to it. `report` compares the groups at 1h, 6h and 24h.

## How close is paper trading to real trading?

Paper mode is built to behave like real trading:

| Real-world effect | How the simulation models it |
|---|---|
| Price impact, pool fees, routing | Every fill is a live Jupiter quote at the exact trade size; no route means no trade |
| Transaction landing time | 3s delay, then a re-quote (`TB_SIM_LATENCY_S`) |
| Failed transactions | If the price moves past the 3% slippage limit during that delay, the trade reverts and the network fee is still charged |
| Sandwich bots (MEV) | Fills land halfway between the live quote and the slippage floor (`TB_SIM_FILL_MODE`) |
| Network fees and account rent | Priority fee on every transaction, token-account rent on every buy, priced in live SOL |
| Honeypots and sell taxes | A buy-then-sell quote before every entry; a large round-trip loss blocks the trade |
| Stale data during research | Fresh pair data and live price re-checked right before buying |
| Slow exits | Positions are checked every 10s on their own thread, independent of Claude research |
| Rugs and pulled liquidity | No route on exit means the position is stuck, as it would be for real |
| Claude API bill | Every call priced from token usage and web searches, and subtracted in the report |
| Restarts and downtime | Cooldowns and the Claude budget persist; offline periods are logged |

Two simulated baseline portfolios trade alongside Claude's with identical rules: `filters_only` (no Claude) and `random` (random picks at Claude's buy rate). If Claude doesn't beat both after its API costs, it isn't adding value.

To measure what's left of the gap, enable **canary trades**: small real trades that mirror one in every N of Claude's buys. The report compares each one with its simulated twin and suggests a fill mode.

```bash
pip install -e '.[live]'
TB_CANARY_EVERY_N=3 TB_CANARY_USD=5 TB_LIVE_CONFIRM=I_ACCEPT_TOTAL_LOSS_RISK python -m tradingbot run
```

**What it still can't capture:** your own trades moving the market at larger sizes, RPC congestion worse than a 3s delay, and anything the tracker misses between its once-a-minute samples.

## Quick start

```bash
pip install -e '.[dev]'
export ANTHROPIC_API_KEY=...

python -m tradingbot scan     # live screen, no Claude calls, no trades
python -m tradingbot once     # one paper-trading tick
python -m tradingbot run      # paper trade continuously (60s ticks)
python -m tradingbot report   # Claude vs baselines, costs, slippage, buy-vs-pass outcomes
python -m tradingbot replay --group bought --set stop_loss_pct=20 --split 2026-10-18
pytest                        # offline test suite
```

`replay` re-runs risk settings over recorded price paths. Tune on data before `--split`, then check on data after it; a setting that only wins before the split is overfitted.

**Run it on a machine that stays on.** [DEPLOY.md](DEPLOY.md) covers Fly.io: one always-on machine, the journal on a volume, a watchdog that restarts stalled loops, and backups. The `Dockerfile` also works on any VPS or container host.

All settings live in `tradingbot/config.py` and can be overridden with `TB_<NAME>` env vars (e.g. `TB_MIN_LIQUIDITY_USD=30000`). See `.env.example`.

## Live trading

Live mode swaps USDC for tokens through Jupiter and signs locally with a Solana keypair. It is **experimental and untested against mainnet**.

```bash
pip install -e '.[live]'
TB_MODE=live TB_LIVE_CONFIRM=I_ACCEPT_TOTAL_LOSS_RISK TB_KEYPAIR_PATH=./keypair.json python -m tradingbot run
```

Don't enable it until `report` shows, over a meaningful sample (100+ analysed tokens), that Claude's buys outperform its passes after costs. Fund the wallet only with an amount you can lose entirely.

## Disclaimer

This is research software, not financial advice. Most memecoins go to zero, and most traders in these markets lose money.
