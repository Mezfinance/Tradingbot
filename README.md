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
- **Measurement.** Six hours after every analysis the price is recorded for buys and passes alike. `report` compares the two groups.

## Quick start

```bash
pip install -e '.[dev]'
export ANTHROPIC_API_KEY=...

python -m tradingbot scan     # live screen, no Claude calls, no trades
python -m tradingbot once     # one paper-trading tick
python -m tradingbot run      # paper trade continuously (60s ticks)
python -m tradingbot report   # win rate, P&L, Claude buy-vs-pass forward returns
pytest                        # offline test suite
```

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
