# Deploying to Fly.io

The bot runs as one always-on Fly Machine with its SQLite journal on a Fly volume. Nothing is exposed to the internet: the health check is private to Fly.

You need `flyctl`, logged in (`fly auth login`).

## 1. First deploy

```bash
# The app name is set in fly.toml (`app = "supermeme"`).
fly apps create supermeme

# The journal volume. Use the same region as primary_region in fly.toml.
fly volumes create tradingbot_data --region iad --size 1

# Your Claude API key, stored encrypted by Fly and never in git.
fly secrets set ANTHROPIC_API_KEY=sk-ant-...

fly deploy

# Exactly one machine: SQLite lives on one volume and must have one writer.
fly scale count 1
```

## 2. Check it's working

```bash
fly logs                                             # live logs
fly checks list                                      # health check status
fly ssh console -C "python -m tradingbot health"     # heartbeat ages
fly ssh console -C "python -m tradingbot report"     # the full report
```

Watch the first hour of logs after setting the API key. That's the first real Claude call, so look for `analysed ... cost=$` lines and no API errors.

## 3. Back up the journal

Fly snapshots the volume daily and keeps 14 days (`snapshot_retention` in fly.toml). To take a copy yourself:

```bash
fly ssh console -C "python -m tradingbot backup /data/backup.db"
fly ssh sftp get /data/backup.db ./tradingbot-$(date +%F).db
```

`backup` uses SQLite's online backup, so it's safe while the bot is running. Run `report` or `replay` locally against the copy:

```bash
TB_DB_PATH=./tradingbot-2026-10-25.db python -m tradingbot report
```

## 4. Change settings

Any setting can be changed with an env var, either in `[env]` in fly.toml (then `fly deploy`) or as a secret (it restarts the machine):

```bash
fly secrets set TB_MIN_LIQUIDITY_USD=30000 TB_MAX_ANALYSES_PER_HOUR=10
```

## 5. Optional: canary trades (real money)

Use a **new wallet** funded only with the canary budget in USDC, plus about 0.05 SOL for fees.

```bash
fly secrets set \
  TB_CANARY_EVERY_N=3 TB_CANARY_USD=5 TB_CANARY_BUDGET_USD=100 \
  TB_LIVE_CONFIRM=I_ACCEPT_TOTAL_LOSS_RISK \
  TB_SOLANA_RPC_URL=https://your-rpc-provider/... \
  TB_KEYPAIR_JSON="$(cat canary-keypair.json)"
```

At start-up the entrypoint writes the key to `/run/secrets` with owner-only permissions and removes it from the bot's environment. It is never written to the volume, so it never appears in snapshots or backups. The public Solana RPC is rate-limited; a free Helius or QuickNode endpoint is more reliable.

## How it stays up

- The **monitor** loop writes a heartbeat every 10s and the **scan** loop after every cycle.
- An in-process **watchdog** exits the bot if the monitor is silent for 5 minutes or the scan loop for 30 minutes.
- Fly's `restart policy = "always"` starts it again. On start-up the bot logs how long it was down and how many positions were unmanaged; `report` shows the total.

## Cost

A `shared-cpu-1x` machine with 512 MB plus a 1 GB volume costs a few dollars a month. The Claude API will cost more; the report's `net after API $` line shows whether it pays for itself.
