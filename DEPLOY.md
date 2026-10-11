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

The bot backs itself up once a day, on the first scan after midnight UTC (`TB_BACKUP_HOUR_UTC`). It also makes one straight away on first start-up.

| Copy | Where | Kept for |
|---|---|---|
| Compressed journal (`tradingbot-YYYY-MM-DD.db.gz`) | `/data/backups` on the volume | Newest 7 (`TB_BACKUP_KEEP_LOCAL`) |
| The same file, uploaded | Your Tigris bucket under `journal/` | Forever (each is small) |
| Whole-volume snapshot | Fly, automatic | 14 days |

**Turn on the bucket upload (once):**

```bash
fly storage create        # creates a Tigris bucket and adds its credentials to the app
git pull && fly deploy
```

`fly storage create` prints the bucket credentials once, so save them in your password manager.

**Check backups:**

```bash
fly ssh console -C "python -m tradingbot backup-list"   # local and bucket copies
fly ssh console -C "python -m tradingbot backup-now"    # make one immediately
```

The `report` also ends with a `LAST BACKUP` line, and a failed backup is logged as `daily backup failed` without stopping trading.

**Get a copy onto your laptop**, either from the volume:

```bash
fly ssh sftp get /data/backups/tradingbot-2026-10-12.db.gz ./
gunzip tradingbot-2026-10-12.db.gz
TB_DB_PATH=./tradingbot-2026-10-12.db python -m tradingbot report
```

or from the bucket, using any S3 tool with the credentials `fly storage create` printed (endpoint `https://t3.storage.dev`, region `auto`).

**Snapshot retention:** if you created the volume by hand before the first deploy, set retention explicitly:

```bash
fly volumes list                                      # find the volume ID
fly volumes update <volume-id> --snapshot-retention 14
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

## If Claude calls fail

Billing or key problems (for example "credit balance is too low") pause Claude calls for 10 minutes and log `Claude API unavailable`. Rate limits and Anthropic outages pause them for 1 minute. In both cases candidates stay queued and are analysed once the API works again, with no restart needed. Fix billing at console.anthropic.com, or a bad key with `fly secrets set ANTHROPIC_API_KEY=...`.

## How it stays up

- The **monitor** loop writes a heartbeat every 10s and the **scan** loop after every cycle.
- An in-process **watchdog** exits the bot if the monitor is silent for 5 minutes or the scan loop for 30 minutes.
- Fly's `restart policy = "always"` starts it again. On start-up the bot logs how long it was down and how many positions were unmanaged; `report` shows the total.

## Cost

A `shared-cpu-1x` machine with 512 MB plus a 1 GB volume costs a few dollars a month. The Claude API will cost more; the report's `net after API $` line shows whether it pays for itself.
