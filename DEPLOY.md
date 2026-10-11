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

## 6. The bot improves itself: what it changes and how to undo it

Learning runs on its own thread, capped at $10/day of Claude spend (`TB_LEARNING_BUDGET_USD_PER_DAY`). Every part works towards one goal, written in the protected `tradingbot/learning/mission.py`: the most long-run net profit after all costs, without risking the account.

| Layer | What changes | How to see it | How to undo it |
|---|---|---|---|
| Post-mortems | Nothing directly. Each closed trade, and each strong pump it passed on, is reviewed | `python -m tradingbot postmortems` | n/a |
| Playbook | Lessons the analyst reads (25% of analyses run without them as a control) | `python -m tradingbot playbook` | Retired automatically when contradicted |
| Settings | Stops, targets, filters, conviction threshold, within hard limits | `python -m tradingbot settings` | Automatic rollback, or `python -m tradingbot settings-rollback` |
| **Code** | **The bot's own source code (except protected files)** | `python -m tradingbot code-versions`, `code-diff vN` | **Automatic rollback, or `code-rollback vN` / `code-rollback image`** |

Run them on Fly with `fly ssh console -C "python -m tradingbot <command>"`.

**Give the bot ideas.** You can queue your own improvement for the engineer; it is built, tested and put on probation like any other change:

```bash
fly ssh console -C 'python -m tradingbot propose "Mention velocity factor" --idea "Count recent Reddit posts mentioning the token via the public search JSON and add it as a factor" --priority 1'
fly ssh console -C "python -m tradingbot proposals"   # queue and results
```

**Social score.** Each analysis records Claude's 0–1 score for organic social demand, plus mentions found, notable accounts and how recent the buzz is. The report's `SOCIAL SCORE vs OUTCOME` section shows whether high-scoring tokens actually do better. `TB_MIN_SOCIAL_SCORE` (default 0, no gate) is a buy threshold the optimiser can raise if the data supports it.

**Self-written code, step by step.** At most once a day, the engineer takes the highest-priority code proposal from the optimiser, edits a copy of the code with Claude, and ships it only if:
- no protected file changed: risk rules, config limits, journal, execution and wallet code, health, backups, CLI, the learning system, and every existing test;
- it compiles and imports, and doesn't touch keys or live-trading switches;
- the **full test suite passes**, including `tests/test_invariants.py` (unsafe tokens always rejected, stops always fire, sizes capped, honeypots blocked, live trading locked).

**Every version is kept, forever.** Each shipped version is stored as an immutable, read-only, checksummed copy in `/data/code/versions/vN`, with its parent, the proposal, a summary and the exact diff. It is also archived to your storage bucket as `code/vN.tar.gz`. Nothing is overwritten or deleted, so you can always go back:

```bash
fly ssh console -C "python -m tradingbot code-versions"        # list all versions, which is active
fly ssh console -C "python -m tradingbot code-diff v3"         # see exactly what v3 changed
fly ssh console -C "python -m tradingbot code-rollback v2"     # run v2 from the next start
fly ssh console -C "python -m tradingbot code-rollback image"  # back to the deployed code
fly apps restart supermeme                                     # apply
```

**Automatic rollback.** A new version is on probation:
- **crash loop:** the start-up script falls back to its parent after 3 failed boots;
- **failed checksums:** same fallback;
- **20 loop errors, or it stops analysing tokens:** the bot rolls it back;
- **worse trading:** after 10 trades, if Claude's edge over the `filters_only` baseline dropped by more than 5% per trade, it rolls back.

**Keeping self-written code across deploys.** A new `fly deploy` always runs the new image; versions built on the old image are set aside, though still listed and archived. To carry them forward, let the bot push each version to a GitHub branch, then merge that branch before deploying:

```bash
fly secrets set GITHUB_TOKEN=<fine-grained token: contents read/write on this repo> \
  GITHUB_REPO=Mezfinance/Tradingbot GITHUB_BASE_BRANCH=<your deploy branch> \
  GITHUB_BRANCH=bot/self-improvements
```

The bot only ever pushes to `GITHUB_BRANCH`, never to your base branch. To switch self-writing off: `fly secrets set TB_ENGINEER_ENABLED=false`.

## If Claude calls fail

Billing or key problems (for example "credit balance is too low") pause Claude calls for 10 minutes and log `Claude API unavailable`. Rate limits and Anthropic outages pause them for 1 minute. In both cases candidates stay queued and are analysed once the API works again, with no restart needed. Fix billing at console.anthropic.com, or a bad key with `fly secrets set ANTHROPIC_API_KEY=...`.

## How it stays up

- The **monitor** loop writes a heartbeat every 10s and the **scan** loop after every cycle.
- An in-process **watchdog** exits the bot if the monitor is silent for 5 minutes or the scan loop for 30 minutes.
- Fly's `restart policy = "always"` starts it again. On start-up the bot logs how long it was down and how many positions were unmanaged; `report` shows the total.

## Cost

A `shared-cpu-1x` machine with 512 MB plus a 1 GB volume costs a few dollars a month. The Claude API will cost more; the report's `net after API $` line shows whether it pays for itself.
