# Router-aware shred decoder — design note

Investigation, 2026-11-21. Not built. This is the plan for decoding Solana shreds ourselves and
recovering the buy/sell flow the live front-run box is currently blind to.

## Why now

- Jito's hosted ShredStream was sunset (Sept 2026). The provider gRPC feed we consume (a
  reassembled, deshredded transaction stream) turns off tomorrow. The replacement feeds
  (Everstake Blockspace, DoubleZero-style) deliver **raw UDP shreds**, so we must reassemble and
  decode ourselves. See Sources.
- Separately, the front-run study (`flurry_study.py`, `--vis-experiment`) found the live box only
  sees **top-level** pumpswap swaps (~40% of trades). ~60% are **aggregator-routed** (Jupiter), where
  the pumpswap swap is a CPI (`invoke [2+]`), not a serialized instruction, so the box recovers the
  buyer (fee payer) but **not the buy/sell direction**. The study classifies direction from pool
  **vault deltas** (`helius.classify`), which are post-execution and not in shreds.

So the box's decode has to work from **instruction bytes alone**, pre-execution, for both direct and
routed swaps. The two needs collapse into one build: own the stack from UDP shred → transaction →
swap classification.

## RESULT (2026-11-21): router/CPI decoder NOT needed — direct-only carries the edge

The clean `--vis-experiment` (999 pools, all tagged, apples-to-apples) settled it: with the target
always the real full-flow ≥20% up-move and only the FEATURES restricted, the precursor-gate edge is
**full +13.0, direct-only +11.6 (~89%), buyer_plus +9.7**. So the box's *existing* top-level pumpswap
decode carries almost the whole edge, and `buyer_plus` (counting aggregator fee-payers as directionless
buyers) is *worse* than direct — adding CPI recovery hurts. **Conclusion: do not build the router-aware
CPI decoder (layer 1 below) or even the fee-payer recovery.** Tomorrow's work collapses to the feed
migration — LocalShred proxy → transactions → the box's existing top-level pumpswap decode → the
precursor gate. The layer-1 spec below is kept for the record but is shelved as not-needed.

**Still unproven (honest):** this is the shred *view* carrying a mid-to-mid, untouched-pool,
positive-skew tail (win ~10-14%, median −1.0), per the invariant. The next real gate is a **live
reactive NET probe** (up-side exit-fill risk), not a build.

## RESULT (2026-11-22): the current reassembled feed is INCOMPLETE — box-exact proves it's not LUT

The live box's `frontmeasure` fires still diverge from the Helius ground truth after the buy/sell
orientation fix (`front_finalize` tib flip): a 40-fire `frontrun-verify` reproduced only 8, with 30
`MISMATCH m2` (the *previous* slot's symbol differs; the current `S` matches). The 2026-11-21
hypothesis was that the residual is the box's `drop_lut` (~3.9% of direct swaps: LUT-pooled direct
pumpswap swaps the box can't resolve but the study's log-bracket `direct` tag keeps). The decisive
test — `frontrun-verify --box-exact`, which makes the ground truth ALSO drop LUT-pooled direct swaps
(`helius._box_visible` / `vis_box_exact`) — came back **DIVERGED 30 / 38, reproduced 8**: replicating
the LUT drop recovered nothing (plain ~33% reproduced, box-exact ~21% on a larger sample, no change).

So **the m2 residual is NOT `drop_lut`; it is deeper — trades lost upstream in the reassembled feed,
invisible to `drop_parse`** (a shred that never arrives is never parsed-or-dropped). The tell is a
completeness signature: 7 of 8 reproductions are `BS` (dominant-buy → net-sellers, eta −3.082); the
box reliably catches the DOMINANT wallet (large, hard to drop) but loses the smaller outsider/new-
wallet trades, so every non-dominant prior slot mis-classifies. That is exactly the wallet-setup
population the motif gate keys on.

**Consequence:** no box-side fix on the current feed closes this — not per-trade tib (a no-op; `tib`
is a pool constant and `front_pending` already replays pre-seed slots), not a LUT resolver (box-exact
shows LUT isn't the gap). The **complete-tx feed (LocalShred) is the only lever left.** And the test
does NOT need the Amsterdam production box: completeness is feed CONTENT, not latency, so it runs from
the existing Nuremberg box against the already-active LocalShred Lite account (see the Nuremberg
measurement runbook). Clean verify → completeness confirmed, edge provable, then buy Amsterdam for
latency; dirty → it wasn't completeness, stop, €0 spent. **Honest residual:** box-exact proves
not-LUT and (with every cheaper cause ruled out — tib no-op, pre-seed handled, CPI matched by
vis='direct', LUT replicated, `drop_parse`=0, `frontrun_live.SlotBank` parity-tested) points to shred
incompleteness, but does not alone separate "missing trades" (LocalShred fixes) from a box-side
aggregate difference (it would not). Only a per-slot trade-level diff — the box's vs the study's trade
set for a mismatch pool — would, and that needs a box logging change on the retiring feed; the
LocalShred verify is the cheaper definitive answer.

## RESULT (2026-11-27): router decoder RE-GREENLIT — live gauge shows ~27–30% routed on fresh pools

The `gate_1790` MEASURE run went live on the box and under-fires badly (~5 fires/hr at tier 1). The
CPI-drop gauge (`localshred_consumer.Consumer.direct_swaps` / `cpi_txs`, box `[health]` line) quantified
why on the real live population: **`cpi-dropped=6179 direct=16308 (27% routed)`** on the fresh-pool
firehose — about **27–30% of pumpswap-touching txs are routed through an aggregator CPI and invisible to
the top-level decode.** A slot whose organic buyers all came through a router reads **dead** (no `N`/`O`
symbol forms), so the precursor never appears and the gate never fires. That is the recall wall, and no
tier change recovers it (the slot is invisible, not low-eta).

**This flips the 2026-11-21 "not needed" verdict — because it measures a different axis.** The 11-21
`--vis-experiment` measured **edge-per-fire** on the *traded/scored* population (direct keeps ~89% of the
edge). This gauge measures **recall** on the *fresh-pool firehose*, where aggregator/bot activity is far
heavier and the dropped buyers are exactly the organic `N`/`O` precursors the gate keys on. Edge-per-fire
and fires-per-hour are different axes; the router decoder is a **recall** fix. The 11-23 "routers are 0.2%
of our flow" measured the scored (mature) population too — the fresh-launch firehose is a different, much
more routed world. Owner's call (2026-11-27): build the CPI-router-aware decode.

**Key de-risk since 11-21: LUT resolution is now live and working** (`lut_fetch.make_live_resolver`,
`localshred_tap.LutResolver.resolve_cached` + a background warmer; box log `[lut] resolved=… miss=0`).
The layer-1 spec below says "maintain no LUT cache for v1" on the assumption the router mints sit in the
STATIC keys. On-chain (11-23) that assumption was partly wrong (the message is v0; mint accounts are
LUT-indexed, which is why the recorded mint indices decoded to program+authority). With LUT resolution
live, `tx_view(resolve=…)` rebuilds the full resolved account list, so the router's token-mint account at
its fixed instruction-index resolves **whether static or LUT-indexed**. That removes the one fragility
the 11-23 check surfaced; the fixed-index read is now sound.

### Staged plan (measure first, decode the Pareto set)

- **Phase 0 — router breakdown. BUILT 2026-11-27.** `Consumer.cpi_routers` tallies each dropped-CPI tx
  to the non-infra, non-pumpswap top-level program that carried it (`INFRA_PROGRAMS` subtracts compute
  budget / system / token / ATA / memo); `Consumer.router_top(n)` ranks them; the box `[routers]` health
  line prints the top 5. This sizes the build — it names which aggregators carry the 27% so Phase 2
  decodes the Pareto set (likely Jupiter + 2–3 others for ~90%), not every router. Run it on the box for
  an hour and read the `[routers]` line.
- **Phase 1 — pin the layouts.** For each top router by volume, get the authoritative account-index
  layout + discriminators. The 11-23 lesson: the recorded Jupiter shortcut was partly wrong (dominant
  discriminator `[187,100,250,204,49,196,175,20]` = `route_v2`, and acct 7/8 were program+authority, not
  mints). Pull from the Jupiter v6 Anchor IDL and cross-validate on-chain against a routed-pumpswap batch.
  Write the confirmed table into "Layer 1 — B" below.
- **Phase 2 — the decoder (repo, tested).** In `localshred_consumer` (or a `router_decode.py`), for each
  top-level instruction whose program is a known router, read `(token mint from the fixed instruction
  account index of the RESOLVED key list, direction from which side is WSOL, buyer = fee payer)` → emit a
  synthetic `Swap` into `on_swap`. Needs a `token_mint → pool` map in `Consumer` (the reverse of the
  `pool → token` it already holds from the arm/decode). Unit-test against pinned real router txs, like the
  pumpswap tests. `buyer_plus` fallback: a router we can't decode, or a token→token / ambiguous route,
  counts the fee payer as a directionless buyer so recall still climbs for the tail.
- **Phase 3 — wire + verify.** Re-run MEASURE → `frontrun-verify`. The CPI-complete decode should (a)
  raise the fire rate and (b) lift the symbol-flip reproduction, since the shred view now sees the same
  buyers the WS vault-delta fit view sees (closing the train/serve distribution mismatch). This re-checks
  the edge-per-fire on the newly-visible slots — not assumed to hold, read from the verify + the live NET.
- **Phase 4 — latency.** The router decode is a discriminator compare + two account reads + one WSOL
  compare per routed tx (trivial); the only real cost (the LUT resolve) is already cached/warmed
  off-thread. Confirm the per-slot decode still beats the N+1 ~400 ms deadline on the live feed.

**Honest caveats (unchanged from the spec):** split routes over-count SOL (direction stays right),
multi-hop and token→token legs degrade to `buyer_plus` (all rare for fresh memecoins); and the whole thing
is still a mid-to-mid, untouched-pool, positive-skew tail — the arbiter is `frontrun-verify` reproduction
+ a live reactive NET run, not the decode being correct.

## Verdict: feasible, and simpler than feared

The de-risking insight from the Jupiter v6 account layout (confirmed below): **we do not parse the
route plan or the `Swap` enum.** For a Jupiter route we read three things straight off the
instruction:

1. **Direction** — which of the route's two mints is WSOL. `source_mint` = WSOL → buy (SOL in);
   `destination_mint` = WSOL → sell (SOL out). The mints are **fixed, static accounts**.
2. **Token identity** — the non-WSOL mint. We watch one pool per token, so the mint maps to our
   pool. We never need the pumpswap pool account in the static keys.
3. **Magnitude** — `in_amount: u64` from the args (SOL size when source is WSOL).

Because mints are read from **fixed account positions** (not args, not the route plan, not LUT-indexed
remaining accounts), the classifier is robust to (a) address-lookup-tables hiding the AMM accounts,
and (b) Jupiter's new `*_v2` arg layout. That removes the two things that would have made this
fragile. The box already has the mirror of this for **direct** swaps (`pumpswap_direct.py`:
discriminator + pool orientation → direction + amount); the router path is the same idea one level up.

## Layer 1 — router-aware swap classify (the actual ask)

Input: a decoded `VersionedTransaction` (from layer 0) — the message's instructions, the static
account keys, and the address-table-lookup references (unresolved).

For each **top-level** instruction, by the program id of its program-id-index:

### A. Direct pumpswap (already decoded by the box today)
Program id `pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA`. Match the 8-byte Anchor discriminator:
- `sell` = `[51,230,133,164,1,127,131,173]`
- `buy_exact_quote_in` = `[198,46,21,82,180,217,232,112]`
- `buy_base_out` (box already handles)

Direction needs the pool orientation (hand-made pools put SOL on either side; `token_is_base`), which
the box holds from the pool account. Amount = the `amount_in`/`min_out` in the data. This is unchanged.

### B. Jupiter v6 route (the new coverage)
Program id `JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4`. Match the discriminator → instruction, then
read the mints from the instruction's **own account index list** (the per-instruction `accounts`
vector indexes into the tx's account keys):

| Instruction | disc (if known) | source mint acct | dest mint acct | signer acct | amount arg |
|---|---|---|---|---|---|
| `shared_accounts_route` | `[193,32,155,51,65,214,156,129]` | **7** | **8** | 2 | `in_amount` u64 |
| `route` | (from IDL) | — (none) | **5** | 1 | `in_amount` u64 |
| `exact_out_route` | (from IDL) | 5 | 6 | 1 | `out_amount` u64 |
| `shared_accounts_exact_out_route` | (from IDL) | 7 | 8 | 2 | `out_amount` u64 |
| `*_v2` variants | (from IDL) | same positions, verify | same | same | same + `positive_slippage_bps` |

`shared_accounts_route` is the dominant path (Jupiter UI + most bots). Its account order (verified,
13 fixed + remaining):

```
0 token_program      3 source_token_account          6 destination_token_account   9  platform_fee_account
1 program_authority  4 program_source_token_account   7 source_mint                 10 token_2022_program
2 user_transfer_authority(signer) 5 program_destination_token_account 8 destination_mint  11 event_authority  12 program
```

Classification:
- `source_mint == WSOL` (`So111…112`) → **buy**, SOL size = `in_amount`.
- `destination_mint == WSOL` → **sell**, SOL size = `quoted_out_amount` (or leave magnitude 0; count only).
- For plain `route` there is no `source_mint` account — use `destination_mint` (acct 5): `== WSOL` →
  sell, else buy (source WSOL implied for a SOL↔token pool).
- Token = the non-WSOL mint → our pool/token.
- Trader = the signer account (acct 2 / 1), cross-checked against fee payer `keys[0]`.

The args after the 8-byte discriminator (borsh): `shared_accounts_route` = `id:u8,
route_plan:Vec<RoutePlanStep>, in_amount:u64, quoted_out_amount:u64, slippage_bps:u16,
platform_fee_bps:u8`. We only need `in_amount`, but `route_plan` is a var-length Vec **before** it, so
to read `in_amount` we must skip the Vec: read the u32 length, skip `len × sizeof(RoutePlanStep)`.
`RoutePlanStep` = `{ swap: Swap(enum), percent: u8, input_index: u8, output_index: u8 }`, and the
`Swap` enum is variable width (some variants carry fields), so a fixed skip is wrong — this is the one
place the enum matters, and only to find `in_amount`. Three options, cheapest first:
  1. **Skip magnitude.** Use direction + buyer + a unit count per routed trade (this is exactly the
     `buyer_plus` visibility the experiment already models). If `buyer_plus` keeps the edge, we never
     decode the Vec at all.
  2. Read `in_amount` from the **tail**: the args end with a fixed `u64,u64,u16,u8` (or `+u16` for v2),
     so `in_amount` is at `data_len - (8+8+2+1)` back from the end (minus the quoted/slippage/fee
     suffix). This avoids walking the Vec entirely — robust as long as the suffix layout is known.
  3. Full borsh decode of `Swap` (needs the enum variant widths). Only if we need exact per-leg SOL.

Option 2 gives magnitude for free and is the recommended default.

### C. Other routers (coverage tail)
Pump.fun's own frontend routes direct (→ path A). The remaining routers — OKX DEX, Photon, Trojan,
BonkBot, Axiom, Bloom — are distinct programs with their own layouts. For each we either (a) write a
small decoder (program id + the one swap instruction's mint/amount positions) or (b) fall back to
`buyer_plus` (fee payer counted as a buyer, no direction). **Measure the breakdown first** (see Open
items) — the box's `front_stats` already has the hook; extend it to bucket the ~60% aggregator slice
by top-level program id. Decode the routers that actually carry volume; leave the long tail on
`buyer_plus`.

### Known limits of the router decode (all rare for fresh memecoins, all degrade to buyer_plus)
- **Split routes**: `in_amount` is the whole input; if only X% goes through pumpswap, SOL is
  over-counted. Direction stays right. (Fix needs the route-plan percents = full enum decode.)
- **Multi-hop**: SOL→tokenA→ourToken. `in_amount` is hop-1 SOL, not the SOL into our pool. Direction
  from source/dest mints still right **only if our token is the dest mint**; if our token is an
  intermediate it isn't at acct 7/8 and we miss it.
- **token→token through our pool**: neither mint is WSOL → ambiguous (both a buy and a sell on our
  pool within one route). Fall back to buyer_plus for these.

## Layer 0 — shreds → transactions

New requirement as of tomorrow. **Source: LocalShred Lite** (dedicatednodes), Amsterdam
(`localshred-lite.nlams.dedicatednodes.io`). We run their Jito fork
`dedicatednodes/shredstream-proxy` (`localshred-lite-proxy shredstream`, auth via `API_KEY` /
`x-localshred-auth`, `--localshred-url http://…:9999`, plain HTTP). It handles the gRPC + auth and
gives us **two taps**, and the fork vendors `solana-ledger 4.3.0` patched so "shred recovery" is
public:

- **gRPC entry service** (`--grpc-service-port`): the proxy itself deshreds + reconstructs entries and
  streams `recovered` / `entries` / `txns` (+ decode-error counts) — the Jito entry-stream
  equivalent. We **skip the whole FEC/RS/deshred build** (the hard layer-0 part). Correction (proto
  read 2026-11-22): the stream is `Entry{ uint64 slot, bytes entries }` where `entries` is bincode
  `Vec<solana_entry::Entry>`, **not** ready `VersionedTransaction`s — so we still bincode-deserialize
  the batch and pull each `Entry.transactions` before layer 1. Co-located (same box) so the decode
  latency is small. **Try this first.**
- **Raw UDP shreds** (`--dest-ip-ports 127.0.0.1:8001,…`): "same wire format a validator's TVU port
  receives, **deduplicated upstream**, every shred of every slot." We deshred ourselves, reusing the
  fork's now-public `solana-ledger 4.3.0` recovery. More control, maybe marginally faster (no gRPC),
  but a real Rust build — only drop here if the entry service can't beat the N+1 deadline.

Either tap is pre-execution (no `meta`, no vault deltas) and either hands layer 1 the same
`VersionedTransaction`s, so **layer 1 (router-aware classify) is ours regardless** — the entry
service gives transactions, not buy/sell direction. Upstream dedup means no shred-dedup for us.
Amsterdam region → put the decoder, executor and Solana send path near Amsterdam for the N+1 budget.

Pipeline for the raw-UDP path (not needed if the gRPC entry service is used):

1. **Ingest** UDP shred packets from the provider (or a self-run receiver). Each is a data or coding
   shred with a common header (slot, fec_set_index, shred_index, variant) + Merkle fields
   (post-1.18: Merkle root + proof, chained Merkle for resiliency).
2. **Group** by `(slot, fec_set_index)` into FEC sets. A set is `N` data + `M` coding (Reed-Solomon)
   shreds.
3. **Recover**: if any data shreds are missing, reconstruct them from coding shreds (RS, up to `M`
   erasures). If too many are missing, that FEC set's entries are **lost** (a gap — partial
   visibility, acceptable; note it).
4. **Deshred**: strip per-shred headers, concatenate the data-shred payloads of the set → a byte slice
   of **entries**.
5. **Deserialize** (bincode): the reassembled slice is `Vec<Entry>`; each `Entry` has
   `Vec<VersionedTransaction>`. (Entry boundaries can span FEC sets within a slot; buffer per slot.)
6. **Emit** each `VersionedTransaction` with its slot → layer 1.

### LUT handling
v0 messages reference accounts via address-table lookups `(table_pubkey, writable_idx[], readonly_idx[])`.
Shreds don't carry table contents. For layer 1 we **don't need them**: the Jupiter mints, the signer,
and the pumpswap program id are all in the **static** key list; LUTs only hide the route's AMM
remaining accounts, which we deliberately don't use. So maintain **no LUT cache** for v1. (If a router
decoder ever needs a LUT-indexed account, resolve lazily + cache the table account data.)

### Why this stays light
We don't execute, don't track state, don't need `meta`. Per tx we touch: the top-level instruction
program ids, one instruction's account-index list, a discriminator compare, two static-key reads, one
u64. The cost is in layer 0 (RS recovery + bincode), which is well-trodden (jito-shredstream-proxy,
`solana-ledger` shred code, Rust crates) — port or wrap, don't reinvent the RS math.

## LocalShred Lite — operational setup (from the dashboard, 2026-11-21)

Run the proxy on the box that hosts the decoder + executor (co-located, near Amsterdam for the N+1 budget).
The box needs a public address and one inbound UDP port open. The key is on the dashboard — **not** in this repo.

**Dataflow — three addresses, three jobs:**
```
  LocalShred senders  45.154.33.229 , 45.154.33.129
     --UDP-->  box : 20000        (--src-bind-port: the client LISTENS here; the ONE inbound firewall rule)
        --forward-->  --dest-ip-ports  (local consumer, e.g. 127.0.0.1:8001 = our shred decoder / the gRPC tap)
```
The client's own source/egress address is auto-registered with LocalShred (≤3 held; a 4th is refused, a 7-day-idle
one frees a slot). The dashboard labels these "Destination IPs" but they are the address the client **connects
from**, not the forward target — don't confuse them with `--dest-ip-ports`.

**Build:** `git clone https://github.com/dedicatednodes/shredstream-proxy.git`, then either
`docker build -t dedicatednodes/localshred-lite-proxy:latest .` or
`cargo build --release --bin localshred-lite-proxy` (binary in `target/release`; needs `protoc` + a C toolchain;
the image is amd64-only, build from source on arm64).

**Run (source binary):**
```
localshred-lite-proxy shredstream \
  --localshred-url http://localshred-lite.nlams.dedicatednodes.io:9999 \
  --api-key <KEY>  --dest-ip-ports 127.0.0.1:8001  [--src-bind-port 20000]
```
Docker: prefer `--network host` with `-e LOCALSHRED_URL / API_KEY / DEST_IP_PORTS`; bridge needs
`-p 20000:20000/udp` + `-e SRC_BIND_PORT=20000`. Endpoint is plain HTTP, port required (no HTTPS). Pass the key via
the `API_KEY` env, not the flag. Replacing a key is immediate with no overlap and clears the registered addresses —
update the client before replacing.

**Firewall:** `sudo ufw allow 20000/udp` inbound; narrow to the senders `45.154.33.229` / `45.154.33.129` (UDP,
dest port 20000 or your `--src-bind-port`). Behind NAT, forward that port to the box.

**Healthy log:** `Starting heartbeat client` → `Shredstream started, listening on 0.0.0.0:20000/udp` →
`Sending shreds to N destinations`, then Received + forwarded counts climbing every 15 s. **Received at 0 while
heartbeats succeed = firewall / NAT, not the key.** `Failed to connect to block engine` = endpoint/key wrong or the
key was replaced. Everything right but nothing forwards = key suspended (the dashboard says so).

**Two taps off this proxy (per the vis verdict, we only need direct-only pumpswap):** `--dest-ip-ports` raw UDP
shreds (we deshred ourselves; top-level pumpswap decode is enough), OR the repo's `--grpc-service-port` entry
service (reconstructed `txns`, no deshred build) — try the entry service first.

## Nuremberg measurement runbook (close the box-exact gap, €0, no new box)

Goal: show the LocalShred feed reproduces the study where the current reassembled feed does NOT (the
2026-11-22 box-exact DIVERGED). Runs on the EXISTING Nuremberg trading box; the Amsterdam production
box waits on a clean result. Background every long command and read it with a separate `cat` (exec
rule). The API key is on the dashboard, never in the repo.

1. **Proxy up (entry-service tap).** On the Nuremberg box, build and run the fork per "operational
   setup" above, but add `--grpc-service-port` so it reconstructs transactions in-proxy:
   ```
   localshred-lite-proxy shredstream \
     --localshred-url http://localshred-lite.nlams.dedicatednodes.io:9999 \
     --api-key $API_KEY --dest-ip-ports 127.0.0.1:8001 --grpc-service-port 9898 --src-bind-port 20000
   ```
   `sudo ufw allow 20000/udp` from the two senders (`45.154.33.229`, `45.154.33.129`). Healthy =
   Received + forwarded counts climbing every 15 s AND the gRPC entry service listening on :9898.

2. **Consumer — decode core BUILT (`tradingbot/localshred_consumer.py` + `tradingbot/localshred_tap.py`,
   tested); only the box-run I/O loop remains.** The core is the box's direct decode moved into the repo:
   - `localshred_consumer.decode_tx(account_keys, instructions, fee_payer)` decodes top-level direct
     pumpswap swaps — program id `pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA`, discriminators `sell`
     `[51,230,133,164,1,127,131,173]` / `buy_exact_quote_in` `[198,46,21,82,180,217,232,112]`
     (`buy_base_out` still TODO from `pump_amm.json`) — with the box orientation flip
     `token_buy = (inst_buy == token_is_base)`, `token_is_base = base_mint(acct 3) != WSOL`, buyer =
     fee payer `keys[0]`.
   - `localshred_consumer.Consumer` routes one `frontrun_live.Runner` per pool, accumulates each pool's
     active slot (`n_buys`/`n_sells`/`buyers`; price/SOL fed 0.0 — the gate is price/magnitude-free),
     finalizes the slot when the pool next trades, and on a fire calls `on_fire` with the box-jsonl
     fields; `localshred_consumer.jsonl_writer(path)` appends the `front-fires.jsonl` shape.
   - `localshred_tap.tx_view(tx)` / `on_transaction(consumer, slot, tx)` map a parsed
     `VersionedTransaction` (solders, duck-typed in tests) into the `decode_tx` view. LUT-indexed pool
     accounts are unresolved → skipped, same blindness as the box (fine: box-exact proved the gap is
     NOT LUT; add a LUT resolver only if the LocalShred verify's residual is then the ~3.9% LUT swaps).

   **Entry-service contract (pinned, `dedicatednodes/shredstream-proxy` `jito_protos/protos/`, 2026-11-22):**
   `service ShredstreamProxy { rpc SubscribeEntries(SubscribeEntriesRequest) returns (stream Entry); }`,
   `message Entry { uint64 slot = 1; bytes entries = 2; }`. **Correction to "consume VersionedTransactions
   directly":** `entries` is **bincode `Vec<solana_entry::Entry>`**, not ready txns — the proxy does
   RS-recovery + deshred + reassembly (layer-0 steps 1–4, the hard part), but we still deserialize the
   batch (step 5) and pull each `Entry.transactions: Vec<VersionedTransaction>` (step 6). Auth: the
   `x-localshred-auth` header = `API_KEY` (env; `--api-key-header` to override). Request has no fields.
   Stubs: `python -m grpc_tools.protoc -I jito_protos/protos --python_out=. --grpc_python_out=. shared.proto shredstream.proto` → `ShredstreamProxyStub`.

   **Remaining box-run loop (validated on the box — needs `grpcio` + `solders`, and solders' deserialize
   behaviour, which can't be unit-tested off the live feed):**
   ```python
   import localshred_consumer as lc, localshred_tap as tap
   from solders.transaction import VersionedTransaction
   stub = ShredstreamProxyStub(grpc.insecure_channel("127.0.0.1:9898"))   # same box
   c = lc.Consumer(lc.load_gate("gate.json"), tier="1", on_fire=lc.jsonl_writer("ls-fires.jsonl"))
   for e in stub.SubscribeEntries(SubscribeEntriesRequest()):             # e.slot, e.entries (bincode Vec<Entry>)
       for tx in deserialize_entries(e.entries):                         # bincode Vec<Entry> -> VersionedTransactions
           tap.on_transaction(c, e.slot, tx)
   ```
   `deserialize_entries` = bincode `Vec<Entry>` (u64 len prefix; each `Entry` = `u64 num_hashes` + `[u8;32]
   hash` + `Vec<VersionedTransaction>`), yielding each tx. Slice each tx via `VersionedTransaction.from_bytes`
   then advance by `len(bytes(tx))` **if** solders tolerates trailing bytes — the one thing to confirm on
   the box (else compute the message length by parsing the bincode message header). `frontrun_live` +
   `precursor_gate` + the decode core are parity-tested; only this gRPC + bincode layer is box-local.

3. **Verify (the answer).** Ship the jsonl to Fly (box → Mac → `fly ssh sftp` put to
   `/data/ls-fires.jsonl`) and run the same feed-agnostic verifier:
   ```
   fly machine exec 863746ae7e73e8 "sh -lc 'nohup python -m tradingbot frontrun-verify \
     --fires /data/ls-fires.jsonl --gate /data/gate.json --tier 1 --fetch --pages 20 \
     > /tmp/v_ls.txt 2>&1 & echo started'" -a supermeme
   ```
   then a separate `cat /tmp/v_ls.txt`. Read `reproduced` / `DIVERGED` against the box's 8 / 30: a
   clean reproduce = the complete feed closes the gap = edge provable on LocalShred (then buy
   Amsterdam for latency); still-diverged = it wasn't completeness (a box aggregate difference, or
   LocalShred itself drops trades) — stop and diff the consumer's decode against the study on a shared
   pool. Decode at leisure; no latency pressure on this gate.

## Latency — the gating constraint

Front-running needs slot N's flow **decided** in time to land a buy at the **front of N+1** (~400 ms
slot). The reassembly adds: wait for enough shreds of N's FEC sets (near slot end) + RS recover +
decode. The old gRPC did this server-side at low latency; self-decoding adds our own CPU + the
wait-for-shreds. Two consequences:

- For **measurement** (does the recovered flow carry the edge), latency is irrelevant — decode at
  leisure. Do this first; it's the go/no-go the `--vis-experiment` is already answering offline, and
  the live decoder confirms it on the real feed.
- For **trading**, we must decide slot N from shreds that arrive **during** N, before all of N is in.
  Design for incremental decode (classify each tx as its FEC set completes; don't wait for the slot to
  close) and accept that a few late shreds mean a slightly incomplete slot-N picture. If we can't beat
  the N+1 deadline, the front-run is dead on this feed regardless of the decode being correct — so
  **measure the per-slot completion latency of the new feed early**.

## Staged build plan

1. **Stand up the proxy + get transactions flowing.** Run `localshred-lite-proxy` with
   `--grpc-service-port`; tap its reconstructed `txns` (no layer-0 build). Record a few minutes, run
   layer 1 over them, and compare the recovered per-slot buy/sell flow against the Helius ground truth
   (`launch_record`/`slot_flow(vis="full")`) for the same pools — the same check `frontrun_verify.py`
   does for box fires, now on our own decode. This is the correctness gate, no latency pressure.
   Concrete steps and the box-exact target: the **Nuremberg measurement runbook** above.
2. **Router coverage.** Extend `front_stats` to bucket the aggregator slice by program id; write
   decoders for the routers that carry volume; everything else → buyer_plus.
3. **Reconcile with the experiment.** The `--vis-experiment` tells us which visibility (`full` /
   `direct` / `buyer_plus`) is needed for the edge. If `direct` already keeps it, layer-1 router
   decode is unnecessary — ship direct only. If `buyer_plus` is needed, we need the fee-payer recovery
   (cheap) but not magnitude. If only `full` has it, we need the full router decode (direction +
   magnitude) — the design above.
4. **Latency harness.** Measure per-slot completion on the live feed; decide whether N+1 trading is
   reachable or whether this stays a measurement/selection tool.
5. **Live wire** into `absfast_tight.py` only once 1–4 pass.

## Open items to confirm

- **gRPC entry service vs raw UDP on LocalShred Lite**: confirm the `--grpc-service-port` entry
  service yields full `VersionedTransaction`s and measure its per-slot completion latency (does it
  beat N+1?). Only if it can't, confirm the raw-UDP framing (one datagram = one raw shred, no fork
  wrapper) with a packet capture and build the deshredder on the fork's public `solana-ledger 4.3.0`.
- **Discriminators** for `route`, `exact_out_route`, `shared_accounts_*_route` and the `*_v2` family,
  and the v2 account order (mints should stay at the same positions — verify). Pull from the Jupiter
  v6 Anchor IDL (`anchor idl fetch JUP6Lkb…` or the published IDL), not secondary pages.
- **Aggregator breakdown by program id** on our real traffic — which routers to decode vs leave on
  buyer_plus.
- **WSOL vs native SOL** edge: confirm routed pumpswap legs always use WSOL mint `So111…112` at
  acct 7/8 (they do via the program source/destination token accounts), so the WSOL-side test holds.

## Sources

- Jupiter V6 program id, instruction set, account layouts: https://www.helius.dev/parsed-data/jupiter/aggregator-v6
- `shared_accounts_route` discriminator `[193,32,155,51,65,214,156,129]`: https://docs.meteora.ag/developer-guides/zap/program/accounts
- Jito ShredStream sunset + raw-UDP replacements: https://orbitflare.com/blog/developers/jito-shredstreams-last-slot , https://chainstack.com/how-to-improve-solana-rpc-latency-with-shredstream/ , https://docs.jito.wtf/lowlatencytxnfeed/
- LocalShred Lite proxy (Jito fork, gRPC entry service + raw UDP, vendors `solana-ledger 4.3.0`): https://github.com/dedicatednodes/shredstream-proxy (README + `docs/localshred-lite-setup.md`)
