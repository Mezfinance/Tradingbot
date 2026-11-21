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

## Layer 0 — raw UDP shreds → transactions

New requirement as of tomorrow. Pipeline:

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

1. **Offline first (no latency pressure).** Feed recorded raw shreds (capture a few minutes from the
   new provider) through layers 0+1; compare the recovered per-slot buy/sell flow against the Helius
   ground truth (`launch_record`/`slot_flow(vis="full")`) for the same pools — the same check
   `frontrun_verify.py` does for box fires, but now on our own decode. This is the correctness gate.
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

- **What the new feed emits** (raw UDP shreds vs partially reassembled) and its per-slot latency —
  dictates how much of layer 0 we build vs the provider does.
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
