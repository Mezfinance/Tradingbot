# Router buy/sell decode — progress notes (short; append concisely)

Code: `tradingbot/router_direction.py`, tests `tests/test_router_direction.py`.
Data: `Router Data/router-samples.jsonl` (62MB, 31,376 labelled txs, 211 routers — never load whole in a
chat context; script aggregates only). Volume denominator: `router-counts.json` (427 routers, 1,134,753 txs).
Validated rules: `Router Data/router_rules.json` (`load_rules` reads it; precision 1.0 on 5 holdout splits).

## Chunk 1 (2026-11-28)
- Sample file covers only 13.3% of routed volume; top routers (Bgo4v, fxckso, Diablo, 5CzA, CtjS, B4vF,
  9KGu ...) have 0 samples, JUP6 has 2. ~130 sampled routers are buy-only (can't validate).
- New rule kind `per_disc` (one sub-rule per instruction: 8-byte Anchor disc or 1-byte native tag).
- Guards: data-blind (identical ix bytes seen as buy and sell -> state-decided arb/MEV bot, reject data
  rules: ZEPHYR, Cvf5Gb7w), MAX_DISCS=8 (no amount memorisation), byte_flag <=3 values, stable across 5 splits.
- Result: 43 rules; 3.7% of routed volume covered, 3.4% effective (was 0.59%). 6Vo3245 decoded
  (ix af05..: byte[24] 01=buy/00=sell; ix fd9c..: sell). Fmhh4: tag 06=sell, tag 08 undecodable.
- Next: fresh box fetch for the top 30 by volume (`router-direction --routers-file router-counts.json
  --top 30 --dump`), then decode those in chunks.

## Top-up fetch (2026-11-28)
- `--skip-sampled` (router-direction --routers-file): fetches only routers lacking >=4 buy AND >=4 sell in
  `tradingbot/router_data/router_sampled.json` (summary of the 62MB JSONL). Counts copy:
  `tradingbot/router_data/router_counts.json`. Top 30 still-missing = 73.3% of routed volume.
- Merge: `--samples "Router Data/router-samples.jsonl,Router Data/router-samples-top30.jsonl"` dedupes by sig.

## Chunk 2 (2026-11-28) — top-30 top-up
- Two runs (`router-samples-top30.jsonl`, `-top30-new.jsonl`) returned txs for only 4 of 30 routers
  (JUP6, 6MWVTis8, term9, DF1ow4ts); the other 26 returned 0 both times (cause unverified; run log pending).
- term9: ix `afaf6d1f…` = buy (302/302, stable); ix `de44ee3b…` (2 accounts, 2 random bytes) undecodable.
- Jupiter (verified discs: `e517cb97…`=route, `bb64facc…`=route_v2): route dest_mint (acct 5) = token was
  labelled sell 3/10 times (multi-hop/arb), so no precision-1.0 rule. Needs route-plan parsing or more data.
- Coverage: 5.0% of routed volume (3.8% effective), 44 rules. Merge dedupe is per (router, sig).

## Chunk 3 (2026-11-28) — raw re-fetch of the 26 empty routers
- `--raw-out` wrote all 78k fetched txs (`Router Data/router-raw26.jsonl.gz`, 14.9MB). Cause of the empty
  result, verified: (1) the routers' flow is on pump_amm **v2** swaps (`buy_v2` b817ee61…, `sell_v2`
  5df6823c…, `buy_exact_quote_in_v2` c2ab1c46…; Anchor names hashed), which the v1-only labeller ignored;
  (2) many top routers are mostly FAILED txs (Bgo4v 97.5%; 52.5% volume-weighted across the 26).
- `label_raw` labels from the v2/v1 disc and drops a tx whose payer token delta disagrees (disc matched
  balances 474/474) or that buys and sells in one tx. `samples_from_raw` streams the gz into samples.
- New fitters: `bit_flag` (side bit in a flags byte), `const_field` (one side always a sentinel),
  `_sentinel_rule` (one-sided: sentinel -> side, else abstain; Bgo4v sells = u64::MAX limit).
- Result: 70 rules, **37.6% of routed volume, 30.3% effective**. Undecodable (state-decided bots, same
  instruction both ways): 4i9Z, E5xD, 8ZkM. Too few successful samples: DiabLokx, CtjS, B4vF, JDk3, 38V4.
- `localshred_consumer.decode_tx` now decodes v2 too (2026-11-28, same account layout as v1); the box copy needs the patch from the HISTORY.md 11-28 entry.
