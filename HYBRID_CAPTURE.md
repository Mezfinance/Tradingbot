# Hybrid capture — the train=serve front-run study

The front-run gate must be **fit on exactly what the box serves live**, or it won't reproduce (the 45% / 12%
hybrid-verify divergence). The leak was the study reconstructing the price path and the history from Helius
`history_window` *after the fact* (a complete, post-hoc "perfect world") and then checking whether the live box
reproduced that reconstruction. It never can, because the offline reconstruction guesses two things the box knows
exactly: which trades the shreds actually carried, and **how deep the confirmation was at decision time**.

Fix: the study data **is the box's own two feeds, captured at decision time**. No `history_window`, no
`localshred-merge`, no Helius reconstruction.

## The live information shape (what the gate may use)

At the N→N+1 boundary the box decides with:

- **current slot N** — the **shred** decode only: per-slot buy/sell/buyers. Incomplete, and **no price** (shreds
  carry no reserves). So the gate is price/magnitude-free on N (`abs_move`/`move`/`price_up` zeroed).
- **≤ the confirmed frontier** — the **Helius-WS confirmed** decode: per-slot buy/sell/buyers **and** price. Exact
  and complete, but **lagged**. The frontier is however many slots back the WS has actually confirmed at that
  instant (N-1, N-3, … — captured, not assumed).

The gate fires on the **wallet setup** (symbols/motifs, and the count/flow `prev_*` features from the confirmed
prior). It does **not** fire on any price-move feature — `abs_move`, `move`, `price_up` and the prior-slot
`prev_abs_move`/`prev_move` are all in `FRONT_MOVE_FIELDS` and excluded by `_is_move_rule`. Firing on a prior move
is a momentum/backrun entry that fills N+1 at the top and eats the reversion (the live bad-fill artifact), so even
though the confirmed N-1 price is known, it is never a gate input.

Target / label: a **+20% up-move in N+1** (within the horizon), measured from the confirmed prices once N+1(+hold)
confirms. Lagged, which is fine — it's the label, not a feature. Real fills are the only thing left, and that is
settled by firing live, not by the study.

## The capture record (box → study)

One JSONL row per pool (box `run_hybrid` capture mode — **box-only, not in this repo**):

```json
{
  "pool": "<pumpswap pool address>",
  "token_mint": "<base mint>",
  "create_ts": 1690000000.0,
  "shred":     [[slot, n_buys, buy_sol, max_buy, n_sells, sell_sol, [buyers...]], ...],
  "confirmed": [[slot, n_buys, buy_sol, max_buy, n_sells, sell_sol, [buyers...], price], ...],
  "frontier":  [[decision_slot_N, confirmed_through_slot], ...],
  "pull_slot": 1234567890
}
```

- **`shred`** — the box's live shred decode, one row per active slot, **7 fields, NO price**. `buyers` is the list
  of per-slot buyer wallets (fee payer `keys[0]`). `buy_sol`/`max_buy` may be `0.0` if amounts aren't decoded; the
  gate is magnitude-free on the current slot, so counts + buyers are sufficient (include amounts if available).
- **`confirmed`** — the Helius-WS confirmed decode, one row per active slot, **8 fields, WITH price**. `price` =
  the slot's **effective (virtual_quote-aware) reserve ratio** `(sol_after + vq) / tok_after` at slot end — the
  same basis the live WS stream uses — carried forward over gaps. Extend `confirmed` **past each decision's
  N+1+hold** (a few slots beyond the last decision) so the target and the scalp entry/exit prices are in the record.
- **`frontier`** — at each N→N+1 decision, the **highest slot the WS had actually confirmed** then. This records
  the real confirmation depth, so the study rebases the prior at that depth and never uses a slot the box didn't
  have. Omit an entry and the adapter defaults that decision to lag-1 (prior = N-1). Emitting it is what lets us
  *measure* N-1 vs N-3 instead of assuming.
- **`pull_slot`** — the first slot with an LP withdrawal (or `null`); a scalp whose window crosses it is −100%.

## Running the study (no Helius)

```
python -m tradingbot flurry-study --frontrun --flows <capture.jsonl> --export-gate /data/gate_live.json
```

`--flows` + `flows_from_file` detect the capture by its `confirmed`/`shred` keys (vs a `rec` row) and route it
through the capture path — `_frontrun` builds the full flow from `confirmed` (ceiling / up-events / target / net
price), the decision slots from `shred`, and the per-slot feature bank from `_hybrid_bank_from_capture` (current
slot from shred, price-free; prior `m2`/`m3` + count `prev_*` from confirmed, capped at `frontier`). No
`localshred-merge`, no `history_window`. `--frontrun` prints the perfect-foresight ceiling, the lead-swept
z-guarded precursor discovery (the logit-gate shape), the enter-on-signal net, and the multivariate logistic;
`--export-gate` writes the train=serve gate the box fires live.

Read the **mean it reports** (ceiling, enter-on-signal net, logit-gate net). Because the study inputs are the
box's own feeds, that mean is honest to the feed we trade on.

### What "holds" means now

There is no offline bank to diverge from (study inputs == live inputs), so the old `frontrun-verify` is moot.
"Holds" is:

1. the **prospective walk-forward inside the capture** — `flows_from_file` splits on `create_ts`, so
   early-session pools train and later-session pools test (a real forward test), and
2. **fire the exported gate live** and read the rate/outcome vs the study mean. Live NET is the arbiter.

## Repo pieces (this change)

- `flurry_study._flow_from_series` — a slot-flow dict from a captured per-slot series (priced / price-free).
- `flurry_study._hybrid_from_flows` — the shared current-slot-shred + confirmed-prior rebasing (used by both the
  offline `_hybrid_bank` and the capture builder, so they can't drift).
- `flurry_study._hybrid_bank_from_capture` — the capture builder, prior capped at the captured `frontier`.
- `flurry_study._frontrun` — a capture branch (confirmed flow for price/target/net, shred flow for decision slots).
- `flurry_study.flows_from_file` — detects a capture row and loads it as a capture dict.
- Tests: `tests/test_flurry_capture.py`.

The box side — `run_hybrid` capture mode emitting the record above, and extending `confirmed` past each decision —
is box-only and not in this repo.
