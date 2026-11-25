"""Helius transactionSubscribe -> per-slot FULL aggregate for the box's hybrid rebase feed (the repo-testable half).

The HybridRunner needs the HISTORY rebased to the complete (Helius) view: every swap on a pool, classified by the
pool's vault deltas (so aggregator-routed CPIs count too, which shreds cannot decode). The box subscribes to each
fresh pool (`transactionSubscribe`, accountInclude=[pool], processed) -- one subscription per pool, not the whole
pumpswap program, to keep the streamed-data credit meter small -- and feeds each tx here:

    agg = FullSlotAggregator(runner.on_full_slot)       # runner = frontrun_live.HybridRunner
    for tx in helius_tx_stream(pool):                   # a transactionSubscribe notification shaped {slot, transaction, meta}
        sw = swap_from_tx(tx, pool, token_mint)
        if sw:
            agg.push(sw)                                # emits the prior slot's FULL aggregate to runner.on_full_slot

This produces the SAME per-slot aggregate shape flurry_study.slot_flow emits for the full view (slot, n_buys,
buy_sol, max_buy, n_sells, sell_sol, buyers, price), so the full SlotBank's records match the study's full bank.
The WS I/O loop + the notification->tx shaping are box-side (run_localshred.py), validated on the box; this decode
+ aggregation is pure and tested here.
"""
# Inlined copies of helius._vault / _keys / SOL_MINT so this module runs on the box's FLAT layout (no tradingbot
# package there). Pinned to the helius originals by tests/test_helius_feature_ws.py so they cannot drift.
SOL_MINT = "So11111111111111111111111111111111111111112"


def _vault(tx: dict, pool: str, mint: str, side: str) -> float:
    for b in (tx.get("meta") or {}).get(side) or []:
        if b.get("owner") == pool and b.get("mint") == mint:
            return float((b.get("uiTokenAmount") or {}).get("uiAmount") or 0)
    return 0.0


def _keys(tx: dict) -> list:
    keys = ((tx.get("transaction") or {}).get("message") or {}).get("accountKeys") or []
    return [k["pubkey"] if isinstance(k, dict) else k for k in keys]


def normalize_notification(result: dict) -> dict:
    """A Helius transactionSubscribe `params.result` -> the flat {slot, transaction:{message}, meta} shape
    swap_from_tx expects. The real notification (confirmed on the box, 11-23) nests one level deeper:
    result = {slot, signature, transaction: {transaction: {signatures, message:{accountKeys}}, meta, version}}.
    The subscription MUST send maxSupportedTransactionVersion >= 1, else Helius returns an error stub with no
    body (slot/signature/error only) and this yields an empty tx. jsonParsed accountKeys are {pubkey,...} dicts,
    which _keys already handles."""
    inner = result.get("transaction") or {}
    return {"slot": result.get("slot"),
            "transaction": inner.get("transaction") or {},   # {signatures, message}
            "meta": inner.get("meta") or {}}


def swap_from_tx(tx: dict, pool: str, token_mint: str) -> dict | None:
    """One Helius tx (shaped {slot, transaction, meta}) -> a swap dict for the full bank, or None (reverted, or no
    change to THIS pool's vaults). buyer = fee payer (keys[0]); the pool's token / SOL vault deltas give direction,
    exactly as helius.launch_record. A buy = tokens leave the pool (tok_delta < 0) and SOL enters (sol_delta > 0)."""
    if (tx.get("meta") or {}).get("err") is not None:
        return None                                      # reverted: slot_flow's full view excludes failed txs
    b0 = _vault(tx, pool, token_mint, "preTokenBalances")
    b1 = _vault(tx, pool, token_mint, "postTokenBalances")
    q0 = _vault(tx, pool, SOL_MINT, "preTokenBalances")
    q1 = _vault(tx, pool, SOL_MINT, "postTokenBalances")
    tok_delta, sol_delta = b1 - b0, q1 - q0
    if tok_delta == 0 and sol_delta == 0:
        return None                                      # not a swap that touched this pool
    keys = _keys(tx)
    return {"slot": tx.get("slot"), "payer": (keys[0] if keys else None),
            "tok_delta": round(tok_delta, 6), "sol_delta": round(sol_delta, 9),
            "tok_after": round(b1, 6), "sol_after": round(q1, 9)}


class FullSlotAggregator:
    """Accumulate pool swaps into per-slot FULL aggregates matching flurry_study.slot_flow's full view, online.
    push(swap) in slot order; when the slot advances, the prior slot's aggregate is emitted to on_slot (the box
    wires on_slot = HybridRunner.on_full_slot). Call flush() at teardown to emit the last open slot."""

    def __init__(self, on_slot):
        self.on_slot = on_slot
        self.cur: dict | None = None
        self.last_price: float | None = None

    @staticmethod
    def _new(slot: int) -> dict:
        return {"slot": slot, "n_buys": 0, "buy_sol": 0.0, "max_buy": 0.0, "n_sells": 0, "sell_sol": 0.0,
                "buyers": [], "price": None}

    def push(self, swap: dict) -> None:
        slot = swap["slot"]
        if self.cur is not None and slot != self.cur["slot"]:
            self._flush()
        if self.cur is None:
            self.cur = self._new(slot)
        s = self.cur
        tokd, sold = swap["tok_delta"], swap["sol_delta"]
        if tokd < 0 and sold > 0:                        # buy: tokens out, SOL in
            s["n_buys"] += 1
            s["buy_sol"] += sold
            s["max_buy"] = max(s["max_buy"], sold)
            s["buyers"].append(swap["payer"])
        elif tokd > 0 and sold < 0:                      # sell: tokens in, SOL out
            s["n_sells"] += 1
            s["sell_sol"] += -sold
        # a pull / seed moves no wallet state but still updates the reserve price
        if swap["tok_after"]:
            s["price"] = swap["sol_after"] / swap["tok_after"]

    def _flush(self) -> None:
        if self.cur is None:
            return
        if self.cur["price"] is None:                    # no readable reserve this slot: carry the last (price is
            self.cur["price"] = self.last_price if self.last_price is not None else 0.0  # gate-irrelevant anyway)
        self.last_price = self.cur["price"]
        self.on_slot(self.cur)
        self.cur = None

    def flush(self) -> None:
        self._flush()

    def flush_if_before(self, slot: int) -> None:
        """Emit the open slot if it is older than `slot`. A global slot watermark (the max slot seen across ALL
        pools) calls this to finalize a quiet pool's slot the instant the chain advances past it, instead of
        waiting for THIS pool's next trade -- cutting the per-pool finalization lag to ~1 slot."""
        if self.cur is not None and self.cur["slot"] < slot:
            self._flush()
