"""Keep only the fresh pumpswap pools in a per-tx capture (frontrun_live.PerTxCapture / box run_ws.py --capture-tx).

The box streams every pumpswap swap in real time, so a pool CREATED DURING the capture session is caught from its
first swap (its first captured slot sits within a few slots of its creation slot), while a pool that existed BEFORE
the session is tuned in mid-life (first captured slot far after creation). The capture's per-tx rows carry the chain
`slot`, so one static per-pool read of the pool's creation slot separates the two:

    keep iff   txs[0].slot - creation_slot <= fresh_slots   (~450 slots ~= 180 s, the live pool_fresh window)

This is NOT the Helius HISTORY reconstruction the owner ruled out (we never refetch or rebuild the trade stream --
the capture's own txs stay untouched). It is a single creation-slot read per pool, the SAME check the live
`pool_fresh` / Runner `_fresh_ok` gate does, so the filtered set stays train == serve. The same pass reads each kept
pool's virtual_quote (i128 @245) and stamps rec["vq"], so the study's up-move LABELS are on the effective
(vq-aware) basis --frontrun forces, not the raw-ratio basis that manufactures early-life artifacts.

Research (backup) Helius key only, never the live key. ~2 credits per pool (one getSignaturesForAddress, one
getMultipleAccounts).
"""

from __future__ import annotations

from .flurry_study import _SLOT, _pool_vq

FRESH_SLOTS = 450  # ~180 s at ~400 ms/slot: the live fresh window. Caught within this of creation = fresh.


def _dedup_richest(rows: list[dict]) -> list[dict]:
    """One capture rec per hand-made pumpswap pool (the capture can emit a pool across windows); keep the richest
    (most txs). Drops rows with no pool/mint/txs and pump.fun graduations (…pump mints), which the study excludes."""
    best: dict[str, dict] = {}
    for r in rows:
        pool, mint = r.get("pool"), r.get("token_mint")
        rec = r.get("rec") or {}
        txs = rec.get("txs") or []
        if not pool or not mint or not txs or mint.endswith("pump"):
            continue
        if pool not in best or len(txs) > len((best[pool].get("rec") or {}).get("txs") or []):
            best[pool] = r
    return list(best.values())


def fresh_filter(rows: list[dict], helius, fresh_slots: int = FRESH_SLOTS, max_credits: int = 0,
                 progress=None) -> tuple[list[dict], dict]:
    """Keep only pools caught fresh and stamp each kept pool's vq. `rows` are PerTxCapture recs
    {pool, token_mint, create_ts, rec:{create_ts, txs:[[sec, slot, ...], ...]}}. Returns (kept_rows, stats).

    stats: total (unique hand-made pools), fresh, stale (first tx too long after creation), unreadable (creation
    slot or first-tx slot missing), credits (research credits spent), capped (True if max_credits stopped it early)."""
    rows = _dedup_richest(rows)
    kept: list[dict] = []
    stats = {"total": len(rows), "fresh": 0, "stale": 0, "unreadable": 0, "credits": 0, "capped": False}
    for i, r in enumerate(rows):
        if max_credits and stats["credits"] + 2 > max_credits:
            stats["capped"] = True
            break
        pool, rec = r["pool"], r["rec"]
        txs = rec.get("txs") or []
        first_slot = txs[0][_SLOT] if txs else None
        cslot = helius.first_slot(pool)
        stats["credits"] += 1
        if cslot is None or first_slot is None:
            stats["unreadable"] += 1
            continue
        if first_slot - cslot > fresh_slots:
            stats["stale"] += 1
            continue
        accts = helius.accounts([pool])
        stats["credits"] += 1
        data = (accts or {}).get(pool)
        rec["vq"] = _pool_vq(data[1]) if data else 0
        kept.append(r)
        stats["fresh"] += 1
        if progress:
            progress(i + 1, len(rows), stats)
    return kept, stats
