"""Slot-level timing of every real canary trade vs the operator's LP pull (11-12, read only, Helius research key).

The question: once our $25 buy lands, how many Solana slots later does the operator's liquidity withdraw land?
A slot is ~400 ms and is the hard floor for a *dependent* sell (our sell spends the tokens the buy produced, so
buy in slot N -> the earliest our sell can land is slot N+1). Block *time* is only 1-second granularity, which
can't tell these apart; slots can. So per trade:

  slot_gap 0   same slot as our buy  -> no separate sell can escape (the pool is drained in the same block)
  slot_gap 1   next slot             -> our fastest possible sell ties at best, usually loses
  slot_gap >=2 two+ slots later      -> a pre-signed fast exit could have sold before the pull

For each canary position with a buy tx and a pool: our buy slot (`getTransaction`), then the earliest liquidity
withdrawal on the pool at or after that slot whose fee payer is NOT one of our own wallets (so the LP route's own
exit withdraw is not mistaken for the rug). CLI `canary-slots [--hours N] [--max-credits 8000] [--rps 2] [--json]`.
"""

from __future__ import annotations

import json
import statistics

from tradingbot import canary_trades as ct
from tradingbot.data import helius as he

WINDOW_S = 600       # look this far past our buy for the pull if the scorer has no pull time
PRE_S = 3            # start the pool scan a few seconds before our buy
MAX_PAGES = 4
PAGE_CREDITS = 10    # getTransactionsForAddress, per 100 txs
TX_CREDITS = 1       # getTransaction


def _our_wallets(cfg) -> set:
    """Every address we sign from, so the LP route's own exit withdraw isn't counted as the rug."""
    out: set = set()
    try:
        from tradingbot.execution.rotating import read_canary_wallets
        for k in read_canary_wallets(cfg):
            out.add(str(k.pubkey()))
    except Exception:  # noqa: BLE001
        pass
    try:
        from tradingbot.execution.jupiter import read_keypair
        kp = read_keypair(cfg)
        if kp:
            out.add(str(kp.pubkey()))
    except Exception:  # noqa: BLE001
        pass
    return out


def _fee_payer(tx: dict) -> str | None:
    try:
        k = tx["transaction"]["message"]["accountKeys"][0]
        return k.get("pubkey") if isinstance(k, dict) else k
    except Exception:  # noqa: BLE001
        return None


def _get_tx(helius, sig: str) -> tuple[int | None, float | None]:
    res = helius._rpc("getTransaction", [sig, {"maxSupportedTransactionVersion": 1, "commitment": "confirmed",
                                               "encoding": "jsonParsed"}], TX_CREDITS)
    if not res:
        return None, None
    return res.get("slot"), res.get("blockTime")


def _pool_txs(helius, pool: str, lo: int, hi: int) -> list:
    txs, token = [], None
    for _ in range(MAX_PAGES):
        opts = {"transactionDetails": "full", "sortOrder": "asc", "limit": 100, "encoding": "jsonParsed",
                "maxSupportedTransactionVersion": 1, "filters": {"blockTime": {"gte": lo, "lte": hi}}}
        if token:
            opts["paginationToken"] = token
        res = helius._rpc("getTransactionsForAddress", [pool, opts], PAGE_CREDITS)
        if res is None:
            break
        page = res.get("data") or []
        txs += page
        token = res.get("paginationToken")
        if not page or not token:
            break
    return txs


def _withdraw(txs: list, after_slot: int | None, ours: set) -> dict | None:
    """The earliest successful liquidity withdrawal at or after our buy slot, by someone other than us."""
    hits = []
    for tx in txs:
        meta = tx.get("meta") or {}
        if meta.get("err"):
            continue
        if not any(he.WITHDRAW_RE.search(line) for line in (meta.get("logMessages") or [])):
            continue
        slot = tx.get("slot")
        if slot is None or (after_slot is not None and slot < after_slot):
            continue
        by = _fee_payer(tx)
        if by in ours:  # the LP route's own exit is a withdraw by us -> not the rug
            continue
        sig = (tx.get("transaction", {}).get("signatures") or [None])[0]
        hits.append({"slot": slot, "ts": tx.get("blockTime"), "by": by, "sig": sig})
    return min(hits, key=lambda h: (h["slot"], h["ts"] or 0)) if hits else None


RUG = -0.9  # lost 90%+ of the stake


def _base(t: dict) -> dict:
    return {"book": t["book"], "symbol": t["symbol"], "pool": t["pool"], "ret": t["ret"],
            "liq": t["liq"], "thin": t["thin"], "buy_tx": t["buy_tx"],
            "cost": t.get("cost"), "scorer_pull_s": t.get("pulled_after_fill_s")}


def _pull_s(r: dict) -> float | None:
    """Seconds from our buy to the pull: the exact withdraw-tx gap, else the scorer's pull time
    (which keeps watching the LP after our exit, so it also catches pulls later than minute 6)."""
    if r.get("time_gap_s") is not None:
        return r["time_gap_s"]
    return r.get("scorer_pull_s")


def _bucket(cost) -> str:
    if cost is None:
        return "?"
    c = round(cost)
    return f"${c}" if c >= 1 else "<$1"


def slots(store, journal, helius, cfg, max_credits: int, now: float | None = None) -> dict:
    tr = ct.trades(journal, store, now)
    real = [t for t in tr if t.get("buy_tx") and t.get("pool")]
    ours = _our_wallets(cfg)
    start = helius.credits_used
    rows, stopped = [], False
    for t in real:
        if helius.credits_used - start + TX_CREDITS + MAX_PAGES * PAGE_CREDITS > max_credits:
            stopped = True
            break
        bslot, btime = _get_tx(helius, t["buy_tx"])
        row = _base(t)
        if bslot is None:
            row["note"] = "buy tx not found (RPC history may not reach it)"
            rows.append(row)
            continue
        row["buy_slot"], row["buy_time"] = bslot, btime
        lo = int((btime or t["fill_ts"]) - PRE_S)
        hi = int((t["pulled_ts"] or (btime or t["fill_ts"]) + WINDOW_S) + 60)
        w = _withdraw(_pool_txs(helius, t["pool"], lo, hi), bslot, ours)
        if w:
            row.update({"pull_slot": w["slot"], "pull_time": w["ts"], "by": w["by"], "pull_tx": w["sig"],
                        "slot_gap": w["slot"] - bslot,
                        "time_gap_s": round(w["ts"] - btime, 2) if w["ts"] is not None and btime is not None else None})
        else:
            row["note"] = "no operator withdraw found at/after our buy"
        rows.append(row)
    return {"credits": helius.credits_used - start, "stopped_by_cap": stopped, "wallets": sorted(ours),
            "trades": rows, "summary": _summary(rows), "by_size": _by_size(rows)}


def _by_size(rows: list) -> dict:
    """Per trade size: how many were pulled, and when — so the LATEST pull (the safe-zone edge) is clear."""
    out: dict = {}
    for r in rows:
        b = _bucket(r.get("cost"))
        g = out.setdefault(b, {"n": 0, "pulled": 0, "rugged": 0, "pulls": [], "last": None, "rugged_no_pull": 0})
        g["n"] += 1
        ps = _pull_s(r)
        rug = r.get("ret") is not None and r["ret"] <= RUG
        if rug:
            g["rugged"] += 1
        if ps is not None:
            g["pulled"] += 1
            g["pulls"].append(ps)
            if g["last"] is None or ps > g["last"][0]:
                g["last"] = (ps, r.get("symbol"))
        elif rug:
            g["rugged_no_pull"] += 1  # lost the stake but no pull time on record -> the max could be later
    for g in out.values():
        p = g.pop("pulls")
        g["pull_s_min"] = round(min(p), 1) if p else None
        g["pull_s_median"] = round(statistics.median(p), 1) if p else None
        g["pull_s_max"] = round(max(p), 1) if p else None
    return out


def _summary(rows: list) -> dict:
    gapped = [r for r in rows if r.get("slot_gap") is not None]
    if not gapped:
        return {"trades": len(rows), "with_pull": 0}
    gaps = [r["slot_gap"] for r in gapped]
    tg = [r["time_gap_s"] for r in gapped if r.get("time_gap_s") is not None]

    def share(pred) -> float:
        return round(100 * sum(1 for g in gaps if pred(g)) / len(gaps), 1)

    return {
        "trades": len(rows), "with_pull": len(gapped),
        "median_slot_gap": statistics.median(gaps),
        "median_time_gap_s": round(statistics.median(tg), 2) if tg else None,
        "unbeatable_le1_slot_pct": share(lambda g: g <= 1),  # same or next slot: a dependent sell can't escape
        "beatable_ge2_slots_pct": share(lambda g: g >= 2),   # a pre-signed fast exit had a window
        "slot_gap_distribution_pct": {
            "same_slot_0": share(lambda g: g == 0), "next_slot_1": share(lambda g: g == 1),
            "2": share(lambda g: g == 2), "3_5": share(lambda g: 3 <= g <= 5), "6_plus": share(lambda g: g >= 6)},
    }


def render(out: dict) -> str:
    s = out["summary"]
    L = ["CANARY SLOT TIMING  (buy slot vs the operator's LP withdraw; a slot is ~400 ms)",
         f"credits {out['credits']}" + ("  [stopped by --max-credits]" if out["stopped_by_cap"] else ""),
         f"trades {s['trades']}, with a pull found {s.get('with_pull', 0)}"]
    if s.get("with_pull"):
        d = s["slot_gap_distribution_pct"]
        L += [f"median slot gap {s['median_slot_gap']}  (median time gap {s['median_time_gap_s']} s)",
              f"same slot (0): {d['same_slot_0']}%   next slot (1): {d['next_slot_1']}%   "
              f"2: {d['2']}%   3-5: {d['3_5']}%   6+: {d['6_plus']}%",
              f"UNBEATABLE (<=1 slot): {s['unbeatable_le1_slot_pct']}%    "
              f"a fast exit had a window (>=2 slots): {s['beatable_ge2_slots_pct']}%"]
    bs = out.get("by_size") or {}
    if bs:
        L += ["", "BY TRADE SIZE  (pull s = seconds from our buy to the pull; MAX = the latest rug = safe-zone edge)",
              f"{'size':>6} {'n':>4} {'pulled':>7} {'rugged':>7} {'min':>6} {'median':>7} {'MAX':>7}  latest rug"]

        def _k(b: str) -> float:
            try:
                return float(b.lstrip("$")) if b.startswith("$") else -1
            except ValueError:
                return 1e9
        for b in sorted(bs, key=_k, reverse=True):
            g = bs[b]
            last = f"+{g['last'][0]:.0f}s ({g['last'][1]})" if g.get("last") else ""
            extra = f"   [{g['rugged_no_pull']} rugged, no pull time]" if g.get("rugged_no_pull") else ""
            L.append(f"{b:>6} {g['n']:>4} {g['pulled']:>7} {g['rugged']:>7} "
                     f"{str(g['pull_s_min'] if g['pull_s_min'] is not None else ''):>6} "
                     f"{str(g['pull_s_median'] if g['pull_s_median'] is not None else ''):>7} "
                     f"{str(g['pull_s_max'] if g['pull_s_max'] is not None else ''):>7}  {last}{extra}")
    L += ["", "TRADES", f"{'symbol':<10} {'$liq':>7} {'ret%':>6} {'buy slot':>12} {'pull slot':>12} "
          f"{'gap':>4} {'s':>5}  note"]
    for r in out["trades"]:
        ret = f"{r['ret'] * 100:.0f}" if r.get("ret") is not None else ""
        gap = r.get("slot_gap")
        L.append(f"{(r['symbol'] or '')[:10]:<10} {(r['liq'] or 0):>7.0f} {ret:>6} "
                 f"{str(r.get('buy_slot') or ''):>12} {str(r.get('pull_slot') or ''):>12} "
                 f"{('' if gap is None else gap):>4} {('' if r.get('time_gap_s') is None else r['time_gap_s']):>5}  "
                 f"{r.get('note', '')}")
    return "\n".join(L)
