"""Were the canary's rugs bad luck, or did real money in the pool set them off? (CLI `canary-rug-check`, read only.)

1. Streaks: the scorer's thin rug-vs-win picks (the canary's universe: flag `rw50`, liquidity at second 4 below
   `launch_canary_max_liq_usd`) in creation order, and launchrw50_sz's own thin trades in buy order: rug rate, the
   longest runs of back-to-back rugs and when they happened.
2. Bought vs skipped: since the canary's first trade it has bought only 1 in N of these picks, chosen by a counter,
   not by anything about the pool. If the pools it bought rug far more often than the ones it skipped in the same
   hours, the real buy (or what operators see of it) is the likely cause. Per canary trade: seconds from our fill to
   the LP pull, the scorer's ideal result, the paper twin's and the canary's own.
"""

from __future__ import annotations

import math
import statistics
import time
from datetime import datetime, timezone

from tradingbot import launch_live
from tradingbot import model_book as mb

RUG_RET = -0.999  # the scorer's rugged: pool SOL < 10% of its peak or LP supply 0 by minute 6 (ret -100%)
STUCK_S = 1800  # an open trade older than this counts as rugged (exit blocked, written off)
TOP_STREAKS = 5


def _ts(x) -> str | None:
    return datetime.fromtimestamp(x, timezone.utc).strftime("%m-%d %H:%M:%S") if x else None


def _streaks(flags: list[tuple[float, bool]]) -> list[dict]:
    """Runs of consecutive True in time order, longest first."""
    runs, start, n = [], None, 0
    for ts, rug in flags + [(None, False)]:
        if rug:
            start = ts if n == 0 else start
            n += 1
            last = ts
        elif n:
            runs.append({"length": n, "from": _ts(start), "to": _ts(last)})
            n = 0
    return sorted(runs, key=lambda r: -r["length"])


def _summary(flags: list[tuple[float, bool]]) -> dict:
    n = len(flags)
    rugs = sum(r for _, r in flags)
    p = rugs / n if n else 0.0
    runs = _streaks(flags)
    windows = sum(1 for i in range(n - 6) if all(r for _, r in flags[i:i + 7]))
    return {"n": n, "rugged": rugs, "rug_pct": round(100 * p, 1),
            "longest_streak": runs[0]["length"] if runs else 0, "top_streaks": runs[:TOP_STREAKS],
            "runs_of_7_or_more": sum(1 for r in runs if r["length"] >= 7),
            "windows_of_7_all_rugged": windows,
            "chance_7_in_a_row_at_this_rate": f"1 in {round(1 / p ** 7):,}" if 0 < p < 1 else None}


def _fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p: chance of >= a rugs among the a+b bought pools if rugs fell at random."""
    n1, n2, k = a + b, c + d, a + c
    total = math.comb(n1 + n2, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / total if total else 1.0


def _trades(journal, like: str, since: float = 0.0) -> list[dict]:
    out = []
    now = time.time()
    for p in journal.query("SELECT id, strategy, token, symbol, opened_at, closed_at, cost_usd, realized_usd, "
                           "exit_reason FROM positions WHERE strategy LIKE ? AND opened_at >= ? ORDER BY opened_at",
                           (like, since)):
        buy = journal.query("SELECT ts FROM fills WHERE position_id=? AND side='buy' ORDER BY ts LIMIT 1", (p["id"],))
        if p["closed_at"]:
            ret = (p["realized_usd"] / p["cost_usd"] - 1) if p["cost_usd"] else None
            rug = bool(p["cost_usd"]) and p["realized_usd"] <= 0.01 * p["cost_usd"]
        else:
            ret, rug = None, now - p["opened_at"] > STUCK_S
        out.append({**dict(p), "fill_ts": buy[0]["ts"] if buy else p["opened_at"], "ret": ret, "rug": rug})
    return out


def rug_check(store, journal, cfg, hours: float | None = None) -> dict:
    launch_live.ensure_schema(store)
    max_liq = cfg.launch_canary_max_liq_usd or 15_000.0
    since = time.time() - hours * 3600 if hours else 0.0
    rows = store.query("SELECT pool, token_mint, create_ts, fill_ts, pulled_ts, ret, liq_usd, rw_pct FROM "
                       "watch_launch_live WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND liq_usd < ? "
                       "AND rw_pct IS NOT NULL AND create_ts >= ? ORDER BY create_ts", (max_liq, since))
    picks = [r for r in rows if launch_live.rw_flags(False, r["rw_pct"], None)["rw50"]]
    out = {"universe": f"scorer's rw50 picks with liquidity < ${max_liq:,.0f} at second 4 (ideal trades, minute-6 exit)",
           "thin picks": _summary([(r["create_ts"], r["ret"] <= RUG_RET) for r in picks])}

    # the paper book's own thin trades (its real simulated fills), matched to the scorer's liquidity
    by_mint = {}
    for r in rows:
        by_mint.setdefault(r["token_mint"], r)
    twin = [t for t in _trades(journal, mb.CANARY_TWIN, since) if t["token"] in by_mint]
    out[f"{mb.CANARY_TWIN} thin trades"] = _summary([(t["opened_at"], t["rug"]) for t in twin])

    # bought vs skipped since the canary started (archived resets count: canary_rw50_sz@<tag>)
    canary = _trades(journal, mb.CANARY4 + "%", since)
    if not canary:
        out["canary"] = "no canary trades yet"
        return out
    start = min(c["opened_at"] for c in canary) - 60
    bought = {c["token"] for c in canary}
    after = [r for r in picks if r["create_ts"] >= start]
    b = [r for r in after if r["token_mint"] in bought]
    s = [r for r in after if r["token_mint"] not in bought]
    rb, rs = sum(r["ret"] <= RUG_RET for r in b), sum(r["ret"] <= RUG_RET for r in s)
    pull_s = lambda g: [r["pulled_ts"] - r["create_ts"] for r in g if r["pulled_ts"] and r["ret"] <= RUG_RET]
    med = lambda xs: round(statistics.median(xs), 1) if xs else None
    out["since the canary started"] = {
        "from": _ts(start),
        "thin picks the canary bought": {"n": len(b), "rugged": rb, "rug_pct": round(100 * rb / len(b), 1) if b else None,
                                         "median pull second (rugged)": med(pull_s(b))},
        "thin picks the canary skipped": {"n": len(s), "rugged": rs, "rug_pct": round(100 * rs / len(s), 1) if s else None,
                                          "median pull second (rugged)": med(pull_s(s))},
        "chance the bought ones rug this often if our buy changed nothing": round(_fisher_greater(rb, len(b) - rb, rs, len(s) - rs), 4),
        "note": ("canary trades not in the scorer's thin rw50 picks are left out of this split; the scorer's ideal "
                 "result is used for both groups, so a failed canary sale shows up as rugged in the table below but "
                 "not here"),
    }
    twin_by = {t["token"]: t for t in _trades(journal, mb.CANARY_TWIN, start)}
    table = []
    for c in canary:
        r = by_mint.get(c["token"]) or (store.query(
            "SELECT pool, token_mint, create_ts, fill_ts, pulled_ts, ret, liq_usd, rw_pct FROM watch_launch_live "
            "WHERE token_mint=? ORDER BY create_ts DESC LIMIT 1", (c["token"],)) or [None])[0]
        t = twin_by.get(c["token"])
        created = r["create_ts"] if r else None
        table.append({
            "symbol": c["symbol"], "token": c["token"], "book": c["strategy"], "created": _ts(created),
            "liq_usd": round(r["liq_usd"]) if r and r["liq_usd"] else None,
            "canary filled at pool second": round(c["fill_ts"] - created, 1) if created else None,
            "twin filled at pool second": round(t["fill_ts"] - created, 1) if t and created else None,
            "LP pulled at pool second": round(r["pulled_ts"] - created, 1) if r and r["pulled_ts"] else None,
            "pulled s after our fill": round(r["pulled_ts"] - c["fill_ts"], 1) if r and r["pulled_ts"] else None,
            "scorer ideal %": round(100 * r["ret"], 1) if r and r["ret"] is not None else None,
            "twin %": round(100 * t["ret"], 1) if t and t["ret"] is not None else None,
            "canary %": round(100 * c["ret"], 1) if c["ret"] is not None else ("open" if not c["rug"] else "stuck"),
            "canary exit": c["exit_reason"],
        })
    out["canary trades"] = table
    return out
