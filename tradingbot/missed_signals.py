"""Signals a launch book was sent but didn't buy (11-09, read only, no Helius).

User: launch-status showed launchrw50_sz buying only ~40% of its picks over 3 h, and the missed picks did better at
ideal fills (+8.4% vs +0.2%). For every finished liquid pool the scorer flagged for the book: bought (real and
ideal result) or not, and if not, why (the engine's `entry_skipped` event for that token and book), with each
reason's ideal outcome, liquidity band, how long after creation the skip came, and what the missed pools would
have made at the book's own sizes (ideal fills, so before the book's real execution gap).
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import numpy as np

from tradingbot import launch_live as ll
from tradingbot import model_book as mb

BOOK = mb.LAUNCH_SIZED  # launchrw50_sz
WINDOW_S = 600  # a buy or skip within 10 min of the pool's creation belongs to that signal
FINISHED_S = 420  # pools younger than this have no minute-6 outcome yet
BANDS = (("$5-15k", 15_000.0), ("$15-25k", 25_000.0), ("$25k+", float("inf")))
LIST_N = 40
NO_EVENT = "not delivered (no skip logged)"


def reason_key(text: str) -> str:
    """Group skip reasons that differ only in numbers: 'no real liquidity (live price 16x ...)', 'round-trip loss
    99.4% ...', 'executor declined: price impact 93.1%'."""
    t = re.sub(r"\s*\(.*\)$", "", text or "").strip()
    t = re.sub(r"[-+]?\d[\d,.]*%?", "#", t)
    return t or "unknown"


def band_of(liq) -> str:
    for name, top in BANDS:
        if liq is not None and liq < top:
            return name
    return BANDS[-1][0]


def book_size(book: str, liq) -> float:
    base = mb.band_usd(book, liq) or 100.0
    return min(base, 0.02 * liq) if liq else base


def _summary(rows, rng, book) -> dict:
    out = ll._group(rows, rng)
    if not out.get("n"):
        return out
    out["usd_at_book_size"] = round(sum(book_size(book, r["liq_usd"]) * r["ret"] for r in rows))
    out["by_band"] = {b: sum(1 for r in rows if band_of(r["liq_usd"]) == b) for b, _ in BANDS}
    secs = [r["skip_s"] for r in rows if r.get("skip_s") is not None]
    if secs:
        out["skip_at_pool_second_median"] = round(float(np.median(secs)), 1)
    return out


def study(store, journal, book: str = BOOK, hours: float = 24.0, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    since = now - hours * 3600
    flag = mb.LAUNCH_CUT.get(book)
    if flag is None:
        return {"book": book, "error": f"{book} is not a launch book"}
    if not store.query("SELECT name FROM sqlite_master WHERE type='table' AND name='watch_launch_live'"):
        return {"book": book, "error": "no scorer table yet (watch_launch_live)"}
    pools = [dict(r) for r in store.query(
        f"SELECT pool, token_mint, create_ts, ret, {ll.FLAG_COLS} FROM watch_launch_live "
        "WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND create_ts >= ? AND create_ts <= ? "
        "ORDER BY create_ts", (since, now - FINISHED_S))]
    signals = [p for p in pools if ll.flag_hit(p, flag)]
    pos = {}
    for p in journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ?", (book, since)):
        pos.setdefault(p["token"], []).append(dict(p))
    skips = {}
    for e in journal.query("SELECT ts, token, detail FROM events WHERE kind='entry_skipped' AND strategy=? "
                           "AND ts >= ?", (book, since)):
        skips.setdefault(e["token"], []).append(dict(e))
    bought, missed = [], []
    for s in signals:
        lo, hi = s["create_ts"], s["create_ts"] + WINDOW_S
        hit = [p for p in pos.get(s["token_mint"], []) if lo <= p["opened_at"] <= hi]
        if hit:
            p = hit[0]
            real = (p["realized_usd"] / p["cost_usd"] - 1) if p["closed_at"] and p["cost_usd"] else None
            bought.append({**s, "real": real, "fill_s": p["opened_at"] - s["create_ts"]})
            continue
        ev = [e for e in skips.get(s["token_mint"], []) if lo <= e["ts"] <= hi]
        e = ev[0] if ev else None
        missed.append({**s, "reason": e["detail"] if e else NO_EVENT, "key": reason_key(e["detail"]) if e else NO_EVENT,
                       "skip_s": (e["ts"] - s["create_ts"]) if e else None})
    rng = np.random.default_rng(0)
    by_reason = {}
    for m in missed:
        by_reason.setdefault(m["key"], []).append(m)
    reasons = {k: {**_summary(v, rng, book), "share_of_signals_pct": round(100 * len(v) / max(1, len(signals)), 1),
                   "example": v[-1]["reason"]}
               for k, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))}
    real_rows = [{"ret": b["real"]} for b in bought if b["real"] is not None]
    closed = [b for b in bought if b["real"] is not None]
    gap = [b["real"] - b["ret"] for b in closed]
    recent = [{"created (UTC)": datetime.fromtimestamp(m["create_ts"], timezone.utc).strftime("%m-%d %H:%M:%S"),
               "token": m["token_mint"], "liq_usd": round(m["liq_usd"] or 0), "ideal_pct": round(100 * m["ret"], 1),
               "skip_s": round(m["skip_s"], 1) if m["skip_s"] is not None else None, "reason": m["reason"]}
              for m in missed[-LIST_N:]]
    return {"book": book, "flag": flag, "hours": hours, "since": since, "until": now,
            "signals": len(signals), "bought": len(bought), "missed": len(missed),
            "bought_pct": round(100 * len(bought) / max(1, len(signals)), 1),
            "bought_ideal": _summary(bought, rng, book),
            "bought_real": ll._group(real_rows, rng),
            "real_minus_ideal_pts": {"mean": round(100 * float(np.mean(gap)), 1),
                                     "median": round(100 * float(np.median(gap)), 1)} if gap else None,
            "fill_at_pool_second_median": round(float(np.median([b["fill_s"] for b in bought])), 1) if bought else None,
            "missed_ideal": _summary(missed, rng, book),
            "by_reason": reasons, "recent_missed": recent}


def _line(name, g, extra="") -> str:
    if not g.get("n"):
        return f"  {name:<46} {0:>5}"
    ci = g.get("ci90")
    ci_s = f"{ci[0]:+.1f}..{ci[1]:+.1f}" if ci else "-"
    return (f"  {name:<46} {g['n']:>5} {g['mean_pct']:>+7.1f} {g['median_pct']:>+7.1f} {g['win_pct']:>5} "
            f"{g['rugged_pct']:>5} {g['returns (% of trades)']['+50% or more']:>5} {ci_s:>14} "
            f"{format(g['usd_at_book_size'], ',') if 'usd_at_book_size' in g else '-':>8}{extra}")


def render(out: dict) -> str:
    if out.get("error"):
        return out["error"]
    when = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M")
    lines = [f"MISSED SIGNALS  {out['book']} (flag {out['flag']}), {when(out['since'])} .. {when(out['until'])} UTC",
             f"signals {out['signals']:,}, bought {out['bought']:,} ({out['bought_pct']}%), missed {out['missed']:,}; "
             f"bought filled at pool second {out['fill_at_pool_second_median']} (median)", "",
             f"  {'group (ideal fills unless marked real)':<46} {'n':>5} {'mean%':>7} {'med%':>7} {'win':>5} "
             f"{'rug':>5} {'+50':>5} {'90% CI':>14} {'$ @size':>8}",
             _line("bought: ideal, same pools", out["bought_ideal"]),
             _line("bought: real fills", out["bought_real"]),
             _line("missed: ideal", out["missed_ideal"])]
    gap = out.get("real_minus_ideal_pts")
    if gap:
        lines.append(f"  bought real minus ideal: mean {gap['mean']:+.1f} pts, median {gap['median']:+.1f} pts")
    lines += ["", "WHY MISSED (ideal outcome of each group; skip second = seconds after the pool's creation)",
              f"  {'reason':<46} {'n':>5} {'mean%':>7} {'med%':>7} {'win':>5} {'rug':>5} {'+50':>5} {'90% CI':>14} "
              f"{'$ @size':>8}  share  skip s   <15k/15-25k/25k+"]
    for k, g in out["by_reason"].items():
        b = g.get("by_band", {})
        extra = (f"  {g['share_of_signals_pct']:>4}% {g.get('skip_at_pool_second_median', '-'):>7}   "
                 f"{b.get('$5-15k', 0)}/{b.get('$15-25k', 0)}/{b.get('$25k+', 0)}")
        lines.append(_line(k[:46], g, extra))
    lines += ["", f"LAST {LIST_N} MISSED", f"  {'created':<15} {'liq $':>8} {'ideal%':>7} {'skip s':>7}  reason  token"]
    for m in out["recent_missed"]:
        lines.append(f"  {m['created (UTC)']:<15} {m['liq_usd']:>8,} {m['ideal_pct']:>+7.1f} "
                     f"{m['skip_s'] if m['skip_s'] is not None else '-':>7}  {m['reason']}  {m['token']}")
    lines += ["", "$ @size = the ideal result at the book's own trade size (2% pool cap), before its real execution gap. "
              "'transaction failed' is the simulator's revert: the re-quote after the landing delay fell below the "
              "first quote's slippage floor, so it tends to hit pools rising fast in those seconds."]
    return "\n".join(lines)
