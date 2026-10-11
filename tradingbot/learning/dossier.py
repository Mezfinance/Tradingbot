"""Evidence packs for post-mortems: everything the journal knows about one trade or candidate."""

from __future__ import annotations

import json
import statistics

from tradingbot.journal import Journal

MAX_PATH_POINTS = 80
AFTER_EXIT_S = 6 * 3600
MARKET_WINDOW_S = 3 * 3600


def _loads(raw):
    try:
        return json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _decision(j: Journal, decision_id: int | None) -> dict | None:
    if decision_id is None:
        return None
    rows = j.query("SELECT * FROM decisions WHERE id=?", (decision_id,))
    if not rows:
        return None
    d = rows[0]
    snapshot = _loads(d["snapshot"]) or {}
    snapshot.pop("socials", None)
    return {
        "decided_at": d["ts"],
        "stage": d["stage"],
        "factor_score": d["factor_score"],
        "factor_components": _loads(d["components"]),
        "market_snapshot": snapshot,
        "claude_verdict": _loads(d["verdict"]),
    }


def price_path(j: Journal, token: str, start_ts: float, end_ts: float, ref_price: float) -> list[list[float]]:
    """[[minutes since start, % vs ref price], ...], evenly thinned to at most MAX_PATH_POINTS."""
    rows = j.query("SELECT ts, price FROM prices WHERE token=? AND ts>=? AND ts<=? ORDER BY ts",
                   (token, start_ts, end_ts))
    if not rows or not ref_price:
        return []
    step = max(1, len(rows) // MAX_PATH_POINTS)
    picked = rows[::step]
    if picked[-1] is not rows[-1]:
        picked.append(rows[-1])
    return [[round((r["ts"] - start_ts) / 60, 1), round((r["price"] / ref_price - 1) * 100, 1)] for r in picked]


def market_context(j: Journal, around_ts: float, exclude_token: str) -> dict:
    """How other candidates moved around the same time: separates our pick from a market-wide move."""
    rows = j.query(
        """SELECT start_price, p1h, p6h FROM tracking
           WHERE start_ts BETWEEN ? AND ? AND token != ? AND start_price > 0""",
        (around_ts - MARKET_WINDOW_S, around_ts + MARKET_WINDOW_S, exclude_token))

    def med(col):
        vals = [(r[col] / r["start_price"] - 1) * 100 for r in rows if r[col] is not None]
        return round(statistics.median(vals), 1) if vals else None

    return {"other_candidates": len(rows), "median_1h_change_pct": med("p1h"), "median_6h_change_pct": med("p6h")}


def trade_dossier(j: Journal, position_id: int) -> dict | None:
    rows = j.query("SELECT * FROM positions WHERE id=?", (position_id,))
    if not rows:
        return None
    p = rows[0]
    fills = j.query("SELECT * FROM fills WHERE position_id=? ORDER BY ts", (position_id,))
    entry = p["entry_price"]
    fill_info = []
    for f in fills:
        slip = None
        if f["trigger_price"]:
            sign = 1 if f["side"] == "buy" else -1
            slip = round(sign * (f["price"] / f["trigger_price"] - 1) * 100, 2)
        fill_info.append({
            "side": f["side"], "reason": f["reason"], "minutes_after_entry": round((f["ts"] - p["opened_at"]) / 60, 1),
            "price_vs_entry_pct": round((f["price"] / entry - 1) * 100, 1) if entry else None,
            "usd": round(f["usd"], 2), "fee_usd": round(f["fee_usd"] or 0, 3), "slippage_vs_trigger_pct": slip,
        })
    closed = p["closed_at"] or p["opened_at"]
    during = price_path(j, p["token"], p["opened_at"], closed, entry)
    after = price_path(j, p["token"], closed, closed + AFTER_EXIT_S, entry)
    held = [pt[1] for pt in during] or [0.0]
    others = j.query(
        """SELECT strategy, realized_usd, cost_usd, exit_reason, closed_at FROM positions
           WHERE token=? AND id != ? AND ABS(opened_at - ?) < 3600""",
        (p["token"], position_id, p["opened_at"]))
    return {
        "kind": "trade",
        "symbol": p["symbol"],
        "token": p["token"],
        "strategy": p["strategy"],
        "result": {
            "return_pct": round((p["realized_usd"] / p["cost_usd"] - 1) * 100, 2) if p["cost_usd"] else None,
            "pnl_usd": round(p["realized_usd"] - p["cost_usd"], 2),
            "cost_usd": round(p["cost_usd"], 2),
            "exit_reason": p["exit_reason"],
            "hold_minutes": round((closed - p["opened_at"]) / 60, 1),
            "take_profits_hit": p["tp_hits"],
            "best_during_hold_pct": max(held),
            "worst_during_hold_pct": min(held),
            "best_in_6h_after_exit_pct": max((pt[1] for pt in after), default=None),
            "last_seen_6h_after_exit_pct": after[-1][1] if after else None,
        },
        "fills": fill_info,
        "price_path_during_hold": during,
        "price_path_after_exit": after,
        "entry_decision": _decision(j, p["decision_id"]),
        "same_token_in_other_books": [
            {"strategy": o["strategy"], "exit_reason": o["exit_reason"],
             "return_pct": round((o["realized_usd"] / o["cost_usd"] - 1) * 100, 2) if o["cost_usd"] else None}
            for o in others],
        "market_context": market_context(j, p["opened_at"], p["token"]),
    }


def missed_dossier(j: Journal, decision_id: int) -> dict | None:
    rows = j.query("SELECT * FROM tracking WHERE decision_id=?", (decision_id,))
    if not rows:
        return None
    t = rows[0]
    start = t["start_price"]

    def pct(v):
        return None if v is None or not start else round((v / start - 1) * 100, 1)

    return {
        "kind": "missed",
        "token": t["token"],
        "outcome_after_pass": {"change_1h_pct": pct(t["p1h"]), "change_6h_pct": pct(t["p6h"]),
                               "change_24h_pct": pct(t["p24h"]), "best_pct": pct(t["max_price"]),
                               "worst_pct": pct(t["min_price"])},
        "price_path": price_path(j, t["token"], t["start_ts"], t["start_ts"] + 24 * 3600, start),
        "decision": _decision(j, decision_id),
        "market_context": market_context(j, t["start_ts"], t["token"]),
    }
