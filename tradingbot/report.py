"""Performance report built from the journal.

Everything is computed per strategy book so the Claude book can be compared with
the baselines on equal terms. Medians and a bootstrap interval are shown next to
means because memecoin returns are fat-tailed: one 10x trade can make a losing
strategy's average look good.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter

from tradingbot.config import Config
from tradingbot.journal import Journal

MIN_TRADES_FOR_CONCLUSION = 30
BOOTSTRAP_ROUNDS = 2000


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def max_drawdown_pct(series: list[float]) -> float:
    peak, worst = float("-inf"), 0.0
    for v in series:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak * 100)
    return worst


def bootstrap_diff_ci(a: list[float], b: list[float], level: float = 0.90,
                      rounds: int = BOOTSTRAP_ROUNDS, seed: int = 0) -> tuple[float, float] | None:
    """Interval for mean(a) - mean(b). If it excludes 0 the gap is unlikely to be noise."""
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(seed)
    diffs = sorted(
        _mean(rng.choices(a, k=len(a))) - _mean(rng.choices(b, k=len(b))) for _ in range(rounds)
    )
    lo = diffs[int((1 - level) / 2 * rounds)]
    hi = diffs[int((1 + level) / 2 * rounds) - 1]
    return lo, hi


def _reason_key(detail: str) -> str:
    """Collapse numbers so 'price ran +22.1%' and 'price ran +31.0%' group together."""
    return re.sub(r"[-+]?\d[\d.,]*%?", "#", detail or "?")


def strategy_stats(j: Journal, name: str, start_cash: float) -> dict:
    eq_rows = j.query("SELECT equity FROM equity WHERE strategy=? ORDER BY ts", (name,))
    series = [r["equity"] for r in eq_rows]
    cash_raw = j.get_state(f"cash:{name}")
    cash = float(cash_raw) if cash_raw is not None else start_cash
    open_pos = j.open_positions(name)
    equity_now = series[-1] if series else cash + sum(p.qty * p.entry_price for p in open_pos)

    closed = j.query(
        "SELECT realized_usd, cost_usd, exit_reason FROM positions WHERE strategy=? AND closed_at IS NOT NULL",
        (name,),
    )
    pnls = [r["realized_usd"] - r["cost_usd"] for r in closed]
    rets = [(r["realized_usd"] / r["cost_usd"] - 1) * 100 for r in closed if r["cost_usd"]]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p <= 0)

    fees = j.query(
        """SELECT COALESCE(SUM(f.fee_usd), 0) AS s FROM fills f JOIN positions p ON p.id=f.position_id
           WHERE p.strategy=?""", (name,))[0]["s"]
    failed = j.query(
        "SELECT COUNT(*) AS n, COALESCE(SUM(usd), 0) AS s FROM events WHERE kind='tx_failed' AND strategy=?",
        (name,))[0]
    api = j.query("SELECT COALESCE(SUM(usd), 0) AS s FROM costs WHERE strategy=?", (name,))[0]["s"]

    slip = {}
    for side in ("buy", "sell"):
        rows = j.query(
            """SELECT f.price, f.trigger_price FROM fills f JOIN positions p ON p.id=f.position_id
               WHERE p.strategy=? AND f.side=? AND f.trigger_price > 0""", (name, side))
        # Positive = worse than the trigger price (paid more on a buy, got less on a sell).
        sign = 1 if side == "buy" else -1
        slip[side] = _median([sign * (r["price"] / r["trigger_price"] - 1) * 100 for r in rows])

    skips = Counter(_reason_key(r["detail"]) for r in j.query(
        "SELECT detail FROM events WHERE kind='entry_skipped' AND strategy=?", (name,)))

    trading_pnl = equity_now - start_cash
    return {
        "strategy": name,
        "start_equity": start_cash,
        "equity": equity_now,
        "return_pct": trading_pnl / start_cash * 100,
        "max_drawdown_pct": max_drawdown_pct([start_cash] + series),
        "open_positions": [
            {"symbol": p.symbol, "cost_usd": round(p.cost_usd, 2), "qty": p.qty, "entry": p.entry_price,
             "stop": p.stop_price, "tp_hits": p.tp_hits}
            for p in open_pos
        ],
        "closed_trades": len(closed),
        "win_rate": sum(p > 0 for p in pnls) / len(pnls) if pnls else None,
        "median_trade_return_pct": _median(rets),
        "mean_trade_return_pct": _mean(rets),
        "best_trade_pct": max(rets) if rets else None,
        "worst_trade_pct": min(rets) if rets else None,
        "profit_factor": gross_win / gross_loss if gross_loss else None,
        "exit_reasons": dict(Counter(r["exit_reason"] for r in closed)),
        "network_fees_usd": fees + failed["s"],
        "failed_txs": failed["n"],
        "claude_api_usd": api,
        "net_after_api_usd": trading_pnl - api,
        "median_entry_slippage_pct": slip["buy"],
        "median_exit_slippage_pct": slip["sell"],
        "top_skip_reasons": dict(skips.most_common(5)),
        "enough_trades": len(closed) >= MIN_TRADES_FOR_CONCLUSION,
    }


GROUPS = (
    ("bought", "d.stage='analysed' AND d.action='buy'"),
    ("passed", "d.stage='analysed' AND d.action='pass'"),
    ("not analysed", "d.stage='survivor'"),
)


def claude_selection(j: Journal) -> dict:
    """Forward returns of candidates after 1h/6h/24h, plus the best and worst point reached.

    Uses the price paths from tracker.py, so every candidate counts, bought or not.
    """
    out: dict = {}
    for label, where in GROUPS:
        rows = j.query(f"""SELECT t.* FROM tracking t JOIN decisions d ON d.id = t.decision_id
                           WHERE {where} AND t.start_price > 0""")
        g: dict = {"n": len(rows)}
        for col in ("p1h", "p6h", "p24h"):
            rets = [(r[col] / r["start_price"] - 1) * 100 for r in rows if r[col] is not None]
            g[col] = {"n": len(rets), "mean": _mean(rets), "median": _median(rets), "values": rets}
        g["median_max_up_pct"] = _median([(r["max_price"] / r["start_price"] - 1) * 100 for r in rows])
        g["median_max_down_pct"] = _median([(r["min_price"] / r["start_price"] - 1) * 100 for r in rows])
        out[label] = g
    for col in ("p6h", "p24h"):
        out[f"buy_minus_pass_{col}_ci90"] = bootstrap_diff_ci(
            out["bought"][col]["values"], out["passed"][col]["values"])
    for g in (out[label] for label, _ in GROUPS):
        for col in ("p1h", "p6h", "p24h"):
            g[col].pop("values")
    return out


def _swap_return_pct(j: Journal, position_id: int) -> float | None:
    """Return from fill prices only (fees excluded), so a $5 canary compares fairly with a $50 sim trade."""
    rows = j.query("SELECT side, qty, price FROM fills WHERE position_id=?", (position_id,))
    bought = sum(r["qty"] * r["price"] for r in rows if r["side"] == "buy")
    sold = sum(r["qty"] * r["price"] for r in rows if r["side"] == "sell")
    return (sold / bought - 1) * 100 if bought else None


def calibration(j: Journal) -> dict:
    """Real canary trades vs the simulated Claude trade on the same decision."""
    pairs = j.query(
        """SELECT c.id AS canary_id, s.id AS sim_id FROM positions c
           JOIN positions s ON s.decision_id = c.decision_id AND s.strategy='claude'
           WHERE c.strategy='canary' AND c.closed_at IS NOT NULL AND s.closed_at IS NOT NULL""")
    gaps = []
    for p in pairs:
        real, sim = _swap_return_pct(j, p["canary_id"]), _swap_return_pct(j, p["sim_id"])
        if real is not None and sim is not None:
            gaps.append(real - sim)
    gap = _median(gaps)
    advice = None
    if gap is not None and len(gaps) >= 5:
        if gap < -1:
            advice = "simulation is optimistic; try TB_SIM_FILL_MODE=pessimistic or raise TB_SIM_LATENCY_S"
        elif gap > 1:
            advice = "simulation is pessimistic; try TB_SIM_FILL_MODE=expected"
        else:
            advice = "simulation is within 1 point of real fills"
    return {"matched_trades": len(gaps), "median_real_minus_sim_pct": gap, "advice": advice}


def downtime(j: Journal) -> dict:
    rows = j.query("SELECT start_ts, end_ts, open_positions FROM downtime")
    return {
        "outages": len(rows),
        "hours_offline": sum(r["end_ts"] - r["start_ts"] for r in rows) / 3600,
        "outages_with_open_positions": sum(r["open_positions"] > 0 for r in rows),
    }


def build(j: Journal, cfg: Config) -> dict:
    names = [r["strategy"] for r in j.query(
        "SELECT DISTINCT substr(key, 6) AS strategy FROM state WHERE key LIKE 'cash:%' ORDER BY key")]
    if "claude" in names:  # main book first
        names.remove("claude")
        names.insert(0, "claude")
    funnel = j.query(
        "SELECT stage, COUNT(*) AS n FROM decisions GROUP BY stage")
    return {
        "strategies": [strategy_stats(j, n, cfg.starting_cash_usd) for n in names],
        "claude_selection": claude_selection(j),
        "downtime": downtime(j),
        "calibration": calibration(j),
        "last_backup": json.loads(j.get_state("last_backup") or "null"),
        "funnel": {r["stage"]: r["n"] for r in funnel},
    }


def _fmt(v, suffix="", digits=2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{digits}f}{suffix}"
    return f"{v}{suffix}"


def render(rep: dict) -> str:
    lines = []
    strategies = rep["strategies"]
    if strategies:
        lines.append("STRATEGY COMPARISON (same risk rules and fill model for every book)")
        header = f"{'metric':<26}" + "".join(f"{s['strategy']:>16}" for s in strategies)
        lines += [header, "-" * len(header)]
        rows = [
            ("equity $", "equity", ""), ("return %", "return_pct", "%"),
            ("max drawdown %", "max_drawdown_pct", "%"), ("closed trades", "closed_trades", ""),
            ("win rate", "win_rate", ""), ("median trade %", "median_trade_return_pct", "%"),
            ("mean trade %", "mean_trade_return_pct", "%"), ("profit factor", "profit_factor", ""),
            ("network fees $", "network_fees_usd", ""), ("failed txs", "failed_txs", ""),
            ("Claude API $", "claude_api_usd", ""), ("net after API $", "net_after_api_usd", ""),
            ("entry slippage % (med)", "median_entry_slippage_pct", "%"),
            ("exit slippage % (med)", "median_exit_slippage_pct", "%"),
        ]
        for label, key, suffix in rows:
            lines.append(f"{label:<26}" + "".join(f"{_fmt(s[key], suffix):>16}" for s in strategies))
        main = strategies[0]
        for s in strategies[1:]:
            edge = main["net_after_api_usd"] - s["net_after_api_usd"]
            lines.append(f"{main['strategy']} vs {s['strategy']}: {edge:+,.2f} USD after Claude API costs "
                         f"({'Claude adds value' if edge > 0 else 'Claude does not pay for itself'} so far)")
        for s in strategies:
            lines.append("")
            lines.append(f"[{s['strategy']}] exits: {s['exit_reasons'] or '-'}")
            lines.append(f"[{s['strategy']}] skipped entries: {s['top_skip_reasons'] or '-'}")
            for p in s["open_positions"]:
                lines.append(f"[{s['strategy']}] open: {p['symbol']} cost ${p['cost_usd']} tp_hits={p['tp_hits']}")
            if not s["enough_trades"]:
                lines.append(f"[{s['strategy']}] only {s['closed_trades']} closed trades; "
                             f"need {MIN_TRADES_FOR_CONCLUSION}+ before drawing conclusions")

    sel = rep["claude_selection"]
    lines += ["", "CANDIDATE OUTCOMES (price change after analysis; median, with mean in brackets)",
              f"{'group':<14}{'n':>5}{'1h':>18}{'6h':>18}{'24h':>18}{'max up':>10}{'max down':>10}"]
    for label, _ in GROUPS:
        g = sel[label]
        cells = "".join(f"{_fmt(g[c]['median'], '%', 1) + ' (' + _fmt(g[c]['mean'], '%', 1) + ')':>18}"
                        for c in ("p1h", "p6h", "p24h"))
        lines.append(f"{label:<14}{g['n']:>5}{cells}"
                     f"{_fmt(g['median_max_up_pct'], '%', 1):>10}{_fmt(g['median_max_down_pct'], '%', 1):>10}")
    for col in ("p6h", "p24h"):
        ci = sel[f"buy_minus_pass_{col}_ci90"]
        if ci:
            verdict = ("buys beat passes" if ci[0] > 0 else "passes beat buys" if ci[1] < 0
                       else "no significant difference yet")
            lines.append(f"  bought - passed at {col[1:]}, 90% interval: {ci[0]:+.1f}% to {ci[1]:+.1f}% -> {verdict}")
    cal = rep["calibration"]
    if cal["matched_trades"]:
        lines.append(f"\nCALIBRATION: {cal['matched_trades']} real canary trades vs their simulated twins; "
                     f"median real - sim = {_fmt(cal['median_real_minus_sim_pct'], ' pts')}"
                     + (f" -> {cal['advice']}" if cal["advice"] else " (need 5+ for advice)"))
    bk = rep.get("last_backup")
    if bk:
        where = bk["uploaded"] or "volume only"
        lines.append(f"\nLAST BACKUP: {bk['date']} -> {where}" + (f"  FAILED: {bk['error']}" if bk["error"] else ""))
    else:
        lines.append("\nLAST BACKUP: none yet")
    dt = rep["downtime"]
    if dt["outages"]:
        lines.append(f"\nDOWNTIME: {dt['outages']} outage(s), {dt['hours_offline']:.1f}h offline, "
                     f"{dt['outages_with_open_positions']} with positions unmanaged")
    lines += ["", f"FUNNEL: {rep['funnel']}"]
    return "\n".join(lines)
