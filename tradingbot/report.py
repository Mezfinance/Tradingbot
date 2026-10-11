"""Performance report built from the journal.

Everything is computed per strategy book so the Claude book can be compared with
the baselines on equal terms. Medians and a bootstrap interval are shown next to
means because memecoin returns are fat-tailed: one 10x trade can make a losing
strategy's average look good.
"""

from __future__ import annotations

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


def claude_selection(j: Journal) -> dict:
    """Forward returns of tokens Claude bought vs passed (6h after analysis)."""
    def fwd(action: str) -> list[float]:
        rows = j.query(
            """SELECT price, outcome_price FROM decisions WHERE stage='analysed' AND action=?
               AND outcome_price IS NOT NULL AND price > 0""", (action,))
        return [(r["outcome_price"] / r["price"] - 1) * 100 for r in rows]

    buys, passes = fwd("buy"), fwd("pass")
    return {
        "buy_samples": len(buys), "pass_samples": len(passes),
        "buy_mean_pct": _mean(buys), "pass_mean_pct": _mean(passes),
        "buy_median_pct": _median(buys), "pass_median_pct": _median(passes),
        "buy_minus_pass_mean_ci90": bootstrap_diff_ci(buys, passes),
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
    lines += ["", "CLAUDE SELECTION (price change 6h after analysis)",
              f"  bought: n={sel['buy_samples']} mean={_fmt(sel['buy_mean_pct'], '%')} "
              f"median={_fmt(sel['buy_median_pct'], '%')}",
              f"  passed: n={sel['pass_samples']} mean={_fmt(sel['pass_mean_pct'], '%')} "
              f"median={_fmt(sel['pass_median_pct'], '%')}"]
    ci = sel["buy_minus_pass_mean_ci90"]
    if ci:
        verdict = ("buys beat passes" if ci[0] > 0 else "passes beat buys" if ci[1] < 0
                   else "no significant difference yet")
        lines.append(f"  buy - pass, 90% interval: {ci[0]:+.1f}% to {ci[1]:+.1f}%  -> {verdict}")
    lines += ["", f"FUNNEL: {rep['funnel']}"]
    return "\n".join(lines)
