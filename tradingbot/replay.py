"""Replay risk settings over recorded candidate price paths.

  python -m tradingbot replay --group bought --set stop_loss_pct=20 --split 2026-10-18

Each candidate in the chosen group is entered at its recorded start price and run
through `risk.plan_exits` against its tracked path (one sample per scan cycle, ~60s,
for 24h). Costs come from fills the bot actually made: canary (real) fills when
there are 5+, otherwise the simulated Claude book, otherwise defaults.

When overrides are given, results are shown for the current settings and the
overridden ones side by side. Use --split to tune on the earlier period and check
on the later one: a setting that only wins before the split is overfitted.

Limits: no slot or daily-loss limits (per-trade view), and exits are checked at
the tracker's ~60s cadence rather than the live 10s monitor.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from tradingbot import risk
from tradingbot.config import Config
from tradingbot.journal import Journal
from tradingbot.models import Position

DEFAULT_COSTS = (1.5, 1.5, 0.33, 0.11)  # entry slip %, exit slip %, buy fee $, sell fee $
MIN_FILLS_FOR_COSTS = 5
GROUPS = ("bought", "passed", "survivors", "filters", "all")


@dataclass
class CostModel:
    entry_slip_pct: float
    exit_slip_pct: float
    buy_fee_usd: float
    sell_fee_usd: float
    source: str


@dataclass
class Trade:
    decision_id: int
    start_ts: float
    pnl_usd: float
    ret_pct: float
    exit_reason: str


def cost_model(j: Journal) -> CostModel:
    for book in ("canary", "claude"):
        rows = j.query(
            """SELECT f.side, f.price, f.trigger_price, f.fee_usd FROM fills f
               JOIN positions p ON p.id = f.position_id
               WHERE p.strategy=? AND f.trigger_price > 0""", (book,))
        buys = [r for r in rows if r["side"] == "buy"]
        sells = [r for r in rows if r["side"] == "sell"]
        if len(buys) >= MIN_FILLS_FOR_COSTS and len(sells) >= MIN_FILLS_FOR_COSTS:
            return CostModel(
                entry_slip_pct=statistics.median((r["price"] / r["trigger_price"] - 1) * 100 for r in buys),
                exit_slip_pct=statistics.median((1 - r["price"] / r["trigger_price"]) * 100 for r in sells),
                buy_fee_usd=statistics.median(r["fee_usd"] for r in buys),
                sell_fee_usd=statistics.median(r["fee_usd"] for r in sells),
                source=f"{book} fills ({len(buys)} buys, {len(sells)} sells)",
            )
    return CostModel(*DEFAULT_COSTS, source="defaults (not enough fills yet)")


def candidates(j: Journal, group: str, cfg: Config) -> list:
    where, params = {
        "bought": ("d.stage='analysed' AND d.action='buy' AND d.conviction >= ?", (cfg.min_conviction,)),
        "passed": ("d.stage='analysed' AND NOT (d.action='buy' AND d.conviction >= ?)", (cfg.min_conviction,)),
        "survivors": ("d.stage='survivor'", ()),
        "filters": ("d.factor_score >= ?", (cfg.min_factor_score,)),
        "all": ("1=1", ()),
    }[group]
    return j.query(
        f"""SELECT t.decision_id, t.token, t.start_ts, t.start_price FROM tracking t
            JOIN decisions d ON d.id = t.decision_id
            WHERE {where} AND t.start_price > 0 ORDER BY t.start_ts""",
        params,
    )


def simulate(cand, path: list[tuple[float, float]], cfg: Config, costs: CostModel, size_usd: float) -> Trade:
    entry = cand["start_price"] * (1 + costs.entry_slip_pct / 100)
    qty = (size_usd - costs.buy_fee_usd) / entry
    pos = Position(id=None, token_address=cand["token"], symbol="", entry_price=entry, qty=qty, initial_qty=qty,
                   cost_usd=size_usd, opened_at=cand["start_ts"], stop_price=risk.initial_stop(entry, cfg),
                   peak_price=entry)
    proceeds, reason = 0.0, "path_end"
    for ts, price in path:
        for sell_qty, why in risk.plan_exits(pos, price, ts, cfg):
            proceeds += sell_qty * price * (1 - costs.exit_slip_pct / 100) - costs.sell_fee_usd
            pos.qty -= sell_qty
            reason = why
        if pos.qty <= 1e-12 * qty:
            break
    else:
        if pos.qty > 0 and path:  # path ran out with tokens left: mark at the last price
            proceeds += pos.qty * path[-1][1] * (1 - costs.exit_slip_pct / 100) - costs.sell_fee_usd
    pnl = proceeds - size_usd
    return Trade(cand["decision_id"], cand["start_ts"], pnl, pnl / size_usd * 100, reason)


def run(j: Journal, cfg: Config, group: str, size_usd: float, costs: CostModel) -> list[Trade]:
    trades = []
    for c in candidates(j, group, cfg):
        path = [(r["ts"], r["price"]) for r in j.query(
            "SELECT ts, price FROM prices WHERE token=? AND ts>? AND ts<=? ORDER BY ts",
            (c["token"], c["start_ts"], c["start_ts"] + 24 * 3600 + 120))]
        if path:
            trades.append(simulate(c, path, cfg, costs, size_usd))
    return trades


def summarise(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    rets = [t.ret_pct for t in trades]
    wins = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    losses = -sum(t.pnl_usd for t in trades if t.pnl_usd <= 0)
    return {
        "n": len(trades),
        "win_rate": sum(t.pnl_usd > 0 for t in trades) / len(trades),
        "median_pct": statistics.median(rets),
        "mean_pct": sum(rets) / len(rets),
        "total_usd": sum(t.pnl_usd for t in trades),
        "profit_factor": wins / losses if losses else None,
        "exits": dict(Counter(t.exit_reason for t in trades)),
    }


def _line(label: str, s: dict) -> str:
    if not s["n"]:
        return f"  {label:<22} no trades"
    pf = "n/a" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
    return (f"  {label:<22} n={s['n']:<4} win={s['win_rate']:.0%}  median={s['median_pct']:+.1f}%  "
            f"mean={s['mean_pct']:+.1f}%  total=${s['total_usd']:+,.2f}  PF={pf}  exits={s['exits']}")


def report(j: Journal, base: Config, overrides: dict[str, str], group: str, size_usd: float,
           split: str | None) -> str:
    costs = cost_model(j)
    configs = [("current settings", base)]
    if overrides:
        tuned = replace(base).with_overrides(overrides)
        configs.append((", ".join(f"{k}={v}" for k, v in overrides.items()), tuned))
    split_ts = (datetime.fromisoformat(split).replace(tzinfo=timezone.utc).timestamp() if split else None)

    lines = [f"REPLAY group={group} size=${size_usd:.0f}",
             f"costs from {costs.source}: entry slip {costs.entry_slip_pct:.2f}%, exit slip "
             f"{costs.exit_slip_pct:.2f}%, fees ${costs.buy_fee_usd:.2f} buy / ${costs.sell_fee_usd:.2f} sell"]
    for label, cfg in configs:
        trades = run(j, cfg, group, size_usd, costs)
        lines.append(f"\n{label}:")
        if split_ts is None:
            lines.append(_line("all", summarise(trades)))
        else:
            lines.append(_line(f"before {split} (tune)", summarise([t for t in trades if t.start_ts < split_ts])))
            lines.append(_line(f"after {split} (check)", summarise([t for t in trades if t.start_ts >= split_ts])))
    return "\n".join(lines)
