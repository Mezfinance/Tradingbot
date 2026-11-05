"""Performance report built from the journal.

Everything is computed per strategy book so the Claude book can be compared with
the baselines on equal terms. Medians and a bootstrap interval are shown next to
means because memecoin returns are fat-tailed: one 10x trade can make a losing
strategy's average look good.
"""

from __future__ import annotations

import json
import time
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


def _blocked(j: Journal, name: str, pos) -> tuple[float, str] | None:
    """(since, reason) when the position has had no route out (exit_blocked events, or the no-route
    tx_failed rows older versions logged every monitor pass), else None."""
    rows = j.query(
        "SELECT MIN(ts) AS since, MAX(detail) AS detail FROM events WHERE strategy=? AND token=? AND ts >= ? "
        "AND (kind='exit_blocked' OR (kind='tx_failed' AND usd = 0 AND detail LIKE '%route%'))",
        (name, pos.token_address, pos.opened_at or 0))
    if not rows or rows[0]["since"] is None:
        return None
    return rows[0]["since"], rows[0]["detail"]


def _exit_blocked(j: Journal, name: str, pos) -> str | None:
    b = _blocked(j, name, pos)
    if b is None:
        return None
    since = time.strftime("%m-%d %H:%M", time.gmtime(b[0]))
    return f"since {since} UTC ({(time.time() - b[0]) / 3600:.1f}h): {b[1]}"


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
    # A position nobody can sell (no route for STUCK_AFTER_S) counts as a finished trade worth what it
    # already returned: leaving it out would keep every rug out of the win rate and average trade.
    from tradingbot.engine import STUCK_AFTER_S
    stuck = [p for p in open_pos
             if (b := _blocked(j, name, p)) is not None and time.time() - b[0] >= STUCK_AFTER_S]
    closed = list(closed) + [{"realized_usd": p.realized_usd, "cost_usd": p.cost_usd, "exit_reason": "stuck"}
                             for p in stuck]
    pnls = [r["realized_usd"] - r["cost_usd"] for r in closed]
    rets = [(r["realized_usd"] / r["cost_usd"] - 1) * 100 for r in closed if r["cost_usd"]]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p <= 0)

    fees = j.query(
        """SELECT COALESCE(SUM(f.fee_usd), 0) AS s FROM fills f JOIN positions p ON p.id=f.position_id
           WHERE p.strategy=?""", (name,))[0]["s"]
    # A "failed" sell with no route sent nothing (older versions logged one every monitor pass): not a tx.
    failed = j.query(
        "SELECT COUNT(*) AS n, COALESCE(SUM(usd), 0) AS s FROM events WHERE kind='tx_failed' AND strategy=? "
        "AND NOT (usd = 0 AND (detail LIKE '%no route%' OR detail LIKE '%route vanished%'))", (name,))[0]
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
             "stop": p.stop_price, "tp_hits": p.tp_hits, "exit_blocked": _exit_blocked(j, name, p)}
            for p in open_pos
        ],
        "closed_trades": len(closed),
        "stuck_trades": len(stuck),
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
    ("triaged out", "d.stage='triaged_out'"),
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


def learning(j: Journal) -> dict:
    cats = j.query("SELECT kind, category, COUNT(*) AS n FROM postmortems GROUP BY kind, category ORDER BY n DESC")
    spend = j.query("SELECT kind, COALESCE(SUM(usd), 0) AS s FROM costs "
                    "WHERE kind IN ('postmortem','playbook','optimizer','engineer') GROUP BY kind")
    from tradingbot.learning import playbook

    from tradingbot.learning import versions

    lessons = playbook.dump(j)
    active = versions.active(j)
    return {
        "settings_version": None if active is None else {"id": active["id"], "changes": active["changes"]},
        "open_code_proposals": j.query("SELECT COUNT(*) AS n FROM proposals WHERE status='open'")[0]["n"],
        "lessons": {s: sum(l["status"] == s for l in lessons) for s in ("active", "candidate", "retired")},
        "active_lessons": [l["statement"] for l in lessons if l["status"] == "active"],
        "playbook_ab": playbook.arm_report(j) if lessons else None,
        "postmortems": {f"{r['kind']}:{r['category']}": r["n"] for r in cats},
        "spend_usd": {r["kind"]: round(r["s"], 2) for r in spend},
    }


SOCIAL_SPLIT = 0.5


def social(j: Journal) -> dict:
    """Do tokens Claude rated socially strong actually do better? (All analysed tokens.)"""
    rows = j.query(
        """SELECT d.social_score AS s, t.start_price, t.p6h, t.max_price FROM decisions d
           JOIN tracking t ON t.decision_id = d.id
           WHERE d.stage='analysed' AND d.social_score IS NOT NULL AND t.start_price > 0""")
    out = {}
    for label, keep in (("high", lambda s: s >= SOCIAL_SPLIT), ("low", lambda s: s < SOCIAL_SPLIT)):
        grp = [r for r in rows if keep(r["s"])]
        p6 = [(r["p6h"] / r["start_price"] - 1) * 100 for r in grp if r["p6h"] is not None]
        up = [(r["max_price"] / r["start_price"] - 1) * 100 for r in grp]
        out[label] = {"n": len(grp), "median_6h_pct": _median(p6), "median_max_up_pct": _median(up)}
    return out


def triage(j: Journal, cfg: Config) -> dict:
    """What the cheap first pass sent on, what it skipped, and what Opus made of the audited skips."""
    rows = j.query("SELECT stage, action, conviction, triage FROM decisions WHERE triage IS NOT NULL")
    if not rows:
        return {}
    tri = [(r, json.loads(r["triage"])) for r in rows]
    audits = [r for r, t in tri if t.get("audit") and r["stage"] == "analysed"]
    spend = {r["kind"]: r["s"] for r in j.query(
        "SELECT kind, COALESCE(SUM(usd), 0) AS s, COUNT(*) AS n FROM costs "
        "WHERE kind IN ('triage', 'claude_api') GROUP BY kind")}
    n_full = j.query("SELECT COUNT(*) AS n FROM costs WHERE kind='claude_api'")[0]["n"]
    skipped = sum(r["stage"] == "triaged_out" for r, _ in tri)
    return {
        "checked": len(tri),
        "sent_on": sum(t["decision"] == "analyse" for _, t in tri),
        "skipped": skipped,
        "audited_skips": len(audits),
        "audited_skips_opus_would_buy": sum(r["action"] == "buy" and (r["conviction"] or 0) >= cfg.min_conviction
                                            for r in audits),
        "triage_spend_usd": spend.get("triage", 0.0),
        "full_analyses_avoided_usd": skipped * (spend.get("claude_api", 0.0) / n_full if n_full else 0.0),
    }


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
        "strategies": [strategy_stats(j, n, cfg.model_book_cash_usd if n.startswith("model5") else cfg.starting_cash_usd)
                       for n in names],
        "claude_selection": claude_selection(j),
        "downtime": downtime(j),
        "social": social(j),
        "triage": triage(j, cfg),
        "calibration": calibration(j),
        "learning": learning(j),
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
            ("  of which stuck ($0)", "stuck_trades", ""),
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
                blocked = f"  EXIT BLOCKED {p['exit_blocked']} (valued at $0 after 30 min)" if p.get("exit_blocked") else ""
                lines.append(f"[{s['strategy']}] open: {p['symbol']} cost ${p['cost_usd']} tp_hits={p['tp_hits']}{blocked}")
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
    lrn = rep.get("learning") or {}
    if lrn.get("settings_version") and lrn["settings_version"]["changes"] != "{}":
        sv = lrn["settings_version"]
        lines.append(f"\nLEARNED SETTINGS: v{sv['id']} {sv['changes']}  (see `settings` for history)")
    if lrn.get("postmortems"):
        lines.append(f"\nLEARNING: post-mortems {lrn['postmortems']}; spend {lrn['spend_usd']}")
        lines.append(f"  lessons: {lrn['lessons']}")
        for s in lrn["active_lessons"]:
            lines.append(f"  - {s}")
        if lrn["playbook_ab"]:
            lines.append(f"  playbook A/B (Claude trades): {lrn['playbook_ab']}")
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
    soc = rep.get("social") or {}
    if any(g["n"] for g in soc.values()):
        lines.append(f"\nSOCIAL SCORE vs OUTCOME (analysed tokens, split at {SOCIAL_SPLIT})")
        for label in ("high", "low"):
            g = soc[label]
            lines.append(f"  {label:<5} n={g['n']:<4} median 6h {_fmt(g['median_6h_pct'], '%', 1):>8}   "
                         f"median best {_fmt(g['median_max_up_pct'], '%', 1):>8}")
    tri = rep.get("triage") or {}
    if tri:
        lines.append(f"\nTRIAGE: {tri['checked']} checked, {tri['sent_on']} sent to full analysis, "
                     f"{tri['skipped']} skipped; triage cost ${tri['triage_spend_usd']:.2f}, "
                     f"~${tri['full_analyses_avoided_usd']:.2f} of full analyses avoided")
        lines.append(f"  audited skips (sent on anyway): {tri['audited_skips']}, "
                     f"of which the full analysis would buy {tri['audited_skips_opus_would_buy']}")
    dt = rep["downtime"]
    if dt["outages"]:
        lines.append(f"\nDOWNTIME: {dt['outages']} outage(s), {dt['hours_offline']:.1f}h offline, "
                     f"{dt['outages_with_open_positions']} with positions unmanaged")
    lines += ["", f"FUNNEL: {rep['funnel']}"]
    return "\n".join(lines)


# --- model books report (the default `report`; `report --all` is the full one) -------------------------
BAD_TRIGGER_MULT = 20.0


def _model_book_report(j: Journal, cfg: Config, name: str) -> dict:
    from tradingbot import model_book as mb
    from tradingbot.engine import STUCK_AFTER_S
    stats = strategy_stats(j, name, cfg.model_book_cash_usd)
    # Exit slippage without total-loss fills (a rugged pool's sale is a loss, not slippage).
    rows = j.query("""SELECT f.price, f.trigger_price FROM fills f JOIN positions p ON p.id=f.position_id
                      WHERE p.strategy=? AND f.side='sell' AND f.trigger_price > 0""", (name,))
    # Also without fills over BAD_TRIGGER_MULT x their trigger: the monitor's price was wrong (11-04: new pools of
    # existing tokens gave triggers ~10^7 x below the fill), which is not slippage either; counted separately.
    ok = [r for r in rows if 0.05 * r["trigger_price"] < r["price"] <= BAD_TRIGGER_MULT * r["trigger_price"]]
    stats["median_exit_slippage_excl_rugs_pct"] = _median([-(r["price"] / r["trigger_price"] - 1) * 100 for r in ok])
    stats["exits_with_bad_trigger"] = sum(r["price"] > BAD_TRIGGER_MULT * r["trigger_price"] for r in rows)
    now = time.time()
    open_pos = j.open_positions(name)
    stuck = {p.id for p in open_pos
             if (b := _blocked(j, name, p)) is not None and now - b[0] >= STUCK_AFTER_S}
    for p, row in zip(open_pos, stats["open_positions"]):
        row["hours"] = round((now - (p.opened_at or now)) / 3600, 1)
    # When the book started: its first equity snapshot (taken every minute while the engine runs), trade or signal.
    firsts = [j.query(q, (name,))[0]["t"] for q in (
        "SELECT MIN(ts) AS t FROM equity WHERE strategy=?",
        "SELECT MIN(opened_at) AS t FROM positions WHERE strategy=?",
        "SELECT MIN(ts) AS t FROM events WHERE strategy=?")]
    started = min((t for t in firsts if t), default=None)
    return {"name": name, "hold_min": mb.hold_min(cfg, name), "book": stats, "started": started,
            "signals": j.query("SELECT COUNT(*) AS n FROM events WHERE strategy=? AND kind IN ('model_buy', "
                               "'entry_skipped')", (name,))[0]["n"],
            "buys": j.query("SELECT COUNT(*) AS n FROM positions WHERE strategy=?", (name,))[0]["n"],
            "rugs": mb.rug_summary(j, stuck, name)}


def build_model(j: Journal, cfg: Config) -> dict:
    """The model paper books (same signals, separate cash and ledgers), and the model they trade with."""
    from tradingbot import model_book as mb
    names = mb.active_books(cfg)
    path = mb.model_path(cfg)
    model = json.loads(path.read_text()) if path.exists() else None
    return {"books": [_model_book_report(j, cfg, n) for n in names],
            "rug_catcher": mb.rug_catcher_summary(j, [n for n in names if n in mb.SHORT]),
            "model": {"threshold": model["threshold"], **model.get("meta", {})} if model else None,
            "downtime": downtime(j), "last_backup": json.loads(j.get_state("last_backup") or "null")}


def _running_for(started: float | None) -> str:
    if not started:
        return "-"
    h = (time.time() - started) / 3600
    return f"{h:.1f} h" if h < 48 else f"{h / 24:.1f} days"


EXPECTED = {  # 10-28 backtests on the 10-26 backup, for comparison with the live books
    "model5": "24h hold: -26%/trade, 62% rugged (rug-aware labels, 377 picks)",
    "model5_10m": "10-min hold: +17%/trade on 74 picks (10-14..10-24); out of sample (10-21..10-26) +1.7%, "
                  "90% range -8%..+10%",
    "model5_climb": "as model5_10m, out at 3 min unless +5%: +2.5%/trade, 7% rugged (vs 20%), 90% range "
                    "-5%..+10% (95 picks, 10-21..10-26)",
    "model5_ops9": "repeat-operator pools only, 9-min hold, no pool-age exit: +8.1%/trade, 13% rugged, 90% range "
                   "-2%..+18% (60 picks, 10-21..10-26, exact on-chain pulls; chosen after looking)",
    "model5_ops10": "as model5_ops9 with a 10-min hold: +7.6%/trade, 15% rugged, 90% range -4%..+18%",
    "early5": "hand-made pools at first sight (~minute 2), out by pool minute 5. All 134 backtest picks at ~minute 6: "
              "-2.9%/trade, 25% emptied, 90% range -11%..+6% (the +8.8% seen on the 60 with 1-min candles was biased)",
    "early6": "as early5, out by pool minute 6 (all picks ~-3%/trade; candle subset +9.9%, biased)",
    "early7": "as early5, out by pool minute 7 (all picks ~-3%/trade; candle subset +9.0%, biased)",
    "early5r": "as early5 but top 5% of the past day's eligible pools (rolling cut-off). Walk-forward at ~minute 6: "
               "-4.9%/trade, 26% emptied, 90% range -16%..+6% (fixed -6.5%, every eligible pool -4.1%)",
    "early6r": "as early5r, out by pool minute 6",
    "early7r": "as early5r, out by pool minute 7",
    "early6t": "as early6 but the top 2% of the recent training scores. Walk-forward 10-16..11-03 (snapshot "
               "yardstick): +1.1%/trade, 27% emptied, 90% range -7.5%..+9.9% (n=202); -8.3% to 10-29, +15.2% after",
    "early6w": "as early6 but a wider cut-off: top 16% of the recent training scores. Walk-forward "
               "(early-walk-forward, snapshot yardstick): see CLAUDE.md",
    "launch4_2": "hand-made pools scored on their first 4 seconds of chain data (websocket), top 2%, bought at pool "
                 "second ~6 through Jupiter, out at minute 6. Walk-forward 10-28..11-02 without names, n=23: "
                 "+32.9%/trade on GeckoTerminal snapshot prices but +20..+22% on 1-minute trade candles "
                 "(90% range -2..+46%), 13% rugged; random liquid pools -12% on candles",
    "launch4_5": "as launch4_2, top 5%, n=48: +26.7% on snapshot prices, +6% on trade candles (90% range "
                 "-12..+24%), 25% rugged",
    "launch4h_2": "history-trained model (every hand-made pool, chain prices, no hour/weekend), top 2% of the past "
                  "24 h's scores. 7-day walk-forward: +5.0%/trade (90% range -4.6..+14.5%), 25% rugged",
    "launch4_2m5": "launch4_2's picks sold at pool minute 5. launch-exit-study (old model top 8% of 906 live pools, "
                   "11-03/04): minute 5 +13.0%/trade (90% range -1.8..+28.6%), 19% rugged vs minute 6 +6.2%; "
                   "chosen after looking at minutes 1-6",
    "launch4h_5": "as launch4h_2, top 5%: +8.8%/trade (90% range +2.8..+14.7%), 23% rugged; every pool -5.2%, 31%",
    # launch-early-pulls walk-forward (11-03/04, 188 D1-D2 pools, ideal vault prices, ranked within 6 h blocks)
    "launchd12": "history model's bottom 20% (D1-D2), every pool: +2.9%/trade, 30% rugged, 24% big wins (n=188)",
    "launchd12_s10": "D1-D2 minus the riskiest 10% early pullers: +6.6%/trade, 27% rugged (n=169; random skip +2.9%)",
    "launchd12_s20": "D1-D2 minus the riskiest 20%: +5.5%/trade, 24% rugged (n=151; random skip +2.9%)",
    "launchd12_s30": "D1-D2 minus the riskiest 30%: +12.0%/trade, 18% rugged (n=132; random skip +2.7%, 90% range "
                     "-4.4..+10.2%)",
    "launchd12_s50": "D1-D2 minus the riskiest 50%: +12.8%/trade, 11% rugged (n=94; random skip +2.6%)",
    "launchrw50": "rug-vs-win, study rule (6 h ranking, retrained every 6 h on all pools). Live-rule replay on "
                  "2,039 pools (11-03..11-05): kept +11.5%/trade vs every pool -1.1%, 19% rugged vs 33% skipped, "
                  "100% of random keeps worse",
    "launchd12rw50": "rug-vs-win within the past 6 h's history bottom 20%, study rule. Live-rule replay: kept "
                     "+8.3%/trade (n=219), 20% rugged, 96% of random keeps worse",
    "launchrw50_24h": "rug-vs-win, earlier live rule (24 h ranking, nightly 14-day model). Replay (11-04..11-05, "
                      "n=1,179): kept +17.7%/trade vs every pool +3.9%, 17% rugged vs 31% skipped",
    "launchd12rw50_24h": "rug-vs-win within the past 24 h's history bottom 20%, earlier live rule. Replay: kept "
                         "+14.7%/trade (n=112), 12% rugged, 89% of random keeps worse",
}


WIDTH = 100  # the model report fits a laptop terminal at this width


def _cell(v, suffix="", digits=1) -> str:
    return "-" if v is None else f"{v:,.{digits}f}{suffix}" if isinstance(v, float) else f"{v}{suffix}"


def _short_audit(a: dict) -> str:
    m = a["median"]
    liq = m.get("liquidity")
    return (f"n={a['n_with_audit']} holders {_cell(m.get('holders'), '', 0)}, top {_cell(m.get('top_holders_pct'), '%')}, "
            f"dev {_cell(m.get('dev_balance_pct'), '%')}, liq {'$' + format(liq / 1000, ',.1f') + 'k' if liq else '-'}")


def render_model(rep: dict) -> str:
    """Books as rows, so the tables stay within WIDTH however many books run; details wrap at WIDTH."""
    import textwrap
    books, m = rep["books"], rep["model"]
    wrap = lambda text, indent="  ", more="      ": textwrap.fill(text, WIDTH, initial_indent=indent,
                                                             subsequent_indent=more)
    lines = ["MODEL PAPER BOOKS (separate $ and ledgers; simulated fills on real Jupiter quotes)", ""]
    head = f"  {'book':<14}{'hold':>5}{'ran':>9}{'equity $':>10}{'ret%':>7}{'maxDD':>6}{'trades':>7}{'win':>5}" \
           f"{'median':>8}{'mean':>8}{'PF':>6}{'rugged':>11}"
    lines += ["RESULTS", head, "  " + "-" * (len(head) - 2)]
    for b in books:
        s, r = b["book"], b["rugs"]
        lines.append(f"  {b['name']:<14}{b['hold_min']:>4g}m{_running_for(b['started']).replace(' days', 'd'):>9}"
                     f"{_cell(s['equity'], '', 0):>10}{_cell(s['return_pct']):>7}{_cell(s['max_drawdown_pct']):>6}"
                     f"{_cell(s['closed_trades']):>7}{_cell(s['win_rate'], '', 2):>5}"
                     f"{_cell(s['median_trade_return_pct'], '%'):>8}{_cell(s['mean_trade_return_pct'], '%'):>8}"
                     f"{_cell(s['profit_factor'], '', 2):>6}"
                     f"{str(r['rugs']) + ' (' + _cell(r['rug_rate_pct'], '%', 0) + ')':>11}")
    head = f"  {'book':<14}{'signals':>8}{'buys':>6}{'stuck':>6}{'fees $':>8}{'failed tx':>10}{'entry slip':>11}" \
           f"{'exit slip':>10}{'bad trigger':>12}"
    lines += ["", "EXECUTION (slippage = median vs trigger; exit excludes rugs and bad trigger prices)", head,
              "  " + "-" * (len(head) - 2)]
    for b in books:
        s = b["book"]
        lines.append(f"  {b['name']:<14}{b['signals']:>8}{b['buys']:>6}{_cell(s['stuck_trades']):>6}"
                     f"{_cell(s['network_fees_usd'], '', 0):>8}{_cell(s['failed_txs']):>10}"
                     f"{_cell(s['median_entry_slippage_pct'], '%'):>11}"
                     f"{_cell(s['median_exit_slippage_excl_rugs_pct'], '%'):>10}"
                     f"{s.get('exits_with_bad_trigger', 0):>12}")
    lines += ["", "BACKTEST EXPECTATIONS"]
    for b in books:
        lines.append(wrap(f"{b['name']}: {EXPECTED.get(b['name'], '-')}"))
    lines += ["", "BOOK DETAILS"]
    for b in books:
        s, r = b["book"], b["rugs"]
        since = time.strftime("%m-%d %H:%M", time.gmtime(b["started"])) if b["started"] else "not yet"
        lines += ["", f"[{b['name']}] running since (UTC) {since}, running for {_running_for(b['started'])}"]
        if not s["enough_trades"]:
            lines.append(f"  only {s['closed_trades']} finished trades; need {MIN_TRADES_FOR_CONCLUSION}+ before drawing conclusions")
        lines += [wrap(f"exits: {s['exit_reasons'] or '-'}"), wrap(f"skipped: {s['top_skip_reasons'] or '-'}"),
                  wrap(f"losing exits: {r['losing_exit_causes'] or '-'}")]
        for p in s["open_positions"] or []:
            blocked = f"  EXIT BLOCKED {p['exit_blocked']}" if p.get("exit_blocked") else ""
            lines.append(wrap(f"open: {p['symbol']:<12} ${p['cost_usd']:<6} {p['hours']}h{blocked}"))
        audits = [f"{label} {_short_audit(r[key])}" for label, key in (("rugs", "entry_audit_rugs"),
                                                                        ("others", "entry_audit_others"))
                  if r[key]["n_with_audit"]]
        if audits:
            lines.append(wrap("entry audit: " + " | ".join(audits)))
    rc = rep.get("rug_catcher") or {}
    if any(rc.values()):
        lines += ["", "RUG CATCHER (blind test: verdict logged before each signal was traded; trading unchanged)",
                  "  backtest 10-16..10-26: new-wallet pools 32% pulled during the hold (-26%/trade), "
                  "repeat operators 17% (-2.5%)"]
        for book, groups in rc.items():
            for verdict, g in sorted(groups.items()):
                lines.append(f"  {book:<13} {verdict:<20} signals {g['signals']:>4}  trades {g['trades']:>4}  "
                             f"P&L ${g['pnl_usd']:>9,.2f}  per trade {_fmt(g['per_trade_usd'], '', 2):>7}  "
                             f"rugged {_fmt(g['rug_rate_pct'], '%', 0)}")
    lines += ["", "MODEL"]
    if m:
        trained = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(m["trained_ts"])) if m.get("trained_ts") else "?"
        lines.append(f"  trained {trained} on {m.get('rows')} pools ({m.get('days')} days); pump rate "
                     f"{_fmt((m.get('pump_rate') or 0) * 100, '%', 1)}; cut-off {m['threshold']:.3f} (top {m.get('top_pct')}%)")
        lines.append(f"  in-sample top picks: pump rate {_fmt((m.get('top_pump_rate_in_sample') or 0) * 100, '%', 1)}, "
                     f"mean trade {_fmt((m.get('top_trade_ret_in_sample') or 0) * 100, '%', 1)} (optimistic: in-sample)")
    else:
        lines.append("  not trained yet")
    dt, bk = rep["downtime"], rep.get("last_backup")
    lines += ["", f"DOWNTIME: {dt['outages']} outage(s), {dt['hours_offline']:.1f}h offline, "
                  f"{dt['outages_with_open_positions']} with positions unmanaged",
              f"LAST BACKUP: {bk['date']} -> {bk['uploaded'] or 'volume only'}" if bk else "LAST BACKUP: none yet"]
    return "\n".join(lines)
