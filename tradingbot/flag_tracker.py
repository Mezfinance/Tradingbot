"""Flag tracker (11-09, read only, never acts): what each market-monitor flag has meant for each running book.

User: lots of flags have opened; can they turn things on or off automatically (save us from bad periods, lean into
good ones)? Not yet: each measure has 1-3 episodes, there are ~40 measures (some will look predictive by chance),
and the old summary counted the hours before a flag was even known. This module measures what can be acted on:

- the action window of a flag starts when it is CONFIRMED (the hour that completed the streak has finished and the
  monitor has stored it, `HOUR_DELAY_S` later) and ends when its close is known the same way;
- per book and per (measure, direction): every flag is one episode; an episode's gap = the book's size-weighted
  result in the window minus the book's result in every hour outside that measure's windows;
- pass bars fixed before looking (`MIN_EPISODES`, a confidence band adjusted for the number of measures tested,
  the same sign in both halves of the episodes); until then a rule reads 'watching';
- what-if responses priced from trades that already happened (half size, skip, thin pools only, double), so
  nothing new trades. Doubling ignores the extra price impact, so it reads optimistic.

Automatic switching is deliberately not built: a passing rule is reported and the user decides.
"""
from __future__ import annotations

import bisect
import collections
import json
import math
import time
from datetime import datetime, timezone
from statistics import NormalDist

from tradingbot import market_monitor as mm

STATE_KEY = "flag_tracker:summary"  # watch_state: the latest summary, written by the market monitor each step
MIN_EPISODES = 8  # usable episodes before a rule can pass
MIN_EP_TRADES = 5  # an episode needs this many of the book's trades in its window to count
ALPHA = 0.10  # two-sided, divided by the number of rules tested for the book (Bonferroni)
AFTER_H = 6  # also reported: the first hours after confirmation
STUCK_S = 1800  # an open trade older than this counts as -100% (as block-compare / the monitor)
THIN_USD = 15_000.0
H = mm.HOUR_S


def _when(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M")


def windows(flag: dict, now: float) -> tuple[float, float]:
    """(confirmed, end) in seconds: when the flag could first have been acted on, and when its close was known."""
    seen = flag.get("seen_hour")
    if seen is None:
        seen = flag["opened_hour"] + (mm.FLAG_HOURS - 1) * H
    start = seen + H + mm.HOUR_DELAY_S
    end = flag["closed_hour"] + H + mm.HOUR_DELAY_S if flag.get("closed_hour") is not None else now
    return start, max(start, end)


def books(cfg) -> list[str]:
    from tradingbot import model_book as mb
    keep = [b for b in mb.active_books(cfg) if b in mb.LAUNCH_CUT and b not in mb.CANARY_BOOKS]
    if mb.LAUNCH_SIZED in keep:  # sz first: it is the book the decisions are about
        keep.remove(mb.LAUNCH_SIZED)
        keep.insert(0, mb.LAUNCH_SIZED)
    return keep


def trades(journal, book: str, since: float, now: float, liq: dict) -> list[dict]:
    """The book's finished trades (closed, or open longer than STUCK_S = -100%), in buy order."""
    out = []
    for p in journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ? ORDER BY opened_at", (book, since)):
        cost = p["cost_usd"] or 0.0
        if cost <= 0:
            continue
        if p["closed_at"] is not None:
            pnl = (p["realized_usd"] or 0.0) - cost
        elif now - p["opened_at"] > STUCK_S:
            pnl = -cost
        else:
            continue
        out.append({"ts": p["opened_at"], "cost": cost, "pnl": pnl, "token": p["token"],
                    "liq": liq.get(p["token"])})
    return out


def _pct(ts):
    cost = sum(t["cost"] for t in ts)
    return 100 * sum(t["pnl"] for t in ts) / cost if cost else None


def _inside(intervals, ts):
    i = bisect.bisect_right([a for a, _ in intervals], ts) - 1
    return i >= 0 and ts < intervals[i][1]


def _group(book_trades, flags, now, k_tests, z_cut):
    """One (measure, direction) for one book: episodes, gap, confidence band, halves, verdict, what-ifs."""
    spans = sorted(windows(f, now) for f in flags)
    ins = [t for t in book_trades if _inside(spans, t["ts"])]
    base = [t for t in book_trades if not _inside(spans, t["ts"])]
    base_pct = _pct(base)
    eps = []
    for f in sorted(flags, key=lambda f: f["opened_hour"]):
        a, b = windows(f, now)
        ts = [t for t in book_trades if a <= t["ts"] < b]
        after = [t for t in book_trades if a <= t["ts"] < a + AFTER_H * H]
        p = _pct(ts)
        eps.append({"id": f.get("id"), "confirmed": a, "end": b, "open": f.get("closed_hour") is None,
                    "trades": len(ts), "pct": None if p is None else round(p, 1),
                    "gap": None if p is None or base_pct is None or len(ts) < MIN_EP_TRADES else p - base_pct,
                    "after_pct": _pct(after), "after_n": len(after),
                    "pnl": sum(t["pnl"] for t in ts),
                    "thick_pnl": sum(t["pnl"] for t in ts if t["liq"] is None or t["liq"] >= THIN_USD)})
    gaps = [e["gap"] for e in eps if e["gap"] is not None]
    n = len(gaps)
    mean = sum(gaps) / n if n else None
    se = (math.sqrt(sum((g - mean) ** 2 for g in gaps) / (n - 1)) / math.sqrt(n)) if n >= 2 else None
    ci = [mean - z_cut * se, mean + z_cut * se] if se is not None else None
    half = n // 2
    halves = None
    if n >= 4:
        h1, h2 = sum(gaps[:half]) / half, sum(gaps[half:]) / (n - half)
        halves = [round(h1, 1), round(h2, 1)]
    agree = bool(halves) and (halves[0] > 0) == (halves[1] > 0) == (mean > 0)
    clear = ci is not None and (ci[0] > 0 or ci[1] < 0)
    if n < MIN_EPISODES:
        verdict = f"watching ({n}/{MIN_EPISODES} episodes)"
    elif clear and agree:
        verdict = "signal: bad periods" if mean < 0 else "signal: good periods"
    else:
        verdict = "no clear effect"
    pnl_in = sum(e["pnl"] for e in eps)
    after_ts = [t for e in eps for t in book_trades if e["confirmed"] <= t["ts"] < e["confirmed"] + AFTER_H * H]
    after_pct = _pct(after_ts)
    return {"episodes": len(eps), "usable": n, "trades_in": len(ins), "in_pct": _r(_pct(ins)), "base_pct": _r(base_pct),
            "gap": _r(mean), "ci": [_r(c) for c in ci] if ci else None, "halves": halves, "verdict": verdict,
            "passing": verdict.startswith("signal"), "tests": k_tests,
            "after6_gap": _r(after_pct - base_pct) if after_pct is not None and base_pct is not None else None,
            "whatif_usd": {"as traded": round(pnl_in), "half size": round(-0.5 * pnl_in), "skip": round(-pnl_in),
                           "thin pools only": round(-sum(e["thick_pnl"] for e in eps)), "double": round(pnl_in)},
            "skip_helped": sum(1 for e in eps if e["trades"] and e["pnl"] < 0),
            "episodes_detail": [{**{k: v for k, v in e.items() if k not in ("thick_pnl",)},
                                 "gap": _r(e["gap"]), "after_pct": _r(e["after_pct"]), "pnl": round(e["pnl"])}
                                for e in eps]}


def _r(x):
    return None if x is None else round(x, 1)


def study(store, journal, cfg, now: float | None = None, book_list: list[str] | None = None) -> dict:
    """Every flag in `market_flags` against every running launch book's trades since the monitor's baseline."""
    from tradingbot import launch_live as ll
    now = time.time() if now is None else now
    since = getattr(cfg, "market_monitor_since", 0.0) or 0.0
    flags = mm.read_flags(store.query)
    ll.ensure_schema(store)
    liq = {r[0]: r[1] for r in store.query("SELECT token_mint, liq_usd FROM watch_launch_live "
                                           "WHERE liq_usd IS NOT NULL AND create_ts >= ?", (since - 3600,))}
    groups = collections.defaultdict(list)
    for f in flags:
        groups[(f["metric"], f["direction"])].append(f)
    out = {"built_ts": now, "since": since, "flags": len(flags),
           "rule": {"min_episodes": MIN_EPISODES, "min_episode_trades": MIN_EP_TRADES, "alpha": ALPHA,
                    "after_h": AFTER_H, "window": "from the flag's confirmation (its streak's last hour stored) "
                                                  "to when its close was stored"},
           "books": {}, "passing": []}
    for book in (book_list if book_list is not None else books(cfg)):
        bt = trades(journal, book, since, now, liq)
        # rules tested = measures with enough episodes for this book (at least 1, so the cut is defined)
        k = max(1, sum(1 for fs in groups.values() if len(fs) >= MIN_EPISODES))
        z_cut = NormalDist().inv_cdf(1 - ALPHA / (2 * k))
        rows = []
        for (metric, direction), fs in groups.items():
            g = _group(bt, fs, now, k, z_cut)
            rows.append({"metric": metric, "label": mm.LABELS.get(metric, metric), "direction": direction, **g})
            if g["passing"]:
                out["passing"].append({"book": book, "metric": metric, "label": mm.LABELS.get(metric, metric),
                                       "direction": direction, "verdict": g["verdict"], "gap": g["gap"],
                                       "ci": g["ci"], "usable": g["usable"]})
        rows.sort(key=lambda r: (-r["usable"], -abs(r["gap"] or 0)))
        out["books"][book] = {"trades": len(bt), "pct": _r(_pct(bt)), "tests": k, "z_cut": round(z_cut, 2),
                              "rules": rows}
    return out


def update(store, journal, cfg, now: float | None = None) -> dict:
    """Recompute and keep the summary in watch_state for the dashboard (episode details dropped to keep it small)."""
    out = study(store, journal, cfg, now)
    slim = json.loads(json.dumps(out))
    for b in slim["books"].values():
        for r in b["rules"]:
            r.pop("episodes_detail", None)
    store.set_state(STATE_KEY, json.dumps(slim))
    return out


def read(query) -> dict:
    """The stored summary through any read function; {} before the first build."""
    import sqlite3
    try:
        rows = query("SELECT value FROM watch_state WHERE key=?", (STATE_KEY,))
    except sqlite3.OperationalError:
        return {}
    return json.loads(rows[0][0]) if rows and rows[0][0] else {}


def render(out: dict, book: str | None = None) -> str:
    if not out.get("books"):
        return "FLAG TRACKER: no running launch books"
    r = out["rule"]
    lines = [f"FLAG TRACKER  ({out['flags']} flags since {_when(out['since'])} UTC; read only, nothing is switched)",
             f"window = {r['window']}; a rule passes with >= {r['min_episodes']} episodes of >= "
             f"{r['min_episode_trades']} trades, a {100 * (1 - r['alpha']):.0f}% band (adjusted for the number of "
             "rules tested) clear of zero, and the same sign in both halves of the episodes", ""]
    if out["passing"]:
        lines.append("PASSING RULES")
        for p in out["passing"]:
            lines.append(f"  {p['book']:<20} {p['label']:<38} {p['direction']:<5} {p['verdict']:<22} gap "
                         f"{p['gap']:+.1f} pts  band {p['ci'][0]:+.1f}..{p['ci'][1]:+.1f}  ({p['usable']} episodes)")
    else:
        lines.append("PASSING RULES: none yet")
    lines.append("")
    lines.append("BOOKS  (rules = measures with enough episodes to test; size-weighted result over the window)")
    for b, v in out["books"].items():
        passing = sum(1 for x in v["rules"] if x["passing"])
        lines.append(f"  {b:<22} trades {v['trades']:>6}  result {v['pct'] if v['pct'] is not None else '-':>6}%  "
                     f"rules tested {v['tests']:>2} (band z {v['z_cut']})  passing {passing}")
    show = book or next(iter(out["books"]))
    v = out["books"].get(show)
    if v:
        lines += ["", f"{show.upper()} BY MEASURE  (gap = mean of episode gaps, episode minus every hour outside "
                      "the measure's windows; $ = what each response would have changed in the windows)",
                  f"  {'measure':<34} {'dir':<4} {'eps':>7} {'trades':>6} {'in%':>6} {'other%':>6} {'gap':>6} "
                  f"{'band':>13} {'halves':>11} {'6h gap':>6} {'half $':>7} {'skip $':>7} {'thin $':>7} "
                  f"{'2x $':>7}  verdict"]
        for x in v["rules"]:
            band = f"{x['ci'][0]:+.0f}..{x['ci'][1]:+.0f}" if x["ci"] else "-"
            halves = f"{x['halves'][0]:+.0f}/{x['halves'][1]:+.0f}" if x["halves"] else "-"
            w = x["whatif_usd"]
            f = lambda y: "-" if y is None else f"{y:+.1f}"
            lines.append(f"  {x['label'][:34]:<34} {x['direction'][:4]:<4} {x['usable']:>3}/{x['episodes']:<3} "
                         f"{x['trades_in']:>6} {f(x['in_pct']):>6} {f(x['base_pct']):>6} {f(x['gap']):>6} {band:>13} "
                         f"{halves:>11} {f(x['after6_gap']):>6} {w['half size']:>7,} {w['skip']:>7,} "
                         f"{w['thin pools only']:>7,} {w['double']:>7,}  {x['verdict']}")
    lines += ["", "Doubling ignores the extra price impact (thin-sizing: ~5 pts more per trade from $100 to $250), so its "
                  "$ is an upper bound. A passing rule is reported only; switching anything stays your decision."]
    return "\n".join(lines)
