"""Safeguards for bad markets, tested on the year replay (`safeguards`, 11-09, read only; no Helius).

The year replay (risk_history) showed the rug-vs-win books lose in crash months (Sep 2026: every pool -12.5%, the
rules -12..-20% a trade) while making most of their money in hot ones. A safeguard is only worth having if it cuts
the crash months without giving up the good ones, so each rule here is scored on both, on the same replayed trades:

  * which months are bad: the book's own replayed result for the month is below zero;
  * saved = the share of the bad months' losses the rule avoided; kept = the share of the good months' profit it
    still made; plus total $, max drawdown, worst day / month, and the share of trades taken.

Every rule is causal: at a trade's decision (pool second ~6) it only sees pools that finished (minute 6) before then,
and the book's own trades that had already been sold. Rules:

  market Hh < T   pause while the ideal result of every liquid pool finished in the past H hours averages below T
  rugs Hh > T     pause while more than T% of the pools finished in the past H hours were rugged
  edge 24h < T    pause while the rule's own edge (kept half minus skipped half, past 24 h) is below T points
  own Hh < T      pause while the book's own trades sold in the past H hours average below T (size-weighted)
  day stop k%     stop for the rest of the UTC day once the day's realized loss passes k% of a typical day's money
  ... (half)      the same trigger, but trade at half size instead of pausing

A window with fewer than MIN_POOLS sampled pools (or MIN_TRADES own trades) gives no signal: trade on. In the sampled
history every pool stands for 1/share pools (weight w); all sums are weighted, so $ figures are full-size estimates.
Thresholds come from a grid looked at all at once: trust a rule only when its neighbours on the grid agree."""
from __future__ import annotations

import collections
import json
import time
from bisect import bisect_left
from datetime import datetime, timezone

import numpy as np

MIN_POOLS = 20
MIN_TRADES = 20
DAY = 86400.0
REPLAY_SCHEMA = """CREATE TABLE IF NOT EXISTS hist_replay(pool TEXT PRIMARY KEY, data TEXT)"""
REPLAY_KEYS = ("pool", "create_ts", "ret", "liq_usd", "rw", "rw_pct", "rwb_pct", "w", "pulled_ts", "token_mint",
               "entry_px", "peak_sol", "raw", "src", "big")

MARKET = [(6, t) for t in (-5, -10, -15, -20)] + [(24, t) for t in (-5, -10, -15)]
RUGS = [(6, t) for t in (35, 40, 45, 50)] + [(24, t) for t in (35, 40)]
EDGE = [(24, t) for t in (0, 5)]
OWN = [(6, t) for t in (-5, -10, -20)] + [(24, t) for t in (0, -5, -10)]
DAYSTOP = (5, 10, 20, 30)


def save_replay(hist_store, rows: list[dict], meta: dict) -> None:
    """The replay's per-pool result, so later studies skip the ~80-minute replay."""
    hist_store.execute(REPLAY_SCHEMA)
    hist_store.execute("DELETE FROM hist_replay")
    hist_store.executemany("INSERT INTO hist_replay(pool, data) VALUES (?, ?)",
                           [(r["pool"], json.dumps({k: r.get(k) for k in REPLAY_KEYS})) for r in rows])
    hist_store.execute("INSERT OR REPLACE INTO watch_state(key, value) VALUES ('hist_replay:meta', ?)",
                       (json.dumps({**meta, "saved": time.time()}),))


def load_replay(hist_store) -> tuple[list[dict], dict] | None:
    try:
        rows = [json.loads(r["data"]) for r in hist_store.query("SELECT data FROM hist_replay")]
        meta = hist_store.query("SELECT value FROM watch_state WHERE key='hist_replay:meta'")
    except Exception:  # noqa: BLE001 - no cache yet
        return None
    if not rows:
        return None
    return sorted(rows, key=lambda r: r["create_ts"]), (json.loads(meta[0]["value"]) if meta else {})


class _Window:
    """Weighted sums over events by their finish time, for causal 'past H hours' lookups."""

    def __init__(self, fin: list[float], vals: dict, w: list[float]):
        order = np.argsort(fin)
        self.fin = np.asarray(fin, float)[order]
        self.cum = {k: np.concatenate([[0.0], np.cumsum(np.asarray(v, float)[order] * np.asarray(w, float)[order])])
                    for k, v in vals.items()}
        self.cnt = np.concatenate([[0], np.cumsum(np.ones(len(order)))])

    def sums(self, t: float, hours: float) -> tuple[float, dict]:
        lo, hi = np.searchsorted(self.fin, t - hours * 3600, "left"), np.searchsorted(self.fin, t, "left")
        return float(self.cnt[hi] - self.cnt[lo]), {k: float(c[hi] - c[lo]) for k, c in self.cum.items()}


def market_window(rows: list[dict]) -> _Window:
    from tradingbot import launch_live as ll
    fin = [r["create_ts"] + ll.EXIT_S for r in rows]
    keep = [r.get("rw_pct") is not None and r["rw_pct"] < 100 - ll.RW_SKIP for r in rows]
    scored = [r.get("rw_pct") is not None for r in rows]
    vals = {"ret": [r["ret"] for r in rows], "one": [1.0] * len(rows), "rug": [float(r["ret"] <= -0.999) for r in rows],
            "kept_ret": [r["ret"] if k else 0.0 for r, k in zip(rows, keep)], "kept": [float(k) for k in keep],
            "skip_ret": [r["ret"] if s and not k else 0.0 for r, k, s in zip(rows, keep, scored)],
            "skip": [float(s and not k) for k, s in zip(keep, scored)]}
    return _Window(fin, vals, [r.get("w", 1.0) for r in rows])


def _decide(t: dict) -> float:
    return t["opened"] - 4.0   # the trade's decision (pool second ~6; opened = create + 10)


def _rule_factor(kind: str, h: float, thr: float, t: dict, mkt: _Window, own: _Window | None) -> float:
    """1 = trade as usual, 0 = paused (callers turn 0 into 0.5 for the half-size versions)."""
    at = _decide(t)
    if kind in ("market", "rugs", "edge"):
        n, s = mkt.sums(at, h)
        if n < MIN_POOLS:
            return 1.0
        if kind == "market":
            return 0.0 if 100 * s["ret"] / s["one"] < thr else 1.0
        if kind == "rugs":
            return 0.0 if 100 * s["rug"] / s["one"] > thr else 1.0
        if s["kept"] <= 0 or s["skip"] <= 0:
            return 1.0
        return 0.0 if 100 * (s["kept_ret"] / s["kept"] - s["skip_ret"] / s["skip"]) < thr else 1.0
    n, s = own.sums(at, h)
    if n < MIN_TRADES or s["cost"] <= 0:
        return 1.0
    return 0.0 if 100 * s["pnl"] / s["cost"] < thr else 1.0


def apply(trades: list[dict], mkt: _Window, kind: str, h: float = 0.0, thr: float = 0.0, half: bool = False) -> np.ndarray:
    """Size factor per trade (trades sorted by decision time)."""
    if kind == "none":
        return np.ones(len(trades))
    if kind == "daystop":
        return _daystop(trades, thr, half)
    own = None
    if kind == "own":
        own = _Window([t["ts"] for t in trades], {"pnl": [t["ret"] * t["cost"] for t in trades],
                                                  "cost": [t["cost"] for t in trades]}, [t.get("w", 1.0) for t in trades])
    f = np.array([_rule_factor(kind, h, thr, t, mkt, own) for t in trades])
    return np.where(f < 1, 0.5, 1.0) if half else f


def _daystop(trades: list[dict], pct: float, half: bool) -> np.ndarray:
    """Stop for the rest of the UTC day once that day's realized loss (trades already sold) passes pct% of the
    book's median daily money traded."""
    daily = collections.defaultdict(float)
    for t in trades:
        daily[int(t["ts"] // DAY)] += t["cost"] * t.get("w", 1.0)
    limit = pct / 100 * float(np.median(list(daily.values()))) if daily else 0.0
    out = np.ones(len(trades))
    by_day: dict = collections.defaultdict(list)
    for i, t in enumerate(trades):
        by_day[int(_decide(t) // DAY)].append(i)
    for day, idx in by_day.items():
        sold = sorted((trades[i]["ts"], trades[i]["ret"] * trades[i]["cost"] * trades[i].get("w", 1.0), i) for i in idx)
        stopped_at, pnl, j = None, 0.0, 0
        for i in idx:   # in decision order
            at = _decide(trades[i])
            while j < len(sold) and sold[j][0] <= at:
                if out[sold[j][2]] > 0:
                    pnl += sold[j][1] * out[sold[j][2]]
                j += 1
            if stopped_at is None and pnl < -limit:
                stopped_at = at
            if stopped_at is not None:
                out[i] = 0.5 if half else 0.0
    return out


def score(trades: list[dict], f: np.ndarray, base_months: dict) -> dict:
    """$, drawdown and the bad-month / good-month split for one rule's size factors."""
    pnl = np.array([t["ret"] * t["cost"] * t.get("w", 1.0) for t in trades]) * f
    cost = np.array([t["cost"] * t.get("w", 1.0) for t in trades]) * f
    days = collections.defaultdict(float); months = collections.defaultdict(float)
    for t, p in zip(trades, pnl):
        days[int(t["ts"] // DAY)] += p
        months[datetime.fromtimestamp(t["ts"], timezone.utc).strftime("%Y-%m")] += p
    curve = np.cumsum([days[d] for d in sorted(days)]) if days else np.zeros(1)
    dd = float((curve - np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]).min()) if len(curve) else 0.0
    bad = [m for m, v in base_months.items() if v < 0]
    good = [m for m, v in base_months.items() if v > 0]
    lost = -sum(base_months[m] for m in bad); won = sum(base_months[m] for m in good)
    return {"pnl_usd": round(float(pnl.sum())), "mean_pct": round(100 * float(pnl.sum() / cost.sum()), 2) if cost.sum() else None,
            "taken_pct": round(100 * float(np.mean(f > 0))), "size_pct": round(100 * float(f.mean())),
            "max_dd_usd": round(dd), "worst_day_usd": round(min(days.values())) if days else 0,
            "worst_month_usd": round(min(months.values())) if months else 0,
            "bad_months_usd": round(sum(months.get(m, 0.0) for m in bad)),
            "good_months_usd": round(sum(months.get(m, 0.0) for m in good)),
            "saved_pct": round(100 * (1 - (-sum(months.get(m, 0.0) for m in bad)) / lost)) if lost > 0 else None,
            "kept_pct": round(100 * sum(months.get(m, 0.0) for m in good) / won) if won > 0 else None,
            "months": {m: round(v) for m, v in sorted(months.items())}}


def rules() -> list[tuple[str, str, float, float, bool]]:
    out = [("no safeguard", "none", 0, 0, False)]
    out += [(f"market {h}h < {t}%", "market", h, t, False) for h, t in MARKET]
    out += [(f"rugs {h}h > {t}%", "rugs", h, t, False) for h, t in RUGS]
    out += [(f"edge {h}h < {t} pts", "edge", h, t, False) for h, t in EDGE]
    out += [(f"own {h}h < {t}%", "own", h, t, False) for h, t in OWN]
    out += [(f"day stop {k}%", "daystop", 0, k, False) for k in DAYSTOP]
    out += [(f"market {h}h < {t}% (half)", "market", h, t, True) for h, t in ((6, -10), (24, -5), (24, -10))]
    out += [(f"own 24h < {t}% (half)", "own", 24, t, True) for t in (0, -5)]
    return out


def study_book(name: str, trades: list[dict], mkt: _Window) -> dict:
    trades = sorted(trades, key=_decide)
    base = score(trades, np.ones(len(trades)), {})
    base_months = base["months"]
    out = {}
    for label, kind, h, thr, half in rules():
        out[label] = score(trades, apply(trades, mkt, kind, h, thr, half), base_months)
    return {"trades": len(trades), "bad months": [m for m, v in base_months.items() if v < 0], "rules": out}


def study(hist_store, watch_store, journal, cfg, fresh: bool = False, now: float | None = None) -> dict:
    from tradingbot import risk_history as rh
    t0 = time.time()
    cached = None if fresh else load_replay(hist_store)
    if cached:
        rows, meta = cached
        source = f"saved replay ({len(rows):,} pools)"
    else:
        rows, info = rh.load_pools(hist_store, watch_store, cfg)
        if len(rows) < 500:
            return {"note": "fewer than 500 liquid pools: run history-backfill first"}
        meta = rh.replay(rows)
        save_replay(hist_store, rows, meta)
        source = f"replayed now ({len(rows):,} pools) and saved"
    g = rh.gaps(journal, watch_store, now or time.time())
    trades = {"launchrw50_sz": rh.rule_trades(rows, "rw50", "launchrw50_sz", g["launchrw50_sz"]["gap"],
                                              cfg.model_book_liq_share),
              "launchrwband_sz": rh.rule_trades(rows, "rwband", "launchrwband_sz", g["launchrwband_sz"]["gap"],
                                                cfg.model_book_liq_share)}
    have = rh._have_holds(hist_store)
    if have:   # sz with the minute-15 hold in $25-40k, where `--hold15` fetched the states
        trades[rh.HOLD_RULES[0]] = rh.hold_trades(rows, have, g["launchrw50_sz"]["gap"], cfg.model_book_liq_share,
                                                  extra_impact=False)[0]
    mkt = market_window(rows)
    return {"source": source, "took_s": round(time.time() - t0), "pools": len(rows),
            "books": {n: study_book(n, ts, mkt) for n, ts in trades.items() if len(ts) >= 100},
            "note": __doc__.split("\n\n")[1].replace("\n", " ")}


def render(out: dict) -> str:
    if "books" not in out:
        return json.dumps(out, indent=2, default=str)
    L = [f"SAFEGUARDS ON THE YEAR REPLAY ({out['source']}; {out['took_s']} s)", ""]
    for name, b in out["books"].items():
        base = b["rules"]["no safeguard"]
        L += [f"{name}: {b['trades']:,} sampled trades; bad months (the book lost): {', '.join(b['bad months']) or 'none'}",
              f"  {'rule':28}{'$ total':>12}{'mean':>7}{'taken':>7}{'bad mo $':>11}{'saved':>7}{'good mo $':>12}"
              f"{'kept':>6}{'max DD $':>11}{'worst day':>11}{'worst mo':>11}"]
        for label, r in b["rules"].items():
            L.append(f"  {label:28}{r['pnl_usd']:>+12,}{(r['mean_pct'] or 0):>+6.1f}%{r['taken_pct']:>6}%"
                     f"{r['bad_months_usd']:>+11,}{(r['saved_pct'] if r['saved_pct'] is not None else 0):>6}%"
                     f"{r['good_months_usd']:>+12,}{(r['kept_pct'] if r['kept_pct'] is not None else 0):>5}%"
                     f"{r['max_dd_usd']:>+11,}{r['worst_day_usd']:>+11,}{r['worst_month_usd']:>+11,}")
        best = sorted(((lbl, r) for lbl, r in b["rules"].items() if lbl != "no safeguard" and r["kept_pct"] is not None
                       and r["kept_pct"] >= 90 and r["saved_pct"] is not None),
                      key=lambda x: -(x[1]["saved_pct"] or 0))[:5]
        L.append("  saves the most while keeping 90%+ of the good months: "
                 + ("; ".join(f"{lbl} (saved {r['saved_pct']}%, kept {r['kept_pct']}%)" for lbl, r in best) or "none"))
        months = list(base["months"])
        show = ["no safeguard"] + [lbl for lbl, _ in best[:3]]
        L.append("  by month ($): " + " | ".join(show))
        for m in months:
            L.append(f"    {m}  " + "  ".join(f"{b['rules'][s]['months'].get(m, 0):>+11,}" for s in show))
        L.append("")
    L.append("note: " + out["note"])
    L.append("$ = replayed trades at the book's sizes (each sampled history trade weighted by the pools it stands for), "
             "after its measured execution gap; bad / good months = the book's own months without a safeguard; saved = "
             "share of the bad months' loss avoided; kept = share of the good months' profit still made.")
    return "\n".join(L)
