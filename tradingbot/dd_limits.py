"""Drawdown limits on a launch book (11-09, read only, no Helius): which limit costs least in the good times and saves
most in the bad, for the flattest, highest long-run return?

Two runs on the same rule set:

* life: the book's real trades (default launchrw50_sz; closed, exit blocked or open > `STUCK_S` = -100%);
* year (`--year`): the year replay `risk-model --history` saves in hist.db (`safeguards.load_replay`), sz's rule on
  every replayed pool sized like the book, minus its measured execution gap, with each sampled history day rebuilt
  to full size by `risk_history.expand` (trades keep their own time of day; a weighted sampled trade would breach a
  limit by itself). That rebuild borrows most of each day's trades from the 7 days either side, which makes
  neighbouring days alike and flatters limits that read the recent past, so the year is also run with every day
  rebuilt from its own trades only (`year_own`, k=1) and the key limits are compared side by side. The replay trades every flagged pool while the live book skips about half its signals, so the
  limits are scaled by the replay's median daily money traded over the live book's (`scale`): a 5% limit bites as
  hard on a replayed day as on a live one.

Limits (k = % of the starting capital, `CAPITAL` $10k; at each buy only trades already sold count; open trades run
to their exit; a skipped trade changes nothing later):

  day loss k        no buys for the rest of the UTC day once the day's realized P&L is -k or worse
  day trail k       ... once the day's realized P&L is k below its high for the day (the high starts at 0)
  peak trail k      ... once realized equity is k below its high since the last stop; the next day starts afresh
  pause Hh at -k    no buys for H hours once realized P&L since the last restart is -k; then restart
  pause Hh trail k  ... once it is k below its high since the last restart
  past Hh -k        no buys while the realized P&L of the past H hours is -k or worse
  ... half          half size instead of no buys

Per limit: $, return on the capital, trades taken, mean, max drawdown on the trade-by-trade P&L, $ per $ of drawdown,
daily Sharpe, worst day, $ kept on the periods the unlimited book made money and the share of its losing periods'
loss saved (6 h blocks for the life, days and months for the year), each half. Thresholds come from a grid seen all
at once: trust a limit only when it holds in both halves and its neighbours agree."""
from __future__ import annotations

import collections
import heapq
import json
import math
import time
from datetime import datetime, timezone

import numpy as np

BOOK = "launchrw50_sz"
CAPITAL = 10_000.0
STUCK_S = 1800
BLOCK_H = 6
YEAR_FLAGS = {"launchrw50_sz": "rw50", "launchrwband_sz": "rwband"}
DAY_LOSS = (1, 2, 3, 4, 5, 6, 7.5, 10, 12.5, 15, 20)
DAY_TRAIL = (1, 2, 3, 4, 5, 6, 7.5, 10, 15)
PEAK_TRAIL = (5, 7.5, 10, 15, 20)
PAUSE = [(h, k) for k in (3, 5, 7.5, 10) for h in (3, 6, 12)]
PAUSE_TRAIL = [(h, k) for k in (3, 5, 7.5) for h in (3, 6, 12)]
ROLL = ((3, 3), (6, 3), (6, 5), (12, 5), (24, 5), (24, 10))
HALF = (("day", 3), ("day", 5), ("trail", 3), ("trail", 5))
TOP = 10
FAMILIES = ("day loss", "day trail", "peak trail", "pause", "pause trail", "past", "half")


def rules() -> list[tuple[str, str, dict]]:
    """(label, family, rule)."""
    out = [("no limit", "", {"kind": "none"})]
    out += [(f"day loss {k:g}%", "day loss", {"kind": "day", "k": k}) for k in DAY_LOSS]
    out += [(f"day trail {k:g}%", "day trail", {"kind": "trail", "k": k}) for k in DAY_TRAIL]
    out += [(f"peak trail {k:g}%", "peak trail", {"kind": "peak", "k": k}) for k in PEAK_TRAIL]
    out += [(f"pause {h}h at -{k:g}%", "pause", {"kind": "pause", "h": h, "k": k}) for h, k in PAUSE]
    out += [(f"pause {h}h trail {k:g}%", "pause trail", {"kind": "pausetrail", "h": h, "k": k}) for h, k in PAUSE_TRAIL]
    out += [(f"past {h}h -{k:g}%", "past", {"kind": "roll", "h": h, "k": k}) for h, k in ROLL]
    out += [(f"{'day loss' if kd == 'day' else 'day trail'} {k:g}% half", "half", {"kind": kd, "k": k, "half": True})
            for kd, k in HALF]
    return out


def _day(ts: float) -> int:
    return int(ts // 86400)


def apply_rule(tr: list[dict], rule: dict, capital: float) -> list[float]:
    """Size multiplier per trade (0 = not bought), trades in any order (each has opened, ts, cost, ret)."""
    n = len(tr)
    if rule["kind"] == "none":
        return [1.0] * n
    order = sorted(range(n), key=lambda i: tr[i]["opened"])
    mult = [1.0] * n
    k = rule.get("k", 0) / 100 * capital
    off = 0.5 if rule.get("half") else 0.0
    pending: list = []
    window = collections.deque()
    wsum = cum = 0.0
    day_pnl: dict = collections.defaultdict(float)
    day_high: dict = collections.defaultdict(float)
    ref = high = 0.0
    stopped_day, until = None, None
    for i in order:
        a = tr[i]["opened"]
        while pending and pending[0][0] <= a:
            ts, pnl = heapq.heappop(pending)
            cum += pnl
            d0 = _day(ts)
            day_pnl[d0] += pnl
            day_high[d0] = max(day_high[d0], day_pnl[d0])
            window.append((ts, pnl))
            wsum += pnl
            high = max(high, cum)
        d = _day(a)
        kind = rule["kind"]
        if kind == "day":
            m = off if day_pnl[d] <= -k else 1.0
        elif kind == "trail":
            m = off if day_high[d] - day_pnl[d] >= k else 1.0
        elif kind == "peak":
            if stopped_day is not None and d > stopped_day:
                stopped_day, high = None, cum
            if stopped_day is None and high - cum >= k:
                stopped_day = d
            m = off if stopped_day is not None else 1.0
        elif kind in ("pause", "pausetrail"):
            if until is not None and a >= until:
                until, ref, high = None, cum, cum
            if until is None and ((ref - cum) if kind == "pause" else (high - cum)) >= k:
                until = a + rule["h"] * 3600
            m = off if until is not None else 1.0
        else:  # roll
            lo = a - rule["h"] * 3600
            while window and window[0][0] <= lo:
                wsum -= window.popleft()[1]
            m = off if wsum <= -k else 1.0
        mult[i] = m
        if m:
            heapq.heappush(pending, (tr[i]["ts"], m * tr[i]["cost"] * tr[i]["ret"]))
    return mult


# scoring ---------------------------------------------------------------------------------------------------------

def _month(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def _kept_saved(rule_p: np.ndarray, base_p: np.ndarray) -> tuple:
    good, bad = base_p > 0, base_p < 0
    won, lost = base_p[good].sum(), -base_p[bad].sum()
    kept = round(100 * float(rule_p[good].sum() / won)) if won > 0 else None
    saved = round(100 * float((rule_p[bad].sum() + lost) / lost)) if lost > 0 else None
    return kept, saved


class _Index:
    """Per-trade arrays shared by every limit's score: sale order, and the block / day / month of each sale."""

    def __init__(self, tr: list[dict]):
        self.ts = np.array([t["ts"] for t in tr])
        self.cost = np.array([t["cost"] for t in tr], dtype=float)
        self.ret = np.array([t["ret"] for t in tr], dtype=float)
        self.order = np.argsort(self.ts, kind="stable")
        self.keys = {}
        for unit, raw in (("block", (self.ts // (BLOCK_H * 3600)).astype(np.int64)),
                          ("day", (self.ts // 86400).astype(np.int64)),
                          ("month", np.array([_month(x) for x in self.ts]))):
            labels, inv = np.unique(raw, return_inverse=True)
            self.keys[unit] = (labels, inv)
        days = self.keys["day"][0]
        self.day_pos = days - days.min() if len(days) else days
        self.n_days = int(days.max() - days.min() + 1) if len(days) else 0

    def periods(self, pnl: np.ndarray, unit: str) -> np.ndarray:
        labels, inv = self.keys[unit]
        return np.bincount(inv, weights=pnl, minlength=len(labels))


def score(ix: _Index, mult: np.ndarray, base: dict, capital: float, mid: float, unit: str) -> dict:
    pnl = mult * ix.cost * ix.ret
    eq = np.cumsum(pnl[ix.order])
    full = np.concatenate([[0.0], eq])
    dd = float((full - np.maximum.accumulate(full)).min())
    dv = np.zeros(ix.n_days)
    dv[ix.day_pos] = ix.periods(pnl, "day")
    total = float(pnl.sum())
    traded = float((mult * ix.cost).sum())
    first = float(pnl[ix.ts < mid].sum())
    kept, saved = _kept_saved(ix.periods(pnl, unit), base[unit])
    out = {"usd": round(total), "return_pct": round(100 * total / capital, 1),
           "trades": int((mult > 0).sum()), "mean_pct": round(100 * total / traded, 2) if traded else None,
           "max_dd_usd": round(dd), "max_dd_pct": round(100 * dd / capital, 1),
           "ret_per_dd": round(total / -dd, 2) if dd < -1 else None,
           "sharpe": round(float(dv.mean() / dv.std() * math.sqrt(365)), 2) if len(dv) > 1 and dv.std() else None,
           "worst_day_usd": round(float(dv.min())) if len(dv) else 0,
           "kept_pct": kept, "saved_pct": saved,
           "first_half_usd": round(first), "second_half_usd": round(total - first)}
    if unit == "day":
        mp = ix.periods(pnl, "month")
        out["months"] = {str(m): round(float(v)) for m, v in zip(ix.keys["month"][0], mp)}
        out["month_kept_pct"], out["month_saved_pct"] = _kept_saved(mp, base["month"])
    return out


def run(tr: list[dict], capital: float, unit: str) -> dict:
    tr = sorted(tr, key=lambda t: t["opened"])
    ix = _Index(tr)
    ones = np.ones(len(tr))
    base = {u: ix.periods(ix.cost * ix.ret, u) for u in ("block", "day", "month")}
    mid = (tr[0]["opened"] + float(ix.ts.max())) / 2
    res = []
    for label, fam, rule in rules():
        mult = ones if rule["kind"] == "none" else np.array(apply_rule(tr, rule, capital))
        res.append({"limit": label, "family": fam, **score(ix, mult, base, capital, mid, unit)})
    return {"trades": len(tr), "from": tr[0]["opened"], "to": float(ix.ts.max()), "capital": capital,
            "unit": unit, "rules": res}


# data ------------------------------------------------------------------------------------------------------------

def live_trades(journal, book: str, since: float, now: float) -> list[dict]:
    blocked = {r["token"]: r["t"] for r in journal.query(
        "SELECT token, MIN(ts) t FROM events WHERE strategy=? AND kind='exit_blocked' AND ts >= ? GROUP BY token",
        (book, since))}
    out = []
    for p in journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ? AND opened_at < ?", (book, since, now)):
        if not p["cost_usd"]:
            continue
        if p["closed_at"]:
            ret, ts = p["realized_usd"] / p["cost_usd"] - 1, p["closed_at"]
        elif p["token"] in blocked and blocked[p["token"]] >= p["opened_at"]:
            ret, ts = -1.0, blocked[p["token"]]
        elif now - p["opened_at"] >= STUCK_S:
            ret, ts = -1.0, p["opened_at"] + STUCK_S
        else:
            continue
        out.append({"opened": p["opened_at"], "ts": ts, "cost": p["cost_usd"], "ret": max(-1.0, ret)})
    return out


def _median_daily(tr: list[dict]) -> float:
    d: dict = collections.defaultdict(float)
    for t in tr:
        d[_day(t["opened"])] += t["cost"]
    full = sorted(d)[1:-1] or sorted(d)  # first and last days are partial
    return float(np.median([d[x] for x in full])) if full else 0.0


def year_trades(hist_store, watch_store, journal, cfg, book: str, fresh: bool, now: float) -> tuple[list[dict], dict]:
    from tradingbot import risk_history as rh
    from tradingbot import safeguards as sg
    cached = None if fresh else sg.load_replay(hist_store)
    if cached:
        rows, _ = cached
        source = f"saved replay ({len(rows):,} pools)"
    else:
        rows, _ = rh.load_pools(hist_store, watch_store, cfg)
        if len(rows) < 500:
            return [], {"error": "fewer than 500 liquid pools: run history-backfill first"}
        meta = rh.replay(rows)
        sg.save_replay(hist_store, rows, meta)
        source = f"replayed now ({len(rows):,} pools) and saved"
    g = rh.gaps(journal, watch_store, now)
    gap = g.get(book, {"gap": rh.DEFAULT_GAP})["gap"]
    ts = rh.rule_trades(rows, YEAR_FLAGS[book], book, gap, cfg.model_book_liq_share)
    full, info = rh.expand(ts)
    own, own_info = rh.expand(ts, k=1.0)
    return full, {"source": source, "gap": gap, "expand": info, "own_days": own, "own_expand": own_info}


def study(journal, book: str = BOOK, hours: float | None = None, now: float | None = None, year: bool = False,
          hist_store=None, watch_store=None, cfg=None, fresh: bool = False, capital: float = CAPITAL) -> dict:
    now = now or time.time()
    since = now - hours * 3600 if hours else 0.0
    live = live_trades(journal, book, since, now)
    out: dict = {"book": book}
    if live:
        out["life"] = run(live, capital, "block")
    else:
        out["life"] = {"error": "no trades"}
    if year:
        if book not in YEAR_FLAGS:
            out["year"] = {"error": f"no year replay rule for {book} (have {', '.join(YEAR_FLAGS)})"}
            return out
        t0 = time.time()
        tr, info = year_trades(hist_store, watch_store, journal, cfg, book, fresh, now)
        if not tr:
            out["year"] = info
            return out
        own = info.pop("own_days")
        live_day, year_day = _median_daily(live), _median_daily(tr)
        scale = year_day / live_day if live_day and year_day else 1.0
        out["year"] = {**run(tr, capital * scale, "day"), **info, "scale": round(scale, 2),
                       "live_daily_usd": round(live_day), "year_daily_usd": round(year_day),
                       "took_s": round(time.time() - t0)}
        # the same limits with each sampled day rebuilt only from its own trades: the smoothed rebuild borrows
        # most of a day's trades from the 7 days either side, which makes neighbouring days alike and flatters any
        # limit that reads the recent past
        out["year_own"] = {**run(own, capital * scale, "day"), "expand": info["own_expand"]}
    return out


COMPARE = ("no limit", "past 24h -5%", "past 24h -10%", "past 12h -5%", "past 6h -5%", "past 6h -3%",
           "day loss 15%", "day loss 5%")
BAD_MONTHS = ("2026-08", "2026-09", "2026-10")


# render ----------------------------------------------------------------------------------------------------------

def _t(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts))


def _f(v, w, s=""):
    return (f"{v:,}{s}" if isinstance(v, int) else f"{v}{s}" if v is not None else "-").rjust(w)


HEAD = (f"  {'limit':<24}{'$':>11}{'return':>9}{'trades':>9}{'mean':>8}{'max DD':>9}{'$/DD':>7}{'Sharpe':>8}"
        f"{'worst day':>11}{'kept':>6}{'saved':>7}{'half 1 $':>11}{'half 2 $':>11}")


def _row(r: dict) -> str:
    return (f"  {r['limit']:<24}{_f(r['usd'], 11)}{_f(r['return_pct'], 9, '%')}{_f(r['trades'], 9)}"
            f"{_f(r['mean_pct'], 8, '%')}{_f(r['max_dd_pct'], 9, '%')}{_f(r['ret_per_dd'], 7)}{_f(r['sharpe'], 8)}"
            f"{_f(r['worst_day_usd'], 11)}{_f(r['kept_pct'], 6, '%')}{_f(r['saved_pct'], 7, '%')}"
            f"{_f(r['first_half_usd'], 11)}{_f(r['second_half_usd'], 11)}")


def _section(name: str, res: dict) -> list[str]:
    if res.get("error"):
        return [f"{name}: {res['error']}", ""]
    rs = res["rules"]
    base = rs[0]
    per = "6 h block" if res["unit"] == "block" else "day"
    L = [f"{name}: {_t(res['from'])} .. {_t(res['to'])} UTC, {res['trades']:,} trades, limits on "
         f"${res['capital']:,.0f}", f"  kept / saved: share of the unlimited book's winning {per}s' profit still made, "
         f"and of its losing {per}s' loss avoided", HEAD, _row(base)]
    for fam in FAMILIES:
        L += [f"  -- {fam}"] + [_row(r) for r in rs if r["family"] == fam]
    rank = sorted(rs[1:], key=lambda r: -(r["ret_per_dd"] or -99))[:TOP]
    L += ["", f"  BEST {TOP} BY $ PER $ OF DRAWDOWN (unlimited: {base['ret_per_dd']})", HEAD] + [_row(r) for r in rank]
    both = [r for r in rs[1:] if r["first_half_usd"] >= base["first_half_usd"]
            and r["second_half_usd"] >= base["second_half_usd"]]
    L += ["", f"  AT LEAST AS MUCH $ AS NO LIMIT IN BOTH HALVES ({len(both)})", HEAD] + \
         [_row(r) for r in sorted(both, key=lambda r: -r["usd"])[:TOP]]
    keep = sorted([r for r in rs[1:] if (r["kept_pct"] or 0) >= 90 and r["saved_pct"] is not None],
                  key=lambda r: -r["saved_pct"])[:TOP]
    L += ["", f"  SAVES MOST WHILE KEEPING 90%+ OF THE GOOD {per.upper()}S", HEAD] + [_row(r) for r in keep]
    if "months" in base:
        show = [base] + rank[:3]
        L += ["", "  BY MONTH ($): " + " | ".join(r["limit"] for r in show)]
        for m in base["months"]:
            L.append(f"    {m}  " + "".join(f"{r['months'].get(m, 0):>14,}" for r in show))
        L.append("  months kept / saved: " + "; ".join(f"{r['limit']} {_pct(r['month_kept_pct'])} / {_pct(r['month_saved_pct'])}"
                                                        for r in show[1:]))
    return L + [""]


def _pct(v) -> str:
    return "-" if v is None else f"{v}%"


def _compare(smooth: dict, own: dict) -> list[str]:
    """Key limits on both rebuilds side by side: a limit whose gain survives without the smoothing is real."""
    a = {r["limit"]: r for r in smooth["rules"]}
    b = {r["limit"]: r for r in own["rules"]}
    L = ["SMOOTHED vs OWN-DAYS REBUILD (the gain should survive without the smoothing)",
         f"  {'limit':<18}" + "".join(f"{h:>13}" for h in ("$ smoothed", "$ own days", "DD smoothed", "DD own",
                                                          "kept own", "saved own"))
         + "".join(f"{m[2:]:>11}" for m in BAD_MONTHS)]
    for name in COMPARE:
        if name not in a or name not in b:
            continue
        x, y = a[name], b[name]
        L.append(f"  {name:<18}{x['usd']:>13,}{y['usd']:>13,}{x['max_dd_usd']:>13,}{y['max_dd_usd']:>13,}"
                 f"{_f(y['kept_pct'], 12, '%')}{_f(y['saved_pct'], 12, '%')}"
                 + "".join(f"{y['months'].get(m, 0):>11,}" for m in BAD_MONTHS))
    return L + ["  (bad-month columns are the own-days rebuild)", ""]


def render(out: dict) -> str:
    L = [f"DRAWDOWN LIMITS: {out['book']}", ""]
    L += _section("LIFE (real trades)", out["life"])
    if "year" in out:
        y = out["year"]
        if not y.get("error"):
            L.append(f"year replay: {y['source']}, execution gap {100 * y['gap']:.1f} pts, sampled days rebuilt "
                     f"{y['expand']}; replay trades ${y['year_daily_usd']:,}/day vs live ${y['live_daily_usd']:,}/day, "
                     f"so limits are scaled x{y['scale']} ({y['took_s']} s)")
        L += _section("YEAR REPLAY", y)
    if "year_own" in out and not out["year_own"].get("error"):
        o = out["year_own"]
        L.append(f"YEAR REPLAY, EACH DAY FROM ITS OWN TRADES ONLY (no borrowing from nearby days; "
                 f"{o['expand'].get('sampled days', 0)} sampled days)")
        L += _section("YEAR REPLAY (own days)", o)
        L += _compare(out["year"], o)
    L.append("Thresholds were all looked at together and the life is a few days: trust a limit only when it holds in "
             "both halves (and on the year) and the settings either side of it agree.")
    return "\n".join(L)


def to_json(out: dict) -> str:
    return json.dumps(out, indent=2, default=str)
