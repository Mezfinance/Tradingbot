"""Always-on meme market monitor (11-07, read only, never trades).

The launch books' edge depends on who launches hand-made pumpswap pools and how they rug, more than on BTC. This
module turns the data the bot already records into hourly measures of that market and compares the latest hours
with the trailing week, so a change (new operators, a different pull timing, a flood or drought of pools, the
rug-vs-win gap closing) shows up while it happens.

Every finished UTC hour (`HOUR_DELAY_S` after it ends, so pools have reached their minute-6 exit) gets one row in
`market_hours(hour, data)` in watch.db, built from:
  watch_launch_live  every hand-made pool the scorer saw (supply, operators, rugs, buying, ideal trades, rw scores)
  journal positions  the sized book's real trades (`model_book.LAUNCH_SIZED`) opened that hour
  watch_market / watch_macro  SOL and BTC prices, fear & greed
`MarketMonitor.step()` runs on its own thread (`MONITOR_POLL_S`); the first run backfills every hour in the data.
`compare(rows)`: each metric's last `RECENT_H` hours vs the trailing `BASELINE_H` (z = gap / the spread of
`RECENT_H`-hour means over the baseline). CLI `market-monitor [--hours 24] [--json]`."""
from __future__ import annotations

import collections
import json
import logging
import math
import statistics
import time
from datetime import datetime, timezone

log = logging.getLogger(__name__)

HOUR_S = 3600
HOUR_DELAY_S = 15 * 60  # an hour's pools have all reached minute 6 (and their row is done) by then
MONITOR_POLL_S = 600
RECENT_H = 3
BASELINE_H = 7 * 24
MIN_BASELINE_WINDOWS = 24  # fewer RECENT_H-hour windows than this in the baseline: no z
RUG = -0.999
BIG_WIN = 0.5
EARLY_PULL_S = 90
SEED_ROUND = 1.0  # seed SOL rounded to this for the copy-paste template count
TEMPLATE_MIN = 3  # a seed size shared by this many pools in the hour = a template
TOP_OPERATORS = 5
SZ_STUCK_S = 1800  # a sized-book trade still open this long counts as -100% (as block-compare)

VERSION = 2  # stored rows from an older version are rebuilt (2: liquidity bands)
BANDS = (("thin", 0.0, 15_000.0), ("mid", 15_000.0, 25_000.0), ("thick", 25_000.0, float("inf")))  # sz's bands

SCHEMA = "CREATE TABLE IF NOT EXISTS market_hours (hour REAL PRIMARY KEY, data TEXT NOT NULL, ts REAL)"

# (key, label, group); every key is a number in an hour row (None when the hour has no data for it)
METRICS = (
    ("pools", "hand-made pools seen", "supply"),
    ("liquid", "liquid at second 4", "supply"),
    ("liquid_pct", "liquid share %", "supply"),
    ("med_liq_usd", "median liquidity at s4 $", "supply"),
    ("med_seed_sol", "median seed SOL", "supply"),
    ("thin_pct", "liquid pools under $15k %", "liquidity"),
    ("mid_pct", "liquid pools $15-25k %", "liquidity"),
    ("thick_pct", "liquid pools $25k+ %", "liquidity"),
    ("thin_mean", "under $15k: ideal mean %", "liquidity"),
    ("mid_mean", "$15-25k: ideal mean %", "liquidity"),
    ("thick_mean", "$25k+: ideal mean %", "liquidity"),
    ("thin_rug_pct", "under $15k: rugged %", "liquidity"),
    ("thin_sz_mean", "under $15k: sz picks ideal %", "liquidity"),
    ("creators", "distinct creators", "operators"),
    ("top5_pct", f"top {TOP_OPERATORS} creators' share %", "operators"),
    ("repeat_pct", "repeat creators %", "operators"),
    ("template_pct", "pools on a repeated seed size %", "operators"),
    ("rug_pct", "rugged by minute 6 %", "rugs"),
    ("med_pull_s", "median LP pull second", "rugs"),
    ("early_pull_pct", f"LP pulled before {EARLY_PULL_S} s %", "rugs"),
    ("late_rug_pct", f"rugged after {EARLY_PULL_S} s %", "rugs"),
    ("med_buy_sol", "median buy SOL in first 4 s", "buying"),
    ("med_buyers", "median buyers in first 4 s", "buying"),
    ("big_win_pct", "+50% winners %", "buying"),
    ("market_mean", "market ideal mean %", "buying"),
    ("market_median", "market ideal median %", "buying"),
    ("market_sized", "market sized like sz %", "edge"),
    ("rw_kept", "rug-vs-win kept half %", "edge"),
    ("rw_skipped", "rug-vs-win skipped half %", "edge"),
    ("rw_gap", "kept minus skipped pts", "edge"),
    ("sz_picks_ideal", "sz picks at ideal fills %", "edge"),
    ("sz_trades", "sz real trades", "edge"),
    ("sz_real", "sz real result %", "edge"),
    ("sz_fill_gap", "sz real minus its ideal pts", "edge"),
    ("sol_usd", "SOL $", "wider"),
    ("sol_chg_pct", "SOL change in the hour %", "wider"),
    ("btc_usd", "BTC $", "wider"),
    ("btc_chg_pct", "BTC change in the hour %", "wider"),
    ("fear_greed", "fear & greed", "wider"),
)
LABELS = {k: lbl for k, lbl, _ in METRICS}
# metrics pooled over pools in the hour; their RECENT_H value is a weighted mean of hours (weight = the count)
WEIGHT = {"liquid_pct": "pools", "top5_pct": "liquid", "thin_pct": "liquid", "mid_pct": "liquid", "thick_pct": "liquid",
          "thin_mean": "thin_n", "mid_mean": "mid_n", "thick_mean": "thick_n", "thin_rug_pct": "thin_n",
          "thin_sz_mean": "thin_sz_n", "repeat_pct": "liquid", "template_pct": "liquid",
          "rug_pct": "liquid", "early_pull_pct": "liquid", "late_rug_pct": "liquid", "big_win_pct": "liquid",
          "market_mean": "liquid", "rw_kept": "liquid", "rw_skipped": "liquid", "sz_real": "sz_trades"}


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def _pct(n, d):
    return round(100 * n / d, 1) if d else None


def _mean_pct(xs):
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def _wpct(pairs):
    """Size-weighted %: sum(ret x size) / sum(size)."""
    cost = sum(s for _, s in pairs)
    return round(100 * sum(r * s for r, s in pairs) / cost, 1) if cost else None


def hour_metrics(pools: list[dict], trades: list[dict], prices: dict, now: float) -> dict:
    """One hour's measures. `pools` = watch_launch_live rows created in the hour; `trades` = the sized book's
    positions opened in it; `prices` = {sol_open, sol_close, btc_open, btc_close, fear_greed}."""
    from tradingbot import launch_live as ll
    from tradingbot import model_book as mb
    feat = {}
    for p in pools:
        try:
            feat[p["pool"]] = json.loads(p["feat"]) if p.get("feat") else {}
        except ValueError:
            feat[p["pool"]] = {}
    liquid = [p for p in pools if p.get("liquid")]
    done = [p for p in liquid if p.get("status") == "done" and p.get("ret") is not None]
    f = lambda p, k: feat[p["pool"]].get(k)
    unlog = lambda v: math.expm1(v) if v is not None else None
    out: dict = {"v": VERSION, "pools": len(pools), "liquid": len(liquid), "liquid_pct": _pct(len(liquid), len(pools)),
                 "done": len(done), "med_liq_usd": _med([p.get("liq_usd") for p in liquid]),
                 "med_seed_sol": _med([unlog(f(p, "log_seed_sol")) for p in liquid])}

    by_creator = collections.Counter(p.get("creator") for p in liquid if p.get("creator"))
    seeds = collections.Counter(round(unlog(f(p, "log_seed_sol")) / SEED_ROUND) for p in liquid
                                if f(p, "log_seed_sol") is not None)
    templated = sum(1 for p in liquid if f(p, "log_seed_sol") is not None
                    and seeds[round(unlog(f(p, "log_seed_sol")) / SEED_ROUND)] >= TEMPLATE_MIN)
    out.update(creators=len(by_creator),
               top5_pct=_pct(sum(n for _, n in by_creator.most_common(TOP_OPERATORS)), len(liquid)),
               repeat_pct=_pct(sum(1 for p in liquid if f(p, "repeat_creator")), len(liquid)),
               template_pct=_pct(templated, len(liquid)))

    rets = [p["ret"] for p in done]
    pulls = [p["pulled_ts"] - p["create_ts"] for p in done if p.get("pulled_ts")]
    early = [p for p in done if p.get("pulled_ts") and p["pulled_ts"] - p["create_ts"] < EARLY_PULL_S]
    out.update(rug_pct=_pct(sum(r <= RUG for r in rets), len(done)), med_pull_s=_med(pulls),
               early_pull_pct=_pct(len(early), len(done)),
               late_rug_pct=_pct(sum(1 for p in done if p["ret"] <= RUG and p not in early), len(done)),
               med_buy_sol=_med([unlog(f(p, "log_buy_sol")) for p in liquid]),
               med_buyers=_med([unlog(f(p, "log_buyers")) for p in liquid]),
               big_win_pct=_pct(sum(r >= BIG_WIN for r in rets), len(done)),
               market_mean=_mean_pct(rets), market_median=round(100 * statistics.median(rets), 1) if rets else None)

    for name, lo, hi in BANDS:
        inband = [p for p in liquid if p.get("liq_usd") is not None and lo <= p["liq_usd"] < hi]
        dn = [p["ret"] for p in done if p.get("liq_usd") is not None and lo <= p["liq_usd"] < hi]
        out[f"{name}_pct"] = _pct(len(inband), len(liquid))
        out[f"{name}_n"] = len(dn)
        out[f"{name}_mean"] = _mean_pct(dn)
        if name == "thin":
            out["thin_rug_pct"] = _pct(sum(r <= RUG for r in dn), len(dn))
            picks_thin = [p["ret"] for p in done if p.get("liq_usd") is not None and p["liq_usd"] < hi
                          and p.get("rw_pct") is not None and p["rw_pct"] < 100 - ll.RW_SKIP]
            out["thin_sz_n"] = len(picks_thin)
            out["thin_sz_mean"] = _mean_pct(picks_thin)

    sized = [(p["ret"], mb.band_usd(mb.LAUNCH_SIZED, p["liq_usd"])) for p in done if p.get("liq_usd")]
    sized = [(r, s) for r, s in sized if s]
    scored = [p for p in done if p.get("rw_pct") is not None]
    kept = [p for p in scored if p["rw_pct"] < 100 - ll.RW_SKIP]
    skipped = [p for p in scored if p["rw_pct"] >= 100 - ll.RW_SKIP]
    picks = [(p["ret"], mb.band_usd(mb.LAUNCH_SIZED, p["liq_usd"])) for p in kept if p.get("liq_usd")]
    picks = [(r, s) for r, s in picks if s]
    out.update(market_sized=_wpct(sized), rw_kept=_mean_pct([p["ret"] for p in kept]),
               rw_skipped=_mean_pct([p["ret"] for p in skipped]), sz_picks_ideal=_wpct(picks))
    out["rw_gap"] = round(out["rw_kept"] - out["rw_skipped"], 1) \
        if out["rw_kept"] is not None and out["rw_skipped"] is not None else None

    real = []
    for t in trades:
        if not t.get("cost_usd"):
            continue
        if t.get("closed_at"):
            real.append(((t["realized_usd"] - t["cost_usd"]) / t["cost_usd"], t["cost_usd"]))
        elif now - t["opened_at"] >= SZ_STUCK_S:
            real.append((-1.0, t["cost_usd"]))
    out.update(sz_trades=len(real), sz_real=_wpct(real))
    out["sz_fill_gap"] = round(out["sz_real"] - out["sz_picks_ideal"], 1) \
        if out["sz_real"] is not None and out["sz_picks_ideal"] is not None else None

    chg = lambda a, b: round(100 * (b / a - 1), 2) if a and b else None
    out.update(sol_usd=round(prices["sol_close"], 2) if prices.get("sol_close") else None,
               sol_chg_pct=chg(prices.get("sol_open"), prices.get("sol_close")),
               btc_usd=round(prices["btc_close"]) if prices.get("btc_close") else None,
               btc_chg_pct=chg(prices.get("btc_open"), prices.get("btc_close")),
               fear_greed=prices.get("fear_greed"))
    return out


def _window(rows: list[dict], key: str):
    """A metric over consecutive hour rows: weighted by the hour's count for pooled metrics, else a plain mean."""
    w = WEIGHT.get(key)
    vals = [(r[key], r.get(w) or 0 if w else 1) for r in rows if r.get(key) is not None]
    tot = sum(x for _, x in vals)
    return sum(v * x for v, x in vals) / tot if tot else None


def compare(rows: list[dict], recent_h: int = RECENT_H, baseline_h: int = BASELINE_H) -> dict:
    """Each metric: the last `recent_h` hour rows vs the `baseline_h` hours before them. z = (recent - baseline
    mean of `recent_h`-hour windows) / their standard deviation; None until MIN_BASELINE_WINDOWS windows exist.
    `rows` sorted by hour, each with 'hour'."""
    if not rows:
        return {}
    recent = rows[-recent_h:]
    base = [r for r in rows[:-recent_h] if r["hour"] >= recent[0]["hour"] - baseline_h * HOUR_S]
    out = {}
    for key, _, group in METRICS:
        now_v = _window(recent, key)
        wins = [_window(base[i:i + recent_h], key) for i in range(0, max(0, len(base) - recent_h + 1))]
        wins = [w for w in wins if w is not None]
        mean = statistics.fmean(wins) if wins else None
        sd = statistics.pstdev(wins) if len(wins) > 1 else None
        z = (now_v - mean) / sd if now_v is not None and mean is not None and sd and len(wins) >= MIN_BASELINE_WINDOWS \
            else None
        out[key] = {"group": group, "now": round(now_v, 2) if now_v is not None else None,
                    "baseline": round(mean, 2) if mean is not None else None,
                    "z": round(z, 1) if z is not None else None}
    return out


class MarketMonitor:
    """Writes one `market_hours` row per finished hour. `store` = a WatchStore (its own connection is best, as the
    dashboard does); `journal` = the trading journal, read only."""

    def __init__(self, store, journal, cfg):
        self.store, self.journal, self.cfg = store, journal, cfg
        store.execute(SCHEMA)

    def _prices(self, start: float, end: float) -> dict:
        sol = self.store.query("SELECT sol_usd FROM watch_market WHERE ts >= ? AND ts < ? AND sol_usd IS NOT NULL "
                               "ORDER BY ts", (start, end))
        def macro(key):
            return [r[0] for r in self.store.query("SELECT value FROM watch_macro WHERE key=? AND ts >= ? AND ts < ? "
                                                   "ORDER BY ts", (key, start, end))]
        btc, fg = macro("btc_usd"), macro("fear_greed")
        return {"sol_open": sol[0][0] if sol else None, "sol_close": sol[-1][0] if sol else None,
                "btc_open": btc[0] if btc else None, "btc_close": btc[-1] if btc else None,
                "fear_greed": fg[-1] if fg else None}

    def build(self, hour: float, now: float) -> dict:
        from tradingbot import launch_live as ll
        from tradingbot import model_book as mb
        ll.ensure_schema(self.store)
        end = hour + HOUR_S
        pools = [dict(r) for r in self.store.query(
            "SELECT pool, creator, create_ts, liquid, status, ret, pulled_ts, liq_usd, feat, rw_pct "
            "FROM watch_launch_live WHERE create_ts >= ? AND create_ts < ?", (hour, end))]
        trades = [dict(r) for r in self.journal.query(
            "SELECT opened_at, closed_at, cost_usd, realized_usd FROM positions WHERE strategy=? "
            "AND opened_at >= ? AND opened_at < ?", (mb.LAUNCH_SIZED, hour, end))] if self.journal else []
        return hour_metrics(pools, trades, self._prices(hour, end), now)

    def step(self, now: float | None = None) -> int:
        """Store every finished hour without a row (the first run backfills from the scorer's first pool)."""
        now = now if now is not None else time.time()
        last_done = (now - HOUR_DELAY_S) // HOUR_S * HOUR_S - HOUR_S  # start of the newest finished hour
        have = {r[0] for r in self.store.query("SELECT hour, data FROM market_hours")
                if json.loads(r[1]).get("v", 1) >= VERSION}
        first = self.store.query("SELECT MIN(create_ts) FROM watch_launch_live")
        if not first or first[0][0] is None:
            return 0
        hour, n = first[0][0] // HOUR_S * HOUR_S, 0
        while hour <= last_done:
            if hour not in have:
                data = self.build(hour, now)
                self.store.execute("INSERT OR REPLACE INTO market_hours(hour, data, ts) VALUES(?,?,?)",
                                   (hour, json.dumps(data), now))
                n += 1
            hour += HOUR_S
        if n:
            log.info("market monitor: %d hour(s) stored", n)
        return n


def read(query, since: float = 0.0) -> list[dict]:
    """`market_hours` rows through any read function (a read-only connection: never creates the table)."""
    import sqlite3
    try:
        rows = query("SELECT hour, data FROM market_hours WHERE hour >= ? ORDER BY hour", (since,))
    except sqlite3.OperationalError:  # no table yet (the monitor hasn't run)
        return []
    return [{"hour": r[0], **json.loads(r[1])} for r in rows]


PAGE_HOURS = 72  # hourly rows sent to the dashboard


def payload(rows: list[dict], now: float | None = None) -> dict:
    """What the dashboard's Market tab needs: the last PAGE_HOURS rows, the comparison and the metric list."""
    now = now if now is not None else time.time()
    return {"built_ts": now, "rows": [r for r in rows if r["hour"] >= now - PAGE_HOURS * HOUR_S],
            "compare": compare(rows), "metrics": [list(m) for m in METRICS],
            "recent_h": RECENT_H, "baseline_h": BASELINE_H, "last_hour": rows[-1]["hour"] if rows else None}


def load(store, since: float = 0.0) -> list[dict]:
    store.execute(SCHEMA)
    return [{"hour": r["hour"], **json.loads(r["data"])}
            for r in store.query("SELECT hour, data FROM market_hours WHERE hour >= ? ORDER BY hour", (since,))]


SHOW = ("pools", "liquid", "thin_pct", "mid_pct", "thick_pct", "thin_mean", "creators", "top5_pct", "template_pct", "rug_pct", "med_pull_s", "big_win_pct",
        "market_mean", "market_sized", "rw_gap", "sz_trades", "sz_real", "btc_chg_pct")
SHORT = {"pools": "pools", "liquid": "liq", "thin_pct": "<15k%", "mid_pct": "15-25k%", "thick_pct": "25k+%",
         "thin_mean": "<15k mkt%", "creators": "ops", "top5_pct": "top5%", "template_pct": "tmpl%",
         "rug_pct": "rug%", "med_pull_s": "pull s", "big_win_pct": "+50%", "market_mean": "mkt%",
         "market_sized": "mkt sz%", "rw_gap": "rw gap", "sz_trades": "sz n", "sz_real": "sz%",
         "btc_chg_pct": "btc%"}


def status(store, hours: float = 24, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    rows = load(store, now - (hours + BASELINE_H + RECENT_H) * HOUR_S)
    shown = [r for r in rows if r["hour"] >= now - hours * HOUR_S]
    return {"hours": hours, "rows": shown, "compare": compare(rows),
            "recent_h": RECENT_H, "baseline_h": BASELINE_H}


def render(out: dict) -> str:
    lines = [f"MEME MARKET MONITOR  (last {out['hours']:g} h, hand-made pumpswap pools; ideal = scorer prices, "
             "6% cost, minute-6 exit)", ""]
    cols = ["hour (UTC)"] + [SHORT[k] for k in SHOW]
    fmt = lambda v: "-" if v is None else (f"{v:,.0f}" if abs(v) >= 100 else f"{v:g}")
    table = [cols] + [[datetime.fromtimestamp(r["hour"], timezone.utc).strftime("%m-%d %H")] +
                      [fmt(r.get(k)) for k in SHOW] for r in out["rows"]]
    widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
    for row in table:
        lines.append("  ".join(c.rjust(w) for c, w in zip(row, widths)))
    lines += ["", f"LAST {out['recent_h']} H VS THE {out['baseline_h'] // 24} DAYS BEFORE  (|z| >= 2 marked *)"]
    group = None
    for key, label, g in METRICS:
        c = out["compare"].get(key)
        if not c or c["now"] is None:
            continue
        if g != group:
            lines.append(f"  {g}")
            group = g
        z = c["z"]
        mark = " *" if z is not None and abs(z) >= 2 else ""
        lines.append(f"    {label:38s} {fmt(c['now']):>10s}  vs {fmt(c['baseline']):>10s}  "
                     f"z {'-' if z is None else f'{z:+.1f}'}{mark}")
    lines += ["", "ops = distinct creators; tmpl% = pools on a seed size shared by 3+ pools that hour (copy-paste "
              "launches); <15k% etc = share of liquid pools by liquidity at second 4 (sz's bands), <15k mkt% = ideal mean of "
              "the thin band; rw gap = rug-vs-win kept minus skipped half (ideal); sz% = launchrw50_sz real, "
              "size-weighted; z = (last 3 h - baseline mean of 3-hour windows) / their spread"]
    return "\n".join(lines)
