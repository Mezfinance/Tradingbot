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

from tradingbot import pool_bands

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
BATCH_HOURS = 24  # hours built per read on a backfill
STATUS_KEY = "market_monitor:status"  # last step: ran, ok_ts, took, stored, error
SZ_STUCK_S = 1800  # a sized-book trade still open this long counts as -100% (as block-compare)

VERSION = 4  # stored rows from an older version are rebuilt (2: liquidity bands, 3: sz $ P&L and cost, 4: 8 bands)
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

    real, real_liq = [], []
    for t in trades:
        if not t.get("cost_usd"):
            continue
        if t.get("closed_at"):
            real.append(((t["realized_usd"] - t["cost_usd"]) / t["cost_usd"], t["cost_usd"]))
        elif now - t["opened_at"] >= SZ_STUCK_S:
            real.append((-1.0, t["cost_usd"]))
        else:
            continue
        real_liq.append((*real[-1], t.get("liq")))
    out["bands"] = pool_bands.hour_bands(done, real_liq, ll.RW_SKIP)  # 8 pool-size bands (sums; not a METRIC)
    out.update(sz_trades=len(real), sz_real=_wpct(real), sz_pnl=round(sum(r * c for r, c in real), 2),
               sz_cost=round(sum(c for _, c in real), 2))
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


def compare(rows: list[dict], recent_h: int = RECENT_H, baseline_h: int = BASELINE_H, since: float = 0.0) -> dict:
    """Each metric: the last `recent_h` hour rows vs the `baseline_h` hours before them. z = (recent - baseline
    mean of `recent_h`-hour windows) / their standard deviation; None until MIN_BASELINE_WINDOWS windows exist.
    `rows` sorted by hour, each with 'hour'. The baseline starts no earlier than `since` (cfg
    `market_monitor_since`: hours before the Helius budget was uncapped saw only part of the pools), and the
    sized book's measures (`sz_*`) only from its first trade (earlier hours read 0 trades, not a quiet market)."""
    if not rows:
        return {}
    recent = rows[-recent_h:]
    base_all = [r for r in rows[:-recent_h] if r["hour"] >= max(since, recent[0]["hour"] - baseline_h * HOUR_S)]
    sz_from = next((r["hour"] for r in rows if r.get("sz_trades")), None)
    out = {}
    for key, _, group in METRICS:
        now_v = _window(recent, key)
        base = base_all if not key.startswith("sz_") else \
            [r for r in base_all if sz_from is not None and r["hour"] >= sz_from]
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


# Flags (step 3): a measure whose last RECENT_H hours sit FLAG_Z+ standard deviations from the baseline for FLAG_HOURS
# hours in a row (same direction) opens a flag; it closes once |z| < CLOSE_Z or the direction flips. Each flag keeps
# what the sized book did in its hours, so over time the flags say which market shifts actually hurt it.
FLAG_Z = 2.0
FLAG_HOURS = 2
CLOSE_Z = 1.5
FLAG_SKIP = ("sz_trades",  # a count of our own trades, not the market
             "sol_usd", "btc_usd", "fear_greed")  # slow levels (a daily index, trending prices) sit away from a weekly
# mean for days on end; their hourly changes (sol_chg_pct, btc_chg_pct) still flag
FLAG_SCHEMA = ("CREATE TABLE IF NOT EXISTS market_flags (id INTEGER PRIMARY KEY, metric TEXT NOT NULL, "
               "direction TEXT NOT NULL, opened_hour REAL NOT NULL, closed_hour REAL, peak_z REAL, data TEXT)")


def advance_flags(state: dict, hour: float, cmp: dict) -> list[dict]:
    """One finished hour through the flag rules. `state` (per metric: streak, sign, first, open) is updated in
    place; returns events {'kind': 'open'|'update'|'close', 'metric', ...}. A missing z keeps an open flag open."""
    events = []
    for key, _, _ in METRICS:
        if key in FLAG_SKIP:
            continue
        c = cmp.get(key) or {}
        z = c.get("z")
        st = state.setdefault(key, {"streak": 0, "sign": 0, "first": None, "open": False})
        if z is None:
            if not st["open"]:
                st["streak"] = 0
            continue
        sign = 1 if z > 0 else -1
        if st["open"]:
            if abs(z) < CLOSE_Z or sign != st["sign"]:
                events.append({"kind": "close", "metric": key, "hour": hour, "z": z, "now": c.get("now")})
                st.update(open=False, streak=0)
            else:
                events.append({"kind": "update", "metric": key, "hour": hour, "z": z, "now": c.get("now")})
            continue
        if abs(z) >= FLAG_Z:
            if st["streak"] and sign == st["sign"]:
                st["streak"] += 1
            else:
                st.update(streak=1, sign=sign, first=hour)
            if st["streak"] >= FLAG_HOURS:
                st["open"] = True
                events.append({"kind": "open", "metric": key, "hour": st["first"], "z": z, "now": c.get("now"),
                               "baseline": c.get("baseline"), "direction": "up" if sign > 0 else "down"})
        else:
            st["streak"] = 0
    return events


class MarketMonitor:
    """Writes one `market_hours` row per finished hour. `store` = a WatchStore (its own connection is best, as the
    dashboard does); `journal` = the trading journal, read only."""

    def __init__(self, store, journal, cfg):
        self.store, self.journal, self.cfg = store, journal, cfg
        store.execute(SCHEMA)
        store.execute(FLAG_SCHEMA)
        self.events: list[dict] = []  # flag opens/closes from the last step (the Telegram step sends these)

    def build_many(self, hours: list[float], now: float) -> dict:
        """Rows for many hours from one read of each table (an hour at a time scanned whole tables: the first
        backfill of ~120 hours took many minutes), bucketed by hour in memory."""
        from tradingbot import launch_live as ll
        from tradingbot import model_book as mb
        if not hours:
            return {}
        ll.ensure_schema(self.store)
        lo, hi = min(hours), max(hours) + HOUR_S
        want = set(hours)
        bucket = lambda ts: ts // HOUR_S * HOUR_S
        pools, trades = collections.defaultdict(list), collections.defaultdict(list)
        for r in self.store.query(
                "SELECT pool, token_mint, creator, create_ts, liquid, status, ret, pulled_ts, liq_usd, feat, rw_pct "
                "FROM watch_launch_live WHERE create_ts >= ? AND create_ts < ?", (lo, hi)):
            pools[bucket(r["create_ts"])].append(dict(r))
        if self.journal:
            liq_by_mint = {p["token_mint"]: p["liq_usd"] for ps in pools.values() for p in ps
                           if p.get("token_mint") and p.get("liq_usd") is not None}
            for r in self.journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                                        "WHERE strategy=? AND opened_at >= ? AND opened_at < ?",
                                        (mb.LAUNCH_SIZED, lo, hi)):
                trades[bucket(r["opened_at"])].append({**dict(r), "liq": liq_by_mint.get(r["token"])})
        series = collections.defaultdict(lambda: collections.defaultdict(list))
        for ts, v in self.store.query("SELECT ts, sol_usd FROM watch_market WHERE ts >= ? AND ts < ? "
                                      "AND sol_usd IS NOT NULL ORDER BY ts", (lo, hi)):
            series["sol"][bucket(ts)].append(v)
        for key, name in (("btc_usd", "btc"), ("fear_greed", "fg")):
            for ts, v in self.store.query("SELECT ts, value FROM watch_macro WHERE key=? AND ts >= ? AND ts < ? "
                                          "ORDER BY ts", (key, lo, hi)):
                series[name][bucket(ts)].append(v)
        out = {}
        for h in sorted(want):
            sol, btc, fg = series["sol"].get(h, []), series["btc"].get(h, []), series["fg"].get(h, [])
            prices = {"sol_open": sol[0] if sol else None, "sol_close": sol[-1] if sol else None,
                      "btc_open": btc[0] if btc else None, "btc_close": btc[-1] if btc else None,
                      "fear_greed": fg[-1] if fg else None}
            out[h] = hour_metrics(pools.get(h, []), trades.get(h, []), prices, now)
        return out

    def build(self, hour: float, now: float) -> dict:
        return self.build_many([hour], now)[hour]

    def step(self, now: float | None = None) -> int:
        """`_step` plus its outcome in watch_state `STATUS_KEY` (last run, seconds taken, hours stored, last
        error), so the dashboard can tell a failing or stopped monitor from a stale page."""
        started = time.time()
        prev = json.loads(self.store.get_state(STATUS_KEY) or "{}")
        try:
            n = self._step(now)
        except Exception as exc:
            self.store.set_state(STATUS_KEY, json.dumps({
                **prev, "ran": started, "took": round(time.time() - started, 1), "error": f"{type(exc).__name__}: {exc}"[:300],
                "error_ts": started}))
            raise
        self.store.set_state(STATUS_KEY, json.dumps({"ran": started, "ok_ts": started, "took": round(time.time() - started, 1),
                                                     "stored": n, "error": None}))
        log.info("market monitor step: %d hour(s) stored in %.1f s", n, time.time() - started)
        return n

    def _step(self, now: float | None = None) -> int:
        """Store every finished hour without a row (the first run backfills from the scorer's first pool)."""
        now = now if now is not None else time.time()
        last_done = (now - HOUR_DELAY_S) // HOUR_S * HOUR_S - HOUR_S  # start of the newest finished hour
        have = {r[0] for r in self.store.query("SELECT hour, data FROM market_hours")
                if json.loads(r[1]).get("v", 1) >= VERSION}
        first = self.store.query("SELECT MIN(create_ts) FROM watch_launch_live")
        if not first or first[0][0] is None:
            return 0
        hour, missing = first[0][0] // HOUR_S * HOUR_S, []
        while hour <= last_done:
            if hour not in have:
                missing.append(hour)
            hour += HOUR_S
        n = 0
        for i in range(0, len(missing), BATCH_HOURS):  # a day at a time keeps memory flat on long backfills
            rows = self.build_many(missing[i:i + BATCH_HOURS], now)
            self.store.executemany("INSERT OR REPLACE INTO market_hours(hour, data, ts) VALUES(?,?,?)",
                                   [(h, json.dumps(d), now) for h, d in rows.items()])
            n += len(rows)
        if n:
            log.info("market monitor: %d hour(s) stored", n)
        self.events = self.update_flags()
        return n

    def update_flags(self) -> list[dict]:
        """Run every stored hour not yet seen through `advance_flags` and save opens, updates and closes in
        `market_flags`. Starts over when the row VERSION or the baseline start changes."""
        since = getattr(self.cfg, "market_monitor_since", 0.0) or 0.0
        key = f"{VERSION}:{since}"
        if self.store.get_state("market_flags:key") != key:
            self.store.execute("DELETE FROM market_flags")
            self.store.set_state("market_flags:state", "{}")
            self.store.set_state("market_flags:last", "0")
            self.store.set_state("market_flags:key", key)
        last = float(self.store.get_state("market_flags:last") or 0)
        state = json.loads(self.store.get_state("market_flags:state") or "{}")
        rows = load(self.store, last - (BASELINE_H + RECENT_H + 1) * HOUR_S if last else 0.0)
        rows = [r for r in rows if r.get("v", 1) >= VERSION]
        open_ids = {r[0]: r[1] for r in self.store.query(
            "SELECT metric, id FROM market_flags WHERE closed_hour IS NULL")}
        out = []
        span = BASELINE_H + RECENT_H
        for i, r in enumerate(rows):
            if r["hour"] <= last:
                continue
            for ev in advance_flags(state, r["hour"], compare(rows[max(0, i - span + 1):i + 1], since=since)):
                m = ev["metric"]
                if ev["kind"] == "open":
                    cur = self.store.execute(
                        "INSERT INTO market_flags(metric, direction, opened_hour, peak_z, data) VALUES(?,?,?,?,?)",
                        (m, ev["direction"], ev["hour"], ev["z"],
                         json.dumps({"open_value": ev["now"], "baseline": ev["baseline"], "last_value": ev["now"],
                                     "last_z": ev["z"], "seen_hour": r["hour"]})))
                    open_ids[m] = cur.lastrowid
                    out.append({**ev, "id": cur.lastrowid, "label": LABELS.get(m, m)})
                elif m in open_ids:
                    fid = open_ids[m]
                    row = self.store.query("SELECT peak_z, data FROM market_flags WHERE id=?", (fid,))
                    if not row:
                        continue
                    data = {**json.loads(row[0]["data"] or "{}"), "last_value": ev["now"], "last_z": ev["z"]}
                    peak = row[0]["peak_z"] if abs(row[0]["peak_z"] or 0) >= abs(ev["z"]) else ev["z"]
                    closed = ev["hour"] if ev["kind"] == "close" else None
                    self.store.execute("UPDATE market_flags SET peak_z=?, data=?, closed_hour=? WHERE id=?",
                                       (peak, json.dumps(data), closed, fid))
                    if closed is not None:
                        del open_ids[m]
                        out.append({**ev, "id": fid, "label": LABELS.get(m, m)})
            last = r["hour"]
        self.store.set_state("market_flags:last", repr(last))
        self.store.set_state("market_flags:state", json.dumps(state))
        for ev in out:
            log.info("market flag %s: %s %s (z %+.1f)", ev["kind"], ev["label"], ev.get("direction", ""), ev["z"])
        return out


def read(query, since: float = 0.0) -> list[dict]:
    """`market_hours` rows through any read function (a read-only connection: never creates the table)."""
    import sqlite3
    try:
        rows = query("SELECT hour, data FROM market_hours WHERE hour >= ? ORDER BY hour", (since,))
    except sqlite3.OperationalError:  # no table yet (the monitor hasn't run)
        return []
    return [{"hour": r[0], **json.loads(r[1])} for r in rows]


PAGE_HOURS = 72  # hourly rows sent to the dashboard
FLAG_DAYS = 7  # closed flags shown


def read_status(query) -> dict:
    """The monitor's last step (see `MarketMonitor.step`) through any read function; {} before its first run."""
    import sqlite3
    try:
        rows = query("SELECT value FROM watch_state WHERE key=?", (STATUS_KEY,))
    except sqlite3.OperationalError:
        return {}
    return json.loads(rows[0][0]) if rows and rows[0][0] else {}


def read_flags(query) -> list[dict]:
    import sqlite3
    try:
        rows = query("SELECT id, metric, direction, opened_hour, closed_hour, peak_z, data FROM market_flags "
                     "ORDER BY opened_hour", ())
    except sqlite3.OperationalError:
        return []
    return [{"id": r[0], "metric": r[1], "direction": r[2], "opened_hour": r[3], "closed_hour": r[4],
             "peak_z": r[5], **json.loads(r[6] or "{}")} for r in rows]


def _sz_market(rows: list[dict]) -> dict:
    """sz (size-weighted, $ P&L / $ traded) and the market (ideal mean weighted by liquid pools) over hour rows."""
    cost = sum(r.get("sz_cost") or 0 for r in rows)
    pnl = sum(r.get("sz_pnl") or 0 for r in rows)
    n_liq = sum(r.get("liquid") or 0 for r in rows if r.get("market_mean") is not None)
    mkt = sum((r.get("market_mean") or 0) * (r.get("liquid") or 0) for r in rows if r.get("market_mean") is not None)
    return {"hours": len(rows), "sz_trades": sum(r.get("sz_trades") or 0 for r in rows),
            "sz_pct": round(100 * pnl / cost, 1) if cost else None, "sz_pnl": round(pnl, 2),
            "market_pct": round(mkt / n_liq, 1) if n_liq else None}


def flag_summary(rows: list[dict], flags: list[dict], now: float | None = None, days: float = FLAG_DAYS) -> dict:
    """Open flags, flags closed in the last `days`, and per measure and direction: sz and the market in flagged
    hours vs every other hour since sz's first trade. `rows` = market_hours rows (with sz_pnl / sz_cost)."""
    now = now if now is not None else time.time()
    sz_from = next((r["hour"] for r in rows if r.get("sz_trades")), None)
    by_hour = {r["hour"]: r for r in rows}

    def hours_of(f):
        end = f["closed_hour"] if f["closed_hour"] is not None else float("inf")
        return [r for h, r in by_hour.items() if f["opened_hour"] <= h < end]

    def view(f):
        return {**f, "label": LABELS.get(f["metric"], f["metric"]), **_sz_market(hours_of(f))}
    out = {"open": [view(f) for f in flags if f["closed_hour"] is None],
           "recent": [view(f) for f in reversed(flags) if f["closed_hour"] is not None
                      and f["closed_hour"] >= now - days * 86400],
           "by_metric": []}
    groups = collections.defaultdict(list)
    for f in flags:
        groups[(f["metric"], f["direction"])].append(f)
    live = [r for r in rows if sz_from is not None and r["hour"] >= sz_from]
    for (metric, direction), fs in groups.items():
        flagged = {r["hour"] for f in fs for r in hours_of(f)}
        inside = [r for r in live if r["hour"] in flagged]
        outside = [r for r in live if r["hour"] not in flagged]
        a, b = _sz_market(inside), _sz_market(outside)
        out["by_metric"].append({"metric": metric, "label": LABELS.get(metric, metric), "direction": direction,
                                 "flags": len(fs), "hours": len(flagged), "in": a, "out": b,
                                 "sz_gap": round(a["sz_pct"] - b["sz_pct"], 1)
                                 if a["sz_pct"] is not None and b["sz_pct"] is not None else None})
    out["by_metric"].sort(key=lambda g: -(g["hours"]))
    return out


def payload(rows: list[dict], now: float | None = None, since: float = 0.0, flags: list[dict] | None = None,
            status: dict | None = None) -> dict:
    """What the dashboard's Market tab needs: the last PAGE_HOURS rows, the comparison and the metric list."""
    now = now if now is not None else time.time()
    return {"built_ts": now, "rows": [r for r in rows if r["hour"] >= now - PAGE_HOURS * HOUR_S],
            "compare": compare(rows, since=since), "baseline_since": since, "metrics": [list(m) for m in METRICS],
            "recent_h": RECENT_H, "baseline_h": BASELINE_H, "last_hour": rows[-1]["hour"] if rows else None,
            "status": status or {}, "poll_s": MONITOR_POLL_S, "bands": band_windows(rows, now),
            "flags": flag_summary(rows, flags or [], now), "flag_rule": {"z": FLAG_Z, "hours": FLAG_HOURS,
                                                                         "close_z": CLOSE_Z}}


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


def status(store, hours: float = 24, now: float | None = None, since: float = 0.0) -> dict:
    store.execute(FLAG_SCHEMA)
    now = now if now is not None else time.time()
    rows = load(store, now - (hours + BASELINE_H + RECENT_H) * HOUR_S)
    shown = [r for r in rows if r["hour"] >= now - hours * HOUR_S]
    all_rows = load(store, 0.0)
    return {"hours": hours, "rows": shown, "compare": compare(rows, since=since), "baseline_since": since,
            "flags": flag_summary(all_rows, read_flags(store.query), now),
            "bands": band_windows(all_rows, now), "recent_h": RECENT_H, "baseline_h": BASELINE_H}


BAND_WINDOWS = (("last 6 h", 6), ("last 24 h", 24), ("last 7 days", 168), ("whole life", None))


def band_windows(rows: list[dict], now: float) -> dict:
    """`pool_bands.aggregate` over the windows in BAND_WINDOWS (hours whose row carries `bands`)."""
    return {name: pool_bands.aggregate([r for r in rows if h is None or r["hour"] >= now - h * HOUR_S])
            for name, h in BAND_WINDOWS}


def render_bands(bands: dict) -> list[str]:
    lines = ["", "RUG-VS-WIN BY POOL SIZE  (liquidity at second 4; ideal = scorer prices; kept = rug-vs-win keeps it; "
             "sz = launchrw50_sz real, size-weighted; $100 = running total of the ideal trade at $100 a pool)"]
    f = lambda v: "-" if v is None else f"{v:+.1f}"
    for name, rows in bands.items():
        if not any(r["pools"] for r in rows):
            continue
        lines += ["", f"  {name}", "  " + f"{'band':<11}{'pools':>6}{'share':>7}{'all':>8}{'rug%':>6}{'+50%':>6}"
                  f"{'kept%':>7}{'kept':>8}{'skip':>8}{'gap':>8}{'all $100':>10}{'kept $100':>10}{'sz n':>6}{'sz':>8}{'sz $':>9}"]
        for r in rows:
            if not r["pools"] and not r["sz_n"]:
                continue
            lines.append("  " + f"{r['band']:<11}{r['pools']:>6}{(r['share_pct'] if r['share_pct'] is not None else '-'):>7}"
                         f"{f(r['mean_pct']):>8}{(r['rugged_pct'] if r['rugged_pct'] is not None else '-'):>6}"
                         f"{(r['big_pct'] if r['big_pct'] is not None else '-'):>6}"
                         f"{(r['kept_share_pct'] if r['kept_share_pct'] is not None else '-'):>7}"
                         f"{f(r['kept_pct']):>8}{f(r['skipped_pct']):>8}{f(r['gap']):>8}"
                         f"{r['all_100_usd']:>10,}{r['kept_100_usd']:>10,}{r['sz_n']:>6}"
                         f"{f(r['sz_pct']):>8}{r['sz_pnl']:>9,.0f}")
    return lines


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
    since = out.get("baseline_since") or 0
    start = f", baseline from {datetime.fromtimestamp(since, timezone.utc):%m-%d %H:%M} UTC" if since else ""
    lines += ["", f"LAST {out['recent_h']} H VS THE {out['baseline_h'] // 24} DAYS BEFORE{start}  (|z| >= 2 marked *; "
              "sz measures only from its first trade)"]
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
    fl = out.get("flags") or {}
    when = lambda h: datetime.fromtimestamp(h, timezone.utc).strftime("%m-%d %H:00")
    pct = lambda v: "-" if v is None else f"{v:+.1f}%"

    def flag_line(f, closed):
        end = f"to {when(f['closed_hour'])}" if closed else "open"
        return (f"    {f['direction']:>4s}  {f['label']:34s} {when(f['opened_hour'])} {end:14s} {f['hours']:>3d} h  "
                f"peak z {f['peak_z']:+.1f}  value {fmt(f.get('open_value'))} vs {fmt(f.get('baseline'))}  "
                f"sz {pct(f['sz_pct'])} ({f['sz_trades']})  market {pct(f['market_pct'])}")
    lines += ["", f"OPEN FLAGS  (|z| >= {FLAG_Z:g} for {FLAG_HOURS}+ hours; close below {CLOSE_Z:g})"]
    lines += [flag_line(f, False) for f in fl.get("open", [])] or ["    none"]
    lines += ["", f"FLAGS CLOSED IN THE LAST {FLAG_DAYS} DAYS"]
    lines += [flag_line(f, True) for f in fl.get("recent", [])[:30]] or ["    none"]
    if fl.get("by_metric"):
        lines += ["", "WHAT THE FLAGS MEANT FOR SZ  (since its first trade: sz size-weighted, market ideal mean; "
                  "flagged hours vs every other hour)",
                  f"    {'':4s}  {'measure':34s} {'flags':>5s} {'hours':>5s}  {'sz in':>7s} {'sz out':>7s} {'gap':>7s}  "
                  f"{'mkt in':>7s} {'mkt out':>7s}"]
        for g in fl["by_metric"][:25]:
            gap = "-" if g["sz_gap"] is None else f"{g['sz_gap']:+.1f}"
            lines.append(f"    {g['direction']:>4s}  {g['label']:34s} {g['flags']:>5d} {g['hours']:>5d}  "
                         f"{pct(g['in']['sz_pct']):>7s} {pct(g['out']['sz_pct']):>7s} {gap:>7s}  "
                         f"{pct(g['in']['market_pct']):>7s} {pct(g['out']['market_pct']):>7s}")
    lines += ["", "ops = distinct creators; tmpl% = pools on a seed size shared by 3+ pools that hour (copy-paste "
              "launches); <15k% etc = share of liquid pools by liquidity at second 4 (sz's bands), <15k mkt% = ideal mean of "
              "the thin band; rw gap = rug-vs-win kept minus skipped half (ideal); sz% = launchrw50_sz real, "
              "size-weighted; z = (last 3 h - baseline mean of 3-hour windows) / their spread"]
    if out.get("bands"):
        lines += render_bands(out["bands"])
    return "\n".join(lines)
