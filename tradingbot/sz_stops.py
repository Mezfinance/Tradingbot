"""Stops and drawdown limits on a launch book's life (11-09, read only; Helius only with --fetch).

Question (user, after a losing night): which rule costs the least in the good times and saves the most in the bad
ones, for the flattest, highest long-run return? Two families, alone and combined, replayed on the book's real
trades (default launchrw50_sz, its whole life):

* Drawdown limits (no Helius): stop buying for the rest of the UTC day once the day's realized loss passes k% of the
  starting capital (`DAY_STOPS`), once the day's realized P&L falls k% below its high that day (`TRAIL_STOPS`), or
  while the realized loss over the past H hours exceeds k% (`ROLLING`); plus half size instead of skipping. Causal:
  at each buy only trades already sold count, and a skipped trade changes nothing later.
* Stop losses per trade (needs each pool's trades during our hold): `fetch_paths` reads the pool's transactions from
  just before our buy to just after our sale (Helius getTransactionsForAddress with a blockTime window, 10 credits per
  100 txs, cached in watch.db `watch_trade_path`, in a stable hash order so a credit cap leaves an even sample). A
  stop is checked every `CHECK_S` seconds against the pool price at our buy (`STOPS` fixed %, `TRAILS` % below the
  best price since the buy) and sells `SELL_DELAY_S` later at the pool price then. The stopped trade's return =
  real return + (pool price at the stop sale - pool price at the real sale) / pool price at the buy, so the trade
  keeps its own fees and fill gap. A pull in one transaction leaves nothing to stop, as in reality.

Per rule: $, return on the capital, trades taken, 6 h blocks (Sharpe, worst), worst day, max drawdown on the
trade-by-trade P&L, $ kept on the blocks where the book made money and $ saved where it lost, and each half of the
life. Every number here is chosen with the data in view: prefer a rule that wins in both halves and whose neighbours
agree. Live, a stop on a launch book also needs a pool price every few seconds (vault reads; today these books
ignore feed prices and sell only at the pool minute)."""
from __future__ import annotations

import bisect
import collections
import hashlib
import heapq
import json
import math
import time

import numpy as np

from tradingbot.data import pumpswap as ps

BOOK = "launchrw50_sz"
CAPITAL = 10_000.0
STUCK_S = 1800  # open longer than this = -100% (as block-compare)
MATCH_S = 600  # a trade belongs to the scorer row of its token created at most this long before the buy
BLOCK_H = 6
CHECK_S = 5.0  # how often a live stop would read the pool
SELL_DELAY_S = 3.0  # from the trigger to the sale landing
PRE_S = 5  # path starts this long before our buy
POST_S = 10  # and ends this long after our sale
MAX_PAGES = 8  # 800 txs per hold at most
CREDITS_PER_PAGE = 10
CAP = 3.0
STOPS = (10, 15, 20, 25, 30, 40, 50)  # % below the pool price at our buy
TRAILS = (20, 30, 40)  # % below the best pool price since our buy
DAY_STOPS = (2, 3, 5, 7.5, 10, 15)  # % of capital realized loss in the UTC day
TRAIL_STOPS = (3, 5, 7.5, 10)  # % of capital below the day's realized high
ROLLING = ((6, 3), (6, 5), (24, 5), (24, 10))  # (hours, % of capital)
HALF_AT = (3, 5)  # day loss % of capital after which trades are half size
TOP = 12

SCHEMA = "CREATE TABLE IF NOT EXISTS watch_trade_path(pool TEXT PRIMARY KEY, data TEXT, ts REAL)"


# trades ----------------------------------------------------------------------------------------------------------

def trades(journal, store, book: str, since: float, now: float) -> list[dict]:
    """The book's real trades (closed; exit blocked or open > STUCK_S = -100%), with the scorer row of the token."""
    rows: dict = {}
    if store is not None:
        try:
            for r in store.query("SELECT pool, token_mint, create_ts, pulled_ts FROM watch_launch_live "
                                 "WHERE token_mint IS NOT NULL AND create_ts >= ? ORDER BY create_ts",
                                 (since - MATCH_S,)):
                rows.setdefault(r["token_mint"], []).append(dict(r))
        except Exception:  # noqa: BLE001 - no scorer table: drawdown rules still work
            rows = {}
    blocked = {r["token"]: r["t"] for r in journal.query(
        "SELECT token, MIN(ts) t FROM events WHERE strategy=? AND kind='exit_blocked' AND ts >= ? GROUP BY token",
        (book, since))}
    out = []
    for p in journal.query("SELECT id, token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ? AND opened_at < ? ORDER BY opened_at",
                           (book, since, now)):
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
        m = [r for r in rows.get(p["token"], []) if r["create_ts"] <= p["opened_at"] <= r["create_ts"] + MATCH_S]
        sc = m[-1] if m else None
        out.append({"id": p["id"], "token": p["token"], "opened": p["opened_at"], "ts": ts,
                    "sold": p["closed_at"], "cost": p["cost_usd"], "ret": max(-1.0, ret),
                    "pool": sc["pool"] if sc else None, "create_ts": sc["create_ts"] if sc else None,
                    "pulled_ts": sc["pulled_ts"] if sc else None})
    return out


# paths -----------------------------------------------------------------------------------------------------------

def _amount(tx: dict, pool: str, mint: str, side: str) -> float | None:
    for b in (tx.get("meta") or {}).get(side) or []:
        if b.get("owner") == pool and b.get("mint") == mint:
            a = b.get("uiTokenAmount") or {}
            return float(a.get("uiAmount") or 0) if mint == ps.WSOL else float(a.get("amount") or 0)
    return None


def path_rows(txs: list[dict], pool: str, mint: str) -> list[list]:
    """[blockTime, pool SOL, pool tokens (raw)] after each successful tx, plus the state before the first one."""
    rows = []
    for i, tx in enumerate(txs):
        if (tx.get("meta") or {}).get("err") is not None:
            continue
        bt = tx.get("blockTime") or 0
        if not rows:
            s0, t0 = _amount(tx, pool, ps.WSOL, "preTokenBalances"), _amount(tx, pool, mint, "preTokenBalances")
            if s0 is not None and t0 is not None:
                rows.append([bt - 0.5, s0, t0])
        s, t = _amount(tx, pool, ps.WSOL, "postTokenBalances"), _amount(tx, pool, mint, "postTokenBalances")
        if s is not None and t is not None:
            rows.append([bt, s, t])
    return rows


def _order(key: str) -> str:
    return hashlib.sha1(f"szstop:{key}".encode()).hexdigest()


def fetch_paths(store, helius, tr: list[dict], max_credits: int) -> dict:
    """Pool transactions during each trade's hold, cached; stops before `max_credits`."""
    store.execute(SCHEMA)
    have = {r["pool"] for r in store.query("SELECT pool FROM watch_trade_path")}
    todo = sorted({t["pool"]: t for t in tr if t["pool"] and t["sold"] and t["pool"] not in have}.values(),
                  key=lambda t: _order(t["pool"]))
    stats = {"wanted": len(todo), "fetched": 0, "failed": 0, "credits": 0, "stopped_by_cap": False}
    start = helius.credits_used
    for t in todo:
        if helius.credits_used - start + MAX_PAGES * CREDITS_PER_PAGE > max_credits:
            stats["stopped_by_cap"] = True
            break
        lo, hi = int(t["opened"] - PRE_S), int(t["sold"] + POST_S)
        txs, token, ok, done = [], None, True, False
        for _ in range(MAX_PAGES):
            opts = {"transactionDetails": "full", "sortOrder": "asc", "limit": 100, "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 1, "filters": {"blockTime": {"gte": lo, "lte": hi}}}
            if token:
                opts["paginationToken"] = token
            res = helius._rpc("getTransactionsForAddress", [t["pool"], opts], CREDITS_PER_PAGE)
            if res is None:
                ok = False
                break
            page = res.get("data") or []
            txs += page
            token = res.get("paginationToken")
            if not page or not token:
                done = True
                break
        if not ok:
            stats["failed"] += 1
            continue
        rows = path_rows(txs, t["pool"], t["token"])
        store.execute("INSERT OR REPLACE INTO watch_trade_path(pool, data, ts) VALUES(?,?,?)",
                      (t["pool"], json.dumps({"rows": rows, "complete": done, "lo": lo, "hi": hi}), time.time()))
        stats["fetched"] += 1
        if stats["fetched"] % 50 == 0:
            print(f"paths {stats['fetched']}/{len(todo)}, credits {helius.credits_used - start}", flush=True)
    stats["credits"] = helius.credits_used - start
    return stats


def load_paths(store) -> dict:
    try:
        return {r["pool"]: json.loads(r["data"]) for r in store.query("SELECT pool, data FROM watch_trade_path")}
    except Exception:  # noqa: BLE001 - no table yet
        return {}


def _price_at(rows: list[list], ts: float, times: list | None = None) -> float | None:
    """Pool price (SOL per raw token) after the last tx at or before ts; 0 when the pool is empty."""
    times = times if times is not None else [r[0] for r in rows]
    i = bisect.bisect_right(times, ts) - 1
    if i < 0:
        return None
    last = rows[i]
    return last[1] / last[2] if last[1] and last[2] else 0.0


def samples(t: dict, path: dict | None) -> dict | None:
    """Prices a live stop would see: at the buy, at the real sale, and every CHECK_S with the sale price after it."""
    if path is None or not t["sold"]:
        return None
    rows = path["rows"]
    times = [r[0] for r in rows]
    p0, p_end = _price_at(rows, t["opened"], times), _price_at(rows, t["sold"], times)
    if not p0 or p_end is None:
        return None
    end = t["sold"] if path.get("complete") else min(t["sold"], times[-1] if times else t["opened"])
    checks, at = [], t["opened"] + CHECK_S
    while at < end:
        p = _price_at(rows, at, times)
        if p is None:
            break
        sale = min(at + SELL_DELAY_S, t["sold"])
        sp = 0.0 if t["pulled_ts"] and t["pulled_ts"] <= sale else (_price_at(rows, sale, times) or 0.0)
        checks.append((p, sp))
        at += CHECK_S
    return {"p0": p0, "p_end": p_end, "checks": checks}


def stopped_return(t: dict, smp: dict | None, stop: float | None, trail: float | None) -> tuple[float, bool]:
    """(return with the stop, whether it fired). Without a usable path the real return stands."""
    if smp is None or (stop is None and trail is None):
        return t["ret"], False
    p0, best = smp["p0"], smp["p0"]
    for p, sp in smp["checks"]:
        best = max(best, p)
        if (stop is not None and p <= p0 * (1 - stop)) or (trail is not None and p <= best * (1 - trail)):
            return max(-1.0, min(CAP, t["ret"] + (sp - smp["p_end"]) / p0)), True
    return t["ret"], False


# drawdown rules --------------------------------------------------------------------------------------------------

def _rules() -> list[tuple[str, dict]]:
    out = [("no limit", {"kind": "none"})]
    out += [(f"day loss {k:g}%", {"kind": "day", "k": k}) for k in DAY_STOPS]
    out += [(f"day {k:g}% off its high", {"kind": "trail", "k": k}) for k in TRAIL_STOPS]
    out += [(f"past {h}h loss {k:g}%", {"kind": "roll", "h": h, "k": k}) for h, k in ROLLING]
    out += [(f"half size after day loss {k:g}%", {"kind": "half", "k": k}) for k in HALF_AT]
    return out


def _day(ts: float) -> int:
    return int(ts // 86400)


def apply_rule(tr: list[dict], rets: list[float], rule: dict, capital: float) -> list[float]:
    """Size multiplier per trade (0 = skipped) under a drawdown rule, using only trades sold before each buy."""
    order = sorted(range(len(tr)), key=lambda i: tr[i]["opened"])
    mult = [1.0] * len(tr)
    if rule["kind"] == "none":
        return mult
    pending: list = []  # (sale ts, pnl)
    window = collections.deque()  # (sale ts, pnl) inside the rolling window
    wsum = 0.0
    day_pnl: dict = {}
    day_high: dict = {}
    for i in order:
        t = tr[i]
        while pending and pending[0][0] <= t["opened"]:
            ts, pnl = heapq.heappop(pending)
            window.append((ts, pnl))
            wsum += pnl
            d = _day(ts)
            day_pnl[d] = day_pnl.get(d, 0.0) + pnl
            day_high[d] = max(day_high.get(d, 0.0), day_pnl[d])
        d = _day(t["opened"])
        cur = day_pnl.get(d, 0.0)
        k = rule.get("k", 0) / 100 * capital
        if rule["kind"] == "day":
            m = 0.0 if cur <= -k else 1.0
        elif rule["kind"] == "trail":
            m = 0.0 if day_high.get(d, 0.0) - cur >= k else 1.0
        elif rule["kind"] == "roll":
            lo = t["opened"] - rule["h"] * 3600
            while window and window[0][0] <= lo:
                wsum -= window.popleft()[1]
            m = 0.0 if wsum <= -k else 1.0
        else:  # half
            m = 0.5 if cur <= -k else 1.0
        mult[i] = m
        if m:
            heapq.heappush(pending, (t["ts"], m * t["cost"] * rets[i]))
    return mult


# scoring ---------------------------------------------------------------------------------------------------------

def score(tr: list[dict], rets: list[float], mult: list[float], base: dict, capital: float, mid: float) -> dict:
    pnl = [m * t["cost"] * r for t, r, m in zip(tr, rets, mult)]
    by_sale = sorted(zip([t["ts"] for t in tr], pnl))
    eq = np.cumsum([p for _, p in by_sale]) if by_sale else np.zeros(1)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))
    dd = float((np.concatenate([[0.0], eq]) - peak).min())
    blocks, days = {}, {}
    for (ts, p) in by_sale:
        blocks[int(ts // (BLOCK_H * 3600))] = blocks.get(int(ts // (BLOCK_H * 3600)), 0.0) + p
        days[_day(ts)] = days.get(_day(ts), 0.0) + p
    keys = base["blocks"]
    b = np.array([blocks.get(k, 0.0) for k in keys])
    bb = np.array([base["block_pnl"][k] for k in keys])
    good, bad = bb > 0, bb < 0
    sd = b.std()
    total = float(sum(pnl))
    first = sum(p for (ts, p) in by_sale if ts < mid)
    traded = sum(m * t["cost"] for t, m in zip(tr, mult))
    return {"usd": round(total), "return_pct": round(100 * total / capital, 1),
            "trades": sum(1 for m in mult if m), "traded_usd": round(traded),
            "mean_pct": round(100 * total / traded, 2) if traded else None,
            "max_dd_usd": round(dd), "max_dd_pct": round(100 * dd / capital, 1),
            "ret_per_dd": round(total / -dd, 2) if dd < -1 else None,
            "block_sharpe": round(float(b.mean() / sd * math.sqrt(365 * 24 / BLOCK_H)), 2) if sd else None,
            "worst_block_usd": round(float(b.min())) if len(b) else 0,
            "worst_day_usd": round(min(days.values())) if days else 0,
            "good_kept_pct": round(100 * b[good].sum() / bb[good].sum()) if good.any() else None,
            "bad_saved_pct": round(100 * (b[bad].sum() - bb[bad].sum()) / -bb[bad].sum()) if bad.any() else None,
            "first_half_usd": round(first), "second_half_usd": round(total - first)}


def _base(tr: list[dict]) -> dict:
    bp: dict = {}
    for t in tr:
        k = int(t["ts"] // (BLOCK_H * 3600))
        bp[k] = bp.get(k, 0.0) + t["cost"] * t["ret"]
    keys = sorted(bp)
    if keys:
        keys = list(range(keys[0], keys[-1] + 1))
    return {"blocks": keys, "block_pnl": {k: bp.get(k, 0.0) for k in keys}}


def study(store, journal, book: str = BOOK, hours: float | None = None, now: float | None = None,
          helius=None, max_credits: int = 60_000, capital: float = CAPITAL) -> dict:
    now = now or time.time()
    since = now - hours * 3600 if hours else 0.0
    tr = trades(journal, store, book, since, now)
    if not tr:
        return {"book": book, "error": "no trades"}
    fetch = fetch_paths(store, helius, tr, max_credits) if helius is not None else None
    paths = load_paths(store) if store is not None else {}
    covered = [t for t in tr if t["pool"] in paths and t["sold"]]
    base = _base(tr)
    mid = (tr[0]["opened"] + max(t["ts"] for t in tr)) / 2
    stops = [("no stop", None, None)] + [(f"stop -{s}%", s / 100, None) for s in STOPS] + \
            [(f"trailing -{s}%", None, s / 100) for s in TRAILS]
    smps = [samples(t, paths.get(t["pool"])) for t in tr]
    stop_rets, stop_info = {}, {}
    for name, s, tl in stops:
        rs, fired, saved, cost, rugged = [], 0, 0.0, 0.0, 0
        for t, smp in zip(tr, smps):
            r, f = stopped_return(t, smp, s, tl)
            rs.append(r)
            if f:
                fired += 1
                rugged += t["ret"] <= -0.99
                d = t["cost"] * (r - t["ret"])
                saved += max(d, 0.0)
                cost += min(d, 0.0)
        stop_rets[name] = rs
        stop_info[name] = {"fired": fired, "saved_on_losers_usd": round(saved), "cut_winners_usd": round(cost),
                           "fired_rugged": rugged}
    rules = _rules()
    grid = []
    for sname, _, _ in stops:
        for rname, rule in rules:
            mult = apply_rule(tr, stop_rets[sname], rule, capital)
            grid.append({"stop": sname, "limit": rname, **score(tr, stop_rets[sname], mult, base, capital, mid)})
    # check of the path units: the path's own minute-6 move vs the real return, on covered trades
    check = []
    for t in covered[:400]:
        rows = paths[t["pool"]]["rows"]
        p0, p1 = _price_at(rows, t["opened"]), _price_at(rows, t["sold"])
        if p0 and p1 is not None and t["ret"] > -0.99:
            check.append((p1 / p0 - 1) - t["ret"])
    return {"book": book, "capital": capital, "trades": len(tr), "covered": len(covered),
            "from": tr[0]["opened"], "to": max(t["ts"] for t in tr), "fetch": fetch,
            "path_check": {"n": len(check), "median_pool_move_minus_real_pct":
                           round(100 * float(np.median(check)), 1) if check else None},
            "stop_info": stop_info, "grid": grid}


# render ----------------------------------------------------------------------------------------------------------

def _t(ts):
    return time.strftime("%m-%d %H:%M", time.gmtime(ts))


def _row(g: dict, label: str) -> str:
    f = lambda v, w, s="": (f"{v}{s}" if v is not None else "-").rjust(w)  # noqa: E731
    return (f"{label:<44}{f(g['usd'], 9)}{f(g['return_pct'], 9, '%')}{f(g['trades'], 7)}{f(g['mean_pct'], 8, '%')}"
            f"{f(g['max_dd_pct'], 8, '%')}{f(g['ret_per_dd'], 7)}{f(g['block_sharpe'], 7)}"
            f"{f(g['worst_block_usd'], 8)}{f(g['worst_day_usd'], 8)}{f(g['good_kept_pct'], 7, '%')}"
            f"{f(g['bad_saved_pct'], 7, '%')}{f(g['first_half_usd'], 8)}{f(g['second_half_usd'], 8)}")


HEAD = (f"{'':<44}{'$':>9}{'ret':>9}{'trades':>7}{'mean':>8}{'maxDD':>8}{'r/DD':>7}{'shrp6h':>7}"
        f"{'worst6h':>8}{'wrstday':>8}{'goodkp':>7}{'badsvd':>7}{'half1 $':>8}{'half2 $':>8}")


def render(out: dict) -> str:
    if out.get("error"):
        return f"STOPS AND DRAWDOWN LIMITS: {out['book']}: {out['error']}"
    L = [f"STOPS AND DRAWDOWN LIMITS: {out['book']}, {_t(out['from'])} .. {_t(out['to'])} UTC, "
         f"{out['trades']} trades, capital ${out['capital']:,.0f}",
         f"stop losses use pool paths for {out['covered']} of {out['trades']} trades (the rest keep their real result)"]
    if out.get("fetch"):
        L.append(f"fetch: {out['fetch']}")
    if out["path_check"]["n"]:
        L.append(f"path check: pool move over the hold minus the real return, median "
                 f"{out['path_check']['median_pool_move_minus_real_pct']} pts on {out['path_check']['n']} trades "
                 "(~6-10 = the fees and fill gap; far off = units wrong)")
    L += ["", "columns: maxDD on the trade-by-trade P&L; r/DD = $ / max drawdown; shrp6h = 6 h block Sharpe "
          "(annualised); goodkp = $ kept on blocks the book made money; badsvd = share of the losing blocks' loss "
          "saved; half1/half2 = each half of the life", ""]
    grid = out["grid"]
    base = next(g for g in grid if g["stop"] == "no stop" and g["limit"] == "no limit")
    L += ["DRAWDOWN LIMITS (no stop)", HEAD] + [_row(g, g["limit"]) for g in grid if g["stop"] == "no stop"]
    L += ["", "STOP LOSSES PER TRADE (no limit)", HEAD]
    for g in grid:
        if g["limit"] == "no limit":
            i = out["stop_info"][g["stop"]]
            L.append(_row(g, f"{g['stop']} (fired {i['fired']})"))
    L += ["", "what the stops did: $ saved on trades that went on falling / $ lost on trades that recovered"]
    for k, i in out["stop_info"].items():
        if k != "no stop":
            L.append(f"  {k:<16} fired {i['fired']:>5}  saved {i['saved_on_losers_usd']:>7}  "
                     f"cut winners {i['cut_winners_usd']:>7}  on trades that rugged anyway {i['fired_rugged']}")
    both = [g for g in grid if g["second_half_usd"] >= base["second_half_usd"] and
            g["first_half_usd"] >= base["first_half_usd"] and g is not base]
    rank = sorted(grid, key=lambda g: (g["ret_per_dd"] or -99), reverse=True)[:TOP]
    L += ["", f"BEST {TOP} BY $ PER $ OF DRAWDOWN (stop + limit)", HEAD] + \
         [_row(g, f"{g['stop']} + {g['limit']}") for g in rank]
    L += ["", f"RULES AT LEAST AS GOOD AS NONE IN BOTH HALVES ({len(both)})", HEAD] + \
         [_row(g, f"{g['stop']} + {g['limit']}") for g in sorted(both, key=lambda g: -g["usd"])[:TOP]]
    L += ["", "Every rule here was picked with the whole life in view and the life is a few days: trust a rule only "
          "if it beats 'no limit' in both halves and its neighbours (the next k up and down) agree."]
    return "\n".join(L)
