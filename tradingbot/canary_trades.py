"""Every real canary trade, in every form the canary has taken, with the pool's price path after our buy (11-10).

Read only. Helius is used only with --fetch, on the research (backup) key: it reads each traded pool's transactions
from just before our buy to pool minute 15 (or our sale, if later) into watch.db `watch_canary_path`, once per pool.

Per trade: the book it ran under (archived resets `canary_rw50_sz@<tag>`, `canary_rw50_24h`, `canary_launch4_2`,
...), size, our fill second, sale, net and swap result, exit reason, the scorer's row for the pool (liquidity at
second 4, LP pull time, ideal minute-6 result) and the paper twin's trade on the same token. With a path: what
selling at each offset after our fill would have returned before fees, from the pool's own reserves (our own sale
impact included when our token count can be estimated; 0 once the LP was pulled), the peak before the pull, and the
gap between our real swap result and the pool's return at our real sale (our execution cost).
"""
from __future__ import annotations

import json
import statistics
import time

from tradingbot import model_book as mb
from tradingbot import sz_stops as ss
from tradingbot.canary_rugs import swap_return

LIKE = "canary%"           # every canary book, archived resets included
MATCH_S = 600              # a trade belongs to the scorer row of its token created at most this long before the buy
PRE_S = 5                  # path starts this long before our fill
WINDOW_S = 900             # and runs to pool minute 15 (or our sale, if later)
POST_S = 10
MAX_PAGES = 10             # 1,000 transactions per pool at most
CREDITS_PER_PAGE = 10
OFFSETS = (3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 600)  # seconds after our fill
THIN_USD = 15_000
STUCK_S = 1800             # open longer than this with no sale = -100%
SCHEMA = "CREATE TABLE IF NOT EXISTS watch_canary_path(pool TEXT PRIMARY KEY, data TEXT, ts REAL)"


def _ts(x):
    return time.strftime("%m-%d %H:%M:%S", time.gmtime(x)) if x else None


def trades(journal, store, now: float | None = None) -> list[dict]:
    now = now or time.time()
    rows: dict = {}
    try:
        for r in store.query("SELECT pool, token_mint, create_ts, pulled_ts, liq_usd, sol_at_decide, ret, rw_pct, "
                             "fill_ts FROM watch_launch_live WHERE token_mint IS NOT NULL ORDER BY create_ts"):
            rows.setdefault(r["token_mint"], []).append(dict(r))
    except Exception:  # noqa: BLE001 - no scorer table
        rows = {}
    twins: dict = {}
    for p in journal.query("SELECT token, opened_at, cost_usd, realized_usd, closed_at FROM positions WHERE strategy=?",
                           (mb.CANARY_TWIN,)):
        twins.setdefault(p["token"], []).append(dict(p))
    out = []
    for p in journal.query("SELECT id, strategy, token, symbol, opened_at, closed_at, cost_usd, realized_usd, "
                           "exit_reason FROM positions WHERE strategy LIKE ? ORDER BY opened_at", (LIKE,)):
        fills = [dict(f) for f in journal.query(
            "SELECT side, price, usd, qty, fee_usd, ts, tx FROM fills WHERE position_id=? ORDER BY ts", (p["id"],))]
        buys = [f for f in fills if f["side"] == "buy"]
        sells = [f for f in fills if f["side"] == "sell"]
        fill = buys[0]["ts"] if buys else p["opened_at"]
        m = [r for r in rows.get(p["token"], []) if r["create_ts"] <= p["opened_at"] <= r["create_ts"] + MATCH_S]
        sc = m[-1] if m else {}
        tw = [t for t in twins.get(p["token"], []) if abs(t["opened_at"] - p["opened_at"]) <= MATCH_S]
        tw = tw[0] if tw else None
        if p["closed_at"]:
            ret, state = (p["realized_usd"] / p["cost_usd"] - 1) if p["cost_usd"] else None, "closed"
        elif now - p["opened_at"] > STUCK_S:
            ret, state = -1.0, "stuck"
        else:
            ret, state = None, "open"
        liq, sol, c = sc.get("liq_usd"), sc.get("sol_at_decide"), sc.get("create_ts")
        out.append({
            "id": p["id"], "book": p["strategy"], "token": p["token"], "symbol": p["symbol"],
            "opened": p["opened_at"], "fill_ts": fill, "sold": p["closed_at"], "state": state,
            "cost": p["cost_usd"], "realized": p["realized_usd"], "ret": ret, "swap": swap_return(fills),
            "fees_usd": round(sum(f["fee_usd"] or 0 for f in fills), 4), "exit_reason": p["exit_reason"],
            "buy_tx": buys[0]["tx"] if buys else None, "sell_tx": sells[-1]["tx"] if sells else None,
            "pool": sc.get("pool"), "create_ts": c, "liq": liq, "thin": bool(liq and liq < THIN_USD),
            "sol_usd": liq / (2 * sol) if liq and sol else None, "pulled_ts": sc.get("pulled_ts"),
            "ideal": sc.get("ret"), "rw_pct": sc.get("rw_pct"),
            "fill_s": round(fill - c, 1) if c else None,
            "sold_s": round(p["closed_at"] - c, 1) if c and p["closed_at"] else None,
            "pulled_after_fill_s": round(sc["pulled_ts"] - fill, 1) if sc.get("pulled_ts") else None,
            "twin": ({"ret": (tw["realized_usd"] / tw["cost_usd"] - 1) if tw["closed_at"] and tw["cost_usd"] else None,
                      "cost": tw["cost_usd"], "fill_s": round(tw["opened_at"] - c, 1) if c else None}
                     if tw else None),
        })
    return out


def fetch_paths(store, helius, tr: list[dict], max_credits: int) -> dict:
    store.execute(SCHEMA)
    have = {r["pool"] for r in store.query("SELECT pool FROM watch_canary_path")}
    todo = list({t["pool"]: t for t in tr if t["pool"] and t["pool"] not in have}.values())
    stats = {"wanted": len(todo), "fetched": 0, "failed": 0, "credits": 0, "stopped_by_cap": False}
    start = helius.credits_used
    for t in todo:
        if helius.credits_used - start + MAX_PAGES * CREDITS_PER_PAGE > max_credits:
            stats["stopped_by_cap"] = True
            break
        lo = int(t["fill_ts"] - PRE_S)
        hi = int(max((t["create_ts"] or t["fill_ts"]) + WINDOW_S, t["sold"] or 0) + POST_S)
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
        rows = ss.path_rows(txs, t["pool"], t["token"])
        store.execute("INSERT OR REPLACE INTO watch_canary_path(pool, data, ts) VALUES(?,?,?)",
                      (t["pool"], json.dumps({"rows": rows, "complete": done, "lo": lo, "hi": hi, "txs": len(txs)}),
                       time.time()))
        stats["fetched"] += 1
    stats["credits"] = helius.credits_used - start
    return stats


def load_paths(store) -> dict:
    try:
        return {r["pool"]: json.loads(r["data"]) for r in store.query("SELECT pool, data FROM watch_canary_path")}
    except Exception:  # noqa: BLE001 - no table yet
        return {}


def curve(t: dict, path: dict | None) -> dict | None:
    """Return of selling at each offset after our fill, before fees, from the pool's reserves."""
    if not path or not path.get("rows"):
        return None
    rows = path["rows"]
    times = [r[0] for r in rows]
    p0 = ss._price_at(rows, t["fill_ts"], times)
    if not p0:
        return None
    ours = t["cost"] / t["sol_usd"] / p0 if t.get("sol_usd") else None   # raw tokens bought (spot, before impact)
    end = path.get("hi") or rows[-1][0]

    def ret_at(ts):
        if t["pulled_ts"] and t["pulled_ts"] <= ts:
            return -1.0
        st = ss._state_at(rows, ts, times)
        if st is None:
            return None
        price = st[0] / st[1] if st[0] and st[1] else 0.0
        return price * ss._keep(st, ours) / p0 - 1

    pts = {}
    for o in OFFSETS:
        at = t["fill_ts"] + o
        pts[o] = round(100 * ret_at(at), 1) if at <= end and ret_at(at) is not None else None
    peak, peak_s = None, None
    stop = min(t["pulled_ts"] or end, end)
    for r in rows:
        if t["fill_ts"] < r[0] < stop:
            v = ret_at(r[0])
            if v is not None and (peak is None or v > peak):
                peak, peak_s = v, r[0] - t["fill_ts"]
    at_sale = ret_at(t["sold"]) if t["sold"] and t["sold"] <= end else None
    return {
        "pct_at": pts, "peak_pct": round(100 * peak, 1) if peak is not None else None,
        "peak_s": round(peak_s, 1) if peak_s is not None else None,
        "pool_at_our_sale_pct": round(100 * at_sale, 1) if at_sale is not None else None,
        "our_swap_minus_pool_pts": (round(100 * t["swap"] - 100 * at_sale, 1)
                                    if at_sale is not None and t["swap"] is not None else None),
        "complete": bool(path.get("complete")), "txs": path.get("txs"), "impact_modelled": ours is not None,
    }


def _summary(tr: list[dict]) -> dict:
    with_c = [t for t in tr if t.get("curve")]
    by = {}
    for o in OFFSETS:
        xs = [t["curve"]["pct_at"][o] for t in with_c if t["curve"]["pct_at"].get(o) is not None]
        by[o] = ({"n": len(xs), "mean": round(statistics.mean(xs), 1), "median": round(statistics.median(xs), 1),
                  "above_0_pct": round(100 * sum(x > 0 for x in xs) / len(xs), 1)} if xs else {"n": 0})
    rugs = [t for t in tr if t["pulled_after_fill_s"] is not None]
    return {
        "trades": len(tr), "with_path": len(with_c),
        "pulled after our fill": {
            "n": len(rugs),
            "median s from our fill to the pull": (round(statistics.median([t["pulled_after_fill_s"] for t in rugs]), 1)
                                                   if rugs else None)},
        "sell at offset (pool return before fees, % of our buy)": by,
    }


def study(store, journal, helius=None, max_credits: int = 5000, now: float | None = None) -> dict:
    tr = trades(journal, store, now)
    fetch = fetch_paths(store, helius, tr, max_credits) if helius is not None else None
    paths = load_paths(store)
    for t in tr:
        t["curve"] = curve(t, paths.get(t["pool"]))
    groups = {"all": tr, "thin (< $15k)": [t for t in tr if t["thin"]],
              "$15k+": [t for t in tr if t["liq"] and not t["thin"]]}
    sizes = sorted({round(t["cost"] or 0, 2) for t in tr})
    for s in sizes:
        groups[f"size ${s:g}"] = [t for t in tr if round(t["cost"] or 0, 2) == s]
    return {"books": sorted({t["book"] for t in tr}), "fetch": fetch,
            "summary": {k: _summary(v) for k, v in groups.items() if v}, "trades": tr}


def render(out: dict) -> str:
    L = ["CANARY TRADES (every form, archived resets included)", f"books: {', '.join(out['books']) or 'none'}"]
    if out["fetch"]:
        L.append(f"path fetch: {out['fetch']}")
    for name, s in out["summary"].items():
        L += ["", f"{name}: {s['trades']} trades, {s['with_path']} with a pool path; pulled after our fill "
                  f"{s['pulled after our fill']['n']} (median {s['pulled after our fill']['median s from our fill to the pull']} s)",
              "  sell at  " + " ".join(f"{o:>6}s" for o in OFFSETS)]
        by = s["sell at offset (pool return before fees, % of our buy)"]
        L.append("  mean %   " + " ".join(f"{by[o].get('mean', '-'):>7}" for o in OFFSETS))
        L.append("  median % " + " ".join(f"{by[o].get('median', '-'):>7}" for o in OFFSETS))
        L.append("  above 0% " + " ".join(f"{by[o].get('above_0_pct', '-'):>7}" for o in OFFSETS))
        L.append("  n        " + " ".join(f"{by[o]['n']:>7}" for o in OFFSETS))
    L += ["", "TRADES", f"{'opened':>14} {'book':<28} {'symbol':<10} {'$':>6} {'liq':>7} {'fill s':>6} {'pull+s':>6} "
          f"{'net %':>7} {'swap %':>7} {'peak %':>7} {'peak s':>6} {'gap':>6} exit"]
    for t in out["trades"]:
        c = t["curve"] or {}
        f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
        L.append(f"{_ts(t['opened']):>14} {t['book'][:28]:<28} {(t['symbol'] or '')[:10]:<10} {f(t['cost'], 2):>6} "
                 f"{f(t['liq'], 0):>7} {f(t['fill_s']):>6} {f(t['pulled_after_fill_s']):>6} "
                 f"{f(100 * t['ret'] if t['ret'] is not None else None):>7} "
                 f"{f(100 * t['swap'] if t['swap'] is not None else None):>7} {f(c.get('peak_pct')):>7} "
                 f"{f(c.get('peak_s')):>6} {f(c.get('our_swap_minus_pool_pts')):>6} {t['exit_reason'] or t['state']}")
    return "\n".join(L)
