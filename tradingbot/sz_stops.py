"""Stop losses, climb checks and drawdown limits on a launch book's real trades (11-09, read only; Helius only with --fetch).

Question (user): would a per-trade exit rule have saved sz money, without giving up its winners? The evidence before
this was mixed: a -25% stop on model5 was slightly worse in its backtest and slightly better on 25 live trades, an
exit model on pool behaviour did no better than random exits, while the climb check (sell at minute 3 unless +5%)
beat its twin book live. Nothing had been measured on sz's own trades (thin pools, ~$250, minute-6 exit).

Rules, each checked every `CHECK_S` seconds on the pool price (pool SOL / pool tokens, as a live vault read would see
it), selling `SELL_DELAY_S` later:
  stop -x%        the pool price is x% below the price at our buy (`STOPS`)
  trailing -x%    x% below the best price seen since our buy (`TRAILS`)
  climb Nm +g%    at N minutes after the buy, sell unless a price seen since the buy was more than g% above it
                  (`CLIMBS`; +0% = sell unless it has been above the buy price at all)

Paths: `fetch_paths` reads each traded pool's transactions from `PRE_S` before our buy to `POST_S` after our sale
(Helius getTransactionsForAddress with a blockTime window, 10 credits per 100 txs, at most `MAX_PAGES`), cached in
watch.db `watch_trade_path`, in a stable hash order so a credit cap leaves an even sample. A path that hit the page cap
before our sale counts as uncovered (its rule can't be judged on half a path).

The exited trade's return = real return + (proceeds per token at the rule's sale - proceeds per token at the real sale)
/ pool price at the buy. Proceeds per token take our own sale's constant-product impact at that moment
(`_keep`: tokens / (tokens + ours)) from the pool's reserves then: larger than at the real sale when liquidity was
withdrawn (both reserves shrink), smaller after a sell-off (the token reserve grows). The trade keeps its own fees
and fill gap. A pull in one transaction leaves nothing to sell, as in reality.

Drawdown limits (`limits()`, run by `dd_limits.apply_rule` on the starting capital `CAPITAL_USD`): day loss
`DAY_LOSS` and day trail `DAY_TRAIL` (no buys for the rest of the UTC day once the day's realized P&L is k below 0 /
below its high for the day), alone and on top of every exit rule; an early sale counts toward the day when its money
came back. They need no path, so they act on every trade.

Per rule: $, change vs no rule with a 90% bootstrap interval over trades (paired), the change in each half of the
life, by liquidity band and per 6 h block, how often it fired, $ saved on trades that went on falling, $ lost on
trades that recovered, fires on trades that rugged anyway, mean per trade, max drawdown, worst 6 h block. The grid is
looked at all at once: trust a rule only when its interval clears zero, both halves agree and its neighbours agree.
Live, any of these needs a pool read every few seconds (these books ignore feed prices and sell at the pool minute)."""
from __future__ import annotations

import bisect
import hashlib
import json
import time

import numpy as np

from tradingbot.data import pumpswap as ps

BOOK = "launchrw50_sz"
CAPITAL = 10_000.0
STUCK_S = 1800  # open longer than this = -100% (as block-compare)
MATCH_S = 600  # a trade belongs to the scorer row of its token created at most this long before the buy
BLOCK_H = 6
CHECK_S = 5.0  # how often a live rule would read the pool
SELL_DELAY_S = 3.0  # from the trigger to the sale landing
PRE_S = 5  # path starts this long before our buy
POST_S = 10  # and ends this long after our sale
MAX_PAGES = 8  # 800 txs per hold at most
CREDITS_PER_PAGE = 10
CAP = 3.0
STOPS = (10, 15, 20, 25, 30, 40, 50)  # % below the pool price at our buy
TRAILS = (20, 30, 40)  # % below the best pool price since our buy
CLIMBS = [(m, g) for m in (1, 2, 3, 4) for g in (0, 5, 10)]  # (minutes after the buy, % gain needed by then)
BANDS = ((0, 15_000, "under $15k"), (15_000, 25_000, "$15-25k"), (25_000, float("inf"), "$25k+"))
DRAWS = 2000
CAPITAL_USD = 10_000.0   # drawdown limits are % of this (the book's starting cash)
DAY_LOSS = (1, 2, 3, 5, 7.5, 10)   # % of capital: stop buying for the rest of the UTC day once down this much
DAY_TRAIL = (1, 2, 3, 5, 7.5, 10)  # % of capital below the day's realized high
TOP = 10

SCHEMA = "CREATE TABLE IF NOT EXISTS watch_trade_path(pool TEXT PRIMARY KEY, data TEXT, ts REAL)"


# trades ----------------------------------------------------------------------------------------------------------

def trades(journal, store, book: str, since: float, now: float) -> list[dict]:
    """The book's real trades (closed; exit blocked or open > STUCK_S = -100%), with the scorer row of the token."""
    rows: dict = {}
    if store is not None:
        try:
            for r in store.query("SELECT pool, token_mint, create_ts, pulled_ts, liq_usd, sol_at_decide "
                                 "FROM watch_launch_live WHERE token_mint IS NOT NULL AND create_ts >= ? "
                                 "ORDER BY create_ts", (since - MATCH_S,)):
                rows.setdefault(r["token_mint"], []).append(dict(r))
        except Exception:  # noqa: BLE001 - no scorer table
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
        sc = m[-1] if m else {}
        liq, sol = sc.get("liq_usd"), sc.get("sol_at_decide")
        out.append({"id": p["id"], "token": p["token"], "opened": p["opened_at"], "ts": ts,
                    "sold": p["closed_at"], "cost": p["cost_usd"], "ret": max(-1.0, ret),
                    "pool": sc.get("pool"), "create_ts": sc.get("create_ts"), "pulled_ts": sc.get("pulled_ts"),
                    "liq": liq, "sol_usd": liq / (2 * sol) if liq and sol else None})
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
    st = _state_at(rows, ts, times)
    if st is None:
        return None
    return st[0] / st[1] if st[0] and st[1] else 0.0


def _state_at(rows: list[list], ts: float, times: list | None = None):
    times = times if times is not None else [r[0] for r in rows]
    i = bisect.bisect_right(times, ts) - 1
    return None if i < 0 else (rows[i][1], rows[i][2])


def _keep(state, ours: float | None) -> float:
    """Proceeds per token of selling `ours` raw tokens into a pool holding `state` (SOL, tokens), per token at the
    spot price: tokens / (tokens + ours) on a constant-product pool. 1 when our size is unknown."""
    if not ours or not state or not state[1]:
        return 1.0
    return state[1] / (state[1] + ours)


def samples(t: dict, path: dict | None) -> dict | None:
    """What a live rule would see: the buy price, the real sale's proceeds per token, and every CHECK_S the price
    then and the proceeds per token of a sale landing SELL_DELAY_S later. None = no usable path."""
    if path is None or not t["sold"] or not path.get("complete"):
        return None
    rows = path["rows"]
    times = [r[0] for r in rows]
    p0 = _price_at(rows, t["opened"], times)
    end_state = _state_at(rows, t["sold"], times)
    if not p0 or end_state is None:
        return None
    ours = t["cost"] / t["sol_usd"] / p0 if t.get("sol_usd") else None   # raw tokens bought (spot, before impact)

    def proceeds(ts, state):
        if t["pulled_ts"] and t["pulled_ts"] <= ts:
            return 0.0
        price = state[0] / state[1] if state and state[0] and state[1] else 0.0
        return price * _keep(state, ours)

    checks, at = [], t["opened"] + CHECK_S
    while at < t["sold"]:
        p = _price_at(rows, at, times)
        if p is None:
            break
        sale = min(at + SELL_DELAY_S, t["sold"])
        checks.append((at - t["opened"], p, proceeds(sale, _state_at(rows, sale, times))))
        at += CHECK_S
    return {"p0": p0, "end": proceeds(t["sold"], end_state), "checks": checks, "impact": ours is not None}


def rule_return(t: dict, smp: dict | None, rule: dict) -> tuple[float, bool, float]:
    """(return under the rule, whether it sold early, when its money came back). Without a usable path the real
    trade stands."""
    if smp is None or rule["kind"] == "none":
        return t["ret"], False, t["ts"]
    p0, best, seen = smp["p0"], smp["p0"], 0.0   # seen = best price at the checks after the buy (climb)
    for age, p, sp in smp["checks"]:
        best, seen = max(best, p), max(seen, p)
        if rule["kind"] == "stop":
            hit = p <= p0 * (1 - rule["x"])
        elif rule["kind"] == "trail":
            hit = p <= best * (1 - rule["x"])
        else:  # climb: one check at its minute
            if age < rule["after"]:
                continue
            if seen > p0 * (1 + rule["gain"]):
                return t["ret"], False, t["ts"]
            hit = True
        if hit:
            sold = min(t["opened"] + age + SELL_DELAY_S, t["sold"])
            return max(-1.0, min(CAP, t["ret"] + (sp - smp["end"]) / p0)), True, sold
    return t["ret"], False, t["ts"]


def limits() -> list[tuple[str, dict]]:
    """Drawdown limits from dd-limits, on the starting capital: the day's realized loss, and the day's give-back
    from its realized high (no buys for the rest of the UTC day once hit)."""
    out = [("no limit", {"kind": "none"})]
    out += [(f"day loss {k:g}%", {"kind": "day", "k": k}) for k in DAY_LOSS]
    out += [(f"day trail {k:g}%", {"kind": "trail", "k": k}) for k in DAY_TRAIL]
    return out


def rules() -> list[tuple[str, str, dict]]:
    out = [("no rule", "", {"kind": "none"})]
    out += [(f"stop -{x}%", "stop", {"kind": "stop", "x": x / 100}) for x in STOPS]
    out += [(f"trailing -{x}%", "trailing", {"kind": "trail", "x": x / 100}) for x in TRAILS]
    out += [(f"climb {m}m +{g}%", "climb", {"kind": "climb", "after": m * 60, "gain": g / 100}) for m, g in CLIMBS]
    return out


# scoring ---------------------------------------------------------------------------------------------------------

def _band(liq):
    for lo, hi, name in BANDS:
        if liq is not None and lo <= liq < hi:
            return name
    return "unknown"


def _ci(delta: np.ndarray, rng) -> list:
    if not len(delta) or not delta.any():
        return [0, 0]
    idx = rng.integers(0, len(delta), (DRAWS, len(delta)))
    sums = delta[idx].sum(axis=1)
    return [round(float(np.percentile(sums, 5))), round(float(np.percentile(sums, 95)))]


def score(tr: list[dict], rets: np.ndarray, fired: np.ndarray, base: np.ndarray, mid: float, rng,
          mult: np.ndarray | None = None, ts: np.ndarray | None = None, ci: bool = True) -> dict:
    cost = np.array([t["cost"] for t in tr])
    mult = np.ones(len(tr)) if mult is None else mult
    fired = fired & (mult > 0)
    pnl = mult * cost * rets
    delta = pnl - base
    ts = np.array([t["ts"] for t in tr]) if ts is None else ts
    order = np.argsort(ts, kind="stable")
    eq = np.concatenate([[0.0], np.cumsum(pnl[order])])
    dd = float((eq - np.maximum.accumulate(eq)).min())
    blocks = (ts // (BLOCK_H * 3600)).astype(np.int64)
    keys, inv = np.unique(blocks, return_inverse=True)
    bpnl = np.bincount(inv, weights=pnl, minlength=len(keys))
    bdel = np.bincount(inv, weights=delta, minlength=len(keys))
    real = np.array([t["ret"] for t in tr])
    went_on = fired & (delta > 0)
    traded = float((mult * cost).sum())
    return {"usd": round(float(pnl.sum())), "delta_usd": round(float(delta.sum())),
            "delta_ci": _ci(delta, rng) if ci else None, "trades": int((mult > 0).sum()),
            "mean_pct": round(100 * float(pnl.sum() / traded), 2) if traded else None,
            "ret_per_dd": round(float(pnl.sum()) / -dd, 2) if dd < -1 else None,
            "fired": int(fired.sum()), "saved_usd": round(float(delta[went_on].sum())),
            "cut_winners_usd": round(float(delta[fired & (delta < 0)].sum())),
            "fired_on_rugs": int((fired & (real <= -0.99)).sum()),
            "max_dd_usd": round(dd), "worst_block_usd": round(float(bpnl.min())) if len(bpnl) else 0,
            "blocks_better": int((bdel > 0.5).sum()), "blocks_worse": int((bdel < -0.5).sum()),
            "half1_delta": round(float(delta[ts < mid].sum())), "half2_delta": round(float(delta[ts >= mid].sum())),
            "by_band": {b: round(float(delta[[_band(t["liq"]) == b for t in tr]].sum())) for _, _, b in BANDS}}


def study(store, journal, book: str = BOOK, hours: float | None = None, now: float | None = None,
          helius=None, max_credits: int = 15_000, seed: int = 7) -> dict:
    now = now or time.time()
    since = now - hours * 3600 if hours else 0.0
    tr = trades(journal, store, book, since, now)
    if not tr:
        return {"book": book, "error": "no trades"}
    fetch = fetch_paths(store, helius, tr, max_credits) if helius is not None else None
    paths = load_paths(store) if store is not None else {}
    smps = [samples(t, paths.get(t["pool"])) if t["pool"] else None for t in tr]
    have = [t["pool"] in paths for t in tr]
    cover = {"trades": len(tr), "with_scorer_row": sum(1 for t in tr if t["pool"]),
             "path_fetched": sum(have), "usable": sum(1 for s in smps if s),
             "cut_by_page_cap": sum(1 for t, h in zip(tr, have) if h and not paths[t["pool"]].get("complete")),
             "impact_modelled": sum(1 for s in smps if s and s["impact"]),
             "still_open_or_stuck": sum(1 for t in tr if not t["sold"])}
    rng = np.random.default_rng(seed)
    cost = np.array([t["cost"] for t in tr])
    base = cost * np.array([t["ret"] for t in tr])
    mid = (tr[0]["opened"] + max(t["ts"] for t in tr)) / 2
    from tradingbot import dd_limits
    res, combos = [], []
    for label, fam, rule in rules():
        rs, fs, sts = zip(*(rule_return(t, s, rule) for t, s in zip(tr, smps)))
        rs, fs, sts = np.array(rs), np.array(fs, dtype=bool), np.array(sts)
        res.append({"rule": label, "family": fam, **score(tr, rs, fs, base, mid, rng, ts=sts)})
        # the same trades under each drawdown limit; an early sale counts toward the day when its money came back
        lt = [{"opened": t["opened"], "ts": float(x), "cost": t["cost"], "ret": float(r)}
              for t, r, x in zip(tr, rs, sts)]
        for lname, lim in limits()[1:]:
            mult = np.array(dd_limits.apply_rule(lt, lim, CAPITAL_USD))
            combos.append({"rule": label, "family": fam, "limit": lname, "_args": (rs, fs, sts, mult),
                           **score(tr, rs, fs, base, mid, rng, mult=mult, ts=sts, ci=False)})
    alone = [c for c in combos if c["rule"] == "no rule"]
    usd_alone = {r["rule"]: r["usd"] for r in res}
    # combinations where the limit changed something (otherwise they repeat the exit rule's own row)
    bound = [c for c in combos if c["rule"] != "no rule" and c["usd"] != usd_alone[c["rule"]]]
    best = sorted(bound, key=lambda c: -(c["ret_per_dd"] or -99))[:TOP]
    for c in alone + best:   # intervals only where they are read
        if c.get("delta_ci") is None:
            rs, fs, sts, mult = c["_args"]
            c["delta_ci"] = _ci(mult * cost * rs - base, rng)
    for c in combos:
        c.pop("_args", None)
    # units check: the pool's move over the hold minus the real return, on usable non-rug trades
    check = [(s["end"] / s["p0"] - 1) - t["ret"] for t, s in zip(tr, smps) if s and t["ret"] > -0.99]
    usable = [t for t, s in zip(tr, smps) if s]
    return {"book": book, "from": tr[0]["opened"], "to": max(t["ts"] for t in tr), "coverage": cover,
            "fetch": fetch, "usable_usd": round(float(sum(t["cost"] * t["ret"] for t in usable))),
            "path_check": {"n": len(check), "median_pts": round(100 * float(np.median(check)), 1) if check else None},
            "rules": res, "limits": alone, "combos": combos, "best_combos": best,
            "capital": CAPITAL_USD}


# render ----------------------------------------------------------------------------------------------------------

def _t(ts):
    return time.strftime("%m-%d %H:%M", time.gmtime(ts))


def _f(v, w):
    return (f"{v:,}" if isinstance(v, int) else str(v) if v is not None else "-").rjust(w)


HEAD = (f"  {'rule':<17}{'fired':>6}{'$':>9}{'vs none':>9}{'90% CI':>17}{'half 1':>8}{'half 2':>8}"
        f"{'blk +/-':>9}{'saved':>8}{'cut win':>9}{'on rugs':>8}{'mean':>8}{'max DD':>9}")


LHEAD = (f"  {'limit':<22}{'trades':>7}{'$':>9}{'vs none':>9}{'90% CI':>17}{'half 1':>8}{'half 2':>8}"
         f"{'blk +/-':>9}{'mean':>8}{'max DD':>9}{'$/DD':>7}")


def _lrow(c: dict, label: str | None = None, width: int = 22) -> str:
    ci = f"{c['delta_ci'][0]:,}..{c['delta_ci'][1]:,}" if c.get("delta_ci") else "-"
    return (f"  {(label or c['limit']):<{width}}{c['trades']:>7,}{c['usd']:>9,}{c['delta_usd']:>9,}{ci:>17}"
            f"{c['half1_delta']:>8,}{c['half2_delta']:>8,}{(str(c['blocks_better']) + '/' + str(c['blocks_worse'])):>9}"
            f"{_f(c['mean_pct'], 7)}%{c['max_dd_usd']:>9,}{_f(c['ret_per_dd'], 7)}")


def _row(r: dict) -> str:
    ci = f"{r['delta_ci'][0]:,}..{r['delta_ci'][1]:,}"
    return (f"  {r['rule']:<17}{_f(r['fired'], 6)}{_f(r['usd'], 9)}{_f(r['delta_usd'], 9)}{ci:>17}"
            f"{_f(r['half1_delta'], 8)}{_f(r['half2_delta'], 8)}{(str(r['blocks_better']) + '/' + str(r['blocks_worse'])):>9}"
            f"{_f(r['saved_usd'], 8)}{_f(r['cut_winners_usd'], 9)}{_f(r['fired_on_rugs'], 8)}"
            f"{_f(r['mean_pct'], 7)}%{_f(r['max_dd_usd'], 9)}")


def _name(r: dict) -> str:
    if "limit" not in r:
        return r["rule"]
    return r["limit"] if r["rule"] == "no rule" else f"{r['rule']} + {r['limit']}"


def render(out: dict) -> str:
    if out.get("error"):
        return f"STOPS AND CLIMB CHECKS: {out['book']}: {out['error']}"
    c = out["coverage"]
    L = [f"STOPS AND CLIMB CHECKS: {out['book']}, {_t(out['from'])} .. {_t(out['to'])} UTC",
         f"coverage: {c['usable']} of {c['trades']} trades judged (scorer row {c['with_scorer_row']}, path fetched "
         f"{c['path_fetched']}, cut by the page cap {c['cut_by_page_cap']}, open or stuck {c['still_open_or_stuck']}; "
         f"own sale impact modelled on {c['impact_modelled']}); the rest keep their real result",
         f"the judged trades made ${out['usable_usd']:,} as traded"]
    if out.get("fetch"):
        L.append(f"fetch: {out['fetch']}")
    if out["path_check"]["n"]:
        L.append(f"path check: pool proceeds over the hold minus the real return, median {out['path_check']['median_pts']}"
                 f" pts on {out['path_check']['n']} trades (expected ~+6..+10: fees, fill gap, buy impact; far off ="
                 " units or matching wrong)")
    L += ["", "vs none = $ change against no rule (90% CI from resampling trades); blk +/- = 6 h blocks the rule "
          "helped / hurt; saved = $ gained on trades it sold that went on falling; cut win = $ lost on trades it sold "
          "that recovered; on rugs = fires on trades that rugged anyway", "", HEAD]
    for fam in ("", "stop", "trailing", "climb"):
        if fam:
            L.append(f"  -- {fam}")
        L += [_row(r) for r in out["rules"] if r["family"] == fam]
    L += ["", "BY LIQUIDITY BAND ($ change vs none)", f"  {'rule':<17}" + "".join(f"{b:>12}" for _, _, b in BANDS)]
    for r in out["rules"][1:]:
        L.append(f"  {r['rule']:<17}" + "".join(f"{r['by_band'][b]:>12,}" for _, _, b in BANDS))
    base = out["rules"][0]
    L += ["", f"DRAWDOWN LIMITS ALONE (no exit rule; % of ${out['capital']:,.0f}; no buys for the rest of the UTC day "
          "once hit; open trades run to their exit)", LHEAD, _lrow({**base, "limit": "no limit"})]
    L += [_lrow(c) for c in out["limits"]]
    lims = ["no limit"] + [c["limit"] for c in out["limits"]]
    short = lambda n: n.replace("day loss ", "loss ").replace("day trail ", "trl ").replace("no limit", "none")  # noqa: E731
    grid = {(c["rule"], c["limit"]): c for c in out["combos"]}
    L += ["", "EXIT RULE x DRAWDOWN LIMIT ($ change vs no rule and no limit)",
          f"  {'rule':<17}" + "".join(f"{short(n):>10}" for n in lims)]
    for r in out["rules"]:
        cells = [r["delta_usd"]] + [grid[(r["rule"], n)]["delta_usd"] for n in lims[1:]]
        L.append(f"  {r['rule']:<17}" + "".join(f"{v:>10,}" for v in cells))
    L += ["", f"BEST {TOP} COMBINATIONS BY $ PER $ OF DRAWDOWN, where the limit changed something "
          f"(no rule, no limit: {base['ret_per_dd']})", LHEAD.replace(f"{'limit':<22}", f"{'exit rule + limit':<34}")]
    L += [_lrow(c, f"{c['rule']} + {c['limit']}", 34) for c in out["best_combos"]] or ["  none: no limit ever bound"]
    judged = out["rules"][1:] + out["limits"] + out["best_combos"]
    clear = [r for r in judged if r.get("delta_ci") and r["delta_ci"][0] > 0 and r["half1_delta"] > 0
             and r["half2_delta"] > 0]
    L += ["", "RULES WHOSE INTERVAL CLEARS ZERO AND HELP IN BOTH HALVES: "
          + (", ".join(_name(r) for r in clear) or "none")]
    L.append("Read: a rule is worth building only if it is in that list and the settings either side of it are too. "
             "Drawdown limits count every trade (no path needed); exit rules change only the judged trades.")
    return "\n".join(L)
