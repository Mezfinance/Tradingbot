"""canary-package: the real-money canary's proposed rule set replayed over the year (11-10, read only; Helius only
with --fetch).

The user's package, one rule per liquidity band at second 4, at the canary's own sizes:

  under $15k   sz's rule (rug-vs-win rw_pct < 50)                         $25, sold at pool minute 6
  $15-25k      launchbig15k's rule (best 30% of the band's past day by     $10, minute 6
               the +50%-by-minute-6 model)
  $25-40k      launchrw50_25k10's rule (rw_pct < 50)                      $10, held to pool minute 10
  $40k+        sz's rule                                                   $10, minute 6

compared with the current canary rules (sz's rule in every band at $25 / $10 / $10, all sold at minute 6), each with
and without the canary's drawdown pause (no new buys while the full-size paper twin launchrw50_sz_dd24 is down $500
over the past 24 h, `model_book.DD_LIMITS`, scaled to the replay's money like `dd-limits`).

Inputs: the year replay saved by `risk-model --history` (hist.db `hist_replay`; it must carry each pool's `big`
score, saved since 11-10: otherwise `--fresh` replays now, ~80 min CPU), minute-10 pool states in hist.db
`hist_path` (sec 600; `--fetch` reads the missing ones, 10 credits each, live key, stable hash order). Every trade
pays sz's measured real-minus-ideal gap (`risk_history.gaps`; at $5-25 the price impact is smaller than at sz's
$100-250, so this leans pessimistic) and the canary's fixed real cost per round trip (`FEE_USD`, from the 1-cent
canary's ledger). Each sampled day is rebuilt from its own trades only (`risk_history.expand(k=1)`), so the pause
reads real day-to-day swings rather than the smoothed ones (see `dd-limits`)."""
from __future__ import annotations

import collections
import json
import time
from bisect import bisect_left
from datetime import datetime, timezone

import numpy as np

FEE_USD = 0.30          # real cost per canary round trip (~0.003 SOL incl. the pool's fee-account rent), 11-06 ledger
HOLD10_S = 600
CAPITAL = 10_000.0      # the paper twin's capital: its $500 limit is 5% of it
THIN_MAX = 15_000.0
DEEP_MIN = 40_000.0
SIZES = {"under $15k": 25.0, "$15-25k": 10.0, "$25-40k": 10.0, "$40k+": 10.0}   # user, 11-10: $5 lost to fees
BANDS = tuple(SIZES)
PACKAGES = ("proposed", "current")


def band_of(liq) -> str | None:
    if liq is None:
        return None
    if liq < THIN_MAX:
        return "under $15k"
    if liq < 25_000:
        return "$15-25k"
    if liq < DEEP_MIN:
        return "$25-40k"
    return "$40k+"


def big_ranks(rows: list[dict]) -> dict:
    """Each $15-25k pool's rank of its `big` score among the band's past 24 h (window widened by the sample weight w,
    like the replay's band ranking); None until launch_live.ROLLING_MIN pools are in the window."""
    from tradingbot import launch_live as ll
    lo_b, hi_b = ll.MID_BAND
    ts_, sc = [], []
    out = {}
    for r in rows:
        liq = r.get("liq_usd")
        if r.get("big") is None or liq is None or not (lo_b <= liq < hi_b):
            continue
        lo = bisect_left(ts_, r["create_ts"] - ll.BIG_WINDOW_S * r.get("w", 1.0))
        win = sc[lo:]
        out[r["pool"]] = 100.0 * float(np.mean(np.array(win) < r["big"])) if len(win) >= ll.ROLLING_MIN else None
        ts_.append(r["create_ts"]); sc.append(r["big"])
    return out


def records(rows: list[dict], have10: dict, gap: float, liq_share: float) -> tuple[list[dict], dict]:
    """One record per pool any of the three books would buy: the paper twin's trade (sz's sizes, minute 6) and both
    canary packages' trades, sharing the pool's open time so the twin's pause applies to the canary."""
    from tradingbot import launch_live as ll
    from tradingbot import model_book as mb
    from tradingbot import risk_history as rh
    ranks = big_ranks(rows)
    info = collections.Counter()
    out = []
    for r in rows:
        band = band_of(r.get("liq_usd"))
        if band is None or r.get("rw") is None:
            continue
        rw50 = r.get("rw_pct") is not None and r["rw_pct"] < 100 - ll.RW_SKIP
        big = ranks.get(r["pool"])
        if band == "$15-25k" and r.get("big") is not None and big is None:
            info["$15-25k pools without a big rank yet (thin window: not bought)"] += 1
        rec = {"opened": r["create_ts"] + 10, "ts": r["create_ts"] + ll.EXIT_S, "w": r.get("w", 1.0), "band": band,
               "ret": 0.0, "cost": 0.0}
        if rw50:   # the full-size paper twin: sz's rule and sizes, minute 6
            size = mb.band_usd(mb.LAUNCH_SIZED, r["liq_usd"]) or 100.0
            size = min(size, liq_share * r["liq_usd"])
            if size >= 5:
                rec["ret"], rec["cost"] = max(-1.0, r["ret"] - gap), size
        size = SIZES[band]
        fee = FEE_USD / size
        cur = max(-1.0, r["ret"] - gap - fee) if rw50 else None
        if band == "$15-25k":
            take = big is not None and big >= 100 - ll.BIG_KEEP
            prop, hold = (max(-1.0, r["ret"] - gap - fee) if take else None), ll.EXIT_S
        elif band == "$25-40k":
            prop, hold = None, ll.EXIT_S
            if rw50:
                m10 = rh.hold_ret(r, have10, HOLD10_S)
                info["$25-40k trades"] += 1
                if m10 is None:
                    m10 = r["ret"]
                    info["$25-40k minute-10 states missing (minute 6 used)"] += 1
                prop, hold = max(-1.0, m10 - gap - fee), HOLD10_S
        else:
            prop, hold = cur, ll.EXIT_S
        rec["proposed"] = (prop, size if prop is not None else 0.0, hold - 10)
        rec["current"] = (cur, size if cur is not None else 0.0, ll.EXIT_S - 10)
        if rec["cost"] or prop is not None or cur is not None:
            out.append(rec)
    return out, dict(info)


def _pause(recs: list[dict], capital: float) -> list[float]:
    """The twin's past-24h $500 limit (scaled) at each record's open time; canary trades ride along at zero P&L."""
    from tradingbot import dd_limits as dl
    from tradingbot import model_book as mb
    h, usd = mb.DD_LIMITS[mb.LAUNCH_SIZED_DD]
    return dl.apply_rule(recs, {"kind": "roll", "h": h, "k": 100 * usd / CAPITAL}, capital)


def _stats(trades: list[tuple], days: int) -> dict:
    """trades: (sale ts, cost, ret, band). Totals, per band, daily and monthly P&L, max drawdown, worst day."""
    if not trades:
        return {"trades": 0}
    ts = np.array([t[0] for t in trades]); cost = np.array([t[1] for t in trades]); ret = np.array([t[2] for t in trades])
    pnl = cost * ret
    o = np.argsort(ts, kind="stable")
    cum = np.cumsum(pnl[o])
    dd = float((np.maximum.accumulate(np.concatenate([[0.0], cum])) - np.concatenate([[0.0], cum])).max())
    day = collections.defaultdict(float); month = collections.defaultdict(float)
    by = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
    for (t, c, r, b), p in zip(trades, pnl):
        day[int(t // 86400)] += p
        month[datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m")] += p
        x = by[b]; x[0] += 1; x[1] += c; x[2] += p; x[3] += r <= -0.8
    dv = np.array(list(day.values()))
    return {"trades": len(trades), "per_day": round(len(trades) / max(days, 1), 1), "usd": round(float(pnl.sum())),
            "traded_usd": round(float(cost.sum())), "mean_pct": round(100 * float(pnl.sum() / cost.sum()), 2),
            "max_dd_usd": round(dd), "worst_day_usd": round(float(dv.min())), "best_day_usd": round(float(dv.max())),
            "losing_days_pct": round(100 * float((dv < 0).mean())),
            "sharpe": round(float(dv.mean() / dv.std() * np.sqrt(365)), 2) if dv.std() > 0 else None,
            "bands": {b: {"trades": v[0], "usd": round(v[2]), "mean_pct": round(100 * v[2] / v[1], 2) if v[1] else None,
                          "rugged_pct": round(100 * v[3] / v[0])} for b, v in sorted(by.items(), key=lambda kv: BANDS.index(kv[0]))},
            "months": {m: round(v) for m, v in sorted(month.items())}}


def study(hist_store, watch_store, journal, cfg, fresh: bool = False, helius=None, max_credits: int = 20_000,
          now: float | None = None, progress=None) -> dict:
    from tradingbot import dd_limits as dl
    from tradingbot import risk_history as rh
    from tradingbot import safeguards as sg
    now = now or time.time()
    t0 = time.time()
    cached = None if fresh else sg.load_replay(hist_store)
    if cached and any(r.get("big") is not None for r in cached[0]):
        rows, _ = cached
        source = f"saved replay ({len(rows):,} pools)"
    elif cached and not fresh:
        return {"error": "the saved replay predates the +50% scores (11-10): run canary-package --fresh (replays "
                         "again, ~80 min CPU; scale the machine to performance-2x first) or risk-model --history"}
    else:
        rows, _ = rh.load_pools(hist_store, watch_store, cfg)
        if len(rows) < 500:
            return {"error": "fewer than 500 liquid pools: run history-backfill first"}
        meta = rh.replay(rows)
        sg.save_replay(hist_store, rows, meta)
        source = f"replayed now ({len(rows):,} pools) and saved"
    cands = rh.hold_candidates(rows)   # sz's kept half in $25-40k: the 25k10 rule's pools
    fetched = rh.fetch_holds(hist_store, helius, cands, max_credits, progress, sec=HOLD10_S) if helius else None
    have10 = rh._have_holds(hist_store, HOLD10_S)
    g = rh.gaps(journal, watch_store, now)
    gap = g.get("launchrw50_sz", {"gap": rh.DEFAULT_GAP})["gap"]
    recs, info = records(rows, have10, gap, cfg.model_book_liq_share)
    full, exp = rh.expand(recs, k=1.0)
    for t in full:   # the canary's own sale times follow each record's (relocated) open time
        for p in PACKAGES:
            ret, cost, hold = t[p]
            t[p + "_ts"] = t["opened"] + hold
    live = dl.live_trades(journal, "launchrw50_sz", 0.0, now) if journal is not None else []
    twin = [t for t in full if t["cost"]]
    live_day, year_day = dl._median_daily(live), dl._median_daily(twin)
    scale = year_day / live_day if live_day and year_day else 1.0
    mult = _pause(full, CAPITAL * scale)
    days = max(1, int((max(t["ts"] for t in full) - min(t["opened"] for t in full)) // 86400) + 1) if full else 1
    out = {"source": source, "gap": gap, "fee_usd": FEE_USD, "sizes": SIZES, "expand": exp, "scale": round(scale, 2),
           "pause_usd_scaled": round(500 * scale), "days": days, "info": info, "minute10_fetch": fetched,
           "minute10_states": len(have10), "packages": {}}
    for p in PACKAGES:
        for paused in (False, True):
            tr = [(t[p + "_ts"], t[p][1], t[p][0], t["band"]) for t, m in zip(full, mult)
                  if t[p][0] is not None and (not paused or m)]
            out["packages"][f"{p}{' + pause' if paused else ''}"] = _stats(tr, days)
    out["twin"] = {"no pause": _stats([(t["ts"], t["cost"], t["ret"], t["band"]) for t in twin], days),
                   "with pause": _stats([(t["ts"], t["cost"], t["ret"], t["band"]) for t, m in zip(full, mult)
                                         if t["cost"] and m], days),
                   "paused_share_of_trades_pct": round(100 * (1 - float(np.mean(mult)))) if mult else None}
    out["took_s"] = round(time.time() - t0)
    return out


def render(out: dict) -> str:
    if out.get("error"):
        return "CANARY PACKAGE\n  " + out["error"]
    L = ["CANARY PACKAGE: the proposed rule set over the year, at the canary's sizes", "",
         f"source: {out['source']}; sz's real-minus-ideal gap {100 * out['gap']:.1f} pts and ${out['fee_usd']:.2f} "
         f"real cost per round trip on every trade; days rebuilt from their own trades {out['expand']}",
         "proposed: under $15k sz $25 m6 | $15-25k big15k $10 m6 | $25-40k 25k10 $10 held to minute 10 | $40k+ sz $10 m6",
         "current:  sz's rule in every band at $25 / $10 / $10 / $10, all sold at minute 6",
         f"pause: no new buys while the paper twin (sz at its own sizes) is down ${out['pause_usd_scaled']:,} over "
         f"24 h (the live $500 x{out['scale']} replay money); it paused {out['twin'].get('paused_share_of_trades_pct')}% "
         "of the picks",
         f"minute-10 states on hand: {out['minute10_states']:,}"
         + (f"; this run's fetch: {out['minute10_fetch']}" if out.get("minute10_fetch") else ""),
         f"notes: {out['info']}", ""]
    head = (f"  {'package':<22}{'trades':>9}{'/day':>7}{'$':>10}{'mean':>8}{'max DD':>9}{'worst day':>11}"
            f"{'best day':>10}{'losing d':>9}{'Sharpe':>8}")
    L += ["WHOLE PACKAGE", head]
    for name, s in out["packages"].items():
        if not s.get("trades"):
            L.append(f"  {name:<22}{'no trades':>9}")
            continue
        L.append(f"  {name:<22}{s['trades']:>9,}{s['per_day']:>7}{s['usd']:>10,}{s['mean_pct']:>7}%{s['max_dd_usd']:>9,}"
                 f"{s['worst_day_usd']:>11,}{s['best_day_usd']:>10,}{s['losing_days_pct']:>8}%{str(s['sharpe']):>8}")
    L += ["", "BY BAND ($ / size-weighted mean after gap and fee / rugged)",
          f"  {'package':<22}" + "".join(f"{b:>24}" for b in BANDS)]
    for name, s in out["packages"].items():
        cells = []
        for b in BANDS:
            x = (s.get("bands") or {}).get(b)
            cells.append(f"{x['usd']:>8,} {x['mean_pct']:>6}% {x['rugged_pct']:>3}%" if x else f"{'-':>24}")
        L.append(f"  {name:<22}" + "".join(f"{c:>24}" for c in cells))
    months = sorted({m for s in out["packages"].values() for m in (s.get("months") or {})})
    names = list(out["packages"])
    L += ["", "BY MONTH ($)", f"  {'month':<9}" + "".join(f"{n:>20}" for n in names)]
    for m in months:
        L.append(f"  {m:<9}" + "".join(f"{(out['packages'][n].get('months') or {}).get(m, 0):>20,}" for n in names))
    tw = out["twin"]
    L += ["", "PAPER TWIN (sz at its own sizes; drives the pause)",
          *[f"  {k:<12} trades {v.get('trades', 0):,}  $ {v.get('usd', 0):,}  max DD {v.get('max_dd_usd', 0):,}"
            for k, v in tw.items() if isinstance(v, dict)], "",
          "Read: the bands are judged on the replay's cousin of the live rules (replay vs live rank corr ~0.36 on the "
          "year replay); minute-10 prices only where fetched; the $10 trades carry a fixed fee of ~3%, so those "
          "bands need a real edge just to break even."]
    return "\n".join(L)


def to_json(out: dict) -> str:
    return json.dumps(out, indent=2, default=str)
