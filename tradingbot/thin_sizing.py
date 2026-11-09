"""Thin-band sizing study (11-09, read only, no Helius): which trade size suits pools under $15k at second 4?

For every finished liquid thin pool the launch scorer measured (vault reads at the fill and at minute 6), each sizing
policy's trade is rebuilt from the ideal one: the price impact of our own buy and sale on a constant-product pool
(`launch_live._amm_keep`, reserves from the vault reads), the scorer's flat 6% cost, the +300% cap, rugs -100%, and a
flat extra execution gap (timing, landing) calibrated on launchrw50's real thin trades at $100. Three filters (every
thin pool / sz's rw50 picks / the band rule), ten sizings. Per filter and sizing: $ and % on the money traded, daily
Sharpe, drawdown, worst day, peak money open; per 6 h block; a walk-forward that picks each block's sizing from the
trailing 24 h; and a one-year day bootstrap on a fixed capital (no compounding).

Optimistic in one way: other traders' reaction to our size is not modelled, so large sizes read better than they
would trade. Pessimistic in none. Treat sizes above ~2% of the pool as an upper bound."""
from __future__ import annotations

import collections
import math
import time

import numpy as np

COST = 0.06
CAP = 3.0
RUG = -0.999
BLOCK_H = 6
LOOKBACK_H = 24
DEFAULT_GAP = 0.02  # extra execution gap when there are too few real trades to calibrate
GAP_MIN = 30
GAP_BOOK = "launchrw50"  # flat $100: its real thin trades calibrate the extra gap
SIMS = 400
DAYS = 365
CAPITAL = 10_000.0
INF = float("inf")
SIZINGS = (("$50", 50.0, 0.02), ("$100", 100.0, 0.02), ("$250 (2% cap)", 250.0, 0.02),
           ("$500 (2% cap)", 500.0, 0.02), ("$1,000 (2% cap)", 1000.0, 0.02), ("0.5% of pool", INF, 0.005),
           ("1% of pool", INF, 0.01), ("2% of pool", INF, 0.02), ("3% of pool", INF, 0.03), ("5% of pool", INF, 0.05))
BASE = "$100"
FILTERS = ("every thin pool", "sz picks (rw50)", "band rule")


def load(store, since: float = 0.0) -> list[dict]:
    from tradingbot import launch_live as ll
    ll.ensure_schema(store)
    out = []
    for r in store.query(
            "SELECT pool, token_mint, create_ts, ret, liq_usd, sol_at_decide, fill_sol, exit_sol, fill_px, exit_px, "
            "rw_pct, rwb_pct FROM watch_launch_live WHERE status='done' AND liquid=1 AND ret IS NOT NULL "
            "AND liq_usd IS NOT NULL AND liq_usd < ? AND create_ts >= ? ORDER BY create_ts", (ll.THIN_MAX_USD, since)):
        r = dict(r)
        fl = ll.thin_flags(r["liq_usd"], r["rw_pct"], r["rwb_pct"])
        r["filters"] = {"every thin pool": True, "sz picks (rw50)": fl["thin_rw50"], "band rule": fl["thin_band"],
                        "_band_ranked": r["rwb_pct"] is not None}
        sol_usd = r["liq_usd"] / (2 * r["sol_at_decide"]) if r["sol_at_decide"] else None
        r["r_in"] = (r["fill_sol"] or 0) * sol_usd if sol_usd else 0.0
        r["r_out"] = (r["exit_sol"] or 0) * sol_usd if sol_usd else 0.0
        r["gross"] = (r["exit_px"] / r["fill_px"] - 1) if r["exit_px"] and r["fill_px"] else r["ret"] + COST
        out.append(r)
    return out


def sized_ret(r: dict, size: float, gap: float) -> float:
    """The trade's return at this size: rugs -100%; else the ideal move kept through our own impact, minus costs."""
    from tradingbot.launch_live import _amm_keep
    if r["ret"] <= RUG:
        return -1.0
    keep = _amm_keep(size, r["r_in"], r["gross"], r["r_out"]) if r["r_in"] > 0 else 1.0
    return max(-1.0, min(CAP, (1 + r["gross"]) * keep - 1 - COST - gap))


def size_of(r: dict, usd: float, cap: float) -> float:
    return min(usd, cap * r["liq_usd"])


def calibrate(store, journal, rows: list[dict]) -> dict:
    """Extra gap = launchrw50's real thin trades minus the model at their own size (positive = real is worse)."""
    by_mint = {}
    for r in rows:
        by_mint.setdefault(r["token_mint"], []).append(r)
    diffs = []
    if journal is not None:
        for p in journal.query("SELECT token, opened_at, cost_usd, realized_usd FROM positions WHERE strategy=? "
                               "AND closed_at IS NOT NULL", (GAP_BOOK,)):
            cands = [r for r in by_mint.get(p["token"], []) if r["create_ts"] <= p["opened_at"]]
            if not cands or not p["cost_usd"] or cands[-1]["ret"] <= RUG:
                continue
            real = p["realized_usd"] / p["cost_usd"] - 1
            diffs.append(sized_ret(cands[-1], p["cost_usd"], 0.0) - real)
    if len(diffs) >= GAP_MIN:
        raw = float(np.mean(diffs))
        gap = float(np.clip(raw, 0.0, 0.10))
        return {"gap": gap, "raw": raw, "median": float(np.median(diffs)), "n": len(diffs),
                "source": f"{GAP_BOOK} real thin trades vs the model at their size"}
    return {"gap": DEFAULT_GAP, "n": len(diffs), "source": f"default (fewer than {GAP_MIN} {GAP_BOOK} thin trades)"}


def _peak_open(trades: list[tuple]) -> float:
    ev = sorted([(t, s) for t, _, s in trades] + [(t + 360 - 6.5, -s) for t, _, s in trades])
    cur = peak = 0.0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def _stats(trades: list[tuple], capital: float) -> dict:
    """trades = [(fill_ts, pnl, size)]."""
    if not trades:
        return {"n": 0}
    pnl = np.array([p for _, p, _ in trades])
    size = np.array([s for _, _, s in trades])
    days = collections.defaultdict(float)
    for t, p, _ in trades:
        days[int(t // 86400)] += p
    d = np.array([days[k] for k in sorted(days)])
    cum = np.cumsum(pnl)
    dd = float(np.min(cum - np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:])) if len(cum) else 0.0
    span = max(1.0, (trades[-1][0] - trades[0][0]) / 86400)
    sharpe = float(d.mean() / d.std() * math.sqrt(365)) if len(d) > 1 and d.std() > 0 else None
    return {"n": len(trades), "traded_usd": round(float(size.sum())), "pnl_usd": round(float(pnl.sum())),
            "per_day_usd": round(float(pnl.sum()) / span), "sized_pct": round(100 * pnl.sum() / size.sum(), 2),
            "mean_size_usd": round(float(size.mean()), 1), "win_pct": round(100 * float(np.mean(pnl > 0)), 1),
            "rugged_pct": round(100 * float(np.mean(pnl <= -0.99 * size)), 1),
            "sharpe_daily": round(sharpe, 2) if sharpe is not None else None,
            "max_dd_usd": round(dd), "max_dd_pct_capital": round(100 * dd / capital, 1),
            "worst_day_usd": round(float(d.min())), "best_day_usd": round(float(d.max())), "days": len(d),
            "peak_open_usd": round(_peak_open(trades)), "_daily": d}


def _bootstrap(daily: np.ndarray, capital: float, rng) -> dict:
    if len(daily) < 2:
        return {}
    draws = rng.choice(daily, size=(SIMS, DAYS), replace=True)
    paths = np.cumsum(draws, axis=1)
    final = paths[:, -1]
    # running peak along each path (axis=1); the default axis=0 ran across the paths and made P(DD>30%) 100%
    dd = np.min(paths - np.maximum.accumulate(np.concatenate([np.zeros((SIMS, 1)), paths], axis=1), axis=1)[:, 1:],
                axis=1)
    return {"median_year_usd": round(float(np.median(final))), "p10_year_usd": round(float(np.percentile(final, 10))),
            "p90_year_usd": round(float(np.percentile(final, 90))),
            "p_loss_pct": round(100 * float(np.mean(final < 0)), 1),
            "p_dd_over_30pct": round(100 * float(np.mean(dd < -0.3 * capital)), 1),
            "p_below_half_pct": round(100 * float(np.mean(np.min(paths, axis=1) < -0.5 * capital)), 1),
            "dd_p90_usd": round(float(np.percentile(dd, 10)))}


def _walk(per_block: dict[str, dict[float, float]]) -> dict:
    """Each block's sizing = the one with the most $ over the trailing LOOKBACK_H (BASE until there is history)."""
    blocks = sorted({b for v in per_block.values() for b in v})
    total, picks, wins = 0.0, collections.Counter(), 0
    for b in blocks:
        past = [x for x in blocks if b - LOOKBACK_H * 3600 <= x < b]
        if past:
            name = max(per_block, key=lambda k: (sum(per_block[k].get(x, 0.0) for x in past), k == BASE))
        else:
            name = BASE
        picks[name] += 1
        got = per_block[name].get(b, 0.0)
        total += got
        wins += got > per_block[BASE].get(b, 0.0)
    return {"blocks": len(blocks), "pnl_usd": round(total), "base_pnl_usd": round(sum(per_block[BASE].values())),
            "blocks_beat_base": wins, "picks": dict(picks.most_common())}


def study(store, journal=None, hours: float | None = None, capital: float = CAPITAL, now: float | None = None,
          seed: int = 7) -> dict:
    now = now if now is not None else time.time()
    rows = load(store, now - hours * 3600 if hours else 0.0)
    out = {"pools": len(rows), "capital_usd": capital, "block_h": BLOCK_H, "lookback_h": LOOKBACK_H}
    if not rows:
        return out
    cal = calibrate(store, journal, rows)
    out["gap"] = cal
    out["since"], out["until"] = rows[0]["create_ts"], rows[-1]["create_ts"]
    out["band_ranked_pools"] = sum(r["filters"]["_band_ranked"] for r in rows)
    rng = np.random.default_rng(seed)
    out["filters"] = {}
    for f in FILTERS:
        sel = [r for r in rows if r["filters"][f]]
        res, per_block = {}, {}
        for name, usd, cap in SIZINGS:
            trades = []
            for r in sel:
                s = size_of(r, usd, cap)
                trades.append((r["create_ts"] + 6.5, s * sized_ret(r, s, cal["gap"]), s))
            st = _stats(trades, capital)
            blk = collections.defaultdict(float)
            for t, p, _ in trades:
                blk[t // (BLOCK_H * 3600) * BLOCK_H * 3600] += p
            per_block[name] = dict(blk)
            st["bootstrap"] = _bootstrap(st.pop("_daily", np.array([])), capital, rng)
            res[name] = st
        base = per_block[BASE]
        for name in res:
            res[name]["blocks_beat_base"] = sum(per_block[name].get(b, 0.0) > v for b, v in base.items())
            res[name]["blocks"] = len(base)
        out["filters"][f] = {"pools": len(sel), "sizings": res, "walk_forward": _walk(per_block),
                             "ideal_mean_pct": round(100 * float(np.mean([r["ret"] for r in sel])), 1) if sel else None}
    return out


def render(out: dict) -> str:
    from datetime import datetime, timezone
    if not out.get("pools"):
        return "THIN-BAND SIZING: no finished thin pools yet"
    when = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M")
    g = out["gap"]
    lines = [f"THIN-BAND SIZING  (liquid pools under $15k at second 4, {when(out['since'])} .. {when(out['until'])} UTC, "
             f"{out['pools']:,} pools; capital ${out['capital_usd']:,.0f}, no compounding)",
             f"extra execution gap {100 * g['gap']:.1f} pts per trade ({g['source']}, n={g['n']}"
             + (f"; measured mean {100 * g['raw']:+.1f}, median {100 * g['median']:+.1f}, floored at 0"
                if "raw" in g else "") + "); own price impact "
             "from the pool reserves at the fill and at minute 6; 6% cost, +300% cap, rugs -100%",
             f"band rule needs the band rank (rwb_pct), recorded since 11-08: {out['band_ranked_pools']:,} of the pools"]
    for f, v in out["filters"].items():
        lines += ["", f"{f.upper()}  ({v['pools']:,} pools, ideal mean {v['ideal_mean_pct']:+.1f}%)" if v["pools"]
                  else f"{f.upper()}  (no pools)"]
        if not v["pools"]:
            continue
        lines.append(f"  {'sizing':<17}{'avg $':>7}{'traded $':>10}{'P&L $':>9}{'$/day':>8}{'sized':>8}{'win':>5}"
                     f"{'rug':>5}{'Sharpe':>7}{'max DD':>9}{'worst day':>10}{'peak open':>10}{'beat $100':>10}"
                     f" | {'year p10':>11}{'median':>11}{'p90':>11}{'P(loss)':>8}{'P(DD>30%)':>10}")
        for name, s in v["sizings"].items():
            b = s.get("bootstrap") or {}
            sh = "-" if s.get("sharpe_daily") is None else f"{s['sharpe_daily']:.1f}"
            lines.append(f"  {name:<17}{s['mean_size_usd']:>7,.0f}{s['traded_usd']:>10,}{s['pnl_usd']:>9,}"
                         f"{s['per_day_usd']:>8,}{s['sized_pct']:>7.1f}%{s['win_pct']:>5.0f}{s['rugged_pct']:>5.0f}"
                         f"{sh:>7}{s['max_dd_usd']:>9,}{s['worst_day_usd']:>10,}{s['peak_open_usd']:>10,}"
                         f"{s['blocks_beat_base']:>6}/{s['blocks']:<3} | {b.get('p10_year_usd', 0):>11,}"
                         f"{b.get('median_year_usd', 0):>11,}{b.get('p90_year_usd', 0):>11,}{b.get('p_loss_pct', 0):>7.0f}%"
                         f"{b.get('p_dd_over_30pct', 0):>9.0f}%")
        w = v["walk_forward"]
        lines.append(f"  walk-forward (each {out['block_h']} h block sized by the best $ over the past "
                     f"{out['lookback_h']} h): ${w['pnl_usd']:,} vs $100 ${w['base_pnl_usd']:,}, beat $100 in "
                     f"{w['blocks_beat_base']} of {w['blocks']} blocks; sizes used "
                     + ", ".join(f"{k} x{n}" for k, n in w["picks"].items()))
    lines += ["", "avg $ = mean trade size; sized = P&L / money traded; beat $100 = 6 h blocks with more $ than the $100 "
              "sizing; year = 365 days drawn with replacement from the observed days (fixed capital, no compounding); "
              "DD = drawdown in $. Large sizes ignore other traders reacting to us, so read them as an upper bound."]
    return "\n".join(lines)
