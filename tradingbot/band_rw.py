"""Rug-vs-win trained per pool-size band (11-08 study, read only, no trading effect).

The live rug-vs-win filter ranks every band together and turned out to be mostly a pool-size picker: it keeps 77%
of $5-15k pools and 8-16% of $40k+ ones, and within a band it only clearly helped at $25-40k (`rw-bands`). This
asks whether a filter trained on each band's own pools adds anything inside that band.

Walk-forward in BLOCK_H blocks. For every block and band:
* band model: `launch_live.RugWinModel` (P(rug) - P(+50%)) fitted only on that band's pools created before the
  block (finished: created at least EXIT_S earlier), needs MIN_TRAIN of them with both outcomes present;
* global model: the same model fitted on every band's earlier pools (what the live filter is), ranked within the
  band, so the two differ only in what they were trained on;
* live rule: the deployed `rw_pct` < 50 (global model, global ranking), as a reference.
Skip cut-offs for the two models come from past scores only: the riskiest k% of the band's training pools scored
in the last CUT_WINDOW_H (or all of them when there are fewer than MIN_CUT_POOLS). Each kept half is compared with
every pool in the band, the skipped half, and RANDOM_DRAWS random keeps of the same size from the same pools.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

BLOCK_H = 6
MIN_TRAIN = 150  # a band needs this many finished earlier pools before its model is fitted
CUT_WINDOW_H = 24
MIN_CUT_POOLS = 30
SKIPS = (20, 30, 50, 70)  # % of the band skipped as riskiest
RANDOM_DRAWS = 500
EXIT_S = 360  # a pool is finished (outcome known) at minute 6
# the traded bands; above $150k there are only a handful of pools, so they are pooled into one
STUDY_BANDS = (("$5-15k", 5_000.0, 15_000.0), ("$15-25k", 15_000.0, 25_000.0), ("$25-40k", 25_000.0, 40_000.0),
               ("$40-80k", 40_000.0, 80_000.0), ("$80-150k", 80_000.0, 150_000.0), ("$150k+", 150_000.0, float("inf")))


def _band(liq) -> str | None:
    if liq is None:
        return None
    return next((n for n, lo, hi in STUDY_BANDS if lo <= liq < hi), None)


def _fit(rows):
    """A RugWinModel on `rows` (each with x, ret), or None when an outcome never happens in them."""
    from tradingbot import launch_live as ll
    rug = np.array([float(r["ret"] <= -0.999) for r in rows])
    big = np.array([float(r["ret"] >= ll.RW_BIG) for r in rows])
    if rug.min() == rug.max() or big.min() == big.max():
        return None
    return ll.RugWinModel(ll._fit_capped(rows, rug), ll._fit_capped(rows, big), {})


def _cut(model, train, t, skip):
    """Score at or above which a pool counts among the riskiest `skip`% (from past scores only)."""
    recent = [r for r in train if r["create_ts"] >= t - CUT_WINDOW_H * 3600]
    base = recent if len(recent) >= MIN_CUT_POOLS else train
    return float(np.quantile([model.score(r["x"]) for r in base], 1 - skip / 100))


def _summary(rets) -> dict:
    if not rets:
        return {"n": 0}
    a = np.array(rets)
    return {"n": len(a), "mean_pct": round(100 * float(a.mean()), 1), "median_pct": round(100 * float(np.median(a)), 1),
            "win_pct": round(100 * float((a > 0).mean())), "rugged_pct": round(100 * float((a <= -0.999).mean())),
            "big_pct": round(100 * float((a >= 0.5).mean())), "usd_at_100": round(100 * float(a.sum()))}


def _judge(tested: list[dict], key: str, rng) -> dict:
    """Kept (r[key] is False = not skipped) vs skipped vs every pool vs random keeps of the same size."""
    pool = [r for r in tested if r.get(key) is not None]
    kept = [r["ret"] for r in pool if not r[key]]
    skipped = [r["ret"] for r in pool if r[key]]
    allr = [r["ret"] for r in pool]
    out = {"every pool": _summary(allr), "kept": _summary(kept), "skipped": _summary(skipped)}
    if len(kept) < 5 or len(skipped) < 5:
        out["note"] = "too few pools"
        return out
    a = np.array(allr)
    k = float(np.mean(kept))
    rand = np.array([rng.choice(a, len(kept), replace=False).mean() for _ in range(RANDOM_DRAWS)])
    gaps = [rng.choice(kept, len(kept)).mean() - rng.choice(skipped, len(skipped)).mean() for _ in range(1000)]
    out["kept minus every pool (pts)"] = round(100 * (k - float(a.mean())), 1)
    out["kept minus skipped (pts)"] = {"mean": round(100 * (k - float(np.mean(skipped))), 1),
                                       "ci90": [round(100 * float(np.quantile(gaps, q)), 1) for q in (0.05, 0.95)]}
    out["random keeps worse than kept"] = round(float((rand < k).mean()), 3)
    blocks: dict = {}
    for r in pool:
        blocks.setdefault(r["block"], []).append(r)
    won = [np.mean([x["ret"] for x in g if not x[key]]) > np.mean([x["ret"] for x in g])
           for g in blocks.values() if sum(not x[key] for x in g) >= 3]
    out["blocks where kept beat every pool"] = f"{sum(won)} of {len(won)}"
    return out


def study(store, hours: float | None = None, now: float | None = None, seed: int = 11) -> dict:
    from tradingbot import launch_live as ll
    from tradingbot.watch_analysis import auc
    now = now if now is not None else time.time()
    ll.ensure_schema(store)
    rows = ll._pull_rows(store, now - hours * 3600 if hours else 0.0)
    extra = {r["pool"]: (r["liq_usd"], r["rw_pct"]) for r in store.query(
        "SELECT pool, liq_usd, rw_pct FROM watch_launch_live WHERE status='done' AND liquid=1")}
    for r in rows:
        r["liq"], r["rw_pct"] = extra.get(r["pool"], (None, None))
        r["band"] = _band(r["liq"])
    rows = [r for r in rows if r["band"]]
    if len(rows) < MIN_TRAIN * 2:
        return {"pools": len(rows), "note": f"needs {MIN_TRAIN * 2}+ finished liquid pools"}
    rng = np.random.default_rng(seed)
    block = BLOCK_H * 3600
    first = rows[0]["create_ts"] // block * block
    tested: dict = {n: [] for n, _, _ in STUDY_BANDS}
    t = first + block
    while t <= rows[-1]["create_ts"]:
        test = [r for r in rows if t <= r["create_ts"] < t + block]
        train_all = [r for r in rows if r["create_ts"] < t - EXIT_S]
        glob = _fit(train_all) if len(train_all) >= MIN_TRAIN else None
        for name, _, _ in STUDY_BANDS:
            tb = [r for r in test if r["band"] == name]
            if not tb:
                continue
            train_b = [r for r in train_all if r["band"] == name]
            band = _fit(train_b) if len(train_b) >= MIN_TRAIN else None
            gcuts = {k: _cut(glob, train_b, t, k) for k in SKIPS} if glob and len(train_b) >= MIN_CUT_POOLS else {}
            bcuts = {k: _cut(band, train_b, t, k) for k in SKIPS} if band else {}
            for r in tb:
                rec = {"ret": r["ret"], "block": t, "create_ts": r["create_ts"],
                       "live": None if r["rw_pct"] is None else r["rw_pct"] >= 100 - ll.RW_SKIP}
                if band:
                    rec["band_score"] = band.score(r["x"])
                    for k, c in bcuts.items():
                        rec[f"band{k}"] = rec["band_score"] >= c
                if gcuts:
                    rec["glob_score"] = glob.score(r["x"])
                    for k, c in gcuts.items():
                        rec[f"glob{k}"] = rec["glob_score"] >= c
                tested[name].append(rec)
        t += block

    out_bands = []
    for name, _, _ in STUDY_BANDS:
        tb = tested[name]
        entry: dict = {"band": name, "pools tested": len(tb),
                       "with a band model": sum("band_score" in r for r in tb)}
        both = [r for r in tb if "band_score" in r and "glob_score" in r]
        entry["compared on (both models scored)"] = len(both)
        if len(both) >= 20:
            rets = np.array([r["ret"] for r in both])
            for m in ("band", "glob"):
                sc = np.array([r[f"{m}_score"] for r in both])
                entry[f"AUC {m} model"] = {"rugs (want > 0.5)": _r(auc(sc, (rets <= -0.999).astype(int))),
                                           "big wins (want < 0.5)": _r(auc(sc, (rets >= 0.5).astype(int)))}
            entry["live rule (rw_pct < 50, all bands ranked together)"] = _judge(both, "live", rng)
            for k in SKIPS:
                entry[f"skip riskiest {k}%"] = {"band model": _judge(both, f"band{k}", rng),
                                                "global model ranked in the band": _judge(both, f"glob{k}", rng)}
        else:
            entry["note"] = "too few pools with both models scored"
        out_bands.append(entry)
    return {"window": f"last {hours:g} h" if hours else "whole life",
            "from": datetime.fromtimestamp(rows[0]["create_ts"], timezone.utc).strftime("%m-%d %H:%M"),
            "pools": len(rows), "blocks": int((t - first) // block) - 1, "bands": out_bands,
            "note": (f"walk-forward in {BLOCK_H} h blocks; models fitted on pools created before each block (band "
                     f"model: that band only, needs {MIN_TRAIN}); skip cut-offs from the past {CUT_WINDOW_H} h of the "
                     "band's training scores; ideal trade = the scorer's (second ~6.5 in, minute 6 out, 6% cost, +300% "
                     "cap, rug -100%); compared only on pools both models scored")}


def _r(x):
    return None if x is None else round(x, 3)


def render(out: dict) -> str:
    if "bands" not in out:
        return json.dumps(out, indent=2)
    f = lambda v: "-" if v is None else f"{v:+.1f}"
    lines = [f"RUG-VS-WIN TRAINED PER BAND ({out['window']}, from {out['from']}, {out['pools']} pools, "
             f"{out['blocks']} blocks)", ""]
    for b in out["bands"]:
        lines.append(f"{b['band']}: {b['pools tested']} pools tested, {b['compared on (both models scored)']} with "
                     f"both models")
        if "note" in b:
            lines += [f"  {b['note']}", ""]
            continue
        for m in ("band", "glob"):
            a = b[f"AUC {m} model"]
            lines.append(f"  AUC {'band model  ' if m == 'band' else 'global model'}: rugs {a['rugs (want > 0.5)']}  "
                         f"big wins {a['big wins (want < 0.5)']}")
        every = b["live rule (rw_pct < 50, all bands ranked together)"]["every pool"]
        lines.append(f"  every pool: n {every['n']}  mean {f(every.get('mean_pct'))}%  rug {every.get('rugged_pct')}%  "
                     f"$ at $100 {every.get('usd_at_100')}")
        lines.append(f"  {'rule':<34}{'kept n':>7}{'kept':>8}{'skip':>8}{'vs all':>8}{'gap 90%':>16}{'rand<':>7}"
                     f"{'blocks':>9}{'kept $100':>11}")

        def row(label, j):
            k, s = j["kept"], j["skipped"]
            ci = (j.get("kept minus skipped (pts)") or {}).get("ci90")
            lines.append(f"  {label:<34}{k.get('n', 0):>7}{f(k.get('mean_pct')):>8}{f(s.get('mean_pct')):>8}"
                         f"{f(j.get('kept minus every pool (pts)')):>8}"
                         f"{(f'{ci[0]:+.1f}..{ci[1]:+.1f}' if ci else '-'):>16}"
                         f"{j.get('random keeps worse than kept', '-'):>7}"
                         f"{j.get('blocks where kept beat every pool', '-'):>9}{k.get('usd_at_100', '-'):>11}")
        row("live rule (global, 50%)", b["live rule (rw_pct < 50, all bands ranked together)"])
        for k in SKIPS:
            row(f"band model, skip {k}%", b[f"skip riskiest {k}%"]["band model"])
            row(f"global model in band, skip {k}%", b[f"skip riskiest {k}%"]["global model ranked in the band"])
        lines.append("")
    lines += ["vs all = kept mean - every pool's mean (same pools); gap = kept - skipped with 90% interval; rand< = "
              "share of random keeps of the same size that did worse; blocks = blocks where kept beat every pool",
              out["note"]]
    return "\n".join(lines)
