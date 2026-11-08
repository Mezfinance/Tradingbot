"""What if the sized book took another book's rule in some liquidity bands? (`sz-whatif`, 11-08, read only).

`launchrw50_sz` trades every band with one rule (the rug-vs-win kept half, out at pool minute 6). Two paper books
test other rules in one band each: `launchrw50_25k10` / `launchrw50_25k15` (the same picks in $25-40k pools held to
minute 10 / 15) and `launchbig40k` (the +50% model's best 30% of $40-80k pools, held to minute 15 while up and
rising). Each variant here rebuilds sz from real trades: sz's own trades outside the swapped bands, the other
book's trades inside them, each priced at sz's size for that pool (`model_book.band_usd`), so the variants differ
only in which pools and which exit. Every variant is compared with sz itself over the same window, starting from
the swap books' first buy.

Measures: trades, money traded, $ result, size-weighted mean, results per band, money-minutes held and $ earned per
$1,000 held for an hour (longer holds tie money up longer), and per 6 h block whether the variant beat sz."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

SZ = "launchrw50_sz"
SWAPS = {  # name -> (band from, band to, book whose trades replace sz's in that band)
    "$25-40k held to 10": (25_000.0, 40_000.0, "launchrw50_25k10"),
    "$25-40k held to 15": (25_000.0, 40_000.0, "launchrw50_25k15"),
    "$40-80k +50% filter, held while rising": (40_000.0, 80_000.0, "launchbig40k"),
}
VARIANTS = (
    ("$25-40k held to 10",),
    ("$25-40k held to 15",),
    ("$40-80k +50% filter, held while rising",),
    ("$25-40k held to 10", "$40-80k +50% filter, held while rising"),
    ("$25-40k held to 15", "$40-80k +50% filter, held while rising"),
)
BANDS = (("$5-15k", 5_000.0, 15_000.0), ("$15-25k", 15_000.0, 25_000.0), ("$25-40k", 25_000.0, 40_000.0),
         ("$40-80k", 40_000.0, 80_000.0), ("$80k+", 80_000.0, float("inf")))
STUCK_S = 1800  # open this long = -100% (as band-compare)
BLOCK_H = 6
DRAWS = 2000


def _band(liq) -> str | None:
    if liq is None:
        return None
    return next((n for n, lo, hi in BANDS if lo <= liq < hi), None)


def _trades(journal, book: str, since: float, now: float, scorer: dict) -> list[dict]:
    """Real trades of `book` opened in the window: return, minutes held, the scorer's liquidity for the token."""
    out = []
    for p in journal.query("SELECT token, opened_at, closed_at, cost_usd, realized_usd FROM positions "
                           "WHERE strategy=? AND opened_at >= ? AND opened_at < ?", (book, since, now)):
        if not p["cost_usd"]:
            continue
        if p["closed_at"]:
            ret, held = p["realized_usd"] / p["cost_usd"] - 1, (p["closed_at"] - p["opened_at"]) / 60
        elif now - p["opened_at"] >= STUCK_S:
            ret, held = -1.0, STUCK_S / 60
        else:
            continue  # still running
        rows = [r for r in scorer.get(p["token"], []) if r["create_ts"] <= p["opened_at"]]
        out.append({"book": book, "ret": ret, "held_min": held, "ts": p["opened_at"], "token": p["token"],
                    "cost": p["cost_usd"], "liq": rows[-1]["liq_usd"] if rows else None})
    return out


def _sized(trades: list[dict]) -> list[dict]:
    """Each trade at sz's size for its pool (its own cost when the scorer has no liquidity for it)."""
    from tradingbot import model_book as mb
    for t in trades:
        t["usd"] = mb.band_usd(SZ, t["liq"]) or t["cost"]
    return trades


def _stats(trades: list[dict], rng) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t["ret"] for t in trades])
    u = np.array([t["usd"] for t in trades])
    mm = float(sum(t["usd"] * t["held_min"] for t in trades))
    out = {"n": len(r), "traded_usd": round(float(u.sum())), "pnl_usd": round(float((r * u).sum()), 1),
           "size_weighted_pct": round(100 * float((r * u).sum() / u.sum()), 2),
           "median_pct": round(100 * float(np.median(r)), 1), "win_pct": round(100 * float(np.mean(r > 0))),
           "rugged_pct": round(100 * float(np.mean(r <= -0.8))),
           "money_hours": round(mm / 60), "usd_per_1k_hour": round(1000 * float((r * u).sum()) / (mm / 60), 1)
           if mm else None}
    if len(r) >= 5:
        idx = rng.integers(0, len(r), (DRAWS, len(r)))
        boots = (r[idx] * u[idx]).sum(1) / u[idx].sum(1)
        out["ci90"] = [round(100 * float(np.quantile(boots, q)), 1) for q in (0.05, 0.95)]
    return out


def _by_band(trades: list[dict], rng) -> dict:
    return {n: _stats([t for t in trades if _band(t["liq"]) == n], rng) for n, _, _ in BANDS}


def _blocks(sz: list[dict], alt: list[dict]) -> dict:
    """Per BLOCK_H block: sz's $ result vs the variant's (both at sz's sizes)."""
    size = BLOCK_H * 3600
    by: dict = {}
    for key, ts in (("sz", sz), ("alt", alt)):
        for t in ts:
            by.setdefault(int(t["ts"] // size), {"sz": 0.0, "alt": 0.0})[key] += t["ret"] * t["usd"]
    rows = [{"block": datetime.fromtimestamp(b * size, timezone.utc).strftime("%m-%d %H:%M"),
             "sz_usd": round(v["sz"], 1), "variant_usd": round(v["alt"], 1)} for b, v in sorted(by.items())]
    won = sum(1 for r in rows if r["variant_usd"] > r["sz_usd"])
    lost = sum(1 for r in rows if r["variant_usd"] < r["sz_usd"])
    return {"blocks": rows, "won": won, "lost": lost}


def whatif(store, journal, hours: float | None = None, now: float | None = None) -> dict:
    from tradingbot import launch_live as ll
    now = now if now is not None else time.time()
    ll.ensure_schema(store)
    scorer: dict = {}
    for r in store.query("SELECT token_mint, create_ts, liq_usd FROM watch_launch_live "
                         "WHERE token_mint IS NOT NULL ORDER BY create_ts"):
        scorer.setdefault(r["token_mint"], []).append(dict(r))
    rng = np.random.default_rng(0)
    out = {"variants": {}, "note": (
        "real trades only, each priced at sz's size for its pool ($250 / $25 / $100 below $15k / $15-25k / above); "
        "a swap book's own fills were at its own size ($250), so its impact is a little worse than sz's would be; "
        "open over 30 min = -100%; each variant is compared with sz over the window since its swap books' first buy")}
    for variant in VARIANTS:
        books = [SWAPS[s][2] for s in variant]
        first = journal.query(f"SELECT MIN(opened_at) t, COUNT(*) n FROM positions WHERE strategy IN "
                              f"({','.join('?' * len(books))}) GROUP BY strategy", books)
        name = " + ".join(variant)
        if len(first) < len(books) or not all(r["t"] for r in first):
            out["variants"][name] = {"note": "a swap book has no trades yet: " + ", ".join(books)}
            continue
        since = now - hours * 3600 if hours else max(r["t"] for r in first) - 60
        sz = _sized(_trades(journal, SZ, since, now, scorer))
        swapped = [(SWAPS[s][0], SWAPS[s][1]) for s in variant]
        inside = lambda t: t["liq"] is not None and any(lo <= t["liq"] < hi for lo, hi in swapped)
        alt = [t for t in sz if not inside(t)]
        for s in variant:
            lo, hi, book = SWAPS[s]
            alt += [t for t in _sized(_trades(journal, book, since, now, scorer))
                    if t["liq"] is not None and lo <= t["liq"] < hi]
        alt.sort(key=lambda t: t["ts"])
        a, b = _stats(sz, rng), _stats(alt, rng)
        delta = {}
        if a.get("n") and b.get("n"):
            delta = {"pnl_usd": round(b["pnl_usd"] - a["pnl_usd"], 1),
                     "size_weighted_pts": round(b["size_weighted_pct"] - a["size_weighted_pct"], 2),
                     "usd_per_1k_hour": None if a["usd_per_1k_hour"] is None or b["usd_per_1k_hour"] is None
                     else round(b["usd_per_1k_hour"] - a["usd_per_1k_hour"], 1)}
        out["variants"][name] = {
            "since": datetime.fromtimestamp(since, timezone.utc).strftime("%m-%d %H:%M"),
            "hours": round((now - since) / 3600, 1),
            "sz": a, "variant": b, "variant minus sz": delta,
            "in the swapped bands": {"sz": _stats([t for t in sz if inside(t)], rng),
                                     "variant": _stats([t for t in alt if inside(t)], rng)},
            "sz by band": _by_band(sz, rng), "variant by band": _by_band(alt, rng),
            "blocks": _blocks(sz, alt), "no scorer row (left in sz)": sum(1 for t in sz if t["liq"] is None)}
    return out


def _line(label: str, d: dict) -> str:
    if not d or not d.get("n"):
        return f"  {label:<34} {'-':>5}"
    ci = d.get("ci90")
    return (f"  {label:<34} {d['n']:>5} {d['traded_usd']:>9,} {d['pnl_usd']:>+10,.0f} {d['size_weighted_pct']:>+8.2f} "
            f"{d['median_pct']:>+7.1f} {d['win_pct']:>4}% {d['rugged_pct']:>4}% "
            f"{(d['usd_per_1k_hour'] if d['usd_per_1k_hour'] is not None else float('nan')):>+9.1f}"
            + (f"  [{ci[0]:+.1f}..{ci[1]:+.1f}]" if ci else ""))


HEAD = (f"  {'':<34} {'n':>5} {'traded $':>9} {'$ result':>10} {'mean %':>8} {'median':>7} {'win':>5} {'rug':>5} "
        f"{'$/1k-hr':>9}  90% CI of the mean")


def render(out: dict) -> str:
    lines = ["SZ WHAT-IF: launchrw50_sz with another book's rule in some bands (real trades, priced at sz's sizes)", ""]
    for name, v in out["variants"].items():
        lines.append(f"{name}")
        if "note" in v:
            lines += [f"  {v['note']}", ""]
            continue
        lines += [f"  since {v['since']} UTC ({v['hours']} h)", HEAD, _line("sz as it is", v["sz"]),
                  _line("sz with the swap", v["variant"])]
        d = v["variant minus sz"]
        if d:
            lines.append(f"  variant minus sz: {d['pnl_usd']:+,.0f} $, mean {d['size_weighted_pts']:+.2f} pts, "
                         f"$/1k-hour {d['usd_per_1k_hour'] if d['usd_per_1k_hour'] is not None else '-'}")
        lines += ["  in the swapped bands only:", _line("  sz", v["in the swapped bands"]["sz"]),
                  _line("  swap book", v["in the swapped bands"]["variant"])]
        lines.append("  by band (sz / variant):")
        for n, _, _ in BANDS:
            lines += [_line(f"  {n} sz", v["sz by band"][n]), _line(f"  {n} variant", v["variant by band"][n])]
        bl = v["blocks"]
        lines.append(f"  6 h blocks: variant ahead in {bl['won']}, behind in {bl['lost']}")
        for r in bl["blocks"][-8:]:
            lines.append(f"    {r['block']}  sz {r['sz_usd']:>+8,.0f} $  variant {r['variant_usd']:>+8,.0f} $")
        lines.append("")
    lines.append("note: " + out["note"])
    return "\n".join(lines)
