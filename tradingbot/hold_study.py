"""How long to hold, by pool size (`hold-study`, 11-08, read only).

The launch books all sell at pool minute 6. That exit was chosen on the whole pool population, which is mostly
small pools; bigger pools may rug on a different clock or keep climbing longer. For a random sample of the live
scorer's finished liquid pools in each liquidity band (liquidity at second 4), the pool's state after its last
transaction by each hold (`HOLDS_S`, one Helius `last_before` call each, cached in `watch_launch_path` with the
exit and late-entry studies) gives the return of selling then: vault price vs the scorer's fill price at second ~6,
6% round trip, capped +300%; -100% once the LP was pulled or the pool's SOL fell below 10% of the highest seen so
far (pulls after minute 6 only show up this way, which is fine: a pull empties the SOL side). Minute 6 itself is
the scorer's own result, the books' ideal trade.

Per band and hold: every sampled pool and the rug-vs-win kept half (live `rw_pct` < 50), with mean, median, win,
rugged, +50% wins, a 90% interval, and the paired gap against minute 6 on the same pools (interval included), so
the read is "does holding H beat minute 6 in this band", not two separate averages."""
from __future__ import annotations
import collections
import hashlib
import time
from datetime import datetime, timezone

import numpy as np

from tradingbot import launch_analysis as la
from tradingbot import launch_live as ll
from tradingbot.data import pumpswap as ps

HOLDS_S = (60, 120, 180, 240, 300, 360, 600, 900, 1800, 3600)
SIX = 360
BANDS = (("$5-15k", 5_000, 15_000), ("$15-25k", 15_000, 25_000), ("$25-40k", 25_000, 40_000),
         ("$40-80k", 40_000, 80_000), ("$80-150k", 80_000, 150_000), ("$150k+", 150_000, float("inf")))
PER_BAND = 150
MIN_LIQ = 25_000   # the user's question is about the bigger pools; --min-liq 0 adds the small bands back
CREDITS_PER_CALL = 10


def band_of(liq) -> str | None:
    for name, lo, hi in BANDS:
        if liq is not None and lo <= liq < hi:
            return name
    return None


def sample(rows: list[dict], per_band: int, min_liq: float = 0.0) -> dict:
    """A stable random sample (hash of the pool) of up to `per_band` pools in each band."""
    by: dict = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: hashlib.sha1(f"hold:{r['pool']}".encode()).hexdigest()):
        b = band_of(r["liq_usd"]) if r["liq_usd"] >= min_liq else None
        if b and len(by[b]) < per_band:
            by[b].append(r)
    return by


CAUSES = ("LP pulled", "SOL drained", "pool emptied", "no balance found", "rugged by minute 6")


def path_detail(r: dict, have: dict) -> dict:
    """(return, rug cause) at each hold; (None, None) = state not fetched. Peak only from states up to that hold
    (no look-ahead). Causes: LP pulled (the scorer saw the LP supply hit 0 by then), SOL drained (below 10% of the
    peak so far), pool emptied (a vault read 0), no balance found (the pool's last tx by then didn't list its vaults:
    possibly a data gap, not a rug), rugged by minute 6 (the scorer's own result). A death carries its cause on."""
    out, peak, dead = {}, r["fill_sol"] or 0.0, None
    for sec in HOLDS_S:
        if sec == SIX:
            peak = max(peak, r["peak_sol"] or 0.0)
            if r["ret"] is not None and r["ret"] <= -0.999:
                dead = dead or ("LP pulled" if r["pulled_ts"] and r["pulled_ts"] <= r["create_ts"] + SIX
                                else "rugged by minute 6")
                out[sec] = (-1.0, dead)
            else:
                out[sec] = (r["ret"], None)
            continue
        if dead:
            out[sec] = (-1.0, dead)
            continue
        if r["pulled_ts"] and r["pulled_ts"] <= r["create_ts"] + sec:
            dead = "LP pulled"
            out[sec] = (-1.0, dead)
            continue
        st = have.get((r["pool"], sec))
        if st is None:
            out[sec] = (None, None)
            continue
        sol, tok = st["sol"], st["tok"]
        if sol is None or tok is None:
            dead = "no balance found"
        elif not sol or not tok:
            dead = "pool emptied"
        elif peak > 0 and sol < la.PULLED_SHARE * peak:
            dead = "SOL drained"
        if dead:
            out[sec] = (-1.0, dead)
            continue
        peak = max(peak, sol)
        out[sec] = (min((sol / tok) / r["fill_px"] - 1 - la.COST, 3.0), None)
    return out


def path(r: dict, have: dict) -> dict:
    """Return at each hold (None = state not fetched)."""
    return {sec: v[0] for sec, v in path_detail(r, have).items()}


def _needs(r: dict, have: dict) -> list[int]:
    """Holds still to fetch for this pool, stopping at the first known death (the rest are -100%)."""
    p = path(r, have)
    need = []
    for sec in HOLDS_S:
        if p[sec] is not None and p[sec] <= -0.999:
            break
        if p[sec] is None:
            need.append(sec)
    return need


def _kept(r: dict) -> bool:
    return r["rw_pct"] is not None and r["rw_pct"] < 50


def study(store, helius, hours: float | None = None, per_band: int = PER_BAND, max_credits: int = 40_000,
          now: float | None = None, seed: int = 7, min_liq: float = MIN_LIQ, max_liq: float = float("inf"),
          kept_only: bool = False) -> dict:
    """`kept_only`: sample only the rug-vs-win kept half (live `rw_pct` < 50), so a band's kept pools aren't
    limited to the ~15% a random sample happens to catch; `max_liq` drops bands above it."""
    ll.ensure_schema(store)
    store.execute(ll.PATH_SCHEMA)
    now = now if now is not None else time.time()
    since = now - hours * 3600 if hours else 0.0
    rows = [dict(r) for r in store.query(
        "SELECT pool, token_mint, create_ts, liq_usd, fill_px, fill_sol, peak_sol, ret, pulled_ts, rw_pct "
        "FROM watch_launch_live WHERE status='done' AND liquid=1 AND ret IS NOT NULL AND fill_px IS NOT NULL "
        "AND fill_sol IS NOT NULL AND liq_usd IS NOT NULL AND create_ts >= ? AND create_ts <= ? ORDER BY create_ts",
        (since, now - HOLDS_S[-1] - 120))]
    rows = [r for r in rows if min_liq <= r["liq_usd"] < max_liq and (_kept(r) or not kept_only)]
    if not rows:
        return {"pools": 0, "note": "no finished liquid pools at least an hour old"}
    groups = sample(rows, per_band, min_liq)
    have = {(r["pool"], r["sec"]): {"sol": r["sol"], "tok": r["tok"]}
            for r in store.query("SELECT pool, sec, sol, tok FROM watch_launch_path")}
    stats: collections.Counter = collections.Counter()
    start = helius.credits_used
    # round-robin over the bands, a whole pool at a time, so a credit cap leaves every band about as full
    queues = {b: list(g) for b, g in groups.items()}
    capped = False
    while not capped and any(queues.values()):
        for b in [name for name, _, _ in BANDS]:
            if capped or not queues.get(b):
                continue
            r = queues[b].pop(0)
            for sec in _needs(r, have):
                if helius.credits_used - start + CREDITS_PER_CALL > max_credits:
                    capped = True
                    break
                tx = helius.last_before(r["pool"], r["create_ts"] + sec)
                if tx is False:
                    stats["call failed"] += 1
                    break
                st = {"sol": ll._raw_vault(tx, r["pool"], ps.WSOL) if tx else None,
                      "tok": ll._raw_vault(tx, r["pool"], r["token_mint"]) if tx else None}
                store.execute("INSERT OR REPLACE INTO watch_launch_path(pool, sec, sol, tok, ts) VALUES (?,?,?,?,?)",
                              (r["pool"], sec, st["sol"], st["tok"], time.time()))
                have[(r["pool"], sec)] = st
                stats["fetched"] += 1
                if not st["sol"] or not st["tok"]:
                    break  # emptied: later holds are -100% without a call
    if capped:
        stats["stopped at the credit cap (rerun to continue; nothing is fetched twice)"] = 1
    stats["credits_used"] = helius.credits_used - start
    rng = np.random.default_rng(seed)
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M")
    out = {"pools": len(rows), "window": f"{fmt(rows[0]['create_ts'])} .. {fmt(rows[-1]['create_ts'])} UTC",
           "per_band": per_band, "min_liq": min_liq,
           "max_liq": None if max_liq == float("inf") else max_liq, "kept_only": kept_only, "fetch": dict(stats), "bands": {}}
    for name, _, _ in BANDS:
        g = groups.get(name, [])
        if not g:
            continue
        details = [(r, path_detail(r, have)) for r in g]
        band = {}
        views = ((("rug-vs-win kept (rw_pct < 50)", _kept),) if kept_only else
                 (("every pool", lambda r: True), ("rug-vs-win kept (rw_pct < 50)", _kept)))
        for label, keep in views:
            ds = [(r, d) for r, d in details if keep(r)]
            ps_ = [(r, {sec: v[0] for sec, v in d.items()}) for r, d in ds]
            holds, causes = {}, {}
            for sec in HOLDS_S:
                why = collections.Counter(d[sec][1] for _, d in ds if d[sec][0] is not None and d[sec][1])
                if why:
                    causes[_label(sec)] = {c: why[c] for c in CAUSES if why[c]}
                both = [(p[sec], p[SIX]) for _, p in ps_ if p[sec] is not None and p[SIX] is not None]
                g_ = ll._group([{"ret": a} for a, _ in both], rng)
                if both:
                    g_["big_win_pct"] = round(100 * sum(a >= 0.5 for a, _ in both) / len(both))
                if sec != SIX and len(both) >= 5:
                    d = np.array([a - b for a, b in both])
                    boots = [float(rng.choice(d, len(d)).mean()) for _ in range(2000)]
                    g_["vs_minute6_pts"] = round(100 * float(d.mean()), 1)
                    g_["vs_minute6_ci90"] = [round(100 * float(np.quantile(boots, q)), 1) for q in (0.05, 0.95)]
                holds[_label(sec)] = g_
            band[label] = {"n": len(ps_), "holds": holds, "rug causes": causes}
        out["bands"][name] = band
    out["note"] = ("ideal pool-level trades from the scorer's fill (second ~6): sell price = pool balances after its "
                   "last tx by the hold, 6% round trip, +300% cap, -100% once pulled or the SOL side fell below 10% "
                   "of its peak so far; minute 6 = the scorer's own result; 'vs minute 6' = paired gap on the same "
                   "pools; pools whose state at a hold wasn't fetched are left out of that hold only; rug causes: "
                   "'no balance found' means the pool's last tx by then didn't list its vaults (counted -100%, but "
                   "possibly a data gap rather than a rug)")
    return out


def _label(sec: int) -> str:
    return f"{sec // 60} min" + (" (books)" if sec == SIX else "")


def render(out: dict) -> str:
    if "bands" not in out:
        return str(out)
    rng_ = f"${out['min_liq']:,.0f}+" if not out.get("max_liq") else f"${out['min_liq']:,.0f}-{out['max_liq']:,.0f}"
    L = [f"HOLD TIMES BY POOL SIZE ({out['window']}: {out['pools']:,} finished pools at {rng_} at second 4"
         f"{', rug-vs-win kept only' if out.get('kept_only') else ''}; up to {out['per_band']} sampled per band)",
         "  fetch: " + ", ".join(f"{k} {v}" for k, v in out["fetch"].items()), ""]
    for band, groups in out["bands"].items():
        for label, g in groups.items():
            L.append(f"{band} - {label} (n={g['n']})")
            L.append(f"  {'hold':12}{'n':>5}{'mean':>8}{'90% CI':>16}{'median':>8}{'win':>6}{'rugged':>8}{'+50%':>6}"
                     f"{'vs min 6':>10}{'90% CI':>16}")
            for h, s in g["holds"].items():
                if not s.get("n"):
                    continue
                ci = s.get("ci90"); vci = s.get("vs_minute6_ci90")
                L.append(f"  {h:12}{s['n']:>5}{s['mean_pct']:>+7.1f}%"
                         f"{(f'{ci[0]:+.1f}..{ci[1]:+.1f}' if ci else '-'):>16}{s['median_pct']:>+7.1f}%"
                         f"{s['win_pct']:>5}%{s['rugged_pct']:>7}%{s.get('big_win_pct', 0):>5}%"
                         f"{(format(s['vs_minute6_pts'], '+.1f') if 'vs_minute6_pts' in s else '-'):>10}"
                         f"{(f'{vci[0]:+.1f}..{vci[1]:+.1f}' if vci else '-'):>16}")
            if g.get("rug causes"):
                L.append("  rugs by cause: " + "; ".join(
                    f"{h} " + ", ".join(f"{c} {n}" for c, n in cs.items()) for h, cs in g["rug causes"].items()
                    if h in ("6 min (books)", "10 min", "15 min", "30 min")))
            L.append("")
    L.append(out["note"])
    return "\n".join(L)
