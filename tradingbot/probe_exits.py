"""Exit-rule study for the 1-slot probe's survivor population (11-12, read only).

The probe lands ~$25 for one slot and sells it back; pools that survive the ~48 s watch
(``canary_probe_verdict`` = ``survivor``, plus feint ``alive`` pools) are the ones we would enter
for real at confirmation (~60 s after the probe).  Live tracking has shown survivors *do* climb
modestly (~+25-50%) and then, often, get pulled late (CF2LVJVy climbed to +25% by minute 6 and was
pulled at minute 7), so a fixed minute-6 hold gives the climb back.  This module quantifies three
exit rules the owner asked to compare, over the survivor population:

1. **Take profit.**  How many survivors climb to each TP level before any pull, how many committed
   probes we send per climber, and the net return per probe (folding in the ~$0.30 probe cost over
   every committed probe).  Lets us set a TP that nets a target %.
2. **Trailing stop.**  Exit when the price falls X% off the running peak since entry.  Good for a
   fading pool, useless against a one-slot rug (it is gone before any stop fires), so a pool pulled
   before the trail triggers is -100%.
3. **Shorter fixed hold.**  Enter at confirmation, exit after T seconds.  The value here is the
   pull-time distribution: the longest window after confirmation in which *no* survivor has rugged
   is a hold we can take with ~zero rug exposure.

Price paths come from the dashboard's persisted reserve samples (``pool_paths.json``: ``pts`` =
[(pool_age_s, liq_usd, price)], ~20 s apart, young pools only) joined to the scorer's
``watch_launch_live`` row (creation, pull time, minute-6 return).  ``--fetch`` refines each survivor
with a fine Helius price series (one ``last_before`` call per offset, research key, cached in
``watch_probe_exit``), mirroring ``hold_study``.  Everything is relative to the confirmation entry
price, so the raw SOL/token reserve ratio is enough and no SOL price / decimals are needed.

Read only; never trades, never the live key.
"""
from __future__ import annotations

import json
import math
import time

COST = 0.06                 # round-trip cost applied to every exit (as pool_tracker / the scorer)
PULLED_SHARE = 0.10         # pool SOL below this share of its peak = pulled
CONF_AFTER_S = 60.0         # confirmation ≈ probe age + the 48 s watch + margin (pool_tracker.CONF_AFTER_S)
PROBE_COST = 0.30           # modelled cost of one committed probe (~fees, per CLAUDE.md 11-12)
ENTRY_USD = 25.0            # size we would enter a confirmed survivor at
RET_CAP = 3.0               # cap a single trade at +300% (as the scorer / hold_study)

TP_GRID = (0.15, 0.20, 0.25, 0.30, 0.50, 0.75, 1.00)
TRAIL_GRID = (0.15, 0.20, 0.30, 0.40)
HOLD_GRID_S = (30, 60, 120, 180, 240, 300, 360, 600)
# finer grid for the pull-time distribution / the "safe window" question
PULL_GRID_S = (15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420, 480, 600, 900, 1800)

SCHEMA = "CREATE TABLE IF NOT EXISTS watch_probe_exit(pool TEXT, sec INTEGER, sol REAL, tok REAL, ts REAL, " \
         "PRIMARY KEY(pool, sec))"


# --- population ------------------------------------------------------------------------------------

def _is_survivor(f: dict) -> bool:
    """A pool we would actually enter: a committed 1-slot probe that survived the watch, or a feint
    that read `alive` (not pulled in reaction)."""
    if f.get("kind") == "probe":
        return f.get("verdict") in ("ok", "trapped") and f.get("watch_verdict") == "survivor"
    return f.get("verdict") == "alive"


def _committed(f: dict) -> bool:
    """A probe that actually spent money (so it cost ~PROBE_COST): a landed 1-slot probe or a feint
    that was placed (not a honeypot / skipped / inconclusive)."""
    if f.get("kind") == "probe":
        return f.get("verdict") in ("ok", "trapped")
    return f.get("verdict") not in ("inconclusive", "honeypot", "skipped", None)


def _entry_series(f: dict, s: dict, path: dict | None, fetched: list | None) -> dict | None:
    """Per-survivor entry price and the post-confirmation price series, relative to entry.

    Returns {conf_ts, pull_after_conf (s or None), series=[(t_after_conf, mult)], peak_mult, m6_ret}.
    ``mult`` = price / entry price.  Prefers the fine fetched series; else the dashboard pts path.
    """
    created = s.get("create_ts")
    feint_ts = f.get("feint_ts")
    if not created or not feint_ts:
        return None
    conf_ts = feint_ts + CONF_AFTER_S
    conf_age = conf_ts - created
    pulled_ts = (path or {}).get("pulled_ts") or s.get("pulled_ts")
    pull_after_conf = (pulled_ts - conf_ts) if pulled_ts else None

    pts: list[tuple[float, float]] = []  # (t_after_conf, raw price)
    if fetched:  # fine Helius series: (sec_after_conf, sol, tok)
        peak_sol = max((sol for _, sol, _ in fetched if sol), default=0.0)
        for sec, sol, tok in fetched:
            if not tok or not sol:
                continue
            if peak_sol and sol < PULLED_SHARE * peak_sol:  # pulled from here on
                break
            pts.append((float(sec), sol / tok))
    elif path and path.get("pts"):
        peak_liq = path.get("peak_liq") or 0.0
        floor = PULLED_SHARE * peak_liq
        for age, liq, pr in path["pts"]:
            if pr is None or age < conf_age:
                continue
            if floor and liq is not None and liq < floor:
                break
            pts.append((age - conf_age, float(pr)))
    if not pts:
        return None
    entry_px = pts[0][1]
    if entry_px <= 0:
        return None
    series = [(t, px / entry_px) for t, px in pts if pull_after_conf is None or t <= pull_after_conf]
    if not series:
        return None
    peak_mult = max(m for _, m in series)
    m6 = s.get("ret")
    return {"conf_ts": conf_ts, "pull_after_conf": pull_after_conf, "series": series,
            "peak_mult": peak_mult, "m6_ret": m6}


# --- scenario maths --------------------------------------------------------------------------------

def _clamp(r: float) -> float:
    return max(-1.0, min(r, RET_CAP))


def _tp_ret(e: dict, tp: float) -> float:
    """Take profit at +tp: if the price reached (1+tp) before any pull, exit there; else if it pulled
    before reaching it, -100%; else exit at the last seen price."""
    target = 1.0 + tp
    if e["peak_mult"] >= target:
        return _clamp(tp - COST)
    if e["pull_after_conf"] is not None:  # never hit TP and then pulled
        return -1.0
    return _clamp(e["series"][-1][1] - 1.0 - COST)


def _trail_ret(e: dict, trail: float) -> float:
    """Exit when the price falls `trail` below the running peak since entry; -100% if a pull comes
    first (a one-slot rug is gone before any stop fires)."""
    peak = 1.0
    for _, m in e["series"]:
        peak = max(peak, m)
        if m <= peak * (1.0 - trail):
            return _clamp(m - 1.0 - COST)
    if e["pull_after_conf"] is not None:
        return -1.0
    return _clamp(e["series"][-1][1] - 1.0 - COST)


def _hold_ret(e: dict, t: float) -> float:
    """Enter at confirmation, exit after t seconds (price at the first sample >= t, else the last);
    -100% if the pool pulled by then."""
    if e["pull_after_conf"] is not None and e["pull_after_conf"] <= t:
        return -1.0
    at = [m for (ts, m) in e["series"] if ts <= t]
    m = at[-1] if at else e["series"][0][1]
    return _clamp(m - 1.0 - COST)


def _dist(vals: list[float]) -> dict:
    vals = sorted(vals)
    n = len(vals)
    if not n:
        return {"n": 0}
    mean = sum(vals) / n
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"n": n, "mean": mean, "median": med, "win": sum(1 for v in vals if v > 0) / n,
            "min": vals[0], "max": vals[-1]}


def take_profit(survs: list[dict], committed: int, tp_grid=TP_GRID,
                probe_cost: float = PROBE_COST, entry_usd: float = ENTRY_USD) -> list[dict]:
    """Per TP level: climbers, probes-per-climber (over committed probes), the mean return of an
    entered survivor, and the net $ per committed probe (every committed probe pays `probe_cost`;
    each survivor is entered for `entry_usd`)."""
    out = []
    for tp in tp_grid:
        rets = [_tp_ret(e, tp) for e in survs]
        climbers = sum(1 for e in survs if e["peak_mult"] >= 1.0 + tp)
        surv_pnl = sum(r * entry_usd for r in rets)
        net = surv_pnl - committed * probe_cost
        out.append({"tp_pct": tp, "climbers": climbers,
                    "probes_per_climber": (committed / climbers) if climbers else None,
                    "surv": _dist(rets), "net_per_probe": (net / committed) if committed else None,
                    "net_total": net})
    return out


def trailing(survs: list[dict], trail_grid=TRAIL_GRID) -> list[dict]:
    return [{"trail_pct": tr, "ret": _dist([_trail_ret(e, tr) for e in survs])} for tr in trail_grid]


def fixed_hold(survs: list[dict], hold_grid=HOLD_GRID_S) -> list[dict]:
    return [{"hold_s": t, "ret": _dist([_hold_ret(e, t) for e in survs])} for t in hold_grid]


CLIMB_CHECKS_S = (60, 90, 120)
CLIMB_GAINS = (0.0, 0.05, 0.10, 0.20)
CLIMB_HOLD_S = 360  # ride winners to minute 6 (the scorer's exit)


def _climb_ret(e: dict, check: float, gain: float, hold: float = CLIMB_HOLD_S) -> float:
    """Climb watcher: at `check` s, keep the trade only if it is up >= `gain`, then ride to `hold`; else
    sell at the check.  -100% if pulled before the check (a flat-then-rug like v8AykTHP that pulled at
    minute 3 after sitting still is exited at the check, dodging the rug)."""
    if e["pull_after_conf"] is not None and e["pull_after_conf"] <= check:
        return -1.0
    at_check = [m for (t, m) in e["series"] if t <= check]
    mc = at_check[-1] if at_check else e["series"][0][1]
    if mc - 1.0 >= gain:  # climbing: ride it
        return _hold_ret(e, hold)
    return _clamp(mc - 1.0 - COST)  # flat/falling: bail now (the climb watcher's whole point)


def climb_gate(survs: list[dict], checks=CLIMB_CHECKS_S, gains=CLIMB_GAINS, hold: float = CLIMB_HOLD_S) -> list[dict]:
    out = []
    for c in checks:
        for g in gains:
            out.append({"check_s": c, "gain_pct": g, "ret": _dist([_climb_ret(e, c, g, hold) for e in survs])})
    return out


def pull_distribution(survs: list[dict], grid=PULL_GRID_S) -> dict:
    """Over survivors (pools we would have entered): the share pulled within T seconds of confirmation,
    for each T in the grid.  The largest T with 0 pulls is a hold window with ~no rug exposure."""
    pulls = sorted(e["pull_after_conf"] for e in survs if e["pull_after_conf"] is not None)
    n = len(survs)
    buckets = [{"t_s": t, "pulled_by": sum(1 for p in pulls if p <= t),
                "share": (sum(1 for p in pulls if p <= t) / n) if n else None} for t in grid]
    safe = 0
    for b in buckets:
        if b["pulled_by"] == 0:
            safe = b["t_s"]
        else:
            break
    return {"n_survivors": n, "n_pulled": len(pulls),
            "earliest_pull_s": pulls[0] if pulls else None,
            "median_pull_s": (pulls[len(pulls) // 2] if pulls else None),
            "safe_window_s": safe, "buckets": buckets}


# --- fetch (optional, research key) ----------------------------------------------------------------

def fetch_series(store, helius, survivors_meta: list[dict], max_credits: int, grid=PULL_GRID_S) -> dict:
    """One `last_before` call per (pool, offset) from confirmation, cached in `watch_probe_exit`.
    Returns {pool: [(sec, sol, tok), ...]} (SOL lamports, raw token units).  Research (backup) key only.
    Reads reserves from the tx with the same `_raw_vault` helper the other trajectory studies use."""
    store.execute(SCHEMA)
    have: dict = {}
    for r in store.query("SELECT pool, sec, sol, tok FROM watch_probe_exit"):
        have.setdefault(r["pool"], {})[int(r["sec"])] = (r["sol"], r["tok"])
    if helius is None:
        return {p: sorted([(s, v[0], v[1]) for s, v in d.items() if v[0] and v[1]]) for p, d in have.items()}
    from tradingbot import launch_live as ll
    from tradingbot.data import pumpswap as ps
    start = helius.credits_used
    for m in survivors_meta:
        pool, conf_ts, mint = m["pool"], m["conf_ts"], m.get("token_mint")
        cur = have.setdefault(pool, {})
        for t in grid:
            if t in cur:
                continue
            if helius.credits_used - start + 10 > max_credits:
                break
            tx = helius.last_before(pool, conf_ts + t)
            sol = ll._raw_vault(tx, pool, ps.WSOL) if isinstance(tx, dict) else None
            tok = ll._raw_vault(tx, pool, mint) if (isinstance(tx, dict) and mint) else None
            cur[t] = (sol, tok)
            store.execute("INSERT OR REPLACE INTO watch_probe_exit(pool, sec, sol, tok, ts) VALUES(?,?,?,?,?)",
                          (pool, t, sol, tok, time.time()))
            if not sol or not tok:
                break  # emptied / unreadable: later offsets are -100% without spending a call
    return {p: sorted([(s, v[0], v[1]) for s, v in d.items() if v[0] and v[1]]) for p, d in have.items()}


# --- study -----------------------------------------------------------------------------------------

def _load_paths(cfg) -> dict:
    import os
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(cfg.db_path)), "pool_paths.json")
        with open(p) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 (no file yet / unreadable)
        return {}


def study(journal, store, cfg=None, hours: float = 24.0, helius=None, max_credits: int = 8_000,
          paths: dict | None = None, now: float | None = None) -> dict:
    from tradingbot import pool_tracker as pt
    now = now if now is not None else time.time()
    since = now - hours * 3600
    feints = pt.feint_pools(journal.query, since)
    scorer = pt.scorer_rows(store.query, since)
    paths = paths if paths is not None else (_load_paths(cfg) if cfg is not None else {})

    all_probed = list(feints.values())
    committed = sum(1 for f in all_probed if _committed(f))
    surv_pools = [f for f in all_probed if _is_survivor(f) and scorer.get(f["pool"])]

    fetched = {}
    if helius is not None:
        meta = [{"pool": f["pool"], "conf_ts": f["feint_ts"] + CONF_AFTER_S,
                 "token_mint": scorer[f["pool"]].get("token_mint")} for f in surv_pools]
        fetched = fetch_series(store, helius, meta, max_credits)

    survs, with_path = [], []
    for f in surv_pools:
        e = _entry_series(f, scorer[f["pool"]], paths.get(f["pool"]), fetched.get(f["pool"]))
        rec = {"pool": f["pool"], "mint": f.get("mint"),
               "conf_ts": f["feint_ts"] + CONF_AFTER_S,
               "pull_after_conf": None if e is None else e["pull_after_conf"],
               "m6_ret": scorer[f["pool"]].get("ret")}
        if e is not None:
            rec.update(peak_mult=e["peak_mult"], series_n=len(e["series"]))
            with_path.append(e)
            rec["pull_after_conf"] = e["pull_after_conf"]
        # for the pull distribution, fall back to the scorer's pull time when there is no path
        if rec["pull_after_conf"] is None and scorer[f["pool"]].get("pulled_ts"):
            rec["pull_after_conf"] = scorer[f["pool"]]["pulled_ts"] - rec["conf_ts"]
        survs.append(rec)

    return {"hours": hours, "probes_total": len(all_probed), "probes_committed": committed,
            "survivors": len(surv_pools), "survivors_with_path": len(with_path),
            "pull_distribution": pull_distribution(
                [{"pull_after_conf": r["pull_after_conf"]} for r in survs]),
            "take_profit": take_profit(with_path, committed),
            "trailing": trailing(with_path),
            "fixed_hold": fixed_hold(with_path),
            "climb_gate": climb_gate(with_path),
            "rows": survs}


# --- render --------------------------------------------------------------------------------------

def _pct(v, dp=0) -> str:
    return "—" if v is None else f"{v * 100:+.{dp}f}%"


def render(r: dict) -> str:
    o = [f"PROBE EXITS — survivor exit rules (last {r['hours']:.0f} h)",
         f"{r['probes_total']} probes ({r['probes_committed']} committed, cost ${PROBE_COST:.2f} each); "
         f"{r['survivors']} survivors ({r['survivors_with_path']} with a price path)"]
    pd = r["pull_distribution"]
    o.append("")
    o.append(f"SHORTER HOLD — pulls after confirmation ({pd['n_pulled']}/{pd['n_survivors']} survivors pulled; "
             f"earliest {pd['earliest_pull_s'] or '—'}s, median {pd['median_pull_s'] or '—'}s)")
    o.append(f"  safe window (no survivor pulled within): {pd['safe_window_s']}s")
    o.append("  by T:    " + "  ".join(f"{b['t_s']}s:{_pct(b['share'], 0)}" for b in pd["buckets"] if b["pulled_by"]))
    if r["survivors_with_path"] == 0:
        o.append("\n(no survivor has a reserve path yet — run with --fetch, or let the dashboard watch more pools)")
        return "\n".join(o)

    o.append("\nTAKE PROFIT (enter every survivor at confirmation; -100% if pulled before the TP)")
    o.append(f"  {'TP':>5} {'climbers':>8} {'probes/climber':>14} {'mean ret':>9} {'win':>6} {'net/probe':>10}")
    for t in r["take_profit"]:
        s = t["surv"]
        o.append(f"  {t['tp_pct'] * 100:>4.0f}% {t['climbers']:>8} "
                 f"{(f'{t['probes_per_climber']:.1f}' if t['probes_per_climber'] else '—'):>14} "
                 f"{_pct(s.get('mean')):>9} {_pct(s.get('win'), 0):>6} "
                 f"{(f'${t['net_per_probe']:+.2f}' if t['net_per_probe'] is not None else '—'):>10}")

    o.append("\nTRAILING STOP (off the running peak; -100% if pulled before it fires)")
    o.append(f"  {'trail':>6} {'mean':>8} {'median':>8} {'win':>6} {'n':>4}")
    for t in r["trailing"]:
        s = t["ret"]
        o.append(f"  {t['trail_pct'] * 100:>5.0f}% {_pct(s.get('mean')):>8} {_pct(s.get('median')):>8} "
                 f"{_pct(s.get('win'), 0):>6} {s['n']:>4}")

    o.append("\nSHORTER FIXED HOLD (enter at confirmation, exit after T; -100% if pulled by then)")
    o.append(f"  {'hold':>6} {'mean':>8} {'median':>8} {'win':>6} {'n':>4}")
    for t in r["fixed_hold"]:
        s = t["ret"]
        o.append(f"  {t['hold_s']:>5}s {_pct(s.get('mean')):>8} {_pct(s.get('median')):>8} "
                 f"{_pct(s.get('win'), 0):>6} {s['n']:>4}")

    o.append(f"\nCLIMB GATE (at the check, keep only if up >= gain then ride to {CLIMB_HOLD_S // 60}m, else bail; "
             f"-100% if pulled before the check)")
    o.append(f"  {'check':>6} {'gain':>5} {'mean':>8} {'median':>8} {'win':>6} {'n':>4}")
    for t in r["climb_gate"]:
        s = t["ret"]
        o.append(f"  {t['check_s']:>5}s {t['gain_pct'] * 100:>4.0f}% {_pct(s.get('mean')):>8} "
                 f"{_pct(s.get('median')):>8} {_pct(s.get('win'), 0):>6} {s['n']:>4}")
    return "\n".join(o)
