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

Before any exit rule, the owner settled an **entry filter** (2026-11-13): among survivors, enter only
those whose confirmation-window peak-to-trough price range is >= V (``VOL_GRID``, swept) — i.e. the
pool *moved* during the ~60 s we waited, up or down, a dip-and-recover counting (it is measured as a
range, not a net change).  Dead-flat pools (e.g. BykXeroQ, +2% and no run) are skipped; a down-move
before entry is to our benefit (cheaper basis), so the filter must be volatility, not an up-only
climb gate (a pool that dipped and then ran would be thrown away by "must be up to enter").  The
deployed *exit* is the **climb gate** (``climb_gate``: keep only if up by the check, else bail) riding
winners to minute 6 — the one lever that both banks the climbers (CF2LVJVy +30% at m6) and bails the
flat-then-rugs (CUFUqgSE, flat past the 60 s watch then pulled at 2m).  Take-profit and trailing are
kept only as comparisons; on the evidence so far hold-to-minute-6 captures the wins (the givebacks
happen at or after minute 6), so a TP is on probation, not required.

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
# confirmation-window fetch offsets (relative to conf_ts; the probe is at conf - CONF_AFTER_S, so -58
# ≈ 2 s after the probe), for the entry volatility measure
CONF_FETCH_OFFSETS = (-58, -44, -30, -16, 0)
VOL_GRID = (0.015, 0.02, 0.03, 0.05)   # confirmation-window peak-to-trough thresholds to sweep
DEFAULT_VOL = 0.02                      # the entry filter we run the deployed climb gate on top of
DUMP_STEP_S = 5                         # dense grid step for a single-pool --pool path dump
DUMP_AFTER_CONF_S = 420                 # dump through ~minute 6 after confirmation, plus margin to catch a late pull

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
    """Per-survivor entry price, confirmation-window volatility, and the post-confirmation series.

    Returns {conf_ts, pull_after_conf (s or None), series=[(t_after_conf, mult)], peak_mult, m6_ret,
    conf_range}.  ``mult`` = price / entry price (entry = price at confirmation).  ``conf_range`` =
    peak-to-trough fractional range over the confirmation window [probe, confirmation] (the entry
    filter's measure; None when fewer than two samples land in the window).  Prefers the fine fetched
    series (``sec`` <= 0 = confirmation-window samples relative to conf_ts, ``sec`` > 0 = after conf);
    else the dashboard pts path.
    """
    created = s.get("create_ts")
    feint_ts = f.get("feint_ts")
    if not created or not feint_ts:
        return None
    conf_ts = feint_ts + CONF_AFTER_S
    conf_age = conf_ts - created
    probe_age = feint_ts - created
    pulled_ts = (path or {}).get("pulled_ts") or s.get("pulled_ts")
    pull_after_conf = (pulled_ts - conf_ts) if pulled_ts else None

    conf_prices: list[float] = []        # prices within the confirmation window [probe, conf]
    post_pts: list[tuple[float, float]] = []  # (t_after_conf, raw price), t >= 0
    entry_px: float | None = None
    if fetched:  # (sec, sol, tok); sec <= 0 = conf window (relative to conf_ts), sec > 0 = after conf
        peak_sol = max((sol for sec, sol, _ in fetched if sec > 0 and sol), default=0.0)
        for sec, sol, tok in fetched:
            if not tok or not sol:
                continue
            px = sol / tok
            if sec <= 0:
                conf_prices.append(px)
                if sec == 0:
                    entry_px = px
            else:
                if peak_sol and sol < PULLED_SHARE * peak_sol:  # pulled from here on
                    break
                post_pts.append((float(sec), px))
    elif path and path.get("pts"):
        peak_liq = path.get("peak_liq") or 0.0
        floor = PULLED_SHARE * peak_liq
        for age, liq, pr in path["pts"]:
            if pr is None:
                continue
            if probe_age <= age <= conf_age:
                conf_prices.append(float(pr))
            if age < conf_age:
                continue
            if floor and liq is not None and liq < floor:
                break
            post_pts.append((age - conf_age, float(pr)))
    if not post_pts:
        return None
    if entry_px is None:
        entry_px = post_pts[0][1]
    if entry_px <= 0:
        return None
    series = [(t, px / entry_px) for t, px in post_pts if pull_after_conf is None or t <= pull_after_conf]
    if not series:
        return None
    peak_mult = max(m for _, m in series)
    conf_range = None
    if len(conf_prices) >= 2:
        lo, hi = min(conf_prices), max(conf_prices)
        if lo > 0:
            conf_range = (hi - lo) / lo  # peak-to-trough; a dip-and-recover counts as alive
    m6 = s.get("ret")
    return {"conf_ts": conf_ts, "pull_after_conf": pull_after_conf, "series": series,
            "peak_mult": peak_mult, "m6_ret": m6, "conf_range": conf_range}


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


def volatility_filter(entered: list[dict], committed: int, vol_grid=VOL_GRID,
                      probe_cost: float = PROBE_COST, entry_usd: float = ENTRY_USD,
                      hold: float = CLIMB_HOLD_S) -> dict:
    """Entry filter: among survivors, enter only those whose confirmation-window peak-to-trough range
    (``conf_range``) is >= V, skipping the dead-flat ones.  Per threshold, compare kept vs skipped on
    the hold-to-minute-`hold` exit (``win`` is the share that profits = the climber rate) and the net
    $ per committed probe.  The filter changes which survivors we *enter*, not how many we probe, so
    every committed probe is still paid; only kept survivors earn.  Survivors with no conf-window path
    are excluded from the split (counted as ``no_range``)."""
    have_range = [e for e in entered if e.get("conf_range") is not None]
    out = []
    for v in vol_grid:
        kept = [e for e in have_range if e["conf_range"] >= v]
        skipped = [e for e in have_range if e["conf_range"] < v]
        kr = [_hold_ret(e, hold) for e in kept]
        sr = [_hold_ret(e, hold) for e in skipped]
        net = sum(r * entry_usd for r in kr) - committed * probe_cost
        out.append({"vol_pct": v, "kept": _dist(kr), "skipped": _dist(sr),
                    "net_per_probe": (net / committed) if committed else None, "net_total": net})
    return {"with_range": len(have_range), "no_range": len(entered) - len(have_range),
            "by_threshold": out}


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

def fetch_series(store, helius, survivors_meta: list[dict], max_credits: int, grid=PULL_GRID_S,
                 conf_offsets=CONF_FETCH_OFFSETS) -> dict:
    """One `last_before` call per (pool, offset), cached in `watch_probe_exit`.  Fetches the
    confirmation-window offsets (``conf_offsets``, stored as sec <= 0, for the entry volatility) and
    the post-confirmation offsets (``grid``, sec > 0, for the exit maths).  Returns
    {pool: [(sec, sol, tok), ...]} (SOL lamports, raw token units).  Research (backup) key only.
    Reads reserves from the tx with the same `_raw_vault` helper the other trajectory studies use.
    Roughly len(conf_offsets)+len(grid) calls per survivor (~210 credits), so a full run wants
    --max-credits in the tens of thousands."""
    store.execute(SCHEMA)
    have: dict = {}
    for r in store.query("SELECT pool, sec, sol, tok FROM watch_probe_exit"):
        have.setdefault(r["pool"], {})[int(r["sec"])] = (r["sol"], r["tok"])
    if helius is None:
        return {p: sorted([(s, v[0], v[1]) for s, v in d.items() if v[0] and v[1]]) for p, d in have.items()}
    from tradingbot import launch_live as ll
    from tradingbot.data import pumpswap as ps
    start = helius.credits_used

    def _read(pool, mint, at_ts):
        tx = helius.last_before(pool, at_ts)
        sol = ll._raw_vault(tx, pool, ps.WSOL) if isinstance(tx, dict) else None
        tok = ll._raw_vault(tx, pool, mint) if (isinstance(tx, dict) and mint) else None
        return sol, tok

    for m in survivors_meta:
        pool, conf_ts, mint = m["pool"], m["conf_ts"], m.get("token_mint")
        cur = have.setdefault(pool, {})
        # confirmation-window samples (pre-entry, the pool is alive): read each, no early break
        for off in conf_offsets:
            if off in cur:
                continue
            if helius.credits_used - start + 10 > max_credits:
                break
            sol, tok = _read(pool, mint, conf_ts + off)
            cur[off] = (sol, tok)
            store.execute("INSERT OR REPLACE INTO watch_probe_exit(pool, sec, sol, tok, ts) VALUES(?,?,?,?,?)",
                          (pool, off, sol, tok, time.time()))
        # post-confirmation series (for the exit maths): break when emptied
        for t in grid:
            if t in cur:
                continue
            if helius.credits_used - start + 10 > max_credits:
                break
            sol, tok = _read(pool, mint, conf_ts + t)
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
            rec.update(peak_mult=e["peak_mult"], series_n=len(e["series"]), conf_range=e.get("conf_range"))
            with_path.append(e)
            rec["pull_after_conf"] = e["pull_after_conf"]
        # for the pull distribution, fall back to the scorer's pull time when there is no path
        if rec["pull_after_conf"] is None and scorer[f["pool"]].get("pulled_ts"):
            rec["pull_after_conf"] = scorer[f["pool"]]["pulled_ts"] - rec["conf_ts"]
        survs.append(rec)

    kept = [e for e in with_path if e.get("conf_range") is not None and e["conf_range"] >= DEFAULT_VOL]
    return {"hours": hours, "probes_total": len(all_probed), "probes_committed": committed,
            "survivors": len(surv_pools), "survivors_with_path": len(with_path),
            "pull_distribution": pull_distribution(
                [{"pull_after_conf": r["pull_after_conf"]} for r in survs]),
            "volatility": volatility_filter(with_path, committed),
            "kept_vol_pct": DEFAULT_VOL, "kept_n": len(kept),
            "kept_climb": climb_gate(kept),
            "take_profit": take_profit(with_path, committed),
            "trailing": trailing(with_path),
            "fixed_hold": fixed_hold(with_path),
            "climb_gate": climb_gate(with_path),
            "rows": survs}


# --- single-pool path dump -------------------------------------------------------------------------

def _dump_grid(probe_age: float, conf_age: float) -> list[int]:
    """Dense conf-relative offsets: every DUMP_STEP_S from the probe through ~minute 6 after confirmation."""
    pre = list(range(-int(round(conf_age - probe_age)), 0, DUMP_STEP_S))
    post = list(range(0, DUMP_AFTER_CONF_S + 1, DUMP_STEP_S))
    return sorted(set(pre + post))


def dump_pool(journal, store, cfg, pool: str, helius=None, max_credits: int = 4_000,
              paths: dict | None = None, now: float | None = None) -> dict:
    """Reconstruct one pool's reserve path from the probe through ~minute 6 so the climb-peak timing and any stall
    before a pull can be eyeballed (9KaQvM4Z: did it climb in the first minute, then stall, then rug?). With `helius`
    (research key) it fetches a fine `last_before` series every DUMP_STEP_S; else it uses the cached `watch_probe_exit`
    rows and the dashboard pts. Prices are shown relative to the confirmation entry price; read only."""
    from tradingbot import pool_tracker as pt
    now = now if now is not None else time.time()
    look = now - 30 * 24 * 3600
    f = pt.feint_pools(journal.query, look).get(pool)
    s = pt.scorer_rows(store.query, look).get(pool) or {}
    if not f:
        return {"pool": pool, "error": "no probe/feint event found for this pool in the last 30 days"}
    created = s.get("create_ts")
    feint_ts = f["feint_ts"]
    conf_ts = feint_ts + CONF_AFTER_S
    probe_age = (feint_ts - created) if created else 0.0
    conf_age = (conf_ts - created) if created else CONF_AFTER_S
    mint = s.get("token_mint") or f.get("mint")
    sol_usd = pt._sol_usd(store.query)
    pulled_ts = s.get("pulled_ts")

    samples: list[tuple[float, float, float | None]] = []  # (pool_age_s, price_raw, liq_usd|None)
    if helius is not None:
        grid = _dump_grid(probe_age, conf_age)
        fetched = fetch_series(store, helius, [{"pool": pool, "conf_ts": conf_ts, "token_mint": mint}],
                               max_credits, grid=[g for g in grid if g > 0], conf_offsets=[g for g in grid if g <= 0])
        for sec, sol, tok in fetched.get(pool, []):
            if sol and tok:
                samples.append((conf_age + sec, sol / tok, (2 * sol / 1e9 * sol_usd) if sol_usd else None))
        source = "helius fine series"
    else:
        p = (paths if paths is not None else _load_paths(cfg)).get(pool) or {}
        for a, lq, pr in p.get("pts", []):
            if pr:
                samples.append((a, pr, lq))
        store.execute(SCHEMA)
        for r in store.query("SELECT sec, sol, tok FROM watch_probe_exit WHERE pool=?", (pool,)):
            if r["sol"] and r["tok"]:
                samples.append((conf_age + int(r["sec"]), r["sol"] / r["tok"],
                                (2 * r["sol"] / 1e9 * sol_usd) if sol_usd else None))
        source = "dashboard path + cache (no --fetch)"
    samples = sorted(set(samples))
    if not samples:
        return {"pool": pool, "mint": mint, "error": "no reserve samples (run with --fetch)"}
    entry = next((pr for a, pr, _ in samples if a >= conf_age), samples[0][1])
    peak = max(pr for _, pr, _ in samples)
    return {"pool": pool, "mint": mint, "source": source, "created_ts": created,
            "probe_age_s": round(probe_age, 1), "conf_age_s": round(conf_age, 1),
            "pull_pool_age_s": round(pulled_ts - created, 1) if (pulled_ts and created) else None,
            "entry_px": entry, "peak_px": peak,
            "peak_from_entry": (peak / entry - 1) if entry else None,
            "samples": [{"pool_age_s": round(a, 1), "after_conf_s": round(a - conf_age, 1),
                         "px_rel_conf": (pr / entry) if entry else None, "liq_usd": lq} for a, pr, lq in samples]}


def render_dump(d: dict) -> str:
    if d.get("error"):
        return f"PATH DUMP — {d['pool']}: {d['error']}"
    pull = d.get("pull_pool_age_s")
    o = [f"PATH DUMP — {d['pool']} ({d.get('mint')})  [{d['source']}]",
         f"probe at pool age {d['probe_age_s']}s, confirmation (entry) at {d['conf_age_s']}s, "
         + (f"scorer LP pull at {pull}s" if pull is not None else "no scorer pull on record"),
         f"entry price {d['entry_px']:.3g}, peak {d['peak_px']:.3g} ({_pct(d['peak_from_entry'])} from entry)",
         "",
         f"  {'pool_age':>8} {'after_conf':>10} {'px xEntry':>9} {'liq':>9}  mark"]
    rels = [sm["px_rel_conf"] for sm in d["samples"] if sm["px_rel_conf"] is not None]
    peak_rel = max(rels) if rels else None
    pull_marked = False
    for sm in d["samples"]:
        marks = []
        if abs(sm["after_conf_s"]) < DUMP_STEP_S:
            marks.append("<- entry")
        if peak_rel is not None and sm["px_rel_conf"] == peak_rel:
            marks.append("<- peak")
        if pull is not None and not pull_marked and sm["pool_age_s"] >= pull - DUMP_STEP_S:
            marks.append("<- scorer pull")  # the first sample at/around the scorer's LP-pull time
            pull_marked = True
        px = f"{sm['px_rel_conf']:.3f}" if sm["px_rel_conf"] is not None else "—"
        liq = f"${sm['liq_usd'] / 1000:.1f}k" if sm.get("liq_usd") else "—"
        o.append(f"  {sm['pool_age_s']:>7.0f}s {sm['after_conf_s']:>9.0f}s {px:>9} {liq:>9}  {' '.join(marks)}")
    return "\n".join(o)


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

    v = r["volatility"]
    o.append(f"\nVOLATILITY ENTRY FILTER (enter a survivor only if its confirmation-window peak-to-trough "
             f"range >= V; up OR down, a dip-and-recover counts)")
    o.append(f"  {v['with_range']} survivors have a conf-window path, {v['no_range']} do not "
             f"(the latter are left out of the split)")
    o.append(f"  {'V':>5} | {'kept n':>6} {'mean':>8} {'climb%':>7} | {'skip n':>6} {'mean':>8} {'climb%':>7} | {'net/probe':>10}")
    for b in v["by_threshold"]:
        k, sk = b["kept"], b["skipped"]
        o.append(f"  {b['vol_pct'] * 100:>4.1f}% | {k['n']:>6} {_pct(k.get('mean')):>8} {_pct(k.get('win'), 0):>7} | "
                 f"{sk['n']:>6} {_pct(sk.get('mean')):>8} {_pct(sk.get('win'), 0):>7} | "
                 f"{(f'${b['net_per_probe']:+.2f}' if b['net_per_probe'] is not None else '—'):>10}")

    o.append(f"\nDEPLOYED COMBO — climb gate on the {r['kept_vol_pct'] * 100:.0f}% volatility-kept set "
             f"({r['kept_n']} survivors): enter movers, bail at the check if not up, else ride to "
             f"{CLIMB_HOLD_S // 60}m")
    o.append(f"  {'check':>6} {'gain':>5} {'mean':>8} {'median':>8} {'win':>6} {'n':>4}")
    for t in r["kept_climb"]:
        s = t["ret"]
        o.append(f"  {t['check_s']:>5}s {t['gain_pct'] * 100:>4.0f}% {_pct(s.get('mean')):>8} "
                 f"{_pct(s.get('median')):>8} {_pct(s.get('win'), 0):>6} {s['n']:>4}")

    o.append("\n--- exit-lever comparison on all survivors with a path (unfiltered) ---")
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
