"""Pool Tracker (read only): the live state of every pool the launch canary has feinted.

For each feint probe (event kind ``canary_feint``) it joins the scorer's ``watch_launch_live`` row
(created at, liquidity at second 4, peak SOL, LP pull time, minute-6 return) and, for pools still
young, a live reserve read, to answer per pool:

- live liquidity (USD) and when it was last read;
- created at, and the pool's age now and when we probed it;
- whether the LP was pulled, at what pool age, and how long after we probed;
- the peak liquidity seen after the probe;
- what $25 held to pool minute 6 would have returned (the scorer's minute-6 trade, net of the 6% cost);
- the best entry time (pool age of the lowest price before minute 6) and the return it would have given.

The dashboard's ``fetch_pools`` poller keeps the live reserve path per young pool and streams the
assembled rows; the ``pool-tracker`` CLI does one live read for a snapshot.
"""
from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone

MINUTE6_S = 360.0          # the scorer's minute-6 exit
PULLED_SHARE = 0.10        # pool SOL below this share of its peak = pulled (as launch_analysis.PULLED_SHARE)
COST = 0.06                # round-trip cost used for the ideal returns
STAKE_USD = 25.0           # the feint / commit size we size "would have made" on
CONF_AFTER_S = 60.0        # survival confirmation ≈ probe age + the watch window (48 s) + margin (12 s)
TRACK_WINDOW_S = 3 * 3600  # how far back the tracker looks by default
YOUNG_MAX_AGE_S = 3600.0   # keep reading a pool's reserves live for ~60 min, so the hold curve reaches 30-60 min
HOLD_MINS = [6, 10, 15, 30, 60]  # exit horizons for the survivor hold curve: does holding past minute 6 pay?
DEFAULT_RPC = "https://api.mainnet-beta.solana.com"

SCORER_COLS = ("pool", "token_mint", "create_ts", "decided_ts", "liq_usd", "sol_at_decide",
               "peak_sol", "sol_vault", "token_vault", "fill_px", "exit_px", "exit_sol",
               "pulled_ts", "ret", "status")


# --- reads -----------------------------------------------------------------------------------------

def _sol_usd(wquery) -> float | None:
    try:
        r = wquery("SELECT sol_usd FROM watch_market WHERE sol_usd > 0 ORDER BY ts DESC LIMIT 1", ())
    except Exception:  # noqa: BLE001 (no watch_market table yet)
        return None
    return float(r[0]["sol_usd"]) if r else None


def feint_pools(jquery, since: float) -> dict:
    """The earliest probe of each pool since `since`: feint probes (`canary_feint`) AND 1-slot probes
    (`canary_probe`), with the 1-slot probe's survival verdict (`canary_probe_verdict`, rugger/survivor) overlaid
    once the watch has read it. `kind` tags which probe it was; `verdict` is the probe's own result (feint:
    alive/pulled/inconclusive; 1-slot: ok/trapped/honeypot/buy_failed), `watch_verdict` is the ~48s survival read."""
    rows = jquery("SELECT ts, token, detail, kind FROM events WHERE kind IN ('canary_feint', 'canary_probe') "
                  "AND ts >= ? ORDER BY ts", (since,))
    out: dict = {}
    for r in rows:
        try:
            d = json.loads(r["detail"] or "{}")
        except Exception:  # noqa: BLE001 (a malformed detail: skip)
            d = {}
        pool = d.get("pool")
        if not pool:
            continue
        out.setdefault(pool, {"pool": pool, "mint": r["token"],
                              "kind": "probe" if r["kind"] == "canary_probe" else "feint",
                              "verdict": d.get("verdict"), "watch_verdict": None, "note": d.get("note") or "",
                              "feint_ts": r["ts"], "sol_before": d.get("sol_before"), "sol_after": d.get("sol_after")})
    for r in jquery("SELECT detail FROM events WHERE kind='canary_probe_verdict' AND ts >= ? ORDER BY ts", (since,)):
        try:
            d = json.loads(r["detail"] or "{}")
        except Exception:  # noqa: BLE001
            d = {}
        f = out.get(d.get("pool"))
        # Only a probe that actually committed money (verdict ok/trapped) can be a survivor/rugger: a honeypot or a
        # skipped probe never touched the pool, so a "survivor" read on it is meaningless and misleading. Ignore the
        # survival verdict for those (handles historical rows too, where the watch wrongly ran before the engine fix).
        if f is not None and (f.get("kind") != "probe" or f.get("verdict") in ("ok", "trapped")):
            f["watch_verdict"] = d.get("verdict")  # rugger / survivor from the survival watch
            if d.get("sol_after") is not None:
                f["sol_after"] = d.get("sol_after")
    return out


def scorer_rows(wquery, since: float) -> dict:
    """The scorer's watch_launch_live rows created since `since`, keyed by pool."""
    try:
        rows = wquery(f"SELECT {', '.join(SCORER_COLS)} FROM watch_launch_live WHERE create_ts >= ?", (since,))
    except Exception:  # noqa: BLE001 (no table / missing column on an old db)
        return {}
    return {r["pool"]: {c: r[c] for c in SCORER_COLS} for r in rows}


# --- live reserve reads ----------------------------------------------------------------------------

def _amount(data) -> int | None:
    """Token amount (u64 at byte 64) from a getMultipleAccounts base64 `data` field."""
    from tradingbot.data import pumpswap as ps
    if not data:
        return None
    raw = data[0] if isinstance(data, (list, tuple)) else data
    try:
        return ps.token_amount(base64.b64decode(raw))
    except Exception:  # noqa: BLE001
        return None


def read_vaults(rpc, pools: list[dict]) -> dict:
    """{pool: (sol, raw_token)} for pools with sol_vault/token_vault, via one getMultipleAccounts per 50 pools."""
    out: dict = {}
    addrs, index = [], []
    for p in pools:
        sv, tv = p.get("sol_vault"), p.get("token_vault")
        if sv and tv:
            index.append((p["pool"], len(addrs)))
            addrs += [sv, tv]
    for i in range(0, len(addrs), 100):
        chunk = addrs[i:i + 100]
        res = rpc("getMultipleAccounts", [chunk, {"encoding": "base64"}]) or {}
        vals = res.get("value", []) or []
        for pool, base in index:
            if base < i or base + 1 >= i + len(chunk):
                continue
            s, t = vals[base - i] if base - i < len(vals) else None, vals[base - i + 1] if base - i + 1 < len(vals) else None
            sol = _amount(s.get("data")) if s else 0
            tok = _amount(t.get("data")) if t else 0
            out[pool] = ((sol or 0) / 1e9, float(tok or 0))
    return out


def update_path(path: dict | None, now: float, created: float | None, feint_ts: float, sol: float,
                tok: float, sol_usd: float | None, exit_px: float | None) -> dict:
    """Append one live reserve sample to a pool's path and recompute the live/peak/pull/optimal figures.

    `path` carries `pts` = [(pool_age_s, liq_usd, price)], `peak_liq`, `live_liq`, `pulled_ts`,
    `opt_age` (pool age of the cheapest entry before minute 6) and `opt_ret` (its return to the exit)."""
    path = path or {"pts": [], "peak_liq": 0.0, "pulled_ts": None}
    liq = (2 * sol * sol_usd) if sol_usd else None
    price = (sol / tok) if sol and tok else None
    age = (now - created) if created else (now - feint_ts)
    if liq is not None:
        path["pts"].append((age, liq, price))
        path["pts"] = path["pts"][-240:]
        path["live_liq"] = liq
        path["peak_liq"] = max(path.get("peak_liq") or 0.0, liq)
        if path["pulled_ts"] is None and path["peak_liq"] > 0 and liq < PULLED_SHARE * path["peak_liq"]:
            path["pulled_ts"] = now
    path["last_ts"] = now
    # optimal entry: the lowest price at or after the probe and before minute 6 while the pool is still alive
    # (liquidity at least PULLED_SHARE of its peak), exited near minute 6.
    probe_age = (feint_ts - created) if created else 0.0
    floor = PULLED_SHARE * (path.get("peak_liq") or 0.0)
    window = [(a, pr) for (a, lq, pr) in path["pts"]
              if pr and probe_age - 1 <= a <= MINUTE6_S + 60 and lq >= floor]
    if window:
        exit_px = exit_px or next((pr for a, pr in reversed(window) if a >= MINUTE6_S - 60), window[-1][1])
        opt_age, opt_px = min(window, key=lambda ap: ap[1])
        path["opt_age"] = opt_age
        path["opt_ret"] = (-1.0 if path["pulled_ts"] else min(exit_px / opt_px - 1 - COST, 3.0)) if opt_px else None
        # realistic two-stage entry: the first price at or after the survival confirmation (~probe age + the watch
        # window), held to minute 6 -- what entering a SURVIVOR for real would actually have made, vs opt (the
        # hindsight-best entry). -100% if the pool was pulled (a survivor that later rugged).
        conf = next(((a, pr) for (a, lq, pr) in path["pts"]
                     if pr and a >= probe_age + CONF_AFTER_S and lq >= floor), None)
        if conf and exit_px:
            path["conf_age"] = conf[0]
            path["conf_ret"] = -1.0 if path["pulled_ts"] else min(exit_px / conf[1] - 1 - COST, 3.0)
            # hold curve: enter at confirmation, sell at each horizon (last live price at/before it while alive);
            # -100% if pulled by then. Tells us whether holding a survivor past minute 6 pays (ripsnorters) or just
            # adds rug risk. Capped +1000% (not +300%) so the big winners show.
            pulled_age = (path["pulled_ts"] - created) if (path["pulled_ts"] and created) else None
            curve = {}
            for H in HOLD_MINS:
                h = H * 60
                if pulled_age is not None and pulled_age <= h:
                    curve[H] = -1.0
                    continue
                at = [pr for (a, lq, pr) in path["pts"] if pr and conf[0] <= a <= h and lq >= floor]
                if at:
                    curve[H] = min(at[-1] / conf[1] - 1 - COST, 10.0)
            path["conf_by"] = curve
    return path


# --- assembly --------------------------------------------------------------------------------------

def assemble(feints: dict, scorer: dict, sol_usd: float | None, paths: dict | None, now: float) -> list[dict]:
    """Join each feint to its scorer row and live path into one tracker row. Pure; newest pool first."""
    paths = paths or {}
    out = []
    for pool, f in feints.items():
        s = scorer.get(pool) or {}
        p = paths.get(pool) or {}
        created = s.get("create_ts")
        liq_probe = s.get("liq_usd")
        if liq_probe is None and f.get("sol_before") and sol_usd:
            liq_probe = 2 * f["sol_before"] * sol_usd
        peak_liq = None
        if s.get("liq_usd") and s.get("sol_at_decide"):
            peak_liq = s["liq_usd"] * (s.get("peak_sol") or 0.0) / s["sol_at_decide"]
        if p.get("peak_liq"):
            peak_liq = max(peak_liq or 0.0, p["peak_liq"])
        pulled_ts = p.get("pulled_ts") or s.get("pulled_ts")
        ret_m6 = s.get("ret")
        # for a 1-slot probe the meaningful verdict is the survival read (rugger/survivor); fall back to the probe's
        # own result (ok/trapped/...) until the watch fires. Feints keep alive/pulled/inconclusive.
        disp = f.get("watch_verdict") or f.get("verdict") if f.get("kind") == "probe" else f.get("verdict")
        pv = f.get("verdict")
        # Did money actually leave the wallet into the pool? A honeypot (sim-blocked, nothing broadcast) and a skipped
        # probe sent nothing; a feint that was `inconclusive` could not be placed. `committed` = the buy LANDED (a real
        # position/stake existed), so the dashboard can flag "no money sent" instead of a confusing dash + survivor tag.
        if f.get("kind") == "probe":
            sent = pv not in ("honeypot", "skipped", None)
            committed = pv in ("ok", "trapped")
        else:
            sent = pv not in ("inconclusive", None)
            committed = sent
        out.append({
            "pool": pool, "mint": f.get("mint"), "verdict": disp, "kind": f.get("kind"),
            "watch_verdict": f.get("watch_verdict"), "probe_verdict": f.get("verdict") if f.get("kind") == "probe" else None,
            "sent": sent, "committed": committed,
            "note": f.get("note"),
            "feint_ts": f["feint_ts"], "created_ts": created,
            "probe_pool_age_s": (f["feint_ts"] - created) if created else None,
            "pool_age_s": (now - created) if created else None,
            "liq_probe": liq_probe, "live_liq": p.get("live_liq"), "peak_liq": peak_liq,
            "pulled": pulled_ts is not None,
            "pull_pool_age_s": (pulled_ts - created) if (pulled_ts and created) else None,
            "pull_after_probe_s": (pulled_ts - f["feint_ts"]) if pulled_ts else None,
            "ret_m6": ret_m6, "pnl25": (STAKE_USD * ret_m6) if ret_m6 is not None else None,
            "opt_age_s": p.get("opt_age"), "opt_ret": p.get("opt_ret"),
            "conf_age_s": p.get("conf_age"), "conf_ret": p.get("conf_ret"), "conf_by": p.get("conf_by"),
            "status": s.get("status") or "pending",
        })
    out.sort(key=lambda r: (r["created_ts"] or r["feint_ts"] or 0), reverse=True)
    return out


def summary(rows: list[dict]) -> dict:
    done = [r for r in rows if r["ret_m6"] is not None]
    pnl = [r["pnl25"] for r in rows if r["pnl25"] is not None]
    surv = [r for r in rows if r.get("watch_verdict") == "survivor"]
    entry = [r["conf_ret"] for r in surv if r.get("conf_ret") is not None]  # enter at confirmation, hold to m6
    best = [r["opt_ret"] for r in surv if r.get("opt_ret") is not None]     # hindsight-best entry
    return {
        "pools": len(rows),
        "alive": sum(1 for r in rows if r["verdict"] == "alive"),
        "pulled_at_probe": sum(1 for r in rows if r["verdict"] == "pulled"),
        "inconclusive": sum(1 for r in rows if r["verdict"] == "inconclusive"),
        "survivor": len(surv),
        "rugger": sum(1 for r in rows if r.get("watch_verdict") == "rugger"),
        "pulled_since": sum(1 for r in rows if r["pulled"]),
        "with_outcome": len(done),
        "would_pnl25": round(sum(pnl), 2) if pnl else 0.0,
        "mean_ret_m6": round(sum(r["ret_m6"] for r in done) / len(done), 4) if done else None,
        # the Section-3 go/no-go: of survivors we measured a path for, what entering at confirmation would have made
        "survivor_entry_n": len(entry),
        "survivor_entry_mean": round(sum(entry) / len(entry), 4) if entry else None,
        "survivor_best_mean": round(sum(best) / len(best), 4) if best else None,
        # hold curve: entering a survivor and holding to each horizon -- does a long hold pay (ripsnorters)?
        "survivor_hold_curve": hold_curve(surv),
    }


def hold_curve(surv: list[dict]) -> dict:
    """Per hold horizon, over survivors with a measured price at that age: n, mean/median return, win rate and the
    share that hit +100% or more (`big` = the ripsnorters)."""
    out: dict = {}
    for H in HOLD_MINS:
        vals = [r["conf_by"][H] for r in surv if r.get("conf_by") and r["conf_by"].get(H) is not None]
        if vals:
            sv = sorted(vals)
            out[str(H)] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 4),
                           "median": round(sv[len(sv) // 2], 4),
                           "win": round(sum(1 for v in vals if v > 0) / len(vals), 3),
                           "big": round(sum(1 for v in vals if v >= 1.0) / len(vals), 3)}
    return out


# --- CLI / snapshot --------------------------------------------------------------------------------

def rows_for(jquery, wquery, hours: float = TRACK_WINDOW_S / 3600, now: float | None = None,
             paths: dict | None = None, rpc=None) -> tuple[list[dict], float | None]:
    """Read feints + scorer rows, optionally do one live reserve read for young pools, and assemble."""
    now = now if now is not None else time.time()
    since = now - hours * 3600
    feints = feint_pools(jquery, since)
    scorer = scorer_rows(wquery, since)
    su = _sol_usd(wquery)
    paths = dict(paths or {})
    if rpc is not None:
        young = [scorer[pl] for pl in feints
                 if scorer.get(pl) and scorer[pl].get("create_ts")
                 and now - scorer[pl]["create_ts"] <= YOUNG_MAX_AGE_S and not scorer[pl].get("pulled_ts")]
        try:
            v = read_vaults(rpc, young)
            for pl, (sol, tok) in v.items():
                s = scorer[pl]
                paths[pl] = update_path(paths.get(pl), now, s["create_ts"], feints[pl]["feint_ts"],
                                        sol, tok, su, s.get("exit_px"))
        except Exception:  # noqa: BLE001 (the public RPC can be slow / rate-limited)
            pass
    return assemble(feints, scorer, su, paths, now), su


def _age(s) -> str:
    if s is None:
        return "—"
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"


def _usd(v) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1000:
        return f"${v / 1000:.1f}k"
    return f"${v:.0f}"


def render(rows: list[dict], su: float | None) -> str:
    out = ["POOL TRACKER — feinted pools"]
    s = summary(rows)
    out.append(f"{s['pools']} pools (alive {s['alive']}, pulled-at-probe {s['pulled_at_probe']}, "
               f"inconclusive {s['inconclusive']}); pulled since {s['pulled_since']}; "
               f"{s['with_outcome']} with a minute-6 outcome, mean ret "
               f"{(s['mean_ret_m6'] * 100):+.1f}% -> $25 would make ${s['would_pnl25']:+.2f}")
    if s.get("survivor"):
        em = f"{s['survivor_entry_mean'] * 100:+.1f}%" if s.get("survivor_entry_mean") is not None else "—"
        bm = f"{s['survivor_best_mean'] * 100:+.1f}%" if s.get("survivor_best_mean") is not None else "—"
        out.append(f"survivors {s['survivor']} (ruggers {s['rugger']}); entering at confirmation -> {em} "
                   f"(hindsight best {bm}) over {s['survivor_entry_n']} with a path")
        curve = s.get("survivor_hold_curve") or {}
        if curve:
            out.append("  hold curve (enter at confirmation, sell at minute): " + "  ".join(
                f"{H}m {c['mean'] * 100:+.0f}% (win {c['win'] * 100:.0f}%, +100% {c['big'] * 100:.0f}%, n{c['n']})"
                for H, c in curve.items()))
    if su:
        out.append(f"SOL ${su:.2f}")
    out.append("")
    hdr = f"{'mint':<10} {'verdict':<8} {'age':>7} {'probe@':>6} {'liq':>7} {'live':>7} {'peak':>7} " \
          f"{'pulled@':>7} {'+probe':>7} {'m6%':>7} {'$25':>7} {'enter%':>7} {'best%':>7}"
    out.append(hdr)
    out.append("-" * len(hdr))
    for r in rows[:60]:
        m = (r["mint"] or "")[:10]
        m6 = f"{r['ret_m6'] * 100:+.0f}%" if r["ret_m6"] is not None else "—"
        pnl = f"${r['pnl25']:+.1f}" if r["pnl25"] is not None else "—"
        opt = f"{r['opt_ret'] * 100:+.0f}%" if r["opt_ret"] is not None else "—"
        enter = f"{r['conf_ret'] * 100:+.0f}%" if r.get("conf_ret") is not None else "—"
        out.append(f"{m:<10} {(r['verdict'] or '')[:8]:<8} {_age(r['pool_age_s']):>7} "
                   f"{_age(r['probe_pool_age_s']):>6} {_usd(r['liq_probe']):>7} {_usd(r['live_liq']):>7} "
                   f"{_usd(r['peak_liq']):>7} {_age(r['pull_pool_age_s']):>7} {_age(r['pull_after_probe_s']):>7} "
                   f"{m6:>7} {pnl:>7} {enter:>7} {opt:>7}")
    return "\n".join(out)
