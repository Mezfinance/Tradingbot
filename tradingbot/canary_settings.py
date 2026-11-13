"""Canary settings menu (dashboard): read, preset and edit the real-money launch canary's settings.

The engine and its executor share the live `Config` object (``engine.run_forever`` builds the dashboard with
``self.cfg``), so the fields the engine reads per decision (the probe, sizing and limit knobs) take effect the
moment they are set on the cfg -- marked ``live`` here. The fields bound when the book and executor are built at
start-up (route, budget, fees) are marked ``restart``: setting them updates the cfg and the persisted overlay, so
a restart (``fly deploy``) picks them up. Every change is persisted to ``canary_settings.json`` next to the journal
and re-applied on boot (defensively, never crashing boot). Keys are never touched here -- only numeric/flag knobs.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("tradingbot.canary_settings")

# key, label, type, group, live, lo, hi, help, choices
FIELDS: list[dict] = [
    # --- probe ---
    {"key": "launch_canary_feint", "label": "Feint before buying", "type": "bool", "group": "Probe", "live": True,
     "help": "Poke the pool with a feint before committing (direct route only)."},
    {"key": "launch_canary_feint_only", "label": "Feint only (never commit)", "type": "bool", "group": "Probe",
     "live": True, "help": "Run the feint on every pool but never buy: measure the pull-trigger rate for ~$0.30/pool."},
    {"key": "launch_canary_feint_all", "label": "Feint every liquid pool", "type": "bool", "group": "Probe",
     "live": True, "help": "Probe every pool >= $5k, not just the scorer's rug-vs-win picks."},
    {"key": "launch_canary_feint_usd", "label": "Feint size ($)", "type": "float", "group": "Probe", "live": True,
     "lo": 1, "hi": 100, "help": "The bait buy size."},
    {"key": "launch_canary_feint_wait_s", "label": "Feint wait (s)", "type": "float", "group": "Probe", "live": True,
     "lo": 0, "hi": 30, "help": "Seconds to wait after the poke before re-reading the pool."},
    {"key": "launch_canary_feint_drop_pct", "label": "Pulled drop (%)", "type": "float", "group": "Probe",
     "live": True, "lo": 10, "hi": 100, "help": "SOL-vault drop vs before the feint that counts as pulled."},
    {"key": "launch_canary_feint_hold_s", "label": "Feint hold (s)", "type": "float", "group": "Probe", "live": True,
     "lo": 0, "hi": 60, "help": "0 = one atomic buy+sell (never hold); > 0 = buy, hold, then sell."},
    {"key": "launch_canary_feint_slippage_bps", "label": "Feint slippage (bps)", "type": "int", "group": "Probe",
     "live": True, "lo": 10, "hi": 1000, "help": "Tighter = less dust left by the feint; too tight just reverts."},
    {"key": "launch_canary_probe", "label": "1-slot probe (measure, never commit)", "type": "bool", "group": "Probe",
     "live": True, "help": "Probe every liquid candidate: buy, land money, sell in the next slot (escapes the pull), "
                           "watch, record rugger/survivor. Never commits. Direct route only."},
    {"key": "launch_canary_probe_usd", "label": "Probe size ($)", "type": "float", "group": "Probe", "live": True,
     "lo": 1, "hi": 100, "help": "Big enough to trip the size-reacting operators (~$25)."},
    {"key": "launch_canary_probe_watch_s", "label": "Probe watch (s)", "type": "float", "group": "Probe", "live": True,
     "lo": 5, "hi": 300, "help": "Watch the pool this long after the probe (the measured max $25 pull age ~48s)."},
    {"key": "launch_canary_probe_margin_s", "label": "Probe margin (s)", "type": "float", "group": "Probe",
     "live": True, "lo": 0, "hi": 120, "help": "Extra margin on the watch before calling a pool a survivor."},
    {"key": "launch_canary_probe_blind", "label": "Blind sell (target N+1)", "type": "bool", "group": "Probe",
     "live": True, "help": "Fire the probe's sell blind (after the pause below) to land it one slot after the buy "
     "(gap 1). A confirmed fallback recovers the stake if it reverts. Off = the observe-lag path (gap 2, no reverts)."},
    {"key": "launch_canary_probe_delay_s", "label": "Blind sell pause (s)", "type": "float", "group": "Probe",
     "live": True, "lo": 0, "hi": 1, "help": "Blind only: wait this long after the buy before the sell, so it lands "
     "in the next slot (gap 1) not the same block (gap 0). ~half a Solana slot (0.2)."},
    # --- entry (two-stage) ---
    {"key": "launch_canary_enter", "label": "Enter double-survivors (REAL commit)", "type": "bool", "group": "Entry",
     "live": True, "help": "On a probe survivor: volatility gate, then a SECOND probe; if it survives too, enter for "
                           "real at the entry size. Needs the 1-slot probe on. Off = probe stays measure-only."},
    {"key": "launch_canary_entry_usd", "label": "Entry size ($)", "type": "float", "group": "Entry", "live": True,
     "lo": 1, "hi": 500, "help": "The real commit after both probes pass."},
    {"key": "launch_canary_probe2_usd", "label": "Second probe size ($)", "type": "float", "group": "Entry",
     "live": True, "lo": 1, "hi": 100, "help": "The 'one more hit' probe: some ruggers fire only on the second commit."},
    {"key": "launch_canary_vol_gate", "label": "Volatility gate", "type": "bool", "group": "Entry", "live": True,
     "help": "Enter only if the confirmation-window price range clears the minimum below (via live Helius)."},
    {"key": "launch_canary_vol_min_pct", "label": "Min confirmation range (%)", "type": "float", "group": "Entry",
     "live": True, "lo": 0, "hi": 50, "help": "Peak-to-trough range over [probe, confirmation]; skips dead-flat duds."},
    {"key": "launch_canary_climb_check_s", "label": "Climb check (s after entry)", "type": "float", "group": "Entry",
     "live": True, "lo": 0, "hi": 300, "help": "At this point: if not up by the gain below, bail; else ride to m6. 0 = off."},
    {"key": "launch_canary_climb_min_gain_pct", "label": "Climb min gain (%)", "type": "float", "group": "Entry",
     "live": True, "lo": 0, "hi": 100, "help": "Keep the position at the check only if up at least this much."},
    {"key": "launch_canary_trail_pct", "label": "Trailing stop (% off peak)", "type": "float", "group": "Entry",
     "live": True, "lo": 0, "hi": 90, "help": "Sell when the pool price falls this % off its running peak. 0 = off."},
    {"key": "launch_canary_stall_window_s", "label": "Stall window (s)", "type": "float", "group": "Entry",
     "live": True, "lo": 0, "hi": 120, "help": "Sell when the rolling range over this window drops below the vol gate "
     "(the wash bot going flat before a pull). 0 = off."},
    {"key": "launch_canary_exit_min", "label": "Exit pool minute", "type": "float", "group": "Entry", "live": True,
     "lo": 1, "hi": 60, "help": "Sell the held position at this pool minute (the hold-to-m6 exit)."},
    {"key": "launch_canary_wallet_gate", "label": "Wallet-pattern gate", "type": "bool", "group": "Entry",
     "live": True, "help": "Before the second probe, read the pool's first-~66s trade stream (live Helius) and skip "
     "a survivor whose wallet fingerprint ranks in the riskiest k% of recent probed pools (creator trading its own "
     "pool, few distinct wallets = rug). Fails open when there is too little history to rank."},
    {"key": "launch_canary_wallet_skip_pct", "label": "Wallet gate skip riskiest (%)", "type": "float",
     "group": "Entry", "live": True, "lo": 0, "hi": 100,
     "help": "Skip a survivor whose wallet-pattern score is in the riskiest this % of recent probed pools."},
    {"key": "launch_canary_wallet_entry_s", "label": "Wallet gate read window (s)", "type": "float", "group": "Entry",
     "live": True, "lo": 10, "hi": 180, "help": "Read the pool's trades up to this pool age (the selection-usable cut)."},
    {"key": "launch_canary_lp_entry", "label": "LP deposit entry", "type": "bool", "group": "Entry", "live": False,
     "help": "The commit enters as a liquidity provider (atomic buy + deposit) instead of a buyer, so an operator's "
     "proportional LP withdrawal leaves our SOL half (a rug recovers ~50% vs -100%). Probes stay direct. Needs Enter + "
     "direct route; takes effect after restart."},
    {"key": "launch_canary_safe_zone", "label": "Safe-zone hold-probe", "type": "bool", "group": "Entry", "live": True,
     "help": "Use the held position as a third probe: enter at confirmation, sell at the safe-zone age below (dodging "
     "the ~1m40 holding rug), then watch the pool. Measures the holding-rug rate until Re-entry is on. Needs Enter."},
    {"key": "launch_canary_safe_zone_s", "label": "Safe-zone sell (pool age s)", "type": "float", "group": "Entry",
     "live": True, "lo": 30, "hi": 180, "help": "Sell the hold-probe at this pool age, before the ~1m40 holding-rug cluster."},
    {"key": "launch_canary_reentry_watch_s", "label": "Re-entry watch (pool age s)", "type": "float", "group": "Entry",
     "live": True, "lo": 60, "hi": 360, "help": "Read the pool at this pool age: rugged = holding rugger dodged; survived = clean."},
    {"key": "launch_canary_reentry", "label": "Re-enter clean survivors (REAL)", "type": "bool", "group": "Entry",
     "live": True, "help": "A pool clean through the watch gets the real minute-6 entry. Needs the safe-zone hold-probe on."},
    # --- sizing ---
    {"key": "launch_canary_flat_usd", "label": "Flat buy size ($)", "type": "float", "group": "Sizing", "live": True,
     "lo": 0, "hi": 500, "help": "> 0: every commit at this size (0 = use the band scale)."},
    {"key": "launch_canary_band_scale", "label": "Band scale", "type": "float", "group": "Sizing", "live": True,
     "lo": 0, "hi": 1, "help": "> 0: the twin's band size x this (0.04 = 1/25). Ignored when flat size > 0."},
    {"key": "launch_canary_clips", "label": "Clips per buy", "type": "int", "group": "Sizing", "live": True,
     "lo": 1, "hi": 20, "help": "Split each buy into N equal back-to-back direct swaps (direct route)."},
    # --- limits ---
    {"key": "launch_canary_max_open", "label": "Max open trades", "type": "int", "group": "Limits", "live": True,
     "lo": 1, "hi": 50, "help": "Most positions open at once."},
    {"key": "launch_canary_max_trades", "label": "Trade cap (0 = none)", "type": "int", "group": "Limits",
     "live": True, "lo": 0, "hi": 100000, "help": "Stop buying after this many trades in the book (a test run)."},
    {"key": "launch_canary_max_liq_usd", "label": "Max liquidity ($, 0 = all)", "type": "float", "group": "Limits",
     "live": True, "lo": 0, "hi": 1000000, "help": "Only pools below this liquidity at second 4 (0 = every pool)."},
    {"key": "launch_canary_every_n", "label": "Buy 1 in N signals", "type": "int", "group": "Limits", "live": True,
     "lo": 1, "hi": 1000, "help": "Sample a fraction of the twin's signals (1 = all)."},
    {"key": "launch_canary_daily_loss_usd", "label": "Daily loss stop ($, 0 = none)", "type": "float",
     "group": "Limits", "live": True, "lo": 0, "hi": 100000, "help": "No new buys once today's realized loss hits this."},
    {"key": "launch_canary_follow_dd", "label": "Follow the twin's drawdown pause", "type": "bool", "group": "Limits",
     "live": True, "help": "No new buys while the full-size paper twin is paused by its drawdown limit."},
    # --- execution (restart) ---
    {"key": "launch_canary_route", "label": "Route", "type": "str", "group": "Execution", "live": False,
     "choices": ["direct", "jupiter", "lp"], "help": "How buys execute. direct = swap on the pumpswap pool."},
    {"key": "launch_canary_budget_usd", "label": "Book budget ($)", "type": "float", "group": "Execution",
     "live": False, "lo": 1, "hi": 100000, "help": "The book's starting cash; keep about this much in the wallet."},
    {"key": "launch_canary_priority_fee_sol", "label": "Priority fee cap (SOL)", "type": "float", "group": "Execution",
     "live": False, "lo": 0, "hi": 0.01, "help": "Cap on the priority fee per transaction."},
    {"key": "launch_canary_sell_slippage_bps", "label": "Sell slippage (bps)", "type": "int", "group": "Execution",
     "live": False, "lo": 100, "hi": 5000, "help": "Sells must go through on a fast pool."},
]
FIELD_BY_KEY = {f["key"]: f for f in FIELDS}
GROUPS = ["Probe", "Entry", "Sizing", "Limits", "Execution"]

PRESETS: dict[str, dict] = {
    "feint_measure": {
        "label": "Feint every pool (measure)",
        "about": "Probe every liquid pool and never commit — measure the pull-trigger rate for ~$0.30 a pool.",
        "values": {"launch_canary_route": "direct", "launch_canary_feint": True, "launch_canary_feint_only": True,
                   "launch_canary_feint_all": True, "launch_canary_feint_usd": 25.0, "launch_canary_max_liq_usd": 0.0,
                   "launch_canary_every_n": 1, "launch_canary_max_trades": 0, "launch_canary_follow_dd": False}},
    "commit_thin": {
        "label": "Commit thin pools (feint-gated)",
        "about": "Feint, then commit $25 on pools under $15k that survive the probe.",
        "values": {"launch_canary_route": "direct", "launch_canary_feint": True, "launch_canary_feint_only": False,
                   "launch_canary_feint_all": False, "launch_canary_flat_usd": 25.0, "launch_canary_clips": 1,
                   "launch_canary_max_liq_usd": 15000.0, "launch_canary_every_n": 1}},
    "clips": {
        "label": "Clip buys (5 × $1)",
        "about": "Split each buy into five $1 swaps into one pool — does the running total trigger the pull?",
        "values": {"launch_canary_route": "direct", "launch_canary_feint": False, "launch_canary_feint_only": False,
                   "launch_canary_clips": 5, "launch_canary_flat_usd": 5.0, "launch_canary_max_liq_usd": 15000.0,
                   "launch_canary_max_trades": 10}},
    "probe_measure": {
        "label": "1-slot probe (measure)",
        "about": "Probe every liquid candidate with a $25 buy + next-slot sell (lands money, escapes the pull), never "
                 "commit; watch 48s and record rugger (tripped and dodged) / survivor (a pool to enter).",
        "values": {"launch_canary_route": "direct", "launch_canary_probe": True, "launch_canary_probe_usd": 25.0,
                   "launch_canary_probe_watch_s": 48.0, "launch_canary_probe_margin_s": 12.0,
                   "launch_canary_feint": False, "launch_canary_feint_only": False, "launch_canary_feint_all": False,
                   "launch_canary_max_liq_usd": 0.0, "launch_canary_every_n": 1, "launch_canary_max_trades": 0,
                   "launch_canary_follow_dd": False}},
    "two_stage": {
        "label": "Two-stage entry (probe → enter)",
        "about": "Probe every liquid pool; on a survivor, check the confirmation-window volatility (Helius), skip a "
                 "rug-like wallet pattern, fire a second probe, and if it survives too, enter $25 for real and ride "
                 "to pool minute 6 with a 30% trailing stop.",
        "values": {"launch_canary_route": "direct", "launch_canary_probe": True, "launch_canary_probe_usd": 25.0,
                   "launch_canary_probe_watch_s": 48.0, "launch_canary_probe_margin_s": 12.0,
                   "launch_canary_enter": True, "launch_canary_entry_usd": 25.0, "launch_canary_probe2_usd": 25.0,
                   "launch_canary_vol_gate": True, "launch_canary_vol_min_pct": 5.0,
                   "launch_canary_climb_check_s": 0.0, "launch_canary_climb_min_gain_pct": 0.0,
                   "launch_canary_trail_pct": 30.0, "launch_canary_stall_window_s": 20.0,
                   "launch_canary_wallet_gate": True, "launch_canary_wallet_skip_pct": 50.0,
                   "launch_canary_exit_min": 6.0, "launch_canary_feint": False, "launch_canary_feint_only": False,
                   "launch_canary_feint_all": False, "launch_canary_max_liq_usd": 0.0, "launch_canary_every_n": 1,
                   "launch_canary_max_trades": 0, "launch_canary_follow_dd": False}},
    "sz_replica": {
        "label": "Exact sz replica (1/10)",
        "about": "Trade sz's rule in every band at a tenth of its band sizes, no feint, no daily stop.",
        "values": {"launch_canary_route": "direct", "launch_canary_feint": False, "launch_canary_feint_only": False,
                   "launch_canary_feint_all": False, "launch_canary_flat_usd": 0.0, "launch_canary_band_scale": 0.1,
                   "launch_canary_clips": 1, "launch_canary_max_liq_usd": 0.0, "launch_canary_daily_loss_usd": 0.0}},
}
# The two-stage entry, but the commit enters as a liquidity provider: an operator's proportional LP withdrawal leaves
# our SOL half, so a minute-2 rug recovers ~50% instead of -100%. Probes stay direct; only the entry/exit are LP.
PRESETS["two_stage_lp"] = {
    "label": "Two-stage LP entry (recover the SOL half)",
    "about": "The two-stage entry, but the commit enters as a liquidity provider (atomic buy + deposit) instead of a "
             "buyer. Operators pull by withdrawing their own LP, which leaves our SOL half, so a minute-2 rug recovers "
             "~50% instead of -100%. The probes stay direct round-trips. Takes effect after a restart.",
    "values": {**PRESETS["two_stage"]["values"], "launch_canary_lp_entry": True}}
# The two-stage entry, but the commit is a safe-zone hold-probe: enter at confirmation, sell at ~90s (before the ~1m40
# holding-rug cluster), then watch the pool to tell a holding rugger from a clean survivor. Re-entry off, so it measures
# the holding-rug rate without committing re-entry money (turn Re-enter on once the induce-and-dodge is confirmed).
PRESETS["safe_zone"] = {
    "label": "Safe-zone hold-probe (measure)",
    "about": "The two-stage entry, but the commit is a hold-probe: enter at confirmation, sell at ~90s (before the "
             "~1m40 holding-rug cluster), then watch the pool to separate a holding rugger (dodged) from a clean "
             "survivor. Re-entry is off, so it measures the holding-rug rate before any re-entry money is committed.",
    "values": {**PRESETS["two_stage"]["values"], "launch_canary_safe_zone": True, "launch_canary_safe_zone_s": 90.0,
               "launch_canary_reentry_watch_s": 120.0, "launch_canary_reentry": False}}


def settings_path(cfg) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(cfg.db_path)), "canary_settings.json")


def current(cfg) -> dict:
    return {f["key"]: getattr(cfg, f["key"], None) for f in FIELDS}


def _coerce(field: dict, value):
    t = field["type"]
    if t == "bool":
        return bool(value) if not isinstance(value, str) else value.strip().lower() in ("1", "true", "on", "yes")
    if t == "int":
        v = int(float(value))
    elif t == "float":
        v = float(value)
    else:  # str
        v = str(value)
        if field.get("choices") and v not in field["choices"]:
            raise ValueError(f"{field['label']}: must be one of {', '.join(field['choices'])}")
        return v
    lo, hi = field.get("lo"), field.get("hi")
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ValueError(f"{field['label']}: must be between {lo} and {hi}")
    return v


def validate(changes: dict) -> dict:
    """Coerce and bounds-check a {key: value} map of changes; raise ValueError on a bad value or unknown key."""
    out = {}
    for key, value in (changes or {}).items():
        field = FIELD_BY_KEY.get(key)
        if field is None:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = _coerce(field, value)
    return out


def _coherent(state: dict) -> None:
    """Reject combinations the canary can't run (checked against the resulting state)."""
    if state.get("launch_canary_feint") and state.get("launch_canary_route") != "direct":
        raise ValueError("Feint needs the direct route — set Route to direct first (takes effect after restart).")
    if state.get("launch_canary_feint_all") and not state.get("launch_canary_feint"):
        raise ValueError("Feint every pool needs Feint before buying on.")
    if state.get("launch_canary_feint_only") and not state.get("launch_canary_feint"):
        raise ValueError("Feint only needs Feint before buying on.")
    if state.get("launch_canary_clips", 1) > 1 and state.get("launch_canary_route") != "direct":
        raise ValueError("Clip buys need the direct route.")
    if state.get("launch_canary_probe") and state.get("launch_canary_route") != "direct":
        raise ValueError("The 1-slot probe needs the direct route — set Route to direct first (takes effect after restart).")
    if state.get("launch_canary_enter") and not state.get("launch_canary_probe"):
        raise ValueError("Enter double-survivors needs the 1-slot probe on (it runs after the probe).")
    if state.get("launch_canary_wallet_gate") and not state.get("launch_canary_enter"):
        raise ValueError("The wallet-pattern gate only runs on the two-stage entry — turn Enter double-survivors on.")
    if state.get("launch_canary_lp_entry") and not state.get("launch_canary_enter"):
        raise ValueError("The LP deposit entry is the two-stage entry's commit — turn Enter double-survivors on.")
    if state.get("launch_canary_lp_entry") and state.get("launch_canary_route") != "direct":
        raise ValueError("The LP deposit entry keeps direct probes; set Route to direct (the entry itself is LP).")
    if state.get("launch_canary_safe_zone") and not state.get("launch_canary_enter"):
        raise ValueError("The safe-zone hold-probe is part of the two-stage entry — turn Enter double-survivors on.")
    if state.get("launch_canary_reentry") and not state.get("launch_canary_safe_zone"):
        raise ValueError("Re-entry runs after the safe-zone hold-probe — turn Safe-zone hold-probe on.")


def apply(cfg, journal, changes: dict) -> dict:
    """Validate `changes`, set the live-safe ones on the shared cfg (immediate effect), persist every change to the
    overlay (for restart-bound fields and to survive a redeploy), and report the diff. Returns the changed fields,
    which of them need a restart, and the resulting values."""
    clean = validate(changes)
    cur = current(cfg)
    state = {**cur, **clean}
    _coherent(state)
    changed, restart = {}, []
    for key, value in clean.items():
        if cur.get(key) != value:
            changed[key] = {"from": cur.get(key), "to": value}
            setattr(cfg, key, value)  # live for engine-read fields; harmless for restart ones until the rebuild
            if not FIELD_BY_KEY[key]["live"]:
                restart.append(key)
    if changed:
        _persist(cfg)
    return {"changed": changed, "restart_needed": restart, "values": current(cfg)}


def _persist(cfg) -> None:
    """Write the current values of every editable field to the overlay (only fields differing from a fresh env cfg)."""
    try:
        from tradingbot.config import Config
        base = Config.from_env()
    except Exception:  # noqa: BLE001 (fall back to writing all values)
        base = None
    overlay = {}
    for f in FIELDS:
        v = getattr(cfg, f["key"], None)
        if base is None or getattr(base, f["key"], None) != v:
            overlay[f["key"]] = v
    try:
        with open(settings_path(cfg), "w") as fh:
            json.dump(overlay, fh, indent=1)
    except OSError as exc:
        log.warning("canary settings: could not persist overlay: %s", exc)


def load_overlay(cfg) -> dict:
    try:
        with open(settings_path(cfg)) as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def apply_overlay(cfg) -> list[str]:
    """Apply the persisted overlay onto a freshly built cfg at start-up. Defensive: never raises, skips bad values."""
    applied = []
    for key, value in load_overlay(cfg).items():
        field = FIELD_BY_KEY.get(key)
        if field is None:
            continue
        try:
            setattr(cfg, key, _coerce(field, value))
            applied.append(key)
        except Exception as exc:  # noqa: BLE001 (a bad overlay value must never stop boot)
            log.warning("canary settings: skipped overlay %s=%r (%s)", key, value, exc)
    if applied:
        log.info("canary settings: applied overlay for %s", ", ".join(applied))
    return applied


def reset(cfg) -> None:
    """Clear the overlay and reset the editable fields to their env/deploy values on the live cfg."""
    try:
        os.remove(settings_path(cfg))
    except OSError:
        pass
    try:
        from tradingbot.config import Config
        base = Config.from_env()
        for f in FIELDS:
            setattr(cfg, f["key"], getattr(base, f["key"]))
    except Exception as exc:  # noqa: BLE001
        log.warning("canary settings: reset could not re-read env: %s", exc)


def menu(cfg) -> dict:
    """The payload the dashboard needs to render the settings menu."""
    return {"fields": FIELDS, "groups": GROUPS, "current": current(cfg),
            "presets": {k: {"label": v["label"], "about": v["about"]} for k, v in PRESETS.items()}}
