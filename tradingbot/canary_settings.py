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
     "live": True, "lo": 0, "hi": 100, "help": "The 'one more hit' probe: some ruggers fire only on the second commit. "
     "0 = turn the second probe off (enter straight after the gates)."},
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
    {"key": "launch_canary_hold_for_s", "label": "Hold for (s from fill)", "type": "float", "group": "Entry",
     "live": True, "lo": 0, "hi": 600, "help": "> 0: the safe-zone hold-probe sells this many seconds after the FILL "
     "(a duration), not at the pool age above. Fixes the fill-latency gap (fills land ~76s for a 66s target). 0 = off."},
    {"key": "launch_canary_hold_schedule", "label": "Hold schedule (hold[:wait],...)", "type": "str", "group": "Entry",
     "live": True, "help": "Multi-round holds: 'hold[:wait],...' seconds, one round per item. Each round buys, holds "
     "`hold` s from the fill, sells, waits `wait` s, then re-enters the same pool; a rug during a hold or wait stops it. "
     "Overrides Hold-for and the pool-age sell. e.g. 8:5,8:5,12"},
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
    {"key": "launch_canary_rotate", "label": "Wallet rotation", "type": "bool", "group": "Execution", "live": False,
     "help": "Spread trades across the loaded wallets, freshest first (needs the direct route; runs with one "
             "wallet too, reusing it). Takes effect after a restart."},
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
# The safe-zone hold-probe, but each hold is a DURATION from the fill and repeats over several rounds with a wait
# between: enter, hold a few seconds, sell, wait, re-enter the same pool. Measures whether re-entering after a wait
# gets rugged again (reactive, fires regardless) or survives (the pull aborts when we leave), and the per-round latency.
PRESETS["multi_hold"] = {
    "label": "Multi-round hold probe (measure)",
    "about": "The safe-zone hold-probe, but each hold is a duration from the fill and repeats: enter a probe survivor, "
             "hold a few seconds, sell, wait, re-enter the same pool, over several rounds. Tests whether re-entering "
             "after a wait rugs again (reactive) or survives (the pull aborts when we leave), round by round. Minute-6 "
             "re-entry stays off.",
    "values": {**PRESETS["safe_zone"]["values"], "launch_canary_hold_schedule": "6:4,6:4,6",
               "launch_canary_hold_for_s": 0.0}}


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


# Dependency rules: when ``active(state)`` holds, ``req_key`` must equal ``req_val``, else the combo is incoherent.
# ``label`` names the prerequisite the dashboard offers to turn on in one click. Resolution iterates to a fixed
# point, since enabling one prerequisite (e.g. the safe-zone hold-probe) can activate another rule (it needs Enter,
# which needs the probe, which needs the direct route).
_DEPS: list = [
    (lambda s: s.get("launch_canary_feint"), "launch_canary_route", "direct",
     "Feint needs the direct route — set Route to direct first (takes effect after restart).", "Route = direct"),
    (lambda s: s.get("launch_canary_feint_all"), "launch_canary_feint", True,
     "Feint every pool needs Feint before buying on.", "Feint before buying"),
    (lambda s: s.get("launch_canary_feint_only"), "launch_canary_feint", True,
     "Feint only needs Feint before buying on.", "Feint before buying"),
    (lambda s: s.get("launch_canary_clips", 1) > 1, "launch_canary_route", "direct",
     "Clip buys need the direct route.", "Route = direct"),
    (lambda s: s.get("launch_canary_rotate"), "launch_canary_route", "direct",
     "Wallet rotation is built for the direct route — set Route to direct first (takes effect after restart).",
     "Route = direct"),
    (lambda s: s.get("launch_canary_probe"), "launch_canary_route", "direct",
     "The 1-slot probe needs the direct route — set Route to direct first (takes effect after restart).",
     "Route = direct"),
    (lambda s: s.get("launch_canary_enter"), "launch_canary_probe", True,
     "Enter double-survivors needs the 1-slot probe on (it runs after the probe).", "1-slot probe"),
    (lambda s: s.get("launch_canary_wallet_gate"), "launch_canary_enter", True,
     "The wallet-pattern gate only runs on the two-stage entry — turn Enter double-survivors on.",
     "Enter double-survivors"),
    (lambda s: s.get("launch_canary_lp_entry"), "launch_canary_enter", True,
     "The LP deposit entry is the two-stage entry's commit — turn Enter double-survivors on.", "Enter double-survivors"),
    (lambda s: s.get("launch_canary_lp_entry"), "launch_canary_route", "direct",
     "The LP deposit entry keeps direct probes; set Route to direct (the entry itself is LP).", "Route = direct"),
    (lambda s: s.get("launch_canary_safe_zone"), "launch_canary_enter", True,
     "The safe-zone hold-probe is part of the two-stage entry — turn Enter double-survivors on.",
     "Enter double-survivors"),
    (lambda s: s.get("launch_canary_reentry"), "launch_canary_safe_zone", True,
     "Re-entry runs after the safe-zone hold-probe — turn Safe-zone hold-probe on.", "Safe-zone hold-probe"),
    (lambda s: s.get("launch_canary_hold_for_s", 0) > 0 or s.get("launch_canary_hold_schedule"),
     "launch_canary_safe_zone", True,
     "The hold-for duration / hold schedule drive the safe-zone hold-probe — turn Safe-zone hold-probe on.",
     "Safe-zone hold-probe"),
]


class CoherenceError(ValueError):
    """A settings combination the canary can't run. Carries the prerequisite changes (``needs``: key -> value)
    and their human ``need_labels`` so the dashboard can offer to turn them on and save in one step."""

    def __init__(self, message: str, needs: dict, labels: list):
        super().__init__(message)
        self.needs = needs
        self.need_labels = labels


def _resolve_coherence(state: dict) -> tuple[dict, list]:
    """The prerequisite changes (key -> value) and their labels that would make ``state`` coherent. Iterates to a
    fixed point so a prerequisite that itself has a prerequisite is pulled in too."""
    work = dict(state)
    needs: dict = {}
    labels: list = []
    for _ in range(len(_DEPS) + 1):
        changed = False
        for active, key, val, _msg, label in _DEPS:
            if active(work) and work.get(key) != val:
                work[key] = val
                needs[key] = val
                if label not in labels:
                    labels.append(label)
                changed = True
        if not changed:
            break
    return needs, labels


def _coherent(state: dict) -> None:
    """Reject combinations the canary can't run (checked against the resulting state). On a conflict, raise a
    CoherenceError naming the first unmet prerequisite and carrying every prerequisite that would fix it, so the
    dashboard can offer to enable them and save in one step instead of rejecting a lone dependent edit."""
    for active, key, val, msg, _label in _DEPS:
        if active(state) and state.get(key) != val:
            needs, labels = _resolve_coherence(state)
            raise CoherenceError(msg, needs, labels)


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


def _num(v) -> str:
    """Tidy a number for display: 25.0 -> '25', 0.215 -> '0.215'."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else f"{f:g}"


def _commit_size(s: dict) -> str:
    if float(s.get("launch_canary_flat_usd", 0) or 0) > 0:
        base = f"${_num(s['launch_canary_flat_usd'])} per buy"
    elif float(s.get("launch_canary_band_scale", 0) or 0) > 0:
        base = f"the twin's band size x {_num(s['launch_canary_band_scale'])}"
    else:
        base = "the book's default band size"
    clips = int(s.get("launch_canary_clips", 1) or 1)
    return base + (f", split into {clips} back-to-back swaps" if clips > 1 else "")


def _exit_steps(s: dict, two_stage: bool) -> list:
    """The hold/exit steps after a real commit."""
    g = s.get
    out = []
    emin = _num(g("launch_canary_exit_min", 6))
    if two_stage and g("launch_canary_safe_zone"):
        sched = (g("launch_canary_hold_schedule") or "").strip()
        hold_for = float(g("launch_canary_hold_for_s", 0) or 0)
        if sched:
            how = f"Multi-round hold: {sched}  (seconds hold[:wait] per round, measured from each fill)."
        elif hold_for > 0:
            how = f"Hold {_num(hold_for)}s from the fill, then sell."
        else:
            how = f"Sell at pool age {_num(g('launch_canary_safe_zone_s', 90))}s (dodging the ~1m40 holding rug)."
        out.append({"on": True, "title": "Safe-zone hold-probe", "detail": how})
        out.append({"on": True, "title": "Watch for the holding rug",
                    "detail": f"Re-read the pool at pool age {_num(g('launch_canary_reentry_watch_s', 120))}s: "
                              "rugged = we dodged it; still liquid = clean."})
        if g("launch_canary_reentry"):
            out.append({"on": True, "title": "Re-enter clean survivors (REAL)",
                        "detail": f"A clean pool gets the real entry and rides to pool minute {emin}."})
        else:
            out.append({"on": False, "title": "Re-entry — off",
                        "detail": "Measures the holding-rug rate; clean survivors are not re-entered."})
        return out
    # Final entry (no safe-zone) or a plain/feint commit: hold to the pool-minute exit, with the price exits if armed.
    out.append({"on": True, "title": f"Hold to pool minute {emin}", "detail": f"Sell the held position at pool minute {emin}."})
    if two_stage:
        if float(g("launch_canary_trail_pct", 0) or 0) > 0:
            out.append({"on": True, "title": "Trailing stop",
                        "detail": f"Sell if the price falls {_num(g('launch_canary_trail_pct'))}% off its running peak."})
        if float(g("launch_canary_stall_window_s", 0) or 0) > 0:
            out.append({"on": True, "title": "Stall exit",
                        "detail": f"Sell if the rolling range over {_num(g('launch_canary_stall_window_s'))}s drops below "
                                  f"{_num(g('launch_canary_vol_min_pct', 2))}% (the wash bot going flat before a pull)."})
        if float(g("launch_canary_climb_check_s", 0) or 0) > 0:
            out.append({"on": True, "title": "Climb check",
                        "detail": f"At entry + {_num(g('launch_canary_climb_check_s'))}s: bail unless up "
                                  f"{_num(g('launch_canary_climb_min_gain_pct', 0))}%, else ride to the exit."})
    return out


def trade_journey(state: dict) -> dict:
    """The ordered steps a single canary trade will take under `state` (the current/pending settings), so the operator
    can confirm at a glance that it is set up the way they want. Read only; mirrors the engine's canary pipeline."""
    s = dict(state)
    g = s.get
    probe = bool(g("launch_canary_probe"))
    feint = bool(g("launch_canary_feint"))
    enter = bool(g("launch_canary_enter"))
    route = g("launch_canary_route", "jupiter")
    steps: list = []

    # 1. Which pools are offered to the canary.
    broad = probe or enter or bool(g("launch_canary_feint_all"))
    pools = "Every liquid pool (>= $5k at second 4)" if broad \
        else "The scorer's thin rug-vs-win picks (the sz twin's signals)"
    maxliq = float(g("launch_canary_max_liq_usd", 0) or 0)
    liq_txt = f"only pools under ${_num(maxliq)} liquidity" if maxliq > 0 else "any pool size"
    nth = int(g("launch_canary_every_n", 1) or 1)
    samp = f"; sampling 1 in {nth} signals" if nth > 1 else ""
    steps.append({"on": True, "title": "Candidate pool", "detail": f"{pools} - {liq_txt}{samp}. Route: {route}."})

    if probe:  # 1-slot probe pipeline (the two-stage entry)
        mode = "Two-stage entry (probe -> enter)" if enter else "1-slot probe (measure only)"
        blind = "fire the sell blind to land one slot after the buy (gap 1)" if g("launch_canary_probe_blind") \
            else "sell on the observe-lag path (gap 2)"
        steps.append({"on": True, "title": "1-slot probe",
                      "detail": f"Buy ${_num(g('launch_canary_probe_usd', 25))}, land it, {blind}; the sell escapes the "
                                "reactive pull. Never holds."})
        watch = float(g("launch_canary_probe_watch_s", 48) or 0) + float(g("launch_canary_probe_margin_s", 12) or 0)
        steps.append({"on": True, "title": f"Watch ~{_num(watch)}s",
                      "detail": "Re-read the pool: pulled -> rugger (tripped and dodged); still liquid -> survivor."})
        if not enter:
            steps.append({"on": False, "title": "Stops here - measure only",
                          "detail": "Never commits real money. Turn on “Enter double-survivors” to trade survivors."})
            return _finish(mode, steps, s)
        if g("launch_canary_vol_gate"):
            steps.append({"on": True, "title": "Volatility gate",
                          "detail": f"On a survivor, the confirmation-window price range must be >= "
                                    f"{_num(g('launch_canary_vol_min_pct', 2))}% (live Helius), else skip."})
        else:
            steps.append({"on": False, "title": "Volatility gate - off",
                          "detail": "Survivors are entered whatever the confirmation-window move."})
        if g("launch_canary_wallet_gate"):
            steps.append({"on": True, "title": "Wallet-pattern gate",
                          "detail": f"Read the first ~{_num(g('launch_canary_wallet_entry_s', 66))}s of trades; skip a "
                                    f"survivor in the riskiest {_num(g('launch_canary_wallet_skip_pct', 50))}% pattern."})
        if float(g("launch_canary_probe2_usd", 25) or 0) > 0:
            steps.append({"on": True, "title": "Second 1-slot probe",
                          "detail": f"“One more hit” at ${_num(g('launch_canary_probe2_usd'))} - enter only if it "
                                    "survives too (some ruggers fire on the second commit)."})
        else:
            steps.append({"on": False, "title": "Second probe - off", "detail": "No second probe (size 0)."})
        if g("launch_canary_lp_entry"):
            commit = (f"Enter as a liquidity provider (atomic buy + deposit), ${_num(g('launch_canary_entry_usd', 25))} - "
                      "a proportional rug leaves our SOL half (~50% recovered). Probes stay direct.")
        else:
            commit = f"Buy ${_num(g('launch_canary_entry_usd', 25))} for real on the pool (direct)."
        steps.append({"on": True, "title": "Enter (REAL money)", "detail": commit})
        steps.extend(_exit_steps(s, two_stage=True))
        return _finish(mode, steps, s)

    if feint:  # feint pipeline (no 1-slot probe)
        if g("launch_canary_feint_only"):
            steps.append({"on": True, "title": "Feint every routed pool",
                          "detail": f"Poke with ${_num(g('launch_canary_feint_usd', 25))}, re-read after "
                                    f"{_num(g('launch_canary_feint_wait_s', 4))}s - never commits."})
            steps.append({"on": False, "title": "Stops here - measure only",
                          "detail": "Measures the pull-trigger rate for ~$0.30 a pool."})
            return _finish("Feint (measure only)", steps, s)
        steps.append({"on": True, "title": "Feint the pool",
                      "detail": f"Poke with ${_num(g('launch_canary_feint_usd', 25))}; re-read after "
                                f"{_num(g('launch_canary_feint_wait_s', 4))}s. Commit only if it wasn't pulled."})
        steps.append({"on": True, "title": "Liquidity recheck",
                      "detail": "Read the SOL vault just before the buy; abort if the pool has been pulled since."})
        steps.append({"on": True, "title": "Buy (REAL money)", "detail": f"Buy {_commit_size(s)}."})
        steps.extend(_exit_steps(s, two_stage=False))
        return _finish("Feint, then commit", steps, s)

    # Plain direct/jupiter buy.
    steps.append({"on": True, "title": "Honeypot / route check",
                  "detail": "Simulate a buy + sell on the pool (free); skip a blocked sell or a high round-trip loss."})
    steps.append({"on": True, "title": "Buy (REAL money)", "detail": f"Buy {_commit_size(s)}."})
    steps.extend(_exit_steps(s, two_stage=False))
    return _finish("Direct buy", steps, s)


def _finish(mode: str, steps: list, s: dict) -> dict:
    """Append the limits that apply to every trade and return the journey."""
    bits = [f"at most {int(s.get('launch_canary_max_open', 5) or 5)} open"]
    cap = int(s.get("launch_canary_max_trades", 0) or 0)
    if cap > 0:
        bits.append(f"stop after {cap} trades")
    dl = float(s.get("launch_canary_daily_loss_usd", 0) or 0)
    if dl > 0:
        bits.append(f"no new buys once today's loss hits ${_num(dl)}")
    if s.get("launch_canary_follow_dd"):
        bits.append("pause with the twin's drawdown limit")
    steps.append({"on": True, "title": "Limits (every trade)", "detail": "; ".join(bits) + "."})
    return {"mode": mode, "steps": steps}
