"""Live dashboard (11-04): the paper books' trades streamed to a browser as they happen, plus the report tables.

Served by the health server (`health.start_health_server`) on `health_port`, behind a password: the Fly secret
`DASHBOARD_TOKEN` (no secret = no dashboard; /health stays open for Fly's check). Open
`https://<app>.fly.dev/?key=<token>` once: it sets a cookie, so the plain address works afterwards.

  /             the page (no build step, one HTML file)
  /api/stream   server-sent events: `init` (every position of the running books + the launch scorer's pools of
                the last day), then `delta` (changed rows) and `fill` (new trades) every POLL_S, `summary`
                (backtest notes, downtime) every SUMMARY_S, comments as keep-alives
  /api/summary  the latest summary as JSON

11-05: the page computes every table itself (equity, return, drawdown, win rate, ideal outcomes, real vs ideal)
from the raw rows, so the tables move with each trade. One shared `Live` poller reads the rows on read-only
connections of its own (no journal lock; sqlite releases the GIL while reading) and versions each changed row,
so a page costs no database reads. The summary extras come from a low-priority child process
(`python -m tradingbot dashboard-summary PATH`)."""
from __future__ import annotations

import hmac
import json
import logging
import os
import subprocess
import sys
import threading
import time

from tradingbot.config import Config
from tradingbot.journal import Journal

log = logging.getLogger(__name__)

POLL_S = 2.0  # new fills are looked for this often
SUMMARY_S = 300  # the tables are rebuilt this often
CHILD_TIMEOUT_S = 240
SUMMARY_FILE = "dashboard-summary.json"
KEEPALIVE_S = 15
SHORTS_POLL_S = 30  # MZF Short Only feed (another Fly app) is read this often
SHORTS_TIMEOUT_S = 8
MARKET_POLL_S = 60  # the market monitor's hourly rows (watch.db market_hours) are re-read this often
POOLS_POLL_S = 20  # the pool tracker re-reads young feinted pools' reserves this often (public RPC)
FEED_BACKLOG = 60  # fills sent when a page connects
COOKIE = "tb_dash"


def token() -> str:
    return os.environ.get("DASHBOARD_TOKEN", "")


def _cookies(headers) -> dict:
    out: dict = {}
    for part in (headers.get("Cookie") or "").split(";"):
        name, _, value = part.strip().partition("=")
        if name:
            out[name] = value
    return out


def authorised(headers, query: dict, cfg=None) -> bool:
    """True when the request carries a valid session: a passkey session cookie, or the
    break-glass token as ?key= or in the dashboard cookie."""
    cookies = _cookies(headers)
    secret = token()
    if secret:
        given = (query.get("key") or [""])[0] or cookies.get(COOKIE, "")
        if given and hmac.compare_digest(given, secret):
            return True
    if cfg is not None:
        from tradingbot import dashpasskey
        sess = cookies.get(dashpasskey.SESS_COOKIE, "")
        if sess and dashpasskey.valid_session(cfg, sess):
            return True
    return False


def fills_after(journal, after_id: int, limit: int = 200) -> list[dict]:
    """Buys and sells of running books with id > after_id, oldest first (archived books, `name@id`, left out)."""
    rows = journal.query(
        "SELECT f.id, f.ts, f.side, f.usd, f.reason, p.strategy, p.symbol, p.token, p.cost_usd, p.realized_usd, "
        "p.closed_at, p.opened_at FROM fills f JOIN positions p ON p.id=f.position_id "
        "WHERE f.id > ? AND instr(p.strategy, '@') = 0 ORDER BY f.id LIMIT ?", (after_id, limit))
    out = []
    for r in rows:
        ret = None
        if r["side"] == "sell" and r["closed_at"] and r["cost_usd"]:
            ret = round(100 * (r["realized_usd"] / r["cost_usd"] - 1), 1)
        out.append({"id": r["id"], "ts": r["ts"], "side": r["side"], "book": r["strategy"],
                    "symbol": r["symbol"] or (r["token"] or "")[:6], "token": r["token"],
                    "usd": round(r["usd"] or 0.0, 2), "reason": r["reason"], "ret_pct": ret,
                    "held_s": round(r["ts"] - r["opened_at"]) if r["side"] == "sell" and r["opened_at"] else None})
    return out


def last_fills(journal: Journal, n: int = FEED_BACKLOG) -> list[dict]:
    top = journal.query("SELECT MAX(id) AS m FROM fills")
    newest = top[0]["m"] if top and top[0]["m"] is not None else 0
    return fills_after(journal, max(0, newest - 5 * n), 5 * n)[-n:]


def _feint_row(r) -> dict:
    try:
        d = json.loads(r["detail"] or "{}")
    except (ValueError, TypeError):
        d = {}
    return {"id": r["id"], "ts": r["ts"], "book": r["strategy"], "token": r["token"],
            "verdict": d.get("verdict"), "note": d.get("note") or "", "pool": d.get("pool"),
            "sol_before": d.get("sol_before"), "sol_after": d.get("sol_after")}


def feints_after(journal, after_id: int, limit: int = 300) -> list[dict]:
    """The launch canary's feint probes (events kind `canary_feint`) with id > after_id, oldest first."""
    rows = journal.query("SELECT id, ts, strategy, token, detail FROM events WHERE id > ? AND kind='canary_feint' "
                         "ORDER BY id LIMIT ?", (after_id, limit))
    return [_feint_row(r) for r in rows]


def last_feints(journal, n: int = 300) -> list[dict]:
    rows = journal.query("SELECT id, ts, strategy, token, detail FROM events WHERE kind='canary_feint' "
                         "ORDER BY id DESC LIMIT ?", (n,))
    return [_feint_row(r) for r in reversed(rows)]


def canary_switch(journal, action: str) -> dict:
    """Flip the real-money launch canary's kill switch from the dashboard. It writes the SAME state the canary-stop /
    canary-start CLI writes; the engine reads it on its next pass (no fly/SSH needed). 'stop' halts every canary buy
    and feint at once (open positions still sell at their pool minute); 'start' clears the halt."""
    from tradingbot.engine import CANARY_ALLOW, CANARY_HALT
    if action == "stop":
        journal.set_state(CANARY_HALT, repr(time.time()))
        return {"halted": True, "msg": "Canary halted: no new buys or feints."}
    if action == "start":
        journal.execute("DELETE FROM state WHERE key IN (?, ?)", (CANARY_HALT, CANARY_ALLOW))
        return {"halted": False, "msg": "Canary kill switch cleared: buys and feints resume."}
    raise ValueError(f"unknown canary action {action!r}")


def _probe_detail(r) -> dict:
    try:
        return json.loads(r["detail"] or "{}")
    except (ValueError, TypeError):
        return {}


def probe_summary(journal, n: int = 500) -> dict:
    """Readout of the 1-slot probe measurement run: how many pools we probed, and of those watched out to the window,
    how many rugged (we tripped and dodged) vs survived (a pool to enter), plus the trip rate and the recent probes."""
    probes = journal.query("SELECT id, ts, token, detail FROM events WHERE kind='canary_probe' ORDER BY id DESC LIMIT ?",
                           (n,))
    verdicts = journal.query("SELECT token, detail FROM events WHERE kind='canary_probe_verdict' ORDER BY id DESC "
                             "LIMIT ?", (n,))
    # probe verdict per token (newest first), so a survival read is only honoured for a probe that COMMITTED money.
    pverdict: dict = {}
    for r in probes:
        pverdict.setdefault(r["token"], _probe_detail(r).get("verdict"))
    committed = {"ok", "trapped"}
    vd: dict = {}
    for r in verdicts:
        if pverdict.get(r["token"]) in committed:  # ignore a stale survivor/rugger read on a honeypot/skipped probe
            vd.setdefault(r["token"], _probe_detail(r))  # newest verdict per token
    counts = {"rugger": 0, "survivor": 0, "unknown": 0}
    for d in vd.values():
        counts[d.get("verdict", "unknown")] = counts.get(d.get("verdict", "unknown"), 0) + 1
    decided = counts["rugger"] + counts["survivor"]
    # blind-fire measurement: of the probes that fired the sell blind (not the observe-lag path), how many reverted
    # (needed the confirmed fallback) and how many landed at each slot gap -- so the extra speed can be weighed.
    blind = [_probe_detail(r) for r in probes if _probe_detail(r).get("blind")]
    blind_reverted = sum(1 for d in blind if d.get("blind_reverted"))
    gaps = [d["slot_gap"] for d in blind if d.get("slot_gap") is not None and not d.get("used_fallback")]
    recent = []
    for r in probes[:60]:
        d, v = _probe_detail(r), vd.get(r["token"], {})
        sent = d.get("verdict") not in ("honeypot", "skipped", None)  # did a buy actually broadcast?
        recent.append({"ts": r["ts"], "token": r["token"], "pool": d.get("pool"), "probe": d.get("verdict"),
                       "sent": sent, "wallet": d.get("wallet_addr"),
                       "slot_gap": d.get("slot_gap"), "recovered_sol": d.get("recovered_sol"),
                       "blind_reverted": d.get("blind_reverted"), "used_fallback": d.get("used_fallback"),
                       "watch": v.get("verdict") if sent else None})
    probed_tokens = {r["token"] for r in probes}
    return {"probes": len(probes), "decided": decided, "pending": max(0, len(probed_tokens) - len(vd)),
            "rugger": counts["rugger"], "survivor": counts["survivor"], "unknown": counts.get("unknown", 0),
            "trip_rate": round(counts["rugger"] / decided, 3) if decided else None,
            "blind": len(blind), "blind_reverted": blind_reverted,
            "blind_revert_rate": round(blind_reverted / len(blind), 3) if blind else None,
            "blind_median_gap": (sorted(gaps)[len(gaps) // 2] if gaps else None), "recent": recent}


def canary_one(journal) -> dict:
    """Arm exactly one more canary action (the 'Execute one probe' button): the next liquid candidate is probed (or
    bought, if committing), then it waits. Writes the same state as the canary-one CLI; the engine reads it next pass."""
    from tradingbot.engine import CANARY_ALLOW, CANARY_HALT
    journal.set_state(CANARY_ALLOW, "1")
    journal.execute("DELETE FROM state WHERE key=?", (CANARY_HALT,))
    return {"armed": True, "msg": "One probe armed: the next liquid candidate is probed, then it waits."}


def canary_probe_all(journal, cfg=None) -> dict:
    """Run the probe continuously on every liquid pool (the 'Probe every liquid pool' button): clear the
    one-at-a-time allowance and the kill switch, so the engine probes each liquid candidate with no gate.
    This is the same state canary-start writes. Probing every pool also needs `launch_canary_probe` set on
    in the config (that is what routes every liquid pool to the canary); the UI warns when it is off."""
    from tradingbot.engine import CANARY_ALLOW, CANARY_HALT
    journal.execute("DELETE FROM state WHERE key IN (?, ?)", (CANARY_HALT, CANARY_ALLOW))
    on = bool(getattr(cfg, "launch_canary_probe", False)) if cfg is not None else None
    msg = "Probing every liquid pool: the kill switch and one-at-a-time gate are cleared."
    if on is False:
        msg += " Note: launch_canary_probe is OFF in the config, so no pools will be probed until it is set on and redeployed."
    return {"running": True, "probe_configured": on, "msg": msg}


def probe_smoke(cfg, mint: str, usd=None, pool: str | None = None) -> dict:
    """Run one low-cost real 1-slot probe (buy + next-slot sell) from the dashboard to prove the sell lands on-chain
    before a $25 run. Real money, direct route only; builds the canary executor like the CLI does."""
    from tradingbot import cli, launch_canary
    from tradingbot.config import LIVE_CONFIRM_PHRASE
    from tradingbot.data.jupiter import Jupiter
    if cfg.live_confirm != LIVE_CONFIRM_PHRASE:
        raise ValueError("The probe smoke spends real money; set TB_LIVE_CONFIRM on the machine.")
    if cfg.launch_canary_route != "direct":
        raise ValueError("The 1-slot probe needs the direct route (set Route to direct and redeploy).")
    if not mint:
        raise ValueError("A token mint with a pumpswap pool is required.")
    size = float(usd) if usd else 1.0
    if not 0 < size <= 25:
        raise ValueError("The probe smoke size must be between 0 and 25.")
    ex = cli.canary_executor(cfg, Jupiter(cfg.slippage_bps))
    return launch_canary.probe_smoke(ex, mint, size, pool)


_ROTATE = {"proc": None, "started": 0.0, "args": None}
_ROTATE_LOCK = threading.Lock()
_TOPUP = {"proc": None, "started": 0.0, "args": None}
_TOPUP_LOCK = threading.Lock()


def _provision_log(cfg) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(cfg.db_path)), "provision.log")


def _topup_log_path(cfg) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(cfg.db_path)), "topup.log")


def _rotation_running() -> bool:
    with _ROTATE_LOCK:
        p = _ROTATE["proc"]
        return p is not None and p.poll() is None


def canary_topup(cfg, level, wallets) -> dict:
    """Top the canary's trading wallets up to `level` SOL each from the treasury (pulling from the reserve if short),
    as a background `provision --top-up` process on this machine. `wallets` = 'all' or a list of 1-based wallet
    numbers. Additive and safe mid-trade, so the canary is NOT halted. Keys never reach the page (provision prints
    addresses + amounts only); the SOL lands straight on-chain, so no restart is needed."""
    try:
        level = float(level)
    except (TypeError, ValueError):
        raise ValueError("level must be a number")
    if not 0.01 <= level <= 5:
        raise ValueError("level must be between 0.01 and 5 SOL")
    if wallets in (None, "all", "", "ALL"):
        pick, indices = "all", "all"
    else:
        if not isinstance(wallets, (list, tuple)):
            raise ValueError("wallets must be 'all' or a list of wallet numbers")
        try:
            indices = sorted({int(x) for x in wallets})
        except (TypeError, ValueError):
            raise ValueError("wallet numbers must be whole numbers")
        if not indices or not all(1 <= i <= 10 for i in indices):
            raise ValueError("wallet numbers must be between 1 and 10")
        pick = ",".join(str(i) for i in indices)
    if _rotation_running():
        return {"started": False, "msg": "A wallet rotation is running; wait for it to finish before topping up."}
    with _TOPUP_LOCK:
        p = _TOPUP["proc"]
        if p is not None and p.poll() is None:
            return {"started": False, "msg": "A top-up is already running."}
        with open(_topup_log_path(cfg), "w") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", "tradingbot", "provision", "--top-up", "--level", str(level),
                 "--pick", pick, "--apply"], stdout=fh, stderr=subprocess.STDOUT,
                env={**os.environ, "TB_DB_PATH": cfg.db_path}, start_new_session=True)
        _TOPUP.update(proc=proc, started=time.time(), args={"level": level, "pick": pick})
    log.info("dashboard: wallet top-up started (level %.3f SOL, wallets %s)", level, pick)
    which = "all trading wallets" if pick == "all" else f"wallets {pick}"
    return {"started": True, "level": level, "pick": pick,
            "msg": f"Topping up {which} to {level} SOL each from the treasury (reserve if short). Watch the log below."}


def topup_log(cfg, lines: int = 400) -> dict:
    """The tail of the running (or last) wallet top-up, for the dashboard to show progress. Addresses + amounts only."""
    with _TOPUP_LOCK:
        p = _TOPUP["proc"]
        running = p is not None and p.poll() is None
        code = None if (p is None or running) else p.returncode
        started, args = _TOPUP["started"], _TOPUP["args"]
    text = ""
    try:
        with open(_topup_log_path(cfg)) as fh:
            text = "".join(fh.readlines()[-lines:])
    except OSError:
        pass
    return {"running": running, "returncode": code, "log": text, "started": started, "args": args}


def canary_rotate(journal, cfg, wallets, sol) -> dict:
    """Rotate the canary's trading wallets from the dashboard: halt the canary, then run `provision --apply` as a
    background process on this machine (it sweeps the current wallets + treasury into a fresh treasury, mints
    `wallets` new wallets and funds each with `sol` SOL, topping up from the reserve). Keys never reach the page --
    provision prints only addresses and the plan. The bot must be restarted to trade the new set, then Start."""
    from tradingbot.engine import CANARY_HALT
    try:
        wallets, sol = int(wallets), float(sol)
    except (TypeError, ValueError):
        raise ValueError("wallets must be a whole number and sol a number")
    if not 1 <= wallets <= 10:
        raise ValueError("wallets must be between 1 and 10")
    if not 0.01 <= sol <= 5:
        raise ValueError("sol per wallet must be between 0.01 and 5")
    with _TOPUP_LOCK:
        tp = _TOPUP["proc"]
        if tp is not None and tp.poll() is None:
            return {"started": False, "msg": "A wallet top-up is running; wait for it to finish before rotating."}
    with _ROTATE_LOCK:
        p = _ROTATE["proc"]
        if p is not None and p.poll() is None:
            return {"started": False, "msg": "A wallet rotation is already running."}
        journal.set_state(CANARY_HALT, repr(time.time()))  # stop feints/buys before sweeping the wallets
        with open(_provision_log(cfg), "w") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", "tradingbot", "provision", "--wallets", str(wallets), "--sol", str(sol),
                 "--apply"], stdout=fh, stderr=subprocess.STDOUT,
                env={**os.environ, "TB_DB_PATH": cfg.db_path}, start_new_session=True)
        _ROTATE.update(proc=proc, started=time.time(), args={"wallets": wallets, "sol": sol})
    log.info("dashboard: wallet rotation started (%d wallets, %.3f SOL each)", wallets, sol)
    return {"started": True, "wallets": wallets, "sol": sol,
            "msg": (f"Rotating to {wallets} fresh wallets at {sol} SOL each. The canary is halted; restart the bot "
                    "(fly deploy) to trade the new set, then press Start. Watch the log below.")}


def canary_settings_menu(cfg) -> dict:
    from tradingbot import canary_settings
    cmd = _deploy_command()
    return {**canary_settings.menu(cfg), "overlay": bool(canary_settings.load_overlay(cfg)),
            "deploy_cmd": " ".join(cmd) if cmd else None}


def canary_settings_update(cfg, journal, body: dict) -> dict:
    """Apply a settings change from the dashboard: a preset, a reset, or a {changes} map. Live fields take effect at
    once (the engine shares this cfg); restart-bound ones are persisted and reported as needing a restart."""
    from tradingbot import canary_settings
    if body.get("reset"):
        canary_settings.reset(cfg)
        return {"reset": True, "values": canary_settings.current(cfg),
                "msg": "Canary settings reset to the deploy (fly.toml) values."}
    if body.get("preset"):
        name = body["preset"]
        if name not in canary_settings.PRESETS:
            raise ValueError(f"unknown preset {name!r}")
        res = canary_settings.apply(cfg, journal, canary_settings.PRESETS[name]["values"])
        res["preset"] = name
        return res
    return canary_settings.apply(cfg, journal, body.get("changes") or {})


_DEPLOY = {"proc": None, "started": 0.0}
_DEPLOY_LOCK = threading.Lock()


def _deploy_command() -> list[str] | None:
    """The command the Deploy button runs: `TB_DEPLOY_CMD` if the operator set it, else `fly deploy --detach` only
    when a fly/flyctl binary is actually on this machine. None = no way to deploy from here (the normal case)."""
    cmd = os.environ.get("TB_DEPLOY_CMD", "").strip()
    if cmd:
        import shlex
        return shlex.split(cmd)
    import shutil
    for exe in ("fly", "flyctl"):
        path = shutil.which(exe)
        if path:
            app = os.environ.get("FLY_APP_NAME") or "supermeme"
            return [path, "deploy", "--detach", "-a", app]
    return None


def _deploy_log_path(cfg) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(cfg.db_path)), "deploy.log")


def fly_deploy(cfg) -> dict:
    """Run the deploy command on this machine (best effort, detached) so it survives the machine being replaced.
    `fly deploy` from inside the machine it replaces is awkward: it needs a fly token + the build context on the
    machine, and the rollout tears this process down, so `--detach` lets flyctl submit and exit. Normally you
    deploy from your Mac (`git pull && fly deploy`)."""
    cmd = _deploy_command()
    if cmd is None:
        return {"started": False, "msg": "No deploy command on this machine. Run `git pull && fly deploy` from your "
                "Mac, or set the TB_DEPLOY_CMD secret (and a fly token) to enable this button."}
    with _DEPLOY_LOCK:
        p = _DEPLOY["proc"]
        if p is not None and p.poll() is None:
            return {"started": False, "msg": "A deploy is already running."}
        with open(_deploy_log_path(cfg), "w") as fh:
            fh.write("$ " + " ".join(cmd) + "\n\n")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                                    cwd=os.environ.get("TB_DEPLOY_DIR") or None, env={**os.environ})
        _DEPLOY.update(proc=proc, started=time.time())
    log.info("dashboard: deploy started (%s)", " ".join(cmd))
    return {"started": True, "cmd": " ".join(cmd),
            "msg": "Deploy started. The machine restarts when it lands, so this page will disconnect — reload in a "
                   "minute or two."}


def deploy_log(cfg, lines: int = 400) -> dict:
    with _DEPLOY_LOCK:
        p = _DEPLOY["proc"]
        running = p is not None and p.poll() is None
        code = None if (p is None or running) else p.returncode
        started = _DEPLOY["started"]
    text = ""
    try:
        with open(_deploy_log_path(cfg)) as fh:
            text = "".join(fh.readlines()[-lines:])
    except OSError:
        pass
    cmd = _deploy_command()
    return {"running": running, "returncode": code, "log": text, "started": started,
            "command": " ".join(cmd) if cmd else None}


def rotate_log(cfg, lines: int = 400) -> dict:
    """The tail of the running (or last) wallet rotation, for the dashboard to show progress. Addresses only."""
    with _ROTATE_LOCK:
        p = _ROTATE["proc"]
        running = p is not None and p.poll() is None
        code = None if (p is None or running) else p.returncode
        started, args = _ROTATE["started"], _ROTATE["args"]
    text = ""
    try:
        with open(_provision_log(cfg)) as fh:
            text = "".join(fh.readlines()[-lines:])
    except OSError:
        pass
    return {"running": running, "returncode": code, "log": text, "started": started, "args": args}


WALLETS_POLL_S = 60  # balances change slowly; the public RPC keeps it off the Helius budget
WALLETS_RPC = os.environ.get("TB_WALLETS_RPC") or "https://api.mainnet-beta.solana.com"


def _solana_rpc(url: str, method: str, params: list, timeout: float = 12.0):
    import httpx
    r = httpx.post(url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("result")


def wallet_directory(cfg) -> dict:
    """Addresses only (never keys) of the canary's wallets, with roles and labels: the live trading set numbered
    1..n, the treasury, the reserve (the canary wallet that funds top-ups), and an archive of every other address
    the key-free address book has recorded. Balances and tx counts are added from the chain by fetch_wallets."""
    from tradingbot import health
    info = health.canary_wallets(cfg)
    book = (health.address_book() or {}).get("addresses", {}) or {}
    label_of = {a: (e.get("label") or "") for a, e in book.items()}
    created_of = {a: e.get("created") for a, e in book.items()}
    live = [{"n": i, "role": "trading", "address": a, "label": label_of.get(a) or f"wallet-{i}"}
            for i, a in enumerate(info.get("wallets") or [], 1)]
    if info.get("treasury"):
        live.append({"role": "treasury", "address": info["treasury"],
                     "label": label_of.get(info["treasury"]) or "treasury"})
    try:
        from tradingbot.execution.jupiter import read_keypair
        reserve = str(read_keypair(cfg).pubkey())
        live.append({"role": "reserve", "address": reserve, "label": label_of.get(reserve) or "reserve (canary)"})
    except Exception:  # noqa: BLE001 (no reserve key available here: fine)
        pass
    seen = {w["address"] for w in live}
    archive = sorted(({"role": "archive", "address": a, "label": label_of.get(a) or "", "created": created_of.get(a)}
                      for a in book if a not in seen), key=lambda w: (w["label"], w["address"]))
    return {"live": live, "archive": archive, "rotate": info.get("rotate"), "route": info.get("route"),
            "source": info.get("source"), "error": info.get("error")}


def build_summary(journal: Journal, cfg: Config, store=None, now: float | None = None) -> dict:
    """The slow-changing extras the page can't compute from the rows: backtest notes, downtime, last backup.
    (Book tables and launch outcomes are computed in the browser from the streamed rows since 11-05.)"""
    from tradingbot import model_book as mb
    from tradingbot import report
    books = mb.active_books(cfg)
    return {"built_ts": now if now is not None else time.time(),
            "expected": {b: report.EXPECTED.get(b) for b in books},
            "downtime": report.downtime(journal), "last_backup": json.loads(journal.get_state("last_backup") or "null")}


class Dashboard:
    """Keeps the latest summary (rebuilt on its own thread) and serves the stream."""

    def __init__(self, journal: Journal, cfg: Config, store=None, in_child: bool | None = None):
        self.journal, self.cfg, self.store = journal, cfg, store
        # 11-05: built inside the bot, the summary (full report + launch-status, ~1 min of Python) held the
        # GIL and the journal lock every 2 min: scorer decisions started ~40 s late, buys waited 25-50 s.
        # So it runs in its own low-priority process unless the journal is in memory (tests).
        self.in_child = (cfg.db_path != ":memory:") if in_child is None else in_child
        self.summary: dict | None = None
        self.version = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.live = Live(cfg, journal, store)
        self._external = True  # streams poll for themselves until start() runs the shared poller
        self.shorts: dict | None = None  # MZF Short Only (the shortbook app's feed), refreshed every SHORTS_POLL_S
        self.shorts_ver = 0
        self.market: dict | None = None  # the meme market monitor (market_monitor.payload), every MARKET_POLL_S
        self.market_ver = 0
        self.wallets: dict | None = None  # the canary wallets with chain balances, every WALLETS_POLL_S
        self.wallets_ver = 0
        self.pools: dict | None = None  # the pool tracker (feinted pools + live reserves), every POOLS_POLL_S
        self.pools_ver = 0
        # pool -> live reserve path kept between polls. Persisted to /data so a fly deploy does not wipe the early
        # (~60 s) reserve samples: without them a now-old pool's conf_ret / best entry / live peak can never be
        # rebuilt (the confirmation-window price is long gone), so those columns would blank out after every deploy.
        self._pool_paths: dict = self._load_pool_paths()
        self._paths_logged = False  # log the resolved save path once, so /data placement is verifiable in the logs
        self._mdb = None

    def _pool_paths_file(self) -> str | None:
        if self.cfg.db_path == ":memory:":
            return None  # tests: no disk
        return os.path.join(os.path.dirname(os.path.abspath(self.cfg.db_path)), "pool_paths.json")

    def _load_pool_paths(self) -> dict:
        path = self._pool_paths_file()
        if not path or not os.path.exists(path):
            if path:
                log.info("dashboard: no pool paths to restore at %s (fresh)", path)
            return {}
        try:
            with open(path) as fh:
                raw = json.load(fh)
            out = {}
            for pool, p in raw.items():
                if not isinstance(p, dict):
                    continue
                cb = p.get("conf_by")
                if isinstance(cb, dict):  # JSON stringifies int keys; hold_curve looks them up as ints
                    p["conf_by"] = {int(k): v for k, v in cb.items()}
                out[pool] = p
            log.info("dashboard: restored %d pool paths from %s", len(out), path)
            return out
        except Exception as exc:  # noqa: BLE001 (a corrupt file must never stop the dashboard)
            log.warning("dashboard: could not load pool paths from %s: %s", path, exc)
            return {}

    def _save_pool_paths(self) -> None:
        path = self._pool_paths_file()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)  # /data exists on Fly; belt-and-suspenders
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self._pool_paths, fh)
            os.replace(tmp, path)  # atomic: a deploy mid-write never leaves a half file
            if not self._paths_logged:  # confirm /data placement in the logs the first time it lands
                log.info("dashboard: persisting %d pool paths to %s", len(self._pool_paths), path)
                self._paths_logged = True
        except Exception as exc:  # noqa: BLE001
            log.warning("dashboard: could not save pool paths to %s: %s", path, exc)

    def _build_in_child(self) -> dict:
        path = os.path.join(os.path.dirname(os.path.abspath(self.cfg.db_path)), SUMMARY_FILE)

        def low_priority():
            try:
                os.nice(19)
                with open("/proc/self/oom_score_adj", "w") as fh:
                    fh.write("1000")
            except OSError:
                pass
        res = subprocess.run([sys.executable, "-m", "tradingbot", "dashboard-summary", path],
                             env={**os.environ, "TB_DB_PATH": self.cfg.db_path},
                             preexec_fn=low_priority, timeout=CHILD_TIMEOUT_S,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"child exited {res.returncode}: {res.stderr.decode(errors='replace')[-300:]}")
        with open(path) as fh:
            return json.load(fh)

    def refresh(self) -> None:
        try:
            summary = (self._build_in_child() if self.in_child
                       else build_summary(self.journal, self.cfg, self.store))
        except Exception:  # noqa: BLE001 (keep the last good summary)
            log.exception("dashboard summary failed")
            return
        with self._lock:
            self.summary, self.version = summary, self.version + 1

    def start(self) -> "Dashboard":
        def loop():
            while True:
                self.refresh()
                if self._stop.wait(SUMMARY_S):
                    return
        def live_loop():
            while not self._stop.is_set():
                self.live.poll()
                self._stop.wait(POLL_S)

        def shorts_loop():
            while not self._stop.is_set():
                self.fetch_shorts()
                self._stop.wait(SHORTS_POLL_S)
        def market_loop():
            while not self._stop.is_set():
                self.fetch_market()
                self._stop.wait(MARKET_POLL_S)
        def wallets_loop():
            while not self._stop.is_set():
                self.fetch_wallets()
                self._stop.wait(WALLETS_POLL_S)
        def pools_loop():
            while not self._stop.is_set():
                self.fetch_pools()
                self._stop.wait(POOLS_POLL_S)
        self._external = False
        if self.cfg.market_monitor and self.store is not None:
            threading.Thread(target=market_loop, name="dashboard-market", daemon=True).start()
        if self.cfg.launch_canary or self.cfg.launch_fast_canary:
            threading.Thread(target=wallets_loop, name="dashboard-wallets", daemon=True).start()
            threading.Thread(target=pools_loop, name="dashboard-pools", daemon=True).start()
        threading.Thread(target=loop, name="dashboard", daemon=True).start()
        threading.Thread(target=live_loop, name="dashboard-live", daemon=True).start()
        if self.cfg.shortbook_feed_url:
            threading.Thread(target=shorts_loop, name="dashboard-shorts", daemon=True).start()
        return self

    def fetch_shorts(self, get=None) -> None:
        """MZF Short Only: the shortbook app's read-only feed over Fly's private network. A failure keeps the
        last good data and marks it stale, so the page can say so."""
        url = self.cfg.shortbook_feed_url
        if not url:
            return
        try:
            if get is None:
                import httpx
                resp = httpx.get(url, timeout=SHORTS_TIMEOUT_S)
                resp.raise_for_status()
                data = resp.json()
            else:
                data = get(url)
            data = {**data, "fetched_ts": time.time(), "stale": False}
        except Exception as exc:  # noqa: BLE001 (the short book is another app: it may be down or redeploying)
            log.warning("dashboard: short book feed failed: %s", exc)
            data = {**(self.shorts or {}), "stale": True, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        with self._lock:
            self.shorts, self.shorts_ver = data, self.shorts_ver + 1

    def fetch_market(self) -> None:
        """The market monitor's hourly rows on a read-only connection of its own (an in-memory store in tests)."""
        from tradingbot import flag_tracker
        from tradingbot import market_monitor as mm
        try:
            path = getattr(self.store, "path", None)
            if path and path != ":memory:":
                if self._mdb is None:
                    self._mdb = Live._open(path)
                query = lambda sql, p: self._mdb.execute(sql, p).fetchall()
            else:
                query = self.store.query
            now = time.time()
            data = mm.payload(mm.read(query, 0.0), now, since=self.cfg.market_monitor_since,
                              flags=mm.read_flags(query), status=mm.read_status(query),
                              tracker=flag_tracker.read(query))
        except Exception:  # noqa: BLE001 (keep the last good data)
            log.exception("dashboard: market monitor read failed")
            return
        with self._lock:
            self.market, self.market_ver = data, self.market_ver + 1

    def fetch_wallets(self, rpc=None) -> None:
        """Add SOL balances (one getMultipleAccounts for every address) and tx counts (getSignaturesForAddress for
        the live set only, to bound the calls) to the wallet directory, over the public RPC so no Helius credits are
        spent. A failure keeps the last good data and marks it stale."""
        rpc = rpc or (lambda m, p: _solana_rpc(WALLETS_RPC, m, p))
        try:
            d = wallet_directory(self.cfg)
            addrs = [w["address"] for w in d["live"] + d["archive"]]
            sol = {}
            for i in range(0, len(addrs), 100):
                chunk = addrs[i:i + 100]
                res = rpc("getMultipleAccounts", [chunk, {"encoding": "base64"}]) or {}
                for a, v in zip(chunk, res.get("value", []) or []):
                    sol[a] = (v["lamports"] / 1e9) if v else 0.0
            for w in d["live"] + d["archive"]:
                w["sol"] = round(sol.get(w["address"], 0.0), 6)
            for w in d["live"]:  # tx count only for the live set (one call each), newest 1000
                try:
                    sigs = rpc("getSignaturesForAddress", [w["address"], {"limit": 1000}])
                    w["txs"] = len(sigs) if isinstance(sigs, list) else None
                except Exception:  # noqa: BLE001 (a per-wallet read may be rate-limited)
                    w["txs"] = None
            d["trading_sol"] = round(sum(w.get("sol") or 0 for w in d["live"] if w["role"] == "trading"), 6)
            d["wallets"] = sum(1 for w in d["live"] if w["role"] == "trading")
            d.update(fetched_ts=time.time(), stale=False)
        except Exception as exc:  # noqa: BLE001 (keep the last good data; the public RPC can be slow)
            log.warning("dashboard: wallet balances failed: %s", exc)
            d = {**(self.wallets or {"live": [], "archive": []}), "stale": True,
                 "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        with self._lock:
            self.wallets, self.wallets_ver = d, self.wallets_ver + 1

    def fetch_pools(self, rpc=None) -> None:
        """The pool tracker: every pool the canary has feinted, joined to the scorer's lifecycle row, plus a live
        reserve read for pools still young. Reads feints from the journal and the scorer rows from watch.db on the
        Live poller's read-only connections, and the reserves over the public RPC (no Helius credits). Keeps the
        last good data and marks it stale on a failure."""
        from tradingbot import pool_tracker as pt
        rpc = rpc or (lambda m, p: _solana_rpc(WALLETS_RPC, m, p))
        try:
            now = time.time()
            since = now - pt.TRACK_WINDOW_S
            feints = pt.feint_pools(self.live._q, since)
            wq = (lambda sql, p: self.live.wdb.execute(sql, p).fetchall()) if self.live.wdb is not None \
                else (lambda sql, p: [])
            scorer = pt.scorer_rows(wq, since)
            su = pt._sol_usd(wq)
            # The live reserve read is BEST-EFFORT: a slow / rate-limited public RPC must never hide the pools
            # themselves (feints + probes come from the journal + watch.db, no RPC) -- it only fills the live liq /
            # conf / best-entry columns. Before this, a thrown read failed the whole poll, so the tab froze on stale
            # rows while new probes never appeared (the CLI --no-live and the probe card kept working).
            live_ok = True
            try:
                young = [scorer[pl] for pl in feints
                         if scorer.get(pl) and scorer[pl].get("create_ts")
                         and now - scorer[pl]["create_ts"] <= pt.YOUNG_MAX_AGE_S and not scorer[pl].get("pulled_ts")]
                if young:
                    v = pt.read_vaults(rpc, young)
                    for pl, (sol, tok) in v.items():
                        s = scorer[pl]
                        self._pool_paths[pl] = pt.update_path(self._pool_paths.get(pl), now, s["create_ts"],
                                                              feints[pl]["feint_ts"], sol, tok, su, s.get("exit_px"))
            except Exception as exc:  # noqa: BLE001 (the public RPC can be slow / rate-limited)
                live_ok = False
                log.warning("dashboard: pool tracker live reserve read failed (pools still shown): %s", exc)
            cut = now - pt.YOUNG_MAX_AGE_S - 1800  # drop paths for pools long past their window
            for pl in [pl for pl in self._pool_paths if pl not in feints
                       or (scorer.get(pl) and (scorer[pl].get("create_ts") or 0) < cut)]:
                self._pool_paths.pop(pl, None)
            rows = pt.assemble(feints, scorer, su, self._pool_paths, now)
            data = {"rows": rows[:200], "summary": pt.summary(rows), "sol_usd": su,
                    "fetched_ts": now, "stale": False, "live_read_ok": live_ok}
        except Exception as exc:  # noqa: BLE001 (keep the last good data; the public RPC can be slow)
            log.warning("dashboard: pool tracker failed: %s", exc)
            data = {**(self.pools or {"rows": [], "summary": {}}), "stale": True,
                    "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        finally:
            # persist to /data on EVERY poll, even when the live read / assemble above threw: a slow or rate-limited
            # public RPC must never stop the reserve samples reaching disk, or a fly deploy wipes them.
            self._save_pool_paths()
        with self._lock:
            self.pools, self.pools_ver = data, self.pools_ver + 1

    def current(self) -> tuple[int, dict | None]:
        with self._lock:
            return self.version, self.summary

    def stream(self, write, stop: threading.Event | None = None) -> None:
        """Write server-sent events to `write(bytes)` until it raises (the page closed) or `stop` is set.

        `init` (every position of the running books + the scorer's pools of the last day), `summary` (the
        slow extras), `backlog` (recent fills), then every POLL_S: `delta` (rows that changed), `fill` (new
        buys and sells) and `summary` when rebuilt. The page computes the tables from the rows, so they move
        with every trade. All of it comes from the shared `Live` poller: a page costs no database reads."""
        def send(event: str, data) -> None:
            write(f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode())

        stop = stop or threading.Event()
        live = self.live
        if live.ver == 0:
            live.poll()
        ver, init = live.snapshot()
        send("init", init)
        version, summary = self.current()
        if summary is not None:
            send("summary", summary)
        backlog = live.fills[-FEED_BACKLOG:]
        send("backlog", backlog)
        send("feintlog", live.feints)
        last_feint = live.feints[-1]["id"] if live.feints else live.last_feint
        shorts_ver = 0
        if self.cfg.shortbook_feed_url and self.shorts is not None:
            send("shorts", self.shorts)
            shorts_ver = self.shorts_ver
        market_ver = 0
        if self.market is not None:
            send("market", self.market)
            market_ver = self.market_ver
        wallets_ver = 0
        if self.wallets is not None:
            send("wallets", self.wallets)
            wallets_ver = self.wallets_ver
        pools_ver = 0
        if self.pools is not None:
            send("pools", self.pools)
            pools_ver = self.pools_ver
        last_fill = backlog[-1]["id"] if backlog else live.last_fill
        quiet = time.time()
        while not stop.is_set():
            if self._external:  # no poller thread (tests, or start() not called): poll here
                live.poll()
            sent = False
            new_ver, delta = live.delta(ver)
            if delta:
                send("delta", delta)
                sent = True
            ver = new_ver
            for f in [f for f in live.fills if f["id"] > last_fill]:
                send("fill", f)
                last_fill = f["id"]
                sent = True
            for ft in [ft for ft in live.feints if ft["id"] > last_feint]:
                send("feint", ft)
                last_feint = ft["id"]
                sent = True
            v, summary = self.current()
            if v != version and summary is not None:
                send("summary", summary)
                version = v
                sent = True
            if self.shorts_ver != shorts_ver and self.shorts is not None:
                send("shorts", self.shorts)
                shorts_ver = self.shorts_ver
                sent = True
            if self.market_ver != market_ver and self.market is not None:
                send("market", self.market)
                market_ver = self.market_ver
                sent = True
            if self.wallets_ver != wallets_ver and self.wallets is not None:
                send("wallets", self.wallets)
                wallets_ver = self.wallets_ver
                sent = True
            if self.pools_ver != pools_ver and self.pools is not None:
                send("pools", self.pools)
                pools_ver = self.pools_ver
                sent = True
            if sent:
                quiet = time.time()
            elif time.time() - quiet > KEEPALIVE_S:
                write(b": keep-alive\n\n")
                quiet = time.time()
            stop.wait(POLL_S)


POS_COLS = ("id", "strategy", "symbol", "token", "cost_usd", "realized_usd", "qty", "entry_price", "opened_at",
            "closed_at", "exit_reason")
LAUNCH_COLS = ("pool", "token_mint", "create_ts", "decided_ts", "liquid", "ret", "status", "pulled_ts", "liq_usd",
               "top2", "top5", "top2h", "top5h", "low_h", "risk_pct", "rw_pct", "rw_low_pct", "rw_low6",
               "rw24_pct", "rw24_low_pct", "rwb_pct", "big_pct")
LAUNCH_KEEP_S = 86400  # the scorer's pools sent to the page (it windows them 1/6/24 h)
STUCK_AFTER_S = 0  # as engine.STUCK_AFTER_S: a position with no route out counts as $0 at once


FAMILIES = (  # book families for the dashboard's family view, in display order
    ("Canary", "real money: launchrw50_sz picks bought for real at a fraction of the twin's band sizes, judged on swap prices against the twin"),
    ("Launch picks", "the second-4 launch scorer's top 2% / 5% picks"),
    ("Low decile", "the history model's bottom 20% (D1-D2), minus the riskiest early pullers"),
    ("Rug-vs-win", "skip the riskiest half by P(rug before the exit) - P(+50% win); study rule, 24 h rule, a size-by-band twin, by-band rules, longer holds in $25-40k pools and the +50% filter held to minute 15 in $40-80k pools"),
    ("Thin band", "only liquid pools under $15k at second 4: every pool / sz's rw50 picks / the band rule, each at $100, $250 (2% of the pool at most) and 2% of the pool"),
    ("Early", "hand-made pools at first sight, out by pool minute 5-7"),
    ("model5", "the 5-minute snapshot model's short-hold books"))


def start_cash(cfg, name: str) -> float:
    """What a book started with: the canary has its own small budget, every other model book the same cash."""
    from tradingbot import model_book as mb
    return cfg.launch_canary_budget_usd if name in mb.CANARY_BOOKS else cfg.model_book_cash_usd


def family_of(name: str) -> str:
    from tradingbot import model_book as mb
    if name in mb.CANARY_BOOKS:
        return "Canary"
    if name in mb.LAUNCH_THIN_CUT:
        return "Thin band"
    if name in mb.LAUNCH_RW_CUT or name in mb.LAUNCH_RW24_CUT or name in mb.LAUNCH_SIZED_CUT or name in mb.LAUNCH_SIZED_DD_CUT or name in mb.LAUNCH_BAND_CUT \
            or name in mb.LAUNCH_HOLD_CUT or name in mb.LAUNCH_BIG_CUT:
        return "Rug-vs-win"
    if name in mb.LAUNCH_LOW_CUT:
        return "Low decile"
    if name in mb.LAUNCH_CUT:
        return "Launch picks"
    if name in mb.EARLY_BOOKS:
        return "Early"
    return "model5"


class Live:
    """Raw rows for the page, polled every POLL_S on read-only connections of their own (never the journal's
    lock, and sqlite releases the GIL while it reads). Each changed row gets a new version number, so every
    open page is sent only what changed since its last version."""

    def __init__(self, cfg: Config, journal: Journal | None = None, store=None):
        from tradingbot import model_book as mb
        self.cfg = cfg
        self.books = list(mb.active_books(cfg))
        self.flag_of = {b: mb.LAUNCH_CUT.get(b) for b in self.books}
        self.canary = set(mb.CANARY_BOOKS)
        self.twin = {b: mb.CANARY_TWIN if b in mb.CANARY_BOOKS else None for b in self.books}
        self.holds = {b: mb.hold_min(cfg, b) for b in self.books}
        self.journal = journal
        self.jdb = self._open(cfg.db_path) if cfg.db_path != ":memory:" else None
        path = getattr(store, "path", None)
        if path and path != ":memory:" and os.path.exists(path):
            self.wdb = self._open(path)
        else:  # in-memory store (tests): its own connection
            self.wdb = getattr(store, "db", None)
        self.lock = threading.Lock()
        self.ver = 0
        self.pos: dict[int, tuple[int, list]] = {}
        self.launch: dict[str, tuple[int, list]] = {}
        self.blocked: dict[tuple, list[float]] = {}
        self.fills: list[dict] = []
        self.last_fill = 0
        self.feints: list[dict] = []
        self.last_feint = 0
        self.last_event = None
        self.health: dict | None = None
        self.health_ver = 0
        self._health_ts = 0.0

    @staticmethod
    def _open(path: str):
        import sqlite3
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2, check_same_thread=False)
        db.row_factory = sqlite3.Row
        return db

    def _q(self, sql: str, params: tuple = ()) -> list:
        if self.jdb is not None:
            return self.jdb.execute(sql, params).fetchall()
        return self.journal.query(sql, params) if self.journal is not None else []

    def query(self, sql: str, params: tuple = ()) -> list:  # lets fills_after read through this connection
        return self._q(sql, params)

    def _bump(self, table: dict, key, row: list) -> None:
        old = table.get(key)
        if old is None or old[1] != row:
            self.ver += 1
            table[key] = (self.ver, row)

    def poll(self) -> None:
        try:
            with self.lock:
                self._poll_events()
                self._poll_positions()
                self._poll_fills()
                self._poll_feints()
                self._poll_launch()
                if time.time() - self._health_ts > 10 and self.journal is not None:
                    from tradingbot import health
                    h = health.status(self.journal, self.cfg)[1]
                    self._health_ts = time.time()
                    if h != self.health:
                        self.health, self.health_ver = h, self.ver + 1
                        self.ver += 1
        except Exception:  # noqa: BLE001 (a failed read is retried next poll)
            log.exception("dashboard live poll failed")

    def _poll_events(self) -> None:
        if self.last_event is None:  # first pass: the past week's blocked exits, then only new events by id
            top = self._q("SELECT MAX(id) AS m FROM events")
            rows = self._q("SELECT strategy, token, ts FROM events WHERE kind='exit_blocked' AND ts > ?",
                           (time.time() - 7 * 86400,))
            self.last_event = (top[0]["m"] if top else None) or 0
        else:
            rows = self._q("SELECT id, strategy, token, ts FROM events WHERE id > ? AND kind='exit_blocked'",
                           (self.last_event,))
            top = self._q("SELECT MAX(id) AS m FROM events")
            self.last_event = (top[0]["m"] if top else None) or self.last_event
        for r in rows:
            self.blocked.setdefault((r["strategy"], r["token"]), []).append(r["ts"])

    def _blocked_since(self, book: str, token: str, opened: float | None) -> float | None:
        ts = [t for t in self.blocked.get((book, token), ()) if t >= (opened or 0)]
        return min(ts) if ts else None

    def _poll_positions(self) -> None:
        if not self.books:
            return
        marks = ",".join("?" * len(self.books))
        top = max(self.pos) if self.pos else 0
        recent = time.time() - 60
        rows = self._q(f"SELECT {', '.join(POS_COLS)} FROM positions WHERE strategy IN ({marks}) "
                       f"AND (id > ? OR closed_at IS NULL OR closed_at > ?)", (*self.books, top, recent))
        for r in rows:
            row = [r[c] for c in POS_COLS]
            row.append(self._blocked_since(r["strategy"], r["token"], r["opened_at"]) if r["closed_at"] is None
                       else None)
            row.append(self._swap(r["id"]) if r["strategy"] in self.canary else None)
            self._bump(self.pos, r["id"], row)

    def _swap(self, pid: int) -> float | None:
        """The canary's result from its fill prices alone: a 1-cent trade's net result is all fees and rent."""
        from tradingbot.canary_rugs import swap_return
        ret = swap_return(self._q("SELECT side, price, qty FROM fills WHERE position_id=? ORDER BY ts", (pid,)))
        return None if ret is None else round(ret, 5)

    def _poll_fills(self) -> None:
        if not self.fills and not self.last_fill:
            top = self._q("SELECT MAX(id) AS m FROM fills")
            newest = (top[0]["m"] if top else None) or 0
            self.fills = fills_after(self, max(0, newest - 5 * FEED_BACKLOG), 5 * FEED_BACKLOG)[-FEED_BACKLOG:]
            self.last_fill = self.fills[-1]["id"] if self.fills else newest
            return
        new = fills_after(self, self.last_fill)
        if new:
            self.fills = (self.fills + new)[-300:]
            self.last_fill = new[-1]["id"]

    def _poll_feints(self) -> None:
        if not self.last_feint:
            self.feints = last_feints(self, 300)
            if self.feints:
                self.last_feint = self.feints[-1]["id"]
            else:  # no feints yet: start after the newest event so later feints are caught incrementally
                top = self._q("SELECT MAX(id) AS m FROM events")
                self.last_feint = (top[0]["m"] if top else None) or 0
            return
        new = feints_after(self, self.last_feint)
        if new:
            self.feints = (self.feints + new)[-500:]
            self.last_feint = new[-1]["id"]

    def _poll_launch(self) -> None:
        if self.wdb is None:
            return
        from tradingbot import launch_live
        cols = {r[1] for r in self.wdb.execute("PRAGMA table_info(watch_launch_live)")}
        if not cols:
            return
        sel = ", ".join(c if c in cols else f"NULL AS {c}" for c in LAUNCH_COLS)
        rows = self.wdb.execute(f"SELECT {sel} FROM watch_launch_live WHERE decided_ts > ?",
                                (time.time() - LAUNCH_KEEP_S,)).fetchall()
        for r in rows:
            flags = [k for k in ("top2", "top5", "top2h", "top5h") if r[k]]
            for fl in (launch_live.low_flags(r["low_h"], r["risk_pct"]),
                       launch_live.rw_flags(launch_live.rw_low(r), r["rw_pct"], r["rw_low_pct"]),
                       launch_live.rw24_flags(r["low_h"], r["rw24_pct"], r["rw24_low_pct"]),
                       launch_live.band_flags(r["liq_usd"], r["rw_pct"], r["rwb_pct"]),
                       launch_live.big_flags(r["liq_usd"], r["big_pct"]),
                       launch_live.thin_flags(r["liq_usd"], r["rw_pct"], r["rwb_pct"], r["liquid"])):
                flags += [k for k, on in fl.items() if on]
            ret = r["ret"]
            self._bump(self.launch, r["pool"], [r["pool"], r["token_mint"], r["create_ts"], r["decided_ts"],
                                                1 if r["liquid"] else 0, None if ret is None else round(ret, 4),
                                                r["status"], r["pulled_ts"], r["liq_usd"], flags])
        cut = time.time() - LAUNCH_KEEP_S - 3600
        for k in [k for k, (_, row) in self.launch.items() if (row[3] or 0) < cut]:
            del self.launch[k]

    def snapshot(self) -> tuple[int, dict]:
        with self.lock:
            return self.ver, {
                "now": time.time(), "cash": self.cfg.model_book_cash_usd, "stuck_after_s": STUCK_AFTER_S,
                "shorts_enabled": bool(self.cfg.shortbook_feed_url),
                "market_enabled": bool(self.cfg.market_monitor),
                "books": [{"name": b, "hold_min": self.holds[b], "flag": self.flag_of[b], "family": family_of(b),
                           "cash": start_cash(self.cfg, b), "twin": self.twin[b]}
                          for b in self.books],
                "families": [{"name": n, "about": a} for n, a in FAMILIES],
                "pos_cols": [*POS_COLS, "blocked_since", "swap_ret"], "positions": [row for _, row in self.pos.values()],
                "launch_cols": ["pool", "token", "create_ts", "decided_ts", "liquid", "ret", "status",
                                "pulled_ts", "liq_usd", "flags"],
                "launch": [row for _, row in self.launch.values()], "launch_keep_s": LAUNCH_KEEP_S,
                "canary": bool(self.canary & set(self.books)), "canary_halted": self._canary_halted(),
                "health": self.health}

    def _canary_halted(self) -> bool:
        from tradingbot.engine import CANARY_HALT
        return bool(self._q("SELECT 1 FROM state WHERE key=? LIMIT 1", (CANARY_HALT,)))

    def delta(self, since: int) -> tuple[int, dict | None]:
        with self.lock:
            if self.ver == since:
                return since, None
            out = {"now": time.time(),
                   "positions": [row for v, row in self.pos.values() if v > since],
                   "launch": [row for v, row in self.launch.values() if v > since]}
            if self.health_ver > since:
                out["health"] = self.health
            return self.ver, out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MZFq Quant Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800;900&display=swap" rel="stylesheet">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' fill='%230A0A0A'/%3E%3Cpolygon points='24,26 76,26 76,39 38,61 76,61 76,74 24,74 24,61 62,39 24,39' fill='%23C8FF4D'/%3E%3C/svg%3E">
<style>
/* MZF Protocol Design System, direction Signal: ink first, lime as a fill only, 2px rules, square corners. */
:root{--ground-app:#f2f2f0;--surface:#fff;--sunken:#f2f2f0;--band:#0a0a0a;--ink:#0a0a0a;--ink-2:#4a4a4a;--ink-3:#6b6b6b;
--on-band:#fff;--on-band-2:#b5b5b5;--lime:#c8ff4d;--lime-hover:#b5f22e;--lime-tint:#eeffc2;--on-lime:#0a0a0a;--rule:#0a0a0a;
--rule-soft:#d9d9d6;--pos:#157347;--neg:#c62828;--warn:#b45309;--focus:#0a0a0a;
--sans:"Archivo",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--ground-app:#121212;--surface:#161616;--sunken:#1e1e1e;--band:#f2f2f0;--ink:#fff;
--ink-2:#b5b5b5;--ink-3:#8a8a8a;--on-band:#0a0a0a;--on-band-2:#4a4a4a;--lime-tint:#2a3313;--rule:#fff;--rule-soft:#2e2e2e;
--pos:#5cd68a;--neg:#ff6b6b;--warn:#f5a524;--focus:#c8ff4d}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground-app);color:var(--ink);font:500 16px/1.5 var(--sans)}
.num{font-variant-numeric:tabular-nums}
.label{font-size:12px;line-height:1.2;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.mono{font:500 13px/1.4 var(--mono)}
header{display:flex;flex-wrap:wrap;gap:16px;align-items:center;justify-content:space-between;padding:20px 32px;
background:var(--surface);border-bottom:2px solid var(--rule)}
.lockup{display:flex;align-items:center;gap:12px}
.lockup svg{width:40px;height:40px;flex:none}.mk-sq{fill:#0a0a0a}.mk-z{fill:#c8ff4d}
@media (prefers-color-scheme:dark){.mk-sq{fill:#c8ff4d}.mk-z{fill:#0a0a0a}}
.word{font-weight:900;font-size:26px;letter-spacing:-.04em;line-height:1}
.desc{font-weight:700;font-size:11px;letter-spacing:.22em;line-height:1;margin-top:4px}
.status{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.tag{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border:2px solid var(--rule);background:var(--surface);
color:var(--ink);font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}
.tag.lime{background:var(--lime);color:var(--on-lime);border-color:var(--rule)}
.tag.ink{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.tag.neg{color:var(--neg);border-color:var(--neg)}
.dot{width:8px;height:8px;background:currentColor}
main{max-width:1360px;margin:0 auto;padding:32px;display:grid;gap:24px}
main[hidden],[hidden]{display:none!important}
h1{margin:0;font-size:30px;line-height:1.05;font-weight:800;letter-spacing:-.02em}
.lead{margin:6px 0 0;color:var(--ink-2);font-size:14px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.stat{padding:20px;border:2px solid var(--rule);background:var(--surface);display:flex;flex-direction:column;gap:8px;min-width:0}
.stat .label{color:var(--ink-2)}
.stat .v{font-size:36px;line-height:1;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.stat .h{font-size:12px;line-height:1.4;font-weight:600;color:var(--ink-2)}
.stat.lime{background:var(--lime);color:var(--on-lime)}.stat.lime .label,.stat.lime .h{color:var(--on-lime)}
.cols{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:24px;align-items:start}
@media (max-width:1100px){.cols{grid-template-columns:minmax(0,1fr)}}
.card{border:2px solid var(--rule);background:var(--surface);padding:28px;min-width:0}
.card h2{margin:0 0 4px;font-size:24px;line-height:1.15;font-weight:800;letter-spacing:-.02em}
.card .sub{margin:0 0 16px;color:var(--ink-2);font-size:14px}
.eyebrow{color:var(--ink-2);margin:0 0 6px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th{font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);text-align:right;
padding:8px 9px;border-bottom:2px solid var(--rule);white-space:nowrap;cursor:pointer;user-select:none}
td{padding:10px 9px;text-align:right;border-bottom:1px solid var(--rule-soft);white-space:nowrap}
th:first-child,td:first-child{text-align:left;padding-left:0}th:last-child,td:last-child{padding-right:0}
tr.lead-row td{background:var(--lime-tint)}
.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--ink-3)}
.book{font-weight:700}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:24px}
@media (max-width:520px){.two{grid-template-columns:minmax(0,1fr)}main{padding:16px}header{padding:16px}.card{padding:20px}.stat .v{font-size:30px}}
/* live feed: an inverted band */
.feed{background:var(--band);color:var(--on-band);border:2px solid var(--band);padding:28px}
.feed h2{margin:0;font-size:24px;line-height:1.15;font-weight:800;letter-spacing:-.02em}
.feed .sub{margin:4px 0 16px;color:var(--on-band-2);font-size:14px}
.feed ul{list-style:none;margin:0;padding:0;max-height:72vh;overflow-y:auto}
.feed li{display:grid;grid-template-columns:64px 1fr auto;gap:8px;align-items:center;padding:10px 0;
border-bottom:1px solid color-mix(in srgb,var(--on-band) 18%,transparent);font-size:14px}
.feed li.new{animation:flash 2.4s ease-out}
@keyframes flash{from{background:var(--lime);color:var(--on-lime)}to{background:transparent}}
.feed .t{color:var(--on-band-2)}
.feed .bk{display:block;color:var(--on-band-2);font-size:12px;font-weight:600}
.side{font-size:11px;font-weight:800;letter-spacing:.06em;padding:2px 6px;margin-right:6px;border:2px solid var(--on-band)}
.side.buy{background:var(--lime);color:var(--on-lime);border-color:var(--lime)}
.r{font-weight:800;font-variant-numeric:tabular-nums}
.feed .pos{color:#5cd68a}.feed .neg{color:#ff6b6b}
@media (prefers-color-scheme:dark){.feed .pos{color:#157347}.feed .neg{color:#c62828}}
footer{max-width:1360px;margin:0 auto;padding:0 32px 32px;color:var(--ink-3);font-size:12px;font-weight:600}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.win{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.win .label{margin-right:4px}
.win button{min-height:36px;padding:0 12px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:700 13px/1 inherit;font-family:inherit;cursor:pointer;border-radius:0}
.win button.on{background:var(--lime);color:var(--on-lime)}.win button:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.tabs{display:flex;gap:0;flex-wrap:wrap;background:var(--surface);border-bottom:2px solid var(--rule);padding:0 32px}
.tabs a{display:inline-flex;align-items:center;min-height:44px;padding:0 16px;font-size:14px;font-weight:700;color:var(--ink);
text-decoration:none;border-right:2px solid var(--rule)}.tabs a:first-child{border-left:2px solid var(--rule)}
.tabs a.on{background:var(--lime);color:var(--on-lime)}.tabs a:hover:not(.on){background:var(--sunken)}
@media (max-width:520px){.tabs{padding:0 16px}.tabs a{padding:0 10px;font-size:13px}}
a.lk{color:inherit;text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px}
a.lk:hover{text-decoration-thickness:2px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px;align-items:center}
.filters select,.filters input,.more button{min-height:44px;padding:0 12px;border:2px solid var(--rule);background:var(--surface);
color:var(--ink);font:600 14px/1 var(--sans);border-radius:0}
.filters input{min-width:180px}.more{margin:16px 0 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.more button{cursor:pointer;font-weight:700}.more button:hover{background:var(--sunken)}
.tag.warn{color:var(--warn);border-color:var(--warn)}
.tabs.sub{background:var(--sunken)}.tabs.sub a{min-height:40px;font-size:13px}
.progs{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}
@media (max-width:520px){.progs{grid-template-columns:minmax(0,1fr)}}
.prog h2{font-size:30px!important;line-height:1.05!important}.prog h2 a{color:inherit;text-decoration:none}.prog h2 a:hover{text-decoration:underline}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin:12px 0 20px}
.kv div{border:2px solid var(--rule);padding:12px;display:flex;flex-direction:column;gap:4px;min-width:0}
.kv b{font-size:22px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.kv span{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}
.note{margin:0 0 16px;padding:12px 16px;border:2px solid var(--warn);color:var(--warn);font-size:14px;font-weight:600}
th.l,td.l{text-align:left}.stat .v.nm{font-size:26px;letter-spacing:-.02em}.tag.sm{font-size:11px;padding:2px 6px}
.fams{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}
.fams .card h2 a{color:inherit;text-decoration:none}.fams .card h2 a:hover{text-decoration:underline}
.mini td,.mini th{padding:7px 6px}
.chart svg{width:100%;height:260px;display:block}
.chart .ax{font:600 11px var(--sans);fill:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:10px 0 0;font-size:13px;font-weight:600}
.legend span{display:inline-flex;align-items:center;gap:6px}.legend svg{width:28px;height:8px}
</style></head><body>
<header>
<div class="lockup"><svg viewBox="0 0 100 100" aria-hidden="true"><rect class="mk-sq" width="100" height="100"/><polygon class="mk-z" points="24,26 76,26 76,39 38,61 76,61 76,74 24,74 24,61 62,39 24,39"/></svg>
<div><div class="word">MZFq</div><div class="desc">QUANT DESK</div></div></div>
<div class="status"><span id="conn" class="tag"><span class="dot"></span>Connecting</span><span id="health" class="tag">Bot</span>
<span id="built" class="label muted"></span>
<button id="signout" onclick="signOut()" title="Sign out of this device" style="padding:4px 10px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);cursor:pointer;font:inherit;font-weight:700;font-size:12px">Sign out</button></div>
</header>
<nav class="tabs" aria-label="Programs"><a href="#/" data-v="programs">MZFq</a><a href="#/supermeme" data-v="supermeme" id="tab-meme">Supermeme</a><a href="#/shorts" data-v="shorts">MZF Short Only</a></nav>
<nav class="tabs sub" id="subtabs" aria-label="Supermeme"><a href="#/supermeme" data-v="overview">Overview</a><a href="#/ledger" data-v="ledger">Trade ledger</a><a href="#/books" data-v="books">Books &amp; families</a><a href="#/market" data-v="market">Market</a><a href="#/wallets" data-v="wallets">Wallets</a><a href="#/pools" data-v="pools">Pool tracker</a><a href="#/settings" data-v="settings">Canary</a></nav>
<main id="v-programs" hidden>
<div><p class="label eyebrow">Quant programs · paper trading</p><h1>MZFq</h1>
<p class="lead">Each program holds its strategies; Supermeme's are grouped into families of books. Everything here is paper trading.</p></div>
<div class="stats" id="p-stats"></div>
<div class="progs" id="progs"></div>
</main>
<main id="v-shorts" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/">MZFq</a> · program</p><h1>MZF Short Only</h1>
<p class="lead" id="sh-about">Short-only strategies on centralised-exchange perps.</p></div>
<div id="sh-note"></div>
<div class="stats" id="sh-stats"></div>
<section class="card"><p class="label eyebrow">Strategy</p><h2 id="sh-strat">Strategy 1: listing shorts</h2><p class="sub">Per group, against the research expectation per trade (after costs and funding). Judge after 30+ closed trades.</p><div class="scroll"><table id="sh-groups"></table></div></section>
<section class="card"><p class="label eyebrow">Leverage</p><h2>Return profiles at 1x, 2x, 3x, 4x</h2><p class="sub" id="sh-lsub">The book's own shorts scaled by L (a short already at 2% of open interest can't grow), isolated margin, liquidated when the highest price seen goes 1/L − 2.5% above entry; profits not reinvested. Backtest column: half the historical edge, $1M.</p>
<div class="scroll"><table id="sh-lev"></table></div><div id="sh-levchart"></div></section>
<section class="card"><p class="label eyebrow">Market</p><h2>Picks against the market</h2><p class="sub" id="sh-msub">Every listing that traded, on one yardstick: spot price 5 minutes after the first trade to each horizon, 0.2% round trip, no funding, a short floored at -100%. Picks = the listings the book shorted.</p>
<div class="filters"><select id="sh-mh" aria-label="Horizon"></select></div><div class="scroll"><table id="sh-market"></table></div><p class="sub" id="sh-mgap"></p></section>
<section class="card chart"><p class="label eyebrow">Equity</p><h2>Net asset value</h2><p class="sub">Hourly NAV (cash + open shorts marked to market, funding and fees included). Dashed: the starting capital.</p><div id="sh-chart"></div></section>
<section class="card"><p class="label eyebrow">Positions</p><h2>Open shorts</h2><div class="scroll"><table id="sh-open"></table></div></section>
<section class="card"><p class="label eyebrow">Positions</p><h2>Closed shorts</h2><div class="scroll"><table id="sh-closed"></table></div></section>
<section class="card"><p class="label eyebrow">Detector</p><h2>Recent listings</h2><p class="sub" id="sh-lcount"></p><div class="scroll"><table id="sh-listings"></table></div></section>
</main>
<main id="v-overview">
<div><p class="label eyebrow"><a class="lk" href="#/">MZFq</a> · Supermeme · paper books on Solana launches, simulated fills on live Jupiter quotes</p><h1>Supermeme</h1>
<p class="lead">Every book starts at $10,000. Trades and every table update live, within two seconds of a trade.</p></div>
<div class="stats">
<div class="stat lime"><span class="label">Leading book</span><span class="v nm" id="s-lead">-</span><span class="h" id="s-lead-h"></span></div>
<div class="stat"><span class="label">P&amp;L, all books</span><span class="v num" id="s-pnl">-</span><span class="h" id="s-pnl-h"></span></div>
<div class="stat"><span class="label">Closed trades</span><span class="v num" id="s-trades">-</span><span class="h" id="s-trades-h"></span></div>
<div class="stat"><span class="label">Trades, last hour</span><span class="v num" id="s-rate">0</span><span class="h">Closed in the past 60 minutes</span></div>
</div>
<section class="card" id="feint-card" hidden><p class="label eyebrow">Launch canary · real money</p><h2>Feint probes <span id="feint-switch"></span></h2>
<p class="sub">Every liquid pool gets a $25 buy-and-sell-back probe that keeps nothing. <b>alive</b> = liquidity held, <b>pulled</b> = the operator withdrew in reaction, <b>inconclusive</b> = the probe could not be placed (the note says why; "wallet SOL low" means a wallet needs topping up). Stop the run if pulled climbs, wallets run low, or feints stall.</p>
<div class="stats" id="feint-tiles"></div>
<div class="two"><div class="scroll"><table id="feint-notes"></table></div><div class="scroll"><table id="feint-feed"></table></div></div></section>
<div class="cols">
<section class="card"><p class="label eyebrow">Books</p><h2>Results by book</h2>
<p class="sub">Sorted by return. Click a header to sort. Open trades count at cost until sold; a trade with no route out counts as $0 until a later sale gets through. Rugged = closed at −80% or worse.</p>
<div class="scroll"><table id="books"></table></div></section>
<aside class="feed"><p class="label" style="color:var(--on-band-2);margin:0 0 6px">Live</p><h2>Trades</h2>
<p class="sub" id="feed-sub">Newest first</p><ul id="feed"></ul></aside>
</div>
<div class="win" id="win"><span class="label">Launch window</span><button data-w="3600">1 h</button><button data-w="21600" class="on">6 h</button><button data-w="86400">24 h</button></div>
<div class="two">
<section class="card"><p class="label eyebrow">Launch scorer · last <span id="lh">6</span> h</p><h2>Ideal trades</h2>
<p class="sub">Pool price at second ~6.5 in, minute 6 out, 6% cost. The yardstick the books are measured against.</p>
<div class="scroll"><table id="outcomes"></table></div></section>
<section class="card"><p class="label eyebrow">Execution · last <span id="lh2">6</span> h</p><h2>Real against ideal</h2>
<p class="sub">Each book's fills against the ideal trade on the same pools, and the signals it did not fill.</p>
<div class="scroll"><table id="gaps"></table></div></section>
</div>
</main>
<main id="v-market" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/supermeme">Supermeme</a> · meme market monitor · hand-made pumpswap pools</p><h1>Market</h1>
<p class="lead">How the launch market is behaving, hour by hour, against the past week: who is launching, how fast they pull, how much buying arrives, and whether our edge is holding. Ideal = the scorer's prices (second ~6.5 in, minute 6 out, 6% cost).</p></div>
<div id="mk-note"></div>
<div class="stats" id="mk-stats"></div>
<section class="card"><p class="label eyebrow">Shifts</p><h2>Moved away from the past week</h2><p class="sub" id="mk-shift-sub"></p><div class="scroll"><table id="mk-shifts"></table></div></section>
<section class="card"><p class="label eyebrow">Flags</p><h2>Saved flags</h2><p class="sub" id="mk-flag-sub"></p><div class="scroll"><table id="mk-flags"></table></div>
<h2 style="margin-top:24px">What the flags meant for sz</h2><p class="sub">Since sz's first trade: sz (size-weighted) and the market (ideal mean) in each measure's flagged hours against every other hour. A big negative gap means that kind of shift hurts sz.</p><div class="scroll"><table id="mk-flagsum"></table></div></section>
<section class="card"><p class="label eyebrow">Flag tracker</p><h2>Can a flag drive a switch?</h2><p class="sub" id="mk-trk-sub"></p><div class="scroll"><table id="mk-trk-pass"></table></div><div class="scroll"><table id="mk-trk-books"></table></div><h2 style="margin-top:24px" id="mk-trk-h"></h2><div class="scroll"><table id="mk-trk"></table></div></section>
<div class="two" id="mk-charts"></div>
<section class="card"><p class="label eyebrow">Every measure</p><h2>Last hours against the past week</h2><p class="sub" id="mk-cmp-sub"></p><div class="scroll"><table id="mk-cmp"></table></div></section>
<section class="card"><p class="label eyebrow">Pool size</p><h2>Rug-vs-win by pool size</h2><p class="sub">Liquidity at second 4. Every pool = no filter: the scorer's ideal trade (second ~6.5 in, minute 6 out, 6% cost) on every liquid pool in the band, the yardstick. Rug-vs-win kept = the half the filter keeps (ranked across every band), same measures; vs every pool = kept mean minus every-pool mean. $ at $100 = running total if each pool had been bought at $100. sz = launchrw50_sz's real trades, size-weighted. Bands under $5k are measured only (no book trades them: below the $5k liquidity minimum, and the filter doesn't score them). The last two rows add up the traded bands ($5k+) and the measured-only bands.</p>
<div class="filters"><select id="mk-bwin" aria-label="Window"></select></div><div class="scroll"><table id="mk-bands"></table></div></section>
<section class="card"><p class="label eyebrow">Hourly</p><h2>Hour by hour</h2><p class="sub">Newest first, UTC. Liquid pools by liquidity at second 4 in sz's bands; sz = launchrw50_sz's real trades, size-weighted.</p><div id="mk-imp"></div><div class="scroll"><table id="mk-hours"></table></div></section>
</main>
<main id="v-wallets" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/supermeme">Supermeme</a> · launch canary · real money</p><h1>Wallets</h1>
<p class="lead">The live trading wallets, treasury and reserve with on-chain SOL balances and transaction counts (public RPC, refreshed each minute), and an archive of every wallet and treasury ever used. Addresses only — keys never reach the page.</p></div>
<div id="w-note"></div>
<div class="stats" id="w-tiles"></div>
<section class="card"><p class="label eyebrow">Live set</p><h2>Trading wallets, treasury, reserve</h2><p class="sub">Numbered 1..n in rotation order. SOL = current balance; Txs = signatures on the wallet (feints and trades both count; a fresh wallet funds at one tx, a direct-route rug drains it to ~0).</p><div class="scroll"><table id="w-live"></table></div></section>
<section class="card"><p class="label eyebrow">Top up · real money</p><h2>Top up trading wallets</h2>
<p class="sub">Fund the selected trading wallets up to a target SOL level from the <b>treasury</b> (pulling from the <b>reserve</b> if the treasury is short). Additive and safe while the canary trades — no halt, no restart. A wallet already at the target is skipped. Keys never appear here — only addresses and amounts.</p>
<div style="display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap;margin:4px 0 8px">
<label style="display:flex;flex-direction:column;gap:4px;font-weight:700">Target SOL each<input id="tu-level" type="number" min="0.01" max="5" step="0.01" value="0.08" style="width:120px;padding:6px 8px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:inherit"></label>
<button id="tu-go" onclick="canaryTopUp()" style="padding:8px 18px;border:2px solid #000;background:#c9f24d;color:#000;cursor:pointer;font:inherit;font-weight:800">Top up</button></div>
<div style="font-weight:700;font-size:13px;margin:0 0 4px">Which wallets</div>
<div id="tu-wallets" style="display:flex;gap:12px;flex-wrap:wrap;margin:0 0 8px"></div>
<pre id="tu-log" hidden style="white-space:pre-wrap;word-break:break-word;background:var(--sunken);border:2px solid var(--rule-soft);padding:10px;max-height:320px;overflow:auto;font:500 12px/1.5 var(--mono)"></pre></section>
<section class="card"><p class="label eyebrow">Rotate · real money</p><h2>Rotate trading wallets</h2>
<p class="sub">Mint a fresh set of wallets and a new treasury, sweep every current wallet and the treasury into it, and fund each new wallet (topping up from the reserve). This <b>halts the canary</b>; restart the bot (fly deploy) to trade the new set, then press Start on the overview. Keys never appear here — only addresses and the plan.</p>
<div style="display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap;margin:4px 0 10px">
<label style="display:flex;flex-direction:column;gap:4px;font-weight:700">Wallets<input id="rot-n" type="number" min="1" max="10" value="10" style="width:90px;padding:6px 8px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:inherit"></label>
<label style="display:flex;flex-direction:column;gap:4px;font-weight:700">SOL each<input id="rot-sol" type="number" min="0.01" max="5" step="0.01" value="0.2" style="width:110px;padding:6px 8px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:inherit"></label>
<button id="rot-go" onclick="canaryRotate()" style="padding:8px 18px;border:2px solid #000;background:#c9f24d;color:#000;cursor:pointer;font:inherit;font-weight:800">Rotate wallets</button></div>
<pre id="rot-log" hidden style="white-space:pre-wrap;word-break:break-word;background:var(--sunken);border:2px solid var(--rule-soft);padding:10px;max-height:320px;overflow:auto;font:500 12px/1.5 var(--mono)"></pre></section>
<section class="card"><p class="label eyebrow">Archive</p><h2>Retired wallets and treasuries</h2><p class="sub">Every other address the key-free address book has recorded, with its label. Old trading wallets and treasuries should read ~0 SOL once swept.</p><div class="scroll"><table id="w-arch"></table></div></section>
</main>
<main id="v-pools" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/supermeme">Supermeme</a> · launch canary · real money</p><h1>Pool tracker</h1>
<p class="lead">Every pool the canary has probed (feint or 1-slot), with its live state and the ~48s survival verdict. Liquidity, age and the LP-pull time come from the second-4 scorer; live liquidity and the best entry come from a public-RPC read of the pool's reserves (no Helius credits), refreshed every 20 seconds while a pool is under 25 minutes old.</p></div>
<div id="p-note"></div>
<div class="stats" id="p-tiles"></div>
<section class="card" id="p-hold-card" style="display:none"><p class="label eyebrow">Survivor hold curve</p><h2>Does holding a survivor past minute 6 pay?</h2><p class="sub">Enter a survivor at the ~48s confirmation, then sell at each horizon (last live price at or before it, −100% if it rugged first). <b>big</b> = share that hit +100% or more. If the mean and the big-win share keep rising out to 30–60 min, a long hold beats the minute-6 exit.</p>
<div class="scroll"><table id="p-hold"></table></div></section>
<section class="card"><p class="label eyebrow">Feinted pools</p><h2>Live pool state</h2><p class="sub">Newest pool first. <b>Probe@</b> = pool age when we feinted. <b>Liq</b> = liquidity at second 4; <b>Live</b> = latest reserve read; <b>Peak</b> = most liquidity seen after the probe. <b>Pulled@</b> = pool age when the LP was pulled, <b>+probe</b> = how long after our feint. <b>$25 @ m6</b> = what a $25 buy held to pool minute 6 would have returned (scorer trade, net of 6% cost). <b>Enter @conf</b> = entering a survivor when the ~48s watch clears it (price ~60s after creation), held to minute 6 — what a real two-stage entry would actually make, net of 6% cost, −100% if it later rugged. <b>Best entry</b> = the hindsight-cheapest price before minute 6 (the ceiling, not attainable live).</p>
<div class="filters"><select id="p-filter" aria-label="Filter"><option value="all">All pools</option><option value="survivor">Survivor</option><option value="rugger">Rugger</option><option value="alive">Alive at feint</option><option value="pulled">Pulled</option><option value="live">Still live (&lt;25m)</option><option value="win">$25 in profit</option></select></div>
<div class="scroll"><table id="p-tbl"></table></div></section>
</main>
<main id="v-settings" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/supermeme">Supermeme</a> · launch canary · real money</p><h1>Canary settings</h1>
<p class="lead">Preset modes set every setting at once; edit any field below. <b>Probe, sizing and limit</b> changes take effect immediately (the engine shares this config). <b>Route, budget and fees</b> apply after a restart (fly deploy). Every change is confirmed and saved so it survives a redeploy.</p></div>
<div id="cs-note"></div>
<section class="card"><p class="label eyebrow">Presets</p><h2>Modes</h2><p class="sub">Each applies a coherent set of settings. You are asked to confirm before anything changes.</p><div id="cs-presets" class="stats"></div></section>
<div id="cs-groups"></div>
<section class="card"><div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center"><button id="cs-save" onclick="canarySave()" style="padding:9px 20px;border:2px solid #000;background:#c9f24d;color:#000;cursor:pointer;font:inherit;font-weight:800">Save changes</button><button onclick="canaryReset()" style="padding:9px 20px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);cursor:pointer;font:inherit;font-weight:700">Reset to deploy settings</button><span class="sub" id="cs-dirty"></span></div></section>
<section class="card"><p class="label eyebrow">Deploy</p><h2>Deploy the bot</h2>
<p class="sub">Runs the deploy command on this machine (if one is configured) and restarts it — <b>this page will disconnect</b>; reload in a minute. Normally you deploy from your Mac with <code>git pull &amp;&amp; fly deploy</code>. Restart-only settings above (route, budget, fees) take effect on the next deploy.</p>
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:4px 0 10px"><button id="dep-go" onclick="flyDeploy()" style="padding:9px 20px;border:2px solid #000;background:#c9f24d;color:#000;cursor:pointer;font:inherit;font-weight:800">fly deploy</button><span class="sub mono" id="dep-cmd"></span></div>
<pre id="dep-log" hidden style="white-space:pre-wrap;word-break:break-word;background:var(--sunken);border:2px solid var(--rule-soft);padding:10px;max-height:320px;overflow:auto;font:500 12px/1.5 var(--mono)"></pre></section>
<section class="card"><p class="label eyebrow">1-slot probe · real money</p><h2>1-slot probe (measurement)</h2>
<p class="sub">Probe every liquid candidate: buy $25 so committed money lands, sell it back in the <b>next slot</b> (escapes the pull — every measured rug is ≥2 slots after the buy), watch ~48s and record <b>rugger</b> (tripped and dodged) or <b>survivor</b> (a pool to enter for real). Never commits. Enable with the <b>1-slot probe (measure)</b> preset or the Probe settings above (direct route only). <b>Smoke-test the next-slot sell at low cost first.</b></p>
<div id="pb-tiles" class="stats"></div>
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0 6px"><button id="pb-one" onclick="probeOne()" style="padding:9px 20px;border:2px solid #000;background:#c9f24d;color:#000;cursor:pointer;font:inherit;font-weight:800">Execute one probe</button><button id="pb-all" onclick="probeAll()" style="padding:9px 20px;border:2px solid #000;background:#c9f24d;color:#000;cursor:pointer;font:inherit;font-weight:800">Probe every liquid pool</button><span class="sub" id="pb-one-note">Execute one probe arms a single probe, then waits. Probe every liquid pool runs continuously.</span></div>
<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:6px 0"><input id="pb-mint" placeholder="token mint (deep pool)" style="padding:7px 9px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:inherit;width:220px"><input id="pb-pool" placeholder="pool (optional)" style="padding:7px 9px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:inherit;width:170px"><input id="pb-usd" type="number" value="1" min="0.1" max="25" step="any" style="padding:7px 9px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:inherit;width:80px"><button id="pb-smoke" onclick="probeSmoke()" style="padding:9px 20px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);cursor:pointer;font:inherit;font-weight:700">Smoke test</button><span class="sub" id="pb-smoke-note"></span></div>
<pre id="pb-smoke-out" hidden style="white-space:pre-wrap;word-break:break-word;background:var(--sunken);border:2px solid var(--rule-soft);padding:10px;max-height:240px;overflow:auto;font:500 12px/1.5 var(--mono)"></pre>
<div class="scroll"><table id="pb-recent"></table></div></section>
<section class="card"><p class="label eyebrow">Access</p><h2>Passkeys</h2>
<p class="sub">Sign in with FaceID, TouchID, Windows Hello or a security key instead of the shared link. Add a passkey on each device you use; the private key never leaves that device. The <code>?key=</code> link stays as a break-glass bootstrap and for recovery. Removing every passkey falls back to the link.</p>
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:4px 0 10px"><input id="pk-label" placeholder="e.g. iPhone, MacBook" maxlength="40" style="padding:7px 9px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:inherit;width:190px"><button id="pk-add" onclick="addPasskey()" style="padding:9px 20px;border:2px solid #000;background:#c9f24d;color:#000;cursor:pointer;font:inherit;font-weight:800">Add this device</button><span class="sub" id="pk-note"></span></div>
<div class="scroll"><table id="pk-list"></table></div></section>
</main>
<main id="v-ledger" hidden>
<div><p class="label eyebrow">Every trade of the running books</p><h1>Trade ledger</h1>
<p class="lead">Newest first. Open trades count at cost until sold; a sale with no route out counts as $0 until a later sale gets through.</p></div>
<section class="card"><div class="filters">
<select id="lf-fam" aria-label="Family"></select><select id="lf-book" aria-label="Book"></select>
<select id="lf-status" aria-label="Status"><option value="">Every status</option><option>Open</option><option>Exit blocked ($0)</option><option>Won</option><option>Lost</option></select>
<input id="lf-q" type="search" placeholder="Token or symbol" aria-label="Token or symbol"></div>
<div class="scroll"><table id="ledger"></table></div>
<p class="more"><button id="ledger-more">Show 200 more</button><span id="ledger-count" class="label muted"></span></p></section>
</main>
<main id="v-books" hidden>
<div><p class="label eyebrow">Paper books by family</p><h1>Books &amp; families</h1>
<p class="lead">A family is a group of books trading one idea with different settings. Open a family to compare its books, or a book for its own equity and trades.</p></div>
<div class="fams" id="fams"></div>
</main>
<main id="v-family" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/supermeme">Supermeme</a> · <a class="lk" href="#/books">Books &amp; families</a> · family</p><h1 id="f-name"></h1><p class="lead" id="f-about"></p></div>
<div class="stats" id="f-stats"></div>
<section class="card"><p class="label eyebrow">Books</p><h2>Books in this family</h2><p class="sub">Click a book to open it.</p><div class="scroll"><table id="f-books"></table></div></section>
<section class="card chart"><p class="label eyebrow">Equity</p><h2>Equity by book</h2><p class="sub">Each book's equity after every closed trade, from $10,000.</p><div id="f-chart"></div></section>
<section class="card"><p class="label eyebrow">Trades</p><h2>Family ledger</h2><div class="scroll"><table id="f-ledger"></table></div>
<p class="more"><button id="f-more">Show 200 more</button><span id="f-count" class="label muted"></span></p></section>
</main>
<main id="v-book" hidden>
<div><p class="label eyebrow"><a class="lk" href="#/supermeme">Supermeme</a> · <a class="lk" id="b-fam" href="#/books"></a></p><h1 id="b-name"></h1><p class="lead" id="b-about"></p></div>
<div class="stats" id="b-stats"></div>
<section class="card chart"><p class="label eyebrow">Equity</p><h2>Equity curve</h2><p class="sub">Equity after every closed trade, from $10,000. Dashed: the starting $10,000.</p><div id="b-chart"></div></section>
<section class="card" id="b-gap-card"><p class="label eyebrow">Execution · last <span id="b-lh">6</span> h (launch window on the overview)</p><h2>Real against ideal</h2>
<p class="sub">This book's fills against the scorer's ideal trade on the same pools, and the flagged pools it did not buy.</p><div class="scroll"><table id="b-gap"></table></div></section>
<section class="card" id="b-swap-card" hidden><p class="label eyebrow">Real money · fill prices only</p><h2>Swap result against the twin</h2>
<p class="sub">At a cent a trade the net result is all fees and account rent, so the canary is judged on what the pool paid: sell price over buy price. The twin is the paper book it copies, on the same tokens.</p>
<div class="stats" id="b-swap-stats"></div><div class="scroll"><table id="b-swap"></table></div></section>
<section class="card"><p class="label eyebrow">Trades</p><h2>Book ledger</h2><div class="scroll"><table id="b-ledger"></table></div>
<p class="more"><button id="b-more">Show 200 more</button><span id="b-count" class="label muted"></span></p></section>
</main>
<footer>MZFq · paper trading only, no funds at risk · <span id="foot"></span></footer>
<script>
const $=id=>document.getElementById(id);
const fmt=(v,d=1,s="")=>v==null||Number.isNaN(v)?"-":(typeof v==="number"?v.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}):v)+s;
const sgn=(v,d=1,s="%")=>v==null||Number.isNaN(v)?"-":(v>0?"+":v<0?"−":"")+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d})+s;
const cls=v=>v==null?"":v>0?"pos":v<0?"neg":"";
const esc=t=>String(t??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
/* mean trade weighted by size: total P&L / total cost (a plain average counts a $25 trade like a $250 one) */
const wmean=tr=>{const c=tr.reduce((a,x)=>a+(x.cost||0),0);return c?100*tr.reduce((a,x)=>a+x.pnl,0)/c:null};
const median=a=>{if(!a.length)return null;const b=[...a].sort((x,y)=>x-y),m=b.length>>1;return b.length%2?b[m]:(b[m-1]+b[m])/2};
const S={market:null,marketEnabled:false,shorts:null,shortsEnabled:false,pos:new Map(),launch:new Map(),books:[],fams:[],cash:10000,stuck:0,pc:[],lc:[],skew:0,health:null,extra:null,feints:[],canary:false,canaryHalted:false,wallets:null,pools:null,cset:null,csetLoading:false};
let sortKey="return_pct",sortDir=-1,win=21600,pending=false;
const R={v:"overview",arg:null},LIM={ledger:200,b:200,f:200};
const now=()=>Date.now()/1000-S.skew;
const obj=(cols,row)=>{const o={};cols.forEach((c,i)=>o[c]=row[i]);return o};
const bookLink=n=>`<a class="lk book" href="#/book/${encodeURIComponent(n)}">${esc(n)}</a>`;
const famLink=n=>`<a class="lk" href="#/family/${encodeURIComponent(n)}">${esc(n)}</a>`;
const famOf=n=>(S.books.find(b=>b.name===n)||{}).family||"";
const when=ts=>{const d=new Date(ts*1000);return d.toISOString().slice(5,10)+" "+d.toISOString().slice(11,19)};
const dur=s=>s==null?"-":s<90?Math.round(s)+"s":s<5400?Math.round(s/60)+"m":(s/3600).toFixed(1)+"h";
function merge(d){if(d.now)S.skew=Date.now()/1000-d.now;
(d.positions||[]).forEach(r=>{const o=obj(S.pc,r);S.pos.set(o.id,o)});
(d.launch||[]).forEach(r=>{const o=obj(S.lc,r);S.launch.set(o.pool,o)});
if("health" in d&&d.health)S.health=d.health}
/* one trade: status and result; stuck = no route out for S.stuck seconds, valued at $0 */
function trade(p,t){const cost=p.cost_usd||0,real=p.realized_usd||0;let st,pnl=null,end=null;
if(p.closed_at){end=p.closed_at;pnl=real-cost;st=pnl>0?"Won":"Lost"}
else if(p.blocked_since&&t-p.blocked_since>=S.stuck){end=p.blocked_since+S.stuck;pnl=real-cost;st="Exit blocked ($0)"}
else if(p.blocked_since){st="Exit blocked"}else{st="Open"}
return{id:p.id,book:p.strategy,sym:p.symbol||(p.token||"").slice(0,6),token:p.token,opened:p.opened_at,end,cost,pnl,
pct:pnl==null||!cost?null:100*pnl/cost,st,reason:p.exit_reason,held:(end||t)-p.opened_at,open:!p.closed_at&&st!=="Exit blocked ($0)",real,p}}
function trades(pred){const t=now(),out=[];for(const p of S.pos.values()){const x=trade(p,t);if(!pred||pred(x))out.push(x)}
return out.sort((a,b)=>b.opened-a.opened)}
function bookStats(b){const t=now(),c=b.cash||S.cash;let equity=c,open=0,started=null;const tr=[];
for(const p of S.pos.values()){if(p.strategy!==b.name)continue;started=started==null?p.opened_at:Math.min(started,p.opened_at);
const x=trade(p,t);
if(x.end!=null){tr.push({t:x.end,ret:x.cost?x.pnl/x.cost:0,pnl:x.pnl,cost:x.cost});equity+=x.pnl}
else{open++;equity+=x.real+(p.qty||0)*(p.entry_price||0)-x.cost}}
tr.sort((x,y)=>x.t-y.t);let eq=c,peak=c,dd=0;const pts=started==null?[]:[[started,c]];
for(const x of tr){eq+=x.pnl;peak=Math.max(peak,eq);dd=Math.max(dd,(peak-eq)/peak);pts.push([x.t,eq])}
const rets=tr.map(x=>x.ret*100),gain=tr.filter(x=>x.pnl>0).reduce((a,x)=>a+x.pnl,0),loss=-tr.filter(x=>x.pnl<0).reduce((a,x)=>a+x.pnl,0);
const rugs=tr.filter(x=>x.ret<=-0.8).length;
return{name:b.name,family:b.family,hold_min:b.hold_min,cash:c,started,equity,return_pct:(equity/c-1)*100,trades:tr.length,open,
win_rate:tr.length?tr.filter(x=>x.ret>0).length/tr.length:null,median_pct:median(rets),mean_pct:wmean(tr),eq_mean_pct:mean(rets),
profit_factor:loss>0?gain/loss:null,rug_pct:tr.length?100*rugs/tr.length:null,max_drawdown_pct:100*dd,
last_hour:tr.filter(x=>x.t>t-3600).length,tr,pts,gain,loss,rugs}}
const COLS=[["name","Book",0],["hold_min","Hold",0],["equity","Equity $",0],["return_pct","Return",2],["trades","Trades",0],["open","Open",0],["win_rate","Win",2],["median_pct","Median",1],["mean_pct","Mean",1],["profit_factor","PF",2],["rug_pct","Rugged",0],["max_drawdown_pct","Max DD",1]];
function booksTable(id,list){const rows=[...list].sort((a,b)=>{const x=a[sortKey],y=b[sortKey];return (x==null)-(y==null)||(x<y?-1:x>y?1:0)*sortDir});
const leader=list.filter(b=>b.trades>0).sort((a,b)=>b.return_pct-a.return_pct)[0];const exp=(S.extra&&S.extra.expected)||{};
let h="<tr>"+COLS.map(c=>`<th data-k="${c[0]}">${c[1]}${sortKey===c[0]?(sortDir<0?" ↓":" ↑"):""}</th>`).join("")+"</tr>";
for(const b of rows){h+=`<tr class="${leader&&b.name===leader.name?"lead-row":""}" title="${esc(exp[b.name]||"")}">`+COLS.map(([k,,d])=>{const v=b[k];
if(k==="name")return`<td>${bookLink(v)}</td>`;if(k==="hold_min")return`<td>${fmt(v,0)}m</td>`;if(k==="equity")return`<td>${fmt(v,0)}</td>`;
if(["return_pct","median_pct","mean_pct"].includes(k))return`<td class="${cls(v)}">${sgn(v,d)}</td>`;
if(k==="rug_pct"||k==="max_drawdown_pct")return`<td>${fmt(v,d,"%")}</td>`;return`<td>${fmt(v,d)}</td>`}).join("")+"</tr>"}
$(id).innerHTML=h;$(id).querySelectorAll("th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortDir=sortKey===k?-sortDir:-1;sortKey=k;render()});
return leader}
function tiles(id,items){$(id).innerHTML=items.map(([l,v,h,lime,nm])=>`<div class="stat${lime?" lime":""}"><span class="label">${l}</span><span class="v num${nm?" nm":""}">${v}</span><span class="h">${h||""}</span></div>`).join("")}
function renderOverview(){const all=S.books.map(bookStats);const leader=booksTable("books",all);
if(leader){$("s-lead").innerHTML=bookLink(leader.name);$("s-lead-h").textContent=`${sgn(leader.return_pct,2)} on ${leader.trades} trades`}
const pnl=all.reduce((a,b)=>a+b.equity-b.cash,0);$("s-pnl").innerHTML=`<span class="${cls(pnl)}">${sgn(pnl,0,"")}</span>`;
$("s-pnl-h").textContent=`$ across ${all.length} books`;$("s-trades").textContent=fmt(all.reduce((a,b)=>a+b.trades,0),0);
$("s-trades-h").textContent=`${all.reduce((a,b)=>a+b.open,0)} open now`;$("s-rate").textContent=fmt(all.reduce((a,b)=>a+b.last_hour,0),0);renderLaunch()}
function group(rs){const r=rs.map(x=>x.ret*100);return{n:r.length,mean:mean(r),median:median(r),win:r.length?100*r.filter(x=>x>0).length/r.length:null,
rug:r.length?100*rs.filter(x=>x.ret<=-0.999).length/r.length:null,big:r.length?100*rs.filter(x=>x.ret>=0.5).length/r.length:null}}
const doneIn=()=>{const t=now();return[...S.launch.values()].filter(x=>x.decided_ts>=t-win&&x.liquid&&x.ret!=null)};
function gapRow(b,done){const t=now(),ideal=done.filter(x=>x.flags.includes(b.flag)),byTok=new Map(ideal.map(x=>[x.token,x]));
const mine=[...S.pos.values()].filter(p=>p.strategy===b.name&&p.closed_at&&p.opened_at>=t-win&&byTok.has(p.token));
const held=new Set([...S.pos.values()].filter(p=>p.strategy===b.name).map(p=>p.token));
const real=mine.map(p=>100*(p.realized_usd/p.cost_usd-1)),idl=mine.map(p=>100*byTok.get(p.token).ret),missed=ideal.filter(x=>!held.has(x.token)).map(x=>100*x.ret);
if(!mine.length&&!missed.length)return"";const gap=mine.length?mean(real)-mean(idl):null;
return`<tr><td>${bookLink(b.name)}</td><td>${mine.length}</td><td class="${cls(mean(real))}">${sgn(mean(real))}</td><td>${sgn(mean(idl))}</td><td class="${cls(gap)}">${sgn(gap,1," pts")}</td><td>${missed.length}</td><td>${sgn(mean(missed))}</td></tr>`}
const GAPHEAD="<tr><th>Book</th><th>Trades</th><th>Real</th><th>Ideal</th><th>Gap</th><th>Missed</th><th>Missed ideal</th></tr>";
function renderLaunch(){const done=doneIn();const label=win>=86400?"24":String(win/3600);$("lh").textContent=$("lh2").textContent=label;
const flags=[];const byFlag={};for(const b of S.books)if(b.flag){(byFlag[b.flag]=byFlag[b.flag]||[]).push(b.name);if(!flags.includes(b.flag))flags.push(b.flag)}
let h="<tr><th>Group</th><th>Trades</th><th>Mean</th><th>Median</th><th>Win</th><th>Rugged</th><th>+50%</th></tr>";
const line=(name,g)=>`<tr><td>${name}</td><td>${fmt(g.n,0)}</td><td class="${cls(g.mean)}">${sgn(g.mean)}</td><td>${sgn(g.median)}</td><td>${fmt(g.win,0,"%")}</td><td>${fmt(g.rug,0,"%")}</td><td>${fmt(g.big,0,"%")}</td></tr>`;
h+=line("Every liquid pool",group(done));
for(const f of flags)h+=line(byFlag[f].map(bookLink).join(", "),group(done.filter(x=>x.flags.includes(f))));
$("outcomes").innerHTML=h;$("gaps").innerHTML=GAPHEAD+S.books.filter(b=>b.flag).map(b=>gapRow(b,done)).join("")}
/* ledger */
const STAG={"Open":"lime","Exit blocked":"warn","Exit blocked ($0)":"neg","Won":"","Lost":""};
function ledgerTable(id,rows,limit,showBook){let h=`<tr><th>Bought (UTC)</th>${showBook?'<th class="l">Book</th>':""}<th class="l">Token</th><th class="l">Status</th><th>Result</th><th>Result $</th><th>Held</th><th>Cost $</th><th>Exit</th></tr>`;
for(const x of rows.slice(0,limit)){const tok=x.token?`<a class="lk" href="https://dexscreener.com/solana/${encodeURIComponent(x.token)}" target="_blank" rel="noopener" title="${esc(x.token)}">${esc(x.sym)}</a>`:esc(x.sym);
h+=`<tr><td class="mono">${when(x.opened)}</td>${showBook?`<td class="l">${bookLink(x.book)}</td>`:""}<td class="l">${tok}</td><td class="l"><span class="tag sm ${STAG[x.st]}">${esc(x.st)}</span></td>`+
`<td class="${cls(x.pct)}">${sgn(x.pct)}${x.p.swap_ret==null?"":`<br><span class="muted" title="fill prices only, fees and rent left out">swap ${sgn(100*x.p.swap_ret)}</span>`}</td><td class="${cls(x.pnl)}">${x.pnl==null?"-":sgn(x.pnl,2,"")}</td><td>${dur(x.held)}</td><td>${fmt(x.cost,2)}</td><td class="muted">${esc((x.reason||"").replace(/_/g," "))}</td></tr>`}
$(id).innerHTML=h;return rows.length}
const STATUS_OK=(x,st)=>!st||x.st===st;
function fillSelects(){const fs=$("lf-fam"),bs=$("lf-book"),fv=fs.value,bv=bs.value;
fs.innerHTML='<option value="">Every family</option>'+S.fams.map(f=>`<option ${f.name===fv?"selected":""}>${esc(f.name)}</option>`).join("");
bs.innerHTML='<option value="">Every book</option>'+S.books.filter(b=>!fs.value||b.family===fs.value).map(b=>`<option ${b.name===bv?"selected":""}>${esc(b.name)}</option>`).join("")}
function renderLedger(){const f=$("lf-fam").value,b=$("lf-book").value,st=$("lf-status").value,q=$("lf-q").value.trim().toLowerCase();
const rows=trades(x=>(!f||famOf(x.book)===f)&&(!b||x.book===b)&&STATUS_OK(x,st)&&(!q||(x.sym||"").toLowerCase().includes(q)||(x.token||"").toLowerCase().includes(q)));
const n=ledgerTable("ledger",rows,LIM.ledger,true);$("ledger-count").textContent=`${Math.min(n,LIM.ledger)} of ${n} trades`;$("ledger-more").hidden=n<=LIM.ledger}
/* equity chart: one line per series, inline SVG, ink tones and dash patterns (lime is never a line) */
const STY=[["var(--ink)",""],["var(--ink)","6 4"],["var(--ink-2)",""],["var(--ink-2)","6 4"],["var(--ink-3)",""],["var(--ink-3)","2 3"],["var(--ink-2)","2 3"],["var(--ink)","2 3"]];
function chart(id,series,base=S.cash,empty="No closed trades yet."){series=series.filter(s=>s.pts.length>1);if(!series.length){$(id).innerHTML=`<p class="sub">${esc(empty)}</p>`;return}
const W=800,H=260,L=72,Rr=12,T=12,B=28;let x0=Infinity,x1=-Infinity,y0=base,y1=base;
for(const s of series)for(const[x,y]of s.pts){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y)}
if(x1<=x0)x1=x0+1;const pad=(y1-y0)*0.08||50;y0-=pad;y1+=pad;
const X=x=>L+(x-x0)/(x1-x0)*(W-L-Rr),Y=y=>T+(y1-y)/(y1-y0)*(H-T-B);
let g=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Equity curve">`;
g+=`<line x1="${L}" x2="${W-Rr}" y1="${Y(base)}" y2="${Y(base)}" style="stroke:var(--rule-soft);stroke-width:1.5" stroke-dasharray="5 5" vector-effect="non-scaling-stroke"/>`;
g+=`<line x1="${L}" x2="${L}" y1="${T}" y2="${H-B}" style="stroke:var(--rule);stroke-width:2" vector-effect="non-scaling-stroke"/><line x1="${L}" x2="${W-Rr}" y1="${H-B}" y2="${H-B}" style="stroke:var(--rule);stroke-width:2" vector-effect="non-scaling-stroke"/>`;
const ys=[];for(const v of[base,y1-pad,y0+pad]){if(ys.some(y=>Math.abs(y-Y(v))<14))continue;ys.push(Y(v));g+=`<text class="ax" x="${L-6}" y="${Y(v)+4}" text-anchor="end">${fmt(v,0)}</text>`}
g+=`<text class="ax" x="${L}" y="${H-8}">${when(x0).slice(0,11)}</text><text class="ax" x="${W-Rr}" y="${H-8}" text-anchor="end">${when(x1).slice(0,11)}</text>`;
series.forEach((s,i)=>{const[c,d]=STY[i%STY.length];g+=`<polyline fill="none" style="stroke:${c};stroke-width:2" ${d?`stroke-dasharray="${d}"`:""} vector-effect="non-scaling-stroke" points="${s.pts.map(([x,y])=>X(x).toFixed(1)+","+Y(y).toFixed(1)).join(" ")}"><title>${esc(s.name)}</title></polyline>`});
g+="</svg>";if(series.length>1)g+='<div class="legend">'+series.map((s,i)=>{const[c,d]=STY[i%STY.length];return`<span><svg viewBox="0 0 28 8"><line x1="0" x2="28" y1="4" y2="4" style="stroke:${c};stroke-width:2" ${d?`stroke-dasharray="${d}"`:""}/></svg>${esc(s.name)}${s.ret==null?"":" "+sgn(s.ret,1)}</span>`}).join("")+"</div>";
$(id).innerHTML=g}
/* families index, family view, book view */
function famStats(list){const tr=list.flatMap(b=>b.tr),rets=tr.map(x=>x.ret*100),pnl=list.reduce((a,b)=>a+b.equity-b.cash,0);
return{pnl,trades:tr.length,open:list.reduce((a,b)=>a+b.open,0),win:tr.length?100*tr.filter(x=>x.ret>0).length/tr.length:null,mean:wmean(tr),median:median(rets),
rug:tr.length?100*tr.filter(x=>x.ret<=-0.8).length/tr.length:null,best:[...list].filter(b=>b.trades).sort((a,b)=>b.return_pct-a.return_pct)[0]}}
function renderFams(){const all=S.books.map(bookStats);
$("fams").innerHTML=S.fams.map(f=>{const list=all.filter(b=>b.family===f.name);if(!list.length)return"";const s=famStats(list);
return`<section class="card"><p class="label eyebrow">Family · ${list.length} book${list.length>1?"s":""}</p><h2>${famLink(f.name)}</h2><p class="sub">${esc(f.about)}</p>`+
`<p class="sub"><b class="${cls(s.pnl)}">${sgn(s.pnl,0," $")}</b> · ${fmt(s.trades,0)} trades · mean ${sgn(s.mean)} · rugged ${fmt(s.rug,0,"%")}</p>`+
`<div class="scroll"><table class="mini"><tr><th>Book</th><th>Return</th><th>Trades</th><th>Mean</th></tr>`+
list.sort((a,b)=>b.return_pct-a.return_pct).map(b=>`<tr><td>${bookLink(b.name)}</td><td class="${cls(b.return_pct)}">${sgn(b.return_pct,2)}</td><td>${fmt(b.trades,0)}</td><td class="${cls(b.mean_pct)}">${sgn(b.mean_pct)}</td></tr>`).join("")+"</table></div></section>"}).join("")}
function renderFamily(name){const fam=S.fams.find(f=>f.name===name)||{name,about:""};const list=S.books.filter(b=>b.family===name).map(bookStats);
$("f-name").textContent=name;$("f-about").textContent=fam.about;const s=famStats(list);
tiles("f-stats",[["Leading book",s.best?bookLink(s.best.name):"-",s.best?`${sgn(s.best.return_pct,2)} on ${s.best.trades} trades`:"",true,true],
["P&amp;L, family",`<span class="${cls(s.pnl)}">${sgn(s.pnl,0,"")}</span>`,`$ across ${list.length} books`],
["Closed trades",fmt(s.trades,0),`${s.open} open now`],["Mean trade",`<span class="${cls(s.mean)}">${sgn(s.mean)}</span>`,`median ${sgn(s.median)} · win ${fmt(s.win,0,"%")}`],
["Rugged",fmt(s.rug,0,"%"),"closed at −80% or worse"]]);
booksTable("f-books",list);chart("f-chart",list.map(b=>({name:b.name,pts:b.pts,ret:b.return_pct})),list.length?list[0].cash:S.cash);
const names=new Set(list.map(b=>b.name)),rows=trades(x=>names.has(x.book));const n=ledgerTable("f-ledger",rows,LIM.f,true);
$("f-count").textContent=`${Math.min(n,LIM.f)} of ${n} trades`;$("f-more").hidden=n<=LIM.f}
function renderBook(name){const b=S.books.find(x=>x.name===name);if(!b){$("b-name").textContent=name;$("b-about").textContent="This book is not running.";return}
const s=bookStats(b),exp=(S.extra&&S.extra.expected)||{};$("b-name").textContent=name;$("b-fam").textContent=b.family;$("b-fam").href="#/family/"+encodeURIComponent(b.family);
$("b-about").textContent=exp[name]||`Hold ${fmt(b.hold_min,0)} min.`;
tiles("b-stats",[["Equity",fmt(s.equity,0),`return <b>${sgn(s.return_pct,2)}</b> · max drawdown ${fmt(s.max_drawdown_pct,1,"%")}`,true],
["Closed trades",fmt(s.trades,0),`${s.open} open now · ${s.last_hour} in the past hour`],
["Mean trade",`<span class="${cls(s.mean_pct)}">${sgn(s.mean_pct)}</span>`,`per $ traded · equal-weight ${sgn(s.eq_mean_pct)} · median ${sgn(s.median_pct)} · win ${fmt(s.win_rate==null?null:100*s.win_rate,0,"%")}`],
["Profit factor",fmt(s.profit_factor,2),`won ${fmt(s.gain,0)} $ · lost ${fmt(s.loss,0)} $`],["Rugged",fmt(s.rug_pct,0,"%"),`${s.rugs} trade${s.rugs===1?"":"s"} at −80% or worse`]]);
chart("b-chart",[{name,pts:s.pts,ret:s.return_pct}],s.cash);
if(b.flag){$("b-gap-card").hidden=false;$("b-lh").textContent=win>=86400?"24":String(win/3600);$("b-gap").innerHTML=GAPHEAD+(gapRow(b,doneIn())||'<tr><td class="muted">No finished trades in the window.</td></tr>')}else $("b-gap-card").hidden=true;
$("b-swap-card").hidden=!b.twin;if(b.twin)swapCard(b);
const rows=trades(x=>x.book===name);const n=ledgerTable("b-ledger",rows,LIM.b,false);$("b-count").textContent=`${Math.min(n,LIM.b)} of ${n} trades`;$("b-more").hidden=n<=LIM.b}
/* the canary judged on fill prices against its paper twin on the same tokens */
function swapCard(b){const t=now(),tw=new Map();
for(const p of S.pos.values())if(p.strategy===b.twin){const x=trade(p,t);if(x.pct!=null)tw.set(p.token,x)}
const rows=trades(x=>x.book===b.name&&x.p.swap_ret!=null);const sw=[],tv=[],gap=[];
let h=`<tr><th>Bought (UTC)</th><th class="l">Token</th><th>Canary swap</th><th>Twin result</th><th>Gap</th><th>Canary net</th><th>Net $</th></tr>`;
for(const x of rows){const s=100*x.p.swap_ret,w=tw.get(x.token),wp=w?w.pct:null;sw.push(s);if(wp!=null){tv.push(wp);gap.push(s-wp)}
if(sw.length>LIM.b)continue;
h+=`<tr><td class="mono">${when(x.opened)}</td><td class="l">${esc(x.sym)}</td><td class="${cls(s)}">${sgn(s)}</td><td class="${cls(wp)}">${sgn(wp)}</td><td class="${cls(wp==null?null:s-wp)}">${wp==null?"-":sgn(s-wp)}</td><td class="muted">${sgn(x.pct)}</td><td class="muted">${x.pnl==null?"-":sgn(x.pnl,2,"")}</td></tr>`}
const rug=sw.filter(v=>v<=-80).length;
tiles("b-swap-stats",[["Canary swap",`<span class="${cls(mean(sw))}">${sgn(mean(sw))}</span>`,`mean of ${sw.length} sold · median ${sgn(median(sw))}`],
["Twin, same tokens",`<span class="${cls(mean(tv))}">${sgn(mean(tv))}</span>`,`${tv.length} matched`],
["Gap",`<span class="${cls(mean(gap))}">${sgn(mean(gap))}</span>`,`canary swap minus twin · median ${sgn(median(gap))}`],
["Rugged (swap)",fmt(sw.length?100*rug/sw.length:null,0,"%"),`${rug} sold at −80% or worse`]]);
$("b-swap").innerHTML=rows.length?h:'<tr><td class="muted">No sold canary trades yet.</td></tr>'}
/* programs: Supermeme (all books) and MZF Short Only (the shortbook app's feed) */
function memeTotals(){const all=S.books.map(bookStats),tr=all.flatMap(b=>b.tr),pnl=all.reduce((a,b)=>a+b.equity-b.cash,0);
return{all,pnl,cap:all.reduce((a,b)=>a+b.cash,0),trades:tr.length,open:all.reduce((a,b)=>a+b.open,0),mean:wmean(tr),last_hour:all.reduce((a,b)=>a+b.last_hour,0)}}
function shortsState(){if(!S.shortsEnabled)return["off","Not connected: set TB_SHORTBOOK_FEED_URL on the Supermeme app."];
if(!S.shorts||S.shorts.nav_usd==null)return["wait",S.shorts&&S.shorts.error?`Short book unreachable: ${S.shorts.error}`:"Waiting for the short book's first update."];
if(S.shorts.stale)return["stale",`Showing the last update from ${when(S.shorts.fetched_ts||S.shorts.built_ts)} UTC: the short book is unreachable (${S.shorts.error||"no answer"}).`];
if(!S.shorts.healthy)return["stale","The short book's loop looks stalled (its heartbeat is old)."];return["ok",""]}
function shortStats(){const f=S.shorts;if(!f||f.nav_usd==null)return null;const cl=f.closed||[],rets=cl.map(p=>p.pnl_pct).filter(x=>x!=null);
return{nav:f.nav_usd,pnl:f.nav_usd-f.start_usd,ret:f.return_pct,open:(f.open||[]).length,exposure:(f.open||[]).reduce((a,p)=>a+p.size_usd,0),
closed:f.closed_total||0,mean:mean(rets),dd:f.max_drawdown_pct,pending:f.pending}}
function renderPrograms(){const m=memeTotals(),sh=shortStats(),[st,msg]=shortsState();
const comb=m.pnl+(sh?sh.pnl:0);
tiles("p-stats",[["P&amp;L, all programs",`<span class="${cls(comb)}">${sgn(comb,0,"")}</span>`,"$, paper",true],
["Supermeme",`<span class="${cls(m.pnl)}">${sgn(m.pnl,0,"")}</span>`,`$ on ${fmt(m.cap,0)} $ across ${m.all.length} books`],
["MZF Short Only",sh?`<span class="${cls(sh.pnl)}">${sgn(sh.pnl,0,"")}</span>`:"-",sh?`$ on ${fmt(S.shorts.start_usd,0)} $ · NAV ${fmt(sh.nav,0)}`:esc(msg)],
["Trades, last hour",fmt(m.last_hour,0),"Supermeme books (shorts hold 7 days)"]]);
const fams=S.fams.map(f=>{const list=m.all.filter(b=>b.family===f.name);if(!list.length)return"";const s=famStats(list);
return`<tr><td>${famLink(f.name)}</td><td>${list.length}</td><td class="${cls(s.pnl)}">${sgn(s.pnl,0,"")}</td><td>${fmt(s.trades,0)}</td><td class="${cls(s.mean)}">${sgn(s.mean)}</td></tr>`}).join("");
const meme=`<section class="card prog"><p class="label eyebrow">Program · Solana launch books</p><h2><a href="#/supermeme">Supermeme</a></h2>
<p class="sub">Paper books buying hand-made Solana pools seconds to minutes after launch, out within minutes. ${m.all.length} books in ${S.fams.filter(f=>m.all.some(b=>b.family===f.name)).length} families.</p>
<div class="kv"><div><span>P&amp;L $</span><b class="${cls(m.pnl)}">${sgn(m.pnl,0,"")}</b></div><div><span>Return on capital</span><b class="${cls(m.pnl)}">${sgn(m.cap?100*m.pnl/m.cap:null,2)}</b></div>
<div><span>Closed trades</span><b>${fmt(m.trades,0)}</b></div><div><span>Mean trade</span><b class="${cls(m.mean)}">${sgn(m.mean)}</b></div><div><span>Open now</span><b>${fmt(m.open,0)}</b></div></div>
<div class="scroll"><table class="mini"><tr><th>Family</th><th>Books</th><th>P&amp;L $</th><th>Trades</th><th>Mean</th></tr>${fams}</table></div>
<p class="more"><a class="lk" href="#/books">Books &amp; families</a> · <a class="lk" href="#/ledger">Trade ledger</a></p></section>`;
let shorts=`<section class="card prog"><p class="label eyebrow">Program · centralised-exchange shorts</p><h2><a href="#/shorts">MZF Short Only</a></h2>
<p class="sub">Short-only strategies on perps. Strategy 1 shorts new exchange listings at the listing hour and covers after 7 days.</p>`;
if(st!=="ok")shorts+=`<p class="note">${esc(msg)}</p>`;
if(sh){const strat=(S.shorts.strategies||[])[0]||{groups:{}};
shorts+=`<div class="kv"><div><span>NAV $</span><b>${fmt(sh.nav,0)}</b></div><div><span>Return</span><b class="${cls(sh.ret)}">${sgn(sh.ret,2)}</b></div>
<div><span>Open shorts</span><b>${fmt(sh.open,0)}</b></div><div><span>Closed shorts</span><b>${fmt(sh.closed,0)}</b></div><div><span>Max drawdown</span><b>${fmt(sh.dd==null?null:-sh.dd,1,"%")}</b></div></div>
<div class="scroll"><table class="mini"><tr><th>${esc(strat.name||"Strategy")}</th><th>Open</th><th>Closed</th><th>Mean</th><th>Research</th></tr>`+
Object.entries(strat.groups||{}).map(([g,v])=>`<tr><td>${esc(g)}</td><td>${v.open}</td><td>${v.closed}</td><td class="${cls(v.mean_pct)}">${sgn(v.mean_pct)}</td><td>${sgn(v.research_pct)}</td></tr>`).join("")+
`</table></div>`}
shorts+=`<p class="more"><a class="lk" href="#/shorts">Open MZF Short Only</a></p></section>`;
$("progs").innerHTML=meme+shorts}
const days=s=>s==null?"-":(s/86400).toFixed(1)+"d";
function renderShortLev(){const V=S.shorts&&S.shorts.leverage;if(!V||!V.profiles){$("sh-lev").innerHTML=`<tr><td class="muted">${V&&V.error?esc(V.error):"No leverage profiles yet."}</td></tr>`;$("sh-levchart").innerHTML="";return}
const n=(x,d=2)=>x==null?"-":fmt(x,d);
$("sh-lev").innerHTML='<tr><th class="l">Leverage</th><th>NAV $</th><th>Return</th><th>Max DD</th><th>Sharpe</th><th>Shorts</th><th>Liquidated</th><th>Mean on margin</th><th>Win</th><th>Gross</th><th style="border-left:2px solid var(--rule)">Backtest Sharpe</th><th>DD</th><th>1y median</th><th>P(DD&gt;30%)</th></tr>'+
V.profiles.map(p=>{const b=(V.backtest||{})[String(p.lev)];return`<tr><td class="l book">${p.lev}x</td><td>${fmt(p.nav_usd,0)}</td><td class="${cls(p.return_pct)}">${sgn(p.return_pct,2)}</td><td>${fmt(-p.max_drawdown_pct,1,"%")}</td><td>${n(p.sharpe)}</td><td>${p.shorts}</td><td>${p.liquidated}</td><td class="${cls(p.mean_on_margin_pct)}">${sgn(p.mean_on_margin_pct)}</td><td>${p.win_pct==null?"-":fmt(p.win_pct,0,"%")}</td><td>${p.gross_x==null?"-":n(p.gross_x)+"x"}</td>`+
(b?`<td style="border-left:2px solid var(--rule)">${n(b[0])}</td><td>${b[1]}%</td><td>+${b[2]}%</td><td>${b[3]}%</td>`:'<td style="border-left:2px solid var(--rule)">-</td><td>-</td><td>-</td><td>-</td>')+"</tr>"}).join("");
chart("sh-levchart",V.profiles.map(p=>({name:p.lev+"x",pts:p.equity||[],ret:p.return_pct})),V.start_usd)}
function renderShortMarket(){const M=S.shorts&&S.shorts.market,sel=$("sh-mh");if(!M||!M.horizons){$("sh-market").innerHTML=`<tr><td class="muted">${M&&M.error?esc(M.error):"No market analysis yet (the short book fills prices for every listing as horizons come due)."}</td></tr>`;$("sh-mgap").textContent="";return}
const hs=Object.keys(M.horizons);if(sel.options.length!==hs.length){const cur=sel.value||"168";sel.innerHTML=hs.map(h=>`<option value="${h}"${h===cur?" selected":""}>${h==="168"?"7 days (the book's hold)":h==="72"?"3 days":h==="24"?"24 hours":h+" hour"}</option>`).join("");sel.onchange=renderShortMarket}
const hz=M.horizons[sel.value]||M.horizons[hs[hs.length-1]],pc=v=>v==null?"-":`<span class="${cls(v)}">${sgn(v)}</span>`,ci=c=>c?`<span class="mono">${sgn(c[0],1)} .. ${sgn(c[1],1)}</span>`:"-";
const sec=t=>`<tr><td class="l muted" colspan="7" style="border-top:2px solid var(--rule)">${t}</td></tr>`,row=(k,v,b)=>`<tr${b?' style="font-weight:700"':""}><td class="l book">${esc(k)}</td><td>${fmt(v.n||0,0)}</td><td>${pc(v.mean_pct)}</td><td>${pc(v.median_pct)}</td><td>${v.n?fmt(v.win_pct,0,"%"):"-"}</td><td>${v.n?pc(v.worst_pct):"-"}</td><td>${ci(v.ci90)}</td></tr>`;
const part=o=>Object.entries(o||{}).map(([k,v])=>row(k,v)).join("");
$("sh-market").innerHTML='<tr><th class="l">Group</th><th>n</th><th>Mean</th><th>Median</th><th>Win</th><th>Worst</th><th>90% CI of mean</th></tr>'+
Object.entries(hz.groups).map(([k,v])=>row(k,v,true)).join("")+sec("Skipped, by reason (short)")+part(hz.skip_reasons)+sec("Every listing by kind (short)")+part(hz.kinds)+sec("Every listing by venue (short)")+part(hz.venues);
const g=(t,x)=>x?`${t} ${sgn(x.pts,1," pts")} (90% CI ${sgn(x.ci90[0],1)} .. ${sgn(x.ci90[1],1)})`:"";const r=M.picks_real||{};
$("sh-mgap").textContent=[g("Picks minus skipped:",hz.picks_vs_skipped),g("Picks minus every listing:",hz.picks_vs_every),r.n?`Picks' real book result: ${r.n} closed, mean ${sgn(r.mean_pct)} (perp fills, fees, funding).`:""].filter(Boolean).join(" · ")}
function renderShorts(){const[st,msg]=shortsState();$("sh-note").innerHTML=st==="ok"?"":`<p class="note">${esc(msg)}</p>`;const f=S.shorts,sh=shortStats();
if(!sh){["sh-stats","sh-groups","sh-chart","sh-open","sh-closed","sh-listings"].forEach(i=>$(i).innerHTML="");return}
renderShortMarket();renderShortLev();
const strat=(f.strategies||[])[0]||{groups:{}};$("sh-strat").textContent=strat.name||"Strategy";$("sh-about").textContent=strat.about||"";
tiles("sh-stats",[["Net asset value",fmt(sh.nav,0),`$ · started ${fmt(f.start_usd,0)} $`,true],["Return",`<span class="${cls(sh.ret)}">${sgn(sh.ret,2)}</span>`,`${sgn(sh.pnl,0," $")}`],
["Open shorts",fmt(sh.open,0),`${fmt(sh.exposure,0)} $ short · ${fmt(sh.pending,0)} listings pending`],["Closed shorts",fmt(sh.closed,0),`mean ${sgn(sh.mean)} per trade`],
["Max drawdown",fmt(sh.dd==null?null:-sh.dd,1,"%"),f.started_ts?`running since ${when(f.started_ts).slice(0,11)}`:""]]);
$("sh-groups").innerHTML="<tr><th>Group</th><th>Open</th><th>Closed</th><th>Mean</th><th>Research</th><th>Win</th><th>P&amp;L $</th></tr>"+
Object.entries(strat.groups||{}).map(([g,v])=>`<tr><td class="book">${esc(g)}</td><td>${v.open}</td><td>${v.closed}</td><td class="${cls(v.mean_pct)}">${sgn(v.mean_pct)}</td><td>${sgn(v.research_pct)}</td><td>${fmt(v.win_pct,0,"%")}</td><td class="${cls(v.pnl_usd)}">${sgn(v.pnl_usd,0,"")}</td></tr>`).join("");
chart("sh-chart",[{name:"NAV",pts:f.equity||[],ret:sh.ret}],f.start_usd);
const head="<tr><th>Opened (UTC)</th><th class=\"l\">Token</th><th class=\"l\">Group</th><th class=\"l\">Venue</th><th>Size $</th><th>Entry</th><th>Price</th><th>Move</th><th>Funding $</th><th>P&amp;L $</th><th>P&amp;L</th><th>",
row=(p,last)=>`<tr><td class="mono">${when(p.entry_ts)}</td><td class="l book">${esc(p.base)}</td><td class="l">${esc(p.group)}</td><td class="l">${esc(p.venue)}</td><td>${fmt(p.size_usd,0)}</td><td>${fmt(p.entry_px,4)}</td><td>${fmt(p.px,4)}</td><td class="${cls(p.price_pct)}">${sgn(p.price_pct)}</td><td class="${cls(p.funding_usd)}">${sgn(p.funding_usd,0,"")}</td><td class="${cls(p.pnl_usd)}">${sgn(p.pnl_usd,0,"")}</td><td class="${cls(p.pnl_pct)}">${sgn(p.pnl_pct)}</td><td>${last}</td></tr>`;
const t=now();$("sh-open").innerHTML=(f.open||[]).length?head+"Covers in</th></tr>"+f.open.map(p=>row(p,days(p.due_ts-t))).join(""):'<tr><td class="muted">No open shorts.</td></tr>';
$("sh-closed").innerHTML=(f.closed||[]).length?head+"Covered (UTC)</th></tr>"+f.closed.map(p=>row(p,`<span class="mono">${when(p.exit_ts)}</span>`)).join(""):'<tr><td class="muted">No closed shorts yet.</td></tr>';
$("sh-lcount").textContent=Object.entries(f.listing_counts||{}).map(([k,n])=>`${k} ${n}`).join(" · ");
$("sh-listings").innerHTML="<tr><th>Seen (UTC)</th><th class=\"l\">Venue</th><th class=\"l\">Token</th><th class=\"l\">Kind</th><th class=\"l\">Status</th><th class=\"l\">Reason</th></tr>"+
(f.listings||[]).map(r=>`<tr><td class="mono">${r.seen_ts?when(r.seen_ts):"-"}</td><td class="l">${esc(r.venue)}</td><td class="l book">${esc(r.base)}</td><td class="l">${esc(r.kind||"")}</td><td class="l"><span class="tag sm ${r.status==="entered"?"lime":r.status==="pending"?"warn":""}">${esc(r.status)}</span></td><td class="l muted">${esc(r.reason||"")}</td></tr>`).join("")}
/* meme market monitor: hourly rows + last 3 h against the past week, built by market_monitor.py */
const MK_RET=new Set(["market_mean","market_median","market_sized","rw_kept","rw_skipped","rw_gap","sz_picks_ideal","sz_real","sz_fill_gap","thin_mean","mid_mean","thick_mean","thin_sz_mean","sol_chg_pct","btc_chg_pct"]);
const MK_USD=new Set(["med_liq_usd","sol_usd","btc_usd"]);
const mkLabel=k=>((S.market&&S.market.metrics)||[]).find(m=>m[0]===k)?.[1]||k;
function mkVal(k,v){if(v==null)return"-";if(MK_RET.has(k))return`<span class="${cls(v)}">${sgn(v,1,k.endsWith("gap")?" pts":"%")}</span>`;
if(MK_USD.has(k))return fmt(v,0);if(k.endsWith("_pct"))return fmt(v,1,"%");return Math.abs(v)>=100?fmt(v,0):fmt(v,1)}
const zTag=z=>z==null?'<span class="muted">-</span>':`<span class="tag sm ${Math.abs(z)>=2?"warn":""}">${z>0?"+":"−"}${Math.abs(z).toFixed(1)}</span>`;
function marketState(){if(!S.marketEnabled)return["off","The market monitor is off: set TB_MARKET_MONITOR=1 on the Supermeme app."];
const m=S.market;if(!m||!m.rows.length)return["wait","Waiting for the monitor's first hours (it stores each hour about 15 minutes after it ends; the first run backfills)."];
if(now()-m.built_ts>600)return["stale",`The dashboard has not read the monitor's table since ${when(m.built_ts)} UTC (its read keeps failing; check the logs for "market monitor read failed"). The monitor itself may be fine.`];
if(now()-m.last_hour>3*3600+900){const s=m.status||{},hr=`The newest stored hour is ${when(m.last_hour).slice(0,11)}:00 UTC.`;
if(!s.ran)return["stale",`${hr} The monitor has no recorded run: its thread is not running (needs TB_MARKET_MONITOR=1 and the launch scorer, then a restart).`];
if(now()-s.ran>3*(m.poll_s||600))return["stale",`${hr} The monitor last ran at ${when(s.ran)} UTC and has not run since: its thread has stopped or is stuck. A restart brings it back.`];
if(s.error)return["stale",`${hr} The monitor runs but fails: ${s.error} (last success ${s.ok_ts?when(s.ok_ts)+" UTC":"never"}).`];
return["stale",`${hr} The monitor ran at ${when(s.ran)} UTC without error but stored nothing new: the launch scorer may have no new pools.`]}
return["ok",""]}
function renderTracker(T){const ids=["mk-trk-pass","mk-trk-books","mk-trk"];const books=T.books||{},names=Object.keys(books);if(!names.length){ids.forEach(i=>$(i).innerHTML="");$("mk-trk-h").textContent="";$("mk-trk-sub").textContent="No tracker summary yet: it is rebuilt by the market monitor every run (about every 10 minutes).";return}
const r=T.rule||{},usd=v=>v==null?"-":(v>0?"+$":v<0?"−$":"$")+Math.abs(v).toLocaleString(),f1=v=>v==null?"-":sgn(v,1,"");
$("mk-trk-sub").textContent=`Read only: nothing is switched. Each flag is one episode, measured from when it was confirmed (its streak's last hour stored) to when its close was stored. A rule passes with ${r.min_episodes} or more episodes of ${r.min_episode_trades}+ trades, a ${Math.round(100*(1-(r.alpha||0.1)))}% band (adjusted for the number of rules tested) clear of zero, and the same sign in both halves. Built ${when(T.built_ts)}.`;
const P=T.passing||[];$("mk-trk-pass").innerHTML=P.length?'<tr><th class="l">Book</th><th class="l">Measure</th><th class="l">Direction</th><th class="l">Verdict</th><th>Gap</th><th>Band</th><th>Episodes</th></tr>'+P.map(p=>`<tr><td class="l book">${esc(p.book)}</td><td class="l">${esc(p.label)}</td><td class="l">${p.direction==="up"?"↑ up":"↓ down"}</td><td class="l">${esc(p.verdict)}</td><td class="${cls(p.gap)}">${sgn(p.gap,1," pts")}</td><td>${f1(p.ci[0])}..${f1(p.ci[1])}</td><td>${p.usable}</td></tr>`).join(""):'<tr><td class="muted">No rule has passed yet. Expect 2-3 weeks of flags before any can.</td></tr>';
$("mk-trk-books").innerHTML='<tr><th class="l">Book</th><th>Trades</th><th>Result</th><th>Rules tested</th><th>Passing</th></tr>'+names.map(b=>{const v=books[b];return `<tr><td class="l book">${esc(b)}</td><td>${v.trades.toLocaleString()}</td><td class="${cls(v.pct)}">${sgn(v.pct)}</td><td>${v.tests}</td><td>${v.rules.filter(x=>x.passing).length}</td></tr>`}).join("");
const b0=names[0],v0=books[b0];$("mk-trk-h").textContent=`${b0} by measure`;
$("mk-trk").innerHTML=v0.rules.length?'<tr><th class="l">Measure</th><th class="l">Direction</th><th>Episodes</th><th>Trades in</th><th>In</th><th>Other</th><th>Gap</th><th>Band</th><th>Halves</th><th>6 h gap</th><th>Half size</th><th>Skip</th><th>Thin only</th><th>Double</th><th class="l">Verdict</th></tr>'+v0.rules.map(x=>{const w=x.whatif_usd||{};return `<tr><td class="l book">${esc(x.label)}</td><td class="l">${x.direction==="up"?"↑ up":"↓ down"}</td><td>${x.usable}/${x.episodes}</td><td>${x.trades_in}</td><td class="${cls(x.in_pct)}">${sgn(x.in_pct)}</td><td class="${cls(x.base_pct)}">${sgn(x.base_pct)}</td><td class="${cls(x.gap)}">${sgn(x.gap,1," pts")}</td><td>${x.ci?f1(x.ci[0])+".."+f1(x.ci[1]):"-"}</td><td>${x.halves?f1(x.halves[0])+" / "+f1(x.halves[1]):"-"}</td><td class="${cls(x.after6_gap)}">${sgn(x.after6_gap,1," pts")}</td><td class="${cls(w["half size"])}">${usd(w["half size"])}</td><td class="${cls(w.skip)}">${usd(w.skip)}</td><td class="${cls(w["thin pools only"])}">${usd(w["thin pools only"])}</td><td class="${cls(w.double)}">${usd(w.double)}</td><td class="l">${esc(x.verdict)}</td></tr>`}).join(""):'<tr><td class="muted">No flags yet.</td></tr>'}
function renderBands(){const B=(S.market&&S.market.bands)||{},sel=$("mk-bwin"),names=Object.keys(B);if(!names.length){$("mk-bands").innerHTML='<tr><td class="muted">No band data yet (rows built before the 8-band version are rebuilt on the next monitor run).</td></tr>';return}
if(sel.options.length!==names.length){const cur=sel.value||"whole life";sel.innerHTML=names.map(n=>`<option${n===cur?" selected":""}>${esc(n)}</option>`).join("");sel.onchange=renderBands}
const rows=B[sel.value]||[],pc=v=>v==null?"-":`<span class="${cls(v)}">${sgn(v)}</span>`,pl=v=>v==null?"-":fmt(v,1,"%"),usd=v=>`<span class="${cls(v)}">${v<0?"−":"+"}$${fmt(Math.abs(v),0)}</span>`;
const G=x=>`<th colspan="${x[1]}" style="text-align:center;border-left:2px solid var(--rule)">${x[0]}</th>`,L=' style="border-left:2px solid var(--rule)"';
$("mk-bands").innerHTML='<tr><th></th><th></th><th></th>'+[["Every pool (no filter)",5],["Rug-vs-win kept",7],["Skipped",2],["sz real",3]].map(G).join("")+'</tr>'+
`<tr><th class="l">Band</th><th>Pools</th><th>Share</th><th${L}>Mean</th><th>Win</th><th>Rugged</th><th>+50%</th><th>$ at $100</th><th${L}>Pools</th><th>Mean</th><th>Win</th><th>Rugged</th><th>+50%</th><th>$ at $100</th><th>vs every pool</th><th${L}>Mean</th><th>Gap</th><th${L}>n</th><th>sz</th><th>sz $</th></tr>`+
rows.filter(r=>r.pools||r.sz_n||r.band.startsWith("All")).map(r=>`<tr${r.band.startsWith("All")?' style="font-weight:700;border-top:2px solid var(--ink)"':""}><td class="l book">${esc(r.band)}</td><td>${fmt(r.pools,0)}</td><td>${pl(r.share_pct)}</td>`+
`<td${L}>${pc(r.mean_pct)}</td><td>${pl(r.win_pct)}</td><td>${pl(r.rugged_pct)}</td><td>${pl(r.big_pct)}</td><td>${usd(r.all_100_usd)}</td>`+
`<td${L}>${fmt(r.kept_n,0)}</td><td>${pc(r.kept_pct)}</td><td>${pl(r.kept_win_pct)}</td><td>${pl(r.kept_rugged_pct)}</td><td>${pl(r.kept_big_pct)}</td><td>${usd(r.kept_100_usd)}</td><td>${r.kept_vs_all==null?"-":`<span class="${cls(r.kept_vs_all)}">${sgn(r.kept_vs_all,1," pts")}</span>`}</td>`+
`<td${L}>${pc(r.skipped_pct)}</td><td>${r.gap==null?"-":`<span class="${cls(r.gap)}">${sgn(r.gap,1," pts")}</span>`}</td>`+
`<td${L}>${fmt(r.sz_n,0)}</td><td>${pc(r.sz_pct)}</td><td>${usd(r.sz_pnl)}</td></tr>`).join("")}
function renderMarket(){const[st,msg]=marketState();$("mk-note").innerHTML=st==="ok"?"":`<p class="note">${esc(msg)}</p>`;const m=S.market;
if(!m||!m.rows.length){["mk-stats","mk-shifts","mk-charts","mk-cmp","mk-hours","mk-imp","mk-flags","mk-flagsum","mk-bands","mk-trk-pass","mk-trk-books","mk-trk"].forEach(i=>$(i).innerHTML="");return}
const c=m.compare||{},rh=m.recent_h,wk=Math.round(m.baseline_h/24);
const tile=(k,l)=>{const x=c[k]||{};return[l,mkVal(k,x.now),`week ${mkVal(k,x.baseline)} · z ${x.z==null?"-":(x.z>0?"+":"−")+Math.abs(x.z).toFixed(1)}`]};
const shifts=Object.entries(c).filter(([,x])=>x.z!=null&&Math.abs(x.z)>=2).sort((a,b)=>Math.abs(b[1].z)-Math.abs(a[1].z));
tiles("mk-stats",[["Shifts now",fmt(shifts.length,0),`measures 2+ standard deviations from the past ${wk} days`],
tile("liquid","Liquid pools per hour"),tile("thin_pct","Thin pools (under $15k)"),tile("rug_pct","Rugged by minute 6"),
tile("market_mean","Market, ideal mean"),tile("rw_gap","Rug-vs-win gap"),tile("sz_real","sz real result")]);
$("mk-shift-sub").textContent=`The last ${rh} hours against the ${wk} days before${m.baseline_since?` (from ${when(m.baseline_since).slice(0,11)} UTC at the earliest)`:""}; sz measures count only from its first trade. z = how many standard deviations of ${rh}-hour windows the last ${rh} hours sit from the weekly mean.`;
$("mk-shifts").innerHTML=shifts.length?"<tr><th class=\"l\">Measure</th><th class=\"l\">Group</th><th>Last "+rh+" h</th><th>Past week</th><th>z</th></tr>"+
shifts.map(([k,x])=>`<tr><td class="l book">${esc(mkLabel(k))}</td><td class="l muted">${esc(x.group)}</td><td>${mkVal(k,x.now)}</td><td>${mkVal(k,x.baseline)}</td><td>${zTag(x.z)}</td></tr>`).join("")
:'<tr><td class="muted">Nothing is 2+ standard deviations from the past week.</td></tr>';
const F=m.flags||{open:[],recent:[],by_metric:[]},fr=m.flag_rule||{},hr=h=>when(h).slice(0,8)+"h";
$("mk-flag-sub").textContent=`A flag opens when a measure stays ${fr.z||2}+ standard deviations from the past week for ${fr.hours||2}+ hours in a row, and closes below ${fr.close_z||1.5}. Open flags first, then those closed in the past 7 days.`;
const fl=(f,open)=>`<tr><td class="l"><span class="tag sm ${open?"warn":""}">${open?"Open":"Closed"}</span></td><td class="l book">${esc(f.label)}</td><td class="l">${f.direction==="up"?"↑ up":"↓ down"}</td><td class="mono">${hr(f.opened_hour)}</td><td class="mono">${open?"-":hr(f.closed_hour)}</td><td>${f.hours}</td><td>${zTag(f.peak_z)}</td><td>${mkVal(f.metric,f.open_value)} vs ${mkVal(f.metric,f.baseline)}</td><td class="${cls(f.sz_pct)}">${f.sz_pct==null?"-":sgn(f.sz_pct)}</td><td>${f.sz_trades}</td><td class="${cls(f.market_pct)}">${f.market_pct==null?"-":sgn(f.market_pct)}</td></tr>`;
$("mk-flags").innerHTML=(F.open.length||F.recent.length)?'<tr><th class="l">State</th><th class="l">Measure</th><th class="l">Direction</th><th>From</th><th>To</th><th>Hours</th><th>Peak z</th><th>Value vs week</th><th>sz</th><th>sz trades</th><th>Market</th></tr>'+F.open.map(f=>fl(f,true)).join("")+F.recent.map(f=>fl(f,false)).join(""):'<tr><td class="muted">No flags yet.</td></tr>';
renderTracker(m.tracker||{});renderBands();$("mk-flagsum").innerHTML=F.by_metric.length?'<tr><th class="l">Measure</th><th class="l">Direction</th><th>Flags</th><th>Hours</th><th>sz in</th><th>sz other</th><th>Gap</th><th>Market in</th><th>Market other</th></tr>'+F.by_metric.map(g=>`<tr><td class="l book">${esc(g.label)}</td><td class="l">${g.direction==="up"?"↑ up":"↓ down"}</td><td>${g.flags}</td><td>${g.hours}</td><td class="${cls(g.in.sz_pct)}">${g.in.sz_pct==null?"-":sgn(g.in.sz_pct)}</td><td class="${cls(g.out.sz_pct)}">${g.out.sz_pct==null?"-":sgn(g.out.sz_pct)}</td><td class="${cls(g.sz_gap)}">${g.sz_gap==null?"-":sgn(g.sz_gap,1," pts")}</td><td class="${cls(g.in.market_pct)}">${g.in.market_pct==null?"-":sgn(g.in.market_pct)}</td><td class="${cls(g.out.market_pct)}">${g.out.market_pct==null?"-":sgn(g.out.market_pct)}</td></tr>`).join(""):'<tr><td class="muted">No flags yet.</td></tr>';
const rows=m.rows,pts=k=>rows.filter(r=>r[k]!=null&&!r.impaired).map(r=>[r.hour+1800,r[k]]);
const CH=[["Liquid pools per hour",[["liquid","pools"]],"liquid"],["Thin pools (under $15k), % of liquid",[["thin_pct","under $15k"]],"thin_pct"],
["Rugged by minute 6, %",[["rug_pct","rugged"]],"rug_pct"],["Median LP pull, seconds after creation",[["med_pull_s","pull second"]],"med_pull_s"],
["Ideal mean: every pool vs the sized market, %",[["market_mean","every pool"],["market_sized","sized like sz"]],"market_mean"],["sz real vs its picks at ideal, %",[["sz_real","sz real"],["sz_picks_ideal","sz picks, ideal"]],"sz_real"]];
$("mk-charts").innerHTML=CH.map((ch,i)=>`<section class="card chart"><p class="label eyebrow">Last ${Math.round((m.rows.length))} h</p><h2>${esc(ch[0])}</h2><p class="sub">Dashed: the past-week mean.</p><div id="mk-ch${i}"></div></section>`).join("");
CH.forEach((ch,i)=>{const base=(c[ch[2]]||{}).baseline;chart("mk-ch"+i,ch[1].map(([k,n])=>({name:n,pts:pts(k),ret:null})),base==null?0:base,"Not enough hours yet.")});
$("mk-cmp-sub").textContent=`Last ${rh} hours against the ${wk} days before; |z| of 2 or more is marked. Pooled measures (shares, means) weight each hour by its pools.`;
let h=`<tr><th class="l">Measure</th><th>Last ${rh} h</th><th>Past week</th><th>z</th></tr>`,g=null;
for(const[k,l,grp]of m.metrics){const x=c[k];if(!x||x.now==null)continue;if(grp!==g){g=grp;h+=`<tr><td class="l label muted" colspan="4">${esc(grp)}</td></tr>`}
h+=`<tr><td class="l">${esc(l)}</td><td>${mkVal(k,x.now)}</td><td>${mkVal(k,x.baseline)}</td><td>${zTag(x.z)}</td></tr>`}
$("mk-cmp").innerHTML=h;
const HC=[["pools","Pools"],["liquid","Liquid"],["thin_pct","<15k"],["mid_pct","15-25k"],["thick_pct","25k+"],["creators","Creators"],["top5_pct","Top 5"],["template_pct","Template"],["rug_pct","Rugged"],["med_pull_s","Pull s"],["big_win_pct","+50%"],["market_mean","Market"],["market_sized","Mkt sized"],["thin_mean","<15k mkt"],["rw_gap","RvW gap"],["sz_trades","sz n"],["sz_real","sz"],["btc_chg_pct","BTC"]];
$("mk-hours").innerHTML="<tr><th>Hour</th>"+HC.map(([,l])=>`<th>${l}</th>`).join("")+"</tr>"+
[...rows].reverse().slice(0,48).map(r=>`<tr${r.impaired?` class="muted" style="opacity:.45" title="${esc("Scorer impaired: "+(r.impaired_why||""))}"`:""}><td class="mono">${when(r.hour).slice(0,8)}h${r.impaired?" !":""}</td>`+HC.map(([k])=>`<td>${mkVal(k,r[k])}</td>`).join("")+"</tr>").join("");
const imp=m.impaired||[];$("mk-imp").innerHTML=imp.length?`<p class="note">Greyed hours (!) are left out of the comparison, the flags, the band totals and the charts: the scorer itself was impaired, so their pools are a biased subset. `+imp.map(x=>`${when(x.hour).slice(0,8)}h: ${esc(x.why||"")}`).join(" · ")+"</p>":""}
/* router */
function route(){const h=decodeURIComponent(location.hash.replace(/^#\/?/,"")),[v,...rest]=h.split("/");const arg=rest.join("/");
R.v=({supermeme:"overview",ledger:"ledger",books:"books",book:"book",family:"family",shorts:"shorts",market:"market",wallets:"wallets",pools:"pools",settings:"settings"})[v]||"programs";R.arg=arg||null;LIM.b=LIM.f=200;
for(const m of document.querySelectorAll("main"))m.hidden=m.id!=="v-"+R.v;
const meme=["overview","ledger","books","book","family","market","wallets","pools","settings"].includes(R.v);$("subtabs").hidden=!meme;
document.querySelectorAll(".tabs:not(.sub) a").forEach(a=>a.classList.toggle("on",a.dataset.v===(meme?"supermeme":R.v)));
document.querySelectorAll(".tabs.sub a").forEach(a=>a.classList.toggle("on",a.dataset.v===R.v||(a.dataset.v==="books"&&(R.v==="book"||R.v==="family"))));
alignSub();window.scrollTo(0,0);render()}
/* the Supermeme sub menu starts under the Supermeme tab (its left rule on the tab's left rule); on narrow screens it keeps the page gutter */
function alignSub(){const s=$("subtabs"),t=$("tab-meme");if(s.hidden)return;s.style.paddingLeft="";
const x=t.getBoundingClientRect().left-s.getBoundingClientRect().left-2,w=[...s.children].reduce((a,c)=>a+c.offsetWidth,0);
const cs=getComputedStyle(s);if(x>parseFloat(cs.paddingLeft)&&x+w<=s.clientWidth-parseFloat(cs.paddingRight))s.style.paddingLeft=x+"px"}
function renderHealth(){const ok=S.health?Object.values(S.health).every(x=>x.ok):null;
$("health").innerHTML=`<span class="dot"></span>${ok==null?"Bot":ok?"Bot healthy":"Bot stalled"}`;$("health").className="tag "+(ok==null?"":ok?"ink":"neg")}
const SBTN="padding:4px 14px;border:2px solid #000;cursor:pointer;font:inherit;font-weight:700;margin-left:8px";
function canaryDo(a){if(a==="start"&&!confirm("Resume the real-money launch canary? It will start probing and committing again."))return;
fetch("api/canary/"+a,{method:"POST"}).then(r=>r.json()).then(d=>{if(d&&typeof d.halted==="boolean"){S.canaryHalted=d.halted;renderFeints()}else alert("Could not switch the canary"+(d&&d.error?": "+d.error:"."))}).catch(e=>alert("Could not reach the canary switch: "+e))}
function agoStr(s){return s<60?s+"s":s<3600?Math.round(s/60)+"m":Math.round(s/3600)+"h"}
function renderFeints(){const card=$("feint-card"),F=S.feints||[];if(!S.canary&&!F.length){card.hidden=true;return}card.hidden=false;
$("feint-switch").innerHTML=S.canaryHalted?`<span class="tag neg">HALTED</span><button onclick="canaryDo('start')" style="${SBTN};background:#c9f24d;color:#000">Start</button>`:`<span class="tag lime">running</span><button onclick="canaryDo('stop')" style="${SBTN};background:#000;color:#fff">Stop</button>`;
let alive=0,pulled=0,incon=0,last=0;const notes={};for(const f of F){if(f.verdict==="alive")alive++;else if(f.verdict==="pulled")pulled++;else{incon++;const n=f.note||"inconclusive";notes[n]=(notes[n]||0)+1}if(f.ts>last)last=f.ts}
const dec=alive+pulled,pr=dec?Math.round(100*pulled/dec):null,ago=last?Math.round(Date.now()/1000-last):null;
tiles("feint-tiles",[["Alive",`<span class="pos">${alive}</span>`,"liquidity held"],["Pulled",`<span class="neg">${pulled}</span>`,"pulled in reaction"],["Pulled rate",pr==null?"–":pr+"%","of alive + pulled"],["Inconclusive",String(incon),"probe not placed"],["Last feint",ago==null?"–":agoStr(ago)+" ago",ago!=null&&ago>600?"stalled? check the run":"since the last probe"],["Probes",String(F.length),"in the live feed"]]);
const nr=Object.entries(notes).sort((a,b)=>b[1]-a[1]);
$("feint-notes").innerHTML=`<tr><th class="l">Why inconclusive</th><th>n</th></tr>`+(nr.length?nr.map(([n,c])=>`<tr><td class="l">${esc(n)}</td><td>${c}</td></tr>`).join(""):`<tr><td class="muted">None</td></tr>`);
const rows=F.slice(-60).reverse();
$("feint-feed").innerHTML=`<tr><th>Time</th><th>Pool</th><th>Verdict</th><th class="l">Note</th></tr>`+(rows.length?rows.map(f=>{const t=new Date(f.ts*1000).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});const vc=f.verdict==="alive"?"pos":f.verdict==="pulled"?"neg":"muted";const pool=f.pool?`<a class="lk mono" href="https://dexscreener.com/solana/${encodeURIComponent(f.pool)}" target="_blank" rel="noopener">${esc(f.pool.slice(0,8))}</a>`:(f.token?`<span class="mono">${esc(f.token.slice(0,8))}</span>`:"–");return`<tr><td class="mono">${t}</td><td>${pool}</td><td class="${vc}">${esc(f.verdict||"?")}</td><td class="l">${esc(f.note||"")}</td></tr>`}).join(""):`<tr><td class="muted">No feints yet.</td></tr>`)}
const WRL={trading:"Trading",treasury:"Treasury",reserve:"Reserve"};
function walletLink(a){return a?`<a class="lk mono" href="https://solscan.io/account/${encodeURIComponent(a)}" target="_blank" rel="noopener">${esc(a.slice(0,4)+"…"+a.slice(-4))}</a>`:"–"}
// Which rotation wallet executed a probe/trade: show W{n} from the Wallets directory when known, else the short address. Links to solscan.
function walletNum(a){const live=(S.wallets&&S.wallets.live)||[];for(const w of live){if(w.address===a&&w.n!=null)return w.n}return null}
function walletCell(a){if(!a)return'<span class="muted">—</span>';const n=walletNum(a);return`<a class="lk mono" href="https://solscan.io/account/${encodeURIComponent(a)}" target="_blank" rel="noopener" title="${esc(a)}">${n!=null?("W"+n):esc(a.slice(0,4))}</a>`}
function walletRow(w){return`<tr><td>${w.n!=null?w.n:'<span class="muted">—</span>'}</td><td>${esc(WRL[w.role]||w.role||"")}</td><td>${walletLink(w.address)}</td><td class="num">${w.sol!=null?fmt(w.sol,4):"–"}</td><td class="num">${w.txs!=null?w.txs:"–"}</td><td class="l">${esc(w.label||"")}</td></tr>`}
function renderWallets(){const W=S.wallets,note=$("w-note");
if(!W){note.innerHTML='<p class="note">Loading wallet balances…</p>';["w-tiles","w-live","w-arch"].forEach(i=>$(i).innerHTML="");return}
note.innerHTML=W.stale?`<p class="note">Balances may be stale${W.error?": "+esc(W.error):""}.</p>`:(W.error?`<p class="note">${esc(W.error)}</p>`:"");
const live=W.live||[],arch=W.archive||[],trading=live.filter(w=>w.role==="trading"),tre=live.find(w=>w.role==="treasury"),res=live.find(w=>w.role==="reserve");
const when=W.fetched_ts?new Date(W.fetched_ts*1000).toLocaleTimeString([],{hour12:false}):"–";
tiles("w-tiles",[["Trading wallets",String(trading.length),W.rotate?"rotating · "+(W.route||"direct"):(W.route||"direct")],["Trading SOL",fmt(W.trading_sol||0,3),"across the live set"],["Treasury",tre?fmt(tre.sol||0,3):"–","SOL · idle funds"],["Reserve",res?fmt(res.sol||0,3):"–","SOL · tops up wallets"],["Updated",when,"public RPC, each minute"]]);
$("w-live").innerHTML=`<tr><th>#</th><th>Role</th><th>Address</th><th>SOL</th><th>Txs</th><th class="l">Label</th></tr>`+(live.length?live.map(walletRow).join(""):`<tr><td class="muted">No wallets loaded (the canary may be off or no wallets file is set).</td></tr>`);
tuBuild(trading);checkRotate();checkTopup();
$("w-arch").innerHTML=`<tr><th>Address</th><th class="l">Label</th><th>SOL</th><th>First recorded</th></tr>`+(arch.length?arch.map(w=>`<tr><td>${walletLink(w.address)}</td><td class="l">${esc(w.label||"")}</td><td class="num">${w.sol!=null?fmt(w.sol,4):"–"}</td><td class="mono">${w.created?esc(String(w.created).slice(0,10)):"–"}</td></tr>`).join(""):`<tr><td class="muted">No archived wallets.</td></tr>`)}
const PLIQ=v=>v==null?"–":(Math.abs(v)>=1000?"$"+(v/1000).toFixed(1)+"k":"$"+v.toFixed(0));
const PAGE2=s=>s==null?"–":agoStr(Math.round(s));
const PPCT=r=>r==null?"–":(r>=0?"+":"")+(r*100).toFixed(0)+"%";
function poolRow(r){const neg=r.pnl25!=null&&r.pnl25<0,win=r.pnl25!=null&&r.pnl25>0;
const vt=(r.verdict==="alive"||r.verdict==="survivor")?'<span class="tag lime">'+esc(r.verdict)+'</span>':(r.verdict==="pulled"||r.verdict==="rugger"||r.verdict==="trapped")?'<span class="tag neg">'+esc(r.verdict)+'</span>':r.verdict?'<span class="tag">'+esc(r.verdict)+'</span>':"–";
// A honeypot/skipped probe broadcast nothing (no buy tx), so GAP and recovered SOL are blank BECAUSE $0 was ever sent — say so, or the dashes read as "lost / unknown".
const ns=r.sent===false?' <span class="tag" style="opacity:.75" title="Nothing was broadcast — the honeypot was caught by the free pre-buy simulation, so $0 was committed.">no money sent</span>':'';
const lk=r.pool?`<a class="lk mono" href="https://dexscreener.com/solana/${encodeURIComponent(r.pool)}" target="_blank" rel="noopener">${esc((r.mint||r.pool).slice(0,8))}</a>`:esc((r.mint||"").slice(0,8));
const wal=walletCell(r.wallet);
return`<tr><td>${lk}</td><td>${wal}</td><td>${vt}${ns}</td><td class="num">${PAGE2(r.pool_age_s)}</td><td class="num">${PAGE2(r.probe_pool_age_s)}</td><td class="num">${PLIQ(r.liq_probe)}</td><td class="num">${PLIQ(r.live_liq)}</td><td class="num">${PLIQ(r.peak_liq)}</td><td class="num${r.pulled?' neg':''}">${r.pulled?PAGE2(r.pull_pool_age_s):"–"}</td><td class="num">${r.pulled?PAGE2(r.pull_after_probe_s):"–"}</td><td class="num${neg?' neg':win?' pos':''}">${r.pnl25!=null?(r.pnl25>=0?"+":"")+"$"+r.pnl25.toFixed(1):"–"}<span class="muted"> ${PPCT(r.ret_m6)}</span></td><td class="num"><span class="${r.conf_ret>0?'pos':r.conf_ret<0?'neg':'muted'}">${PPCT(r.conf_ret)}</span></td><td class="num">${r.opt_age_s!=null?PAGE2(r.opt_age_s)+" ":""}<span class="${r.opt_ret>0?'pos':r.opt_ret<0?'neg':'muted'}">${PPCT(r.opt_ret)}</span></td></tr>`}
function canaryRotate(){const n=+($("rot-n").value||0),s=+($("rot-sol").value||0);
if(!(n>=1&&n<=10)){alert("Wallets must be between 1 and 10.");return}
if(!(s>=0.01&&s<=5)){alert("SOL per wallet must be between 0.01 and 5.");return}
if(!confirm(`Rotate to ${n} fresh wallets with ${s} SOL each?\n\nThis sweeps ALL current trading wallets and the treasury into a NEW treasury, mints ${n} new wallets and funds them (topping up from the reserve). The canary will be HALTED. You must restart the bot (fly deploy) to use the new wallets.`))return;
const go=$("rot-go");go.disabled=true;$("rot-log").hidden=false;$("rot-log").textContent="Starting the rotation…";
fetch("api/canary/rotate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({wallets:n,sol:s})}).then(r=>r.json()).then(d=>{
if(d&&d.started){$("rot-log").textContent=(d.msg||"Started.")+"\n\n";pollRotate()}else{go.disabled=false;$("rot-log").textContent=(d&&(d.msg||d.error))||"Could not start the rotation."}
}).catch(e=>{go.disabled=false;$("rot-log").textContent="Could not reach the rotation endpoint: "+e})}
let rotPoll=false;
function pollRotate(){rotPoll=true;fetch("api/canary/rotate/log").then(r=>r.json()).then(d=>{
$("rot-log").hidden=false;$("rot-log").textContent=(d.log||"(no output yet)")+(d.running?"\n\n… running …":d.returncode===0?"\n\n✓ Done. Restart the bot (fly deploy), then press Start on the overview.":"\n\n✗ Rotation exited with code "+d.returncode+". Nothing was swapped if it failed before the final step — read the log.");
const lg=$("rot-log");lg.scrollTop=lg.scrollHeight;if(d.running)setTimeout(pollRotate,2000);else{rotPoll=false;$("rot-go").disabled=false}}).catch(()=>setTimeout(pollRotate,3000))}
function checkRotate(){if(rotPoll)return;fetch("api/canary/rotate/log").then(r=>r.json()).then(d=>{if(d&&d.started){$("rot-log").hidden=false;$("rot-log").textContent=d.log||"";if(d.running){$("rot-go").disabled=true;pollRotate()}}}).catch(()=>{})}
// Top-up: a target SOL level + a wallet picker (All, or individual W1..Wn). Rebuilt only when the wallet set changes, so a background re-render never clears the user's selection.
function tuBuild(trading){const el=$("tu-wallets");if(!el)return;const sig=trading.map(w=>w.n).join(",");if(el.dataset.sig===sig)return;el.dataset.sig=sig;
const box=(inner,label)=>`<label style="display:flex;align-items:center;gap:5px;font-weight:700;cursor:pointer"><input type="checkbox" ${inner}>${label}</label>`;
el.innerHTML=box('id="tu-all" checked onchange="tuAll()"','All')+trading.map(w=>box(`class="tu-w" data-n="${w.n}" disabled onchange="tuOne()"`,"W"+w.n)).join("")}
function tuAll(){const on=$("tu-all").checked;document.querySelectorAll(".tu-w").forEach(c=>{c.disabled=on;if(on)c.checked=false})}
function tuOne(){if([...document.querySelectorAll(".tu-w")].some(c=>c.checked))$("tu-all").checked=false}
function canaryTopUp(){const level=+($("tu-level").value||0);
if(!(level>=0.01&&level<=5)){alert("Target SOL must be between 0.01 and 5.");return}
let wallets="all";
if(!$("tu-all").checked){wallets=[...document.querySelectorAll(".tu-w")].filter(c=>c.checked).map(c=>+c.dataset.n);if(!wallets.length){alert("Pick at least one wallet, or tick All.");return}}
const which=wallets==="all"?"ALL trading wallets":"wallets "+wallets.join(", ");
if(!confirm(`Top up ${which} to ${level} SOL each?\n\nFunds come from the treasury (and the reserve if it is short). Safe while trading — no halt, no restart.`))return;
const go=$("tu-go");go.disabled=true;$("tu-log").hidden=false;$("tu-log").textContent="Starting the top-up…";
fetch("api/canary/topup",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({level,wallets})}).then(r=>r.json()).then(d=>{
if(d&&d.started){$("tu-log").textContent=(d.msg||"Started.")+"\n\n";pollTopup()}else{go.disabled=false;$("tu-log").textContent=(d&&(d.msg||d.error))||"Could not start the top-up."}
}).catch(e=>{go.disabled=false;$("tu-log").textContent="Could not reach the top-up endpoint: "+e})}
let tuPoll=false;
function pollTopup(){tuPoll=true;fetch("api/canary/topup/log").then(r=>r.json()).then(d=>{
$("tu-log").hidden=false;$("tu-log").textContent=(d.log||"(no output yet)")+(d.running?"\n\n… running …":d.returncode===0?"\n\n✓ Done. Balances refresh within a minute.":"\n\n✗ Top-up exited with code "+d.returncode+" — read the log.");
const lg=$("tu-log");lg.scrollTop=lg.scrollHeight;if(d.running)setTimeout(pollTopup,2000);else{tuPoll=false;$("tu-go").disabled=false}}).catch(()=>setTimeout(pollTopup,3000))}
function checkTopup(){if(tuPoll)return;fetch("api/canary/topup/log").then(r=>r.json()).then(d=>{if(d&&d.started){$("tu-log").hidden=false;$("tu-log").textContent=d.log||"";if(d.running){$("tu-go").disabled=true;pollTopup()}}}).catch(()=>{})}
function csVal(f){const el=$("cs-"+f.key);if(!el)return S.cset.current[f.key];
if(f.type==="bool")return el.checked;if(f.type==="int")return Math.round(+el.value);if(f.type==="float")return +el.value;return el.value}
function csField(f){const v=S.cset.current[f.key],badge=f.live?'<span class="tag lime" style="font-size:10px">live</span>':'<span class="tag" style="font-size:10px">restart</span>';
let input;if(f.type==="bool")input=`<input type="checkbox" id="cs-${f.key}" ${v?"checked":""} style="width:20px;height:20px">`;
else if(f.choices)input=`<select id="cs-${f.key}" style="padding:6px 8px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:inherit">`+f.choices.map(c=>`<option ${c===v?"selected":""}>${esc(c)}</option>`).join("")+`</select>`;
else input=`<input type="number" id="cs-${f.key}" value="${v}" ${f.lo!=null?`min="${f.lo}"`:""} ${f.hi!=null?`max="${f.hi}"`:""} step="${f.type==="int"?1:"any"}" style="width:130px;padding:6px 8px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);font:inherit">`;
return`<div style="display:flex;gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid var(--rule-soft)"><div style="flex:1 1 260px;min-width:0"><div style="font-weight:700">${esc(f.label)} ${badge}</div><div class="sub" style="margin:0">${esc(f.help||"")}</div></div><div>${input}</div></div>`}
let csBuilt=false;
function renderSettings(){const note=$("cs-note");
if(!window._pbStarted){window._pbStarted=true;probePoll()}
if(!S.cset){if(!S.csetLoading){S.csetLoading=true;fetch("api/canary/settings").then(r=>r.json()).then(d=>{S.cset=d;S.csetLoading=false;csBuilt=false;render()}).catch(e=>{S.csetLoading=false;note.innerHTML='<p class="note neg">Could not load settings: '+esc(""+e)+'</p>'})}
if(!note.innerHTML)note.innerHTML='<p class="note">Loading settings…</p>';$("cs-presets").innerHTML="";$("cs-groups").innerHTML="";return}
if(csBuilt){csDirty();return}  // don't rebuild inputs under the user (background re-renders would wipe edits)
const C=S.cset;if(!note.dataset.kept)note.innerHTML=C.overlay?'<p class="note">Showing saved settings (a dashboard override is active; Reset returns to the deploy values).</p>':'';
$("cs-presets").innerHTML=Object.entries(C.presets).map(([k,p])=>`<button onclick="csPreset('${k}')" class="stat" style="text-align:left;border:2px solid var(--rule);cursor:pointer;background:var(--surface)"><span class="label">${esc(p.label)}</span><span class="h" style="white-space:normal">${esc(p.about)}</span></button>`).join("");
$("cs-groups").innerHTML=C.groups.map(g=>{const fs=C.fields.filter(f=>f.group===g);return`<section class="card"><p class="label eyebrow">${esc(g)}</p><h2>${esc(g)}</h2>${fs.map(csField).join("")}</section>`}).join("");
$("cs-groups").oninput=csDirty;$("cs-groups").onchange=csDirty;
$("dep-cmd").textContent=C.deploy_cmd?("will run: "+C.deploy_cmd):"no deploy command on this machine — deploy from your Mac";
$("dep-go").disabled=!C.deploy_cmd;csBuilt=true;csDirty();renderPasskeys()}
function csDiff(){const out={};for(const f of S.cset.fields){const v=csVal(f),c=S.cset.current[f.key];if(f.type==="bool"?(!!v!==!!c):(""+v!==""+c))out[f.key]={f,to:v,from:c}}return out}
function csDirty(){const d=csDiff(),n=Object.keys(d).length;$("cs-dirty").textContent=n?n+" unsaved change"+(n>1?"s":""):"No unsaved changes"}
function csFmt(f,v){return f.type==="bool"?(v?"on":"off"):""+v}
function canarySave(){const d=csDiff(),keys=Object.keys(d);if(!keys.length){alert("No changes to save.");return}
const lines=keys.map(k=>`• ${d[k].f.label}: ${csFmt(d[k].f,d[k].from)} → ${csFmt(d[k].f,d[k].to)}${d[k].f.live?"":"  (after restart)"}`).join("\n");
if(!confirm("Apply these canary changes?\n\n"+lines+"\n\nLive changes take effect immediately on the real-money canary; restart ones apply after fly deploy."))return;
const changes={};keys.forEach(k=>changes[k]=d[k].to);
fetch("api/canary/settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({changes})}).then(r=>r.json().then(j=>({ok:r.ok,j}))).then(({ok,j})=>{
if(!ok){$("cs-note").dataset.kept=1;$("cs-note").innerHTML='<p class="note neg">'+esc(j.error||"Rejected.")+'</p>';return}
const cur={};S.cset.fields.forEach(f=>cur[f.key]=j.values[f.key]);S.cset.current=cur;S.cset.overlay=true;
const rb=(j.restart_needed||[]);$("cs-note").dataset.kept=1;$("cs-note").innerHTML='<p class="note">Saved '+Object.keys(j.changed||{}).length+' change(s).'+(rb.length?" Restart the bot (fly deploy) for: "+rb.map(esc).join(", ")+".":" Live now.")+'</p>';csBuilt=false;render()})
.catch(e=>{$("cs-note").dataset.kept=1;$("cs-note").innerHTML='<p class="note neg">Could not save: '+esc(""+e)+'</p>'})}
function csPreset(name){const p=S.cset.presets[name];if(!confirm('Apply preset "'+p.label+'"?\n\n'+p.about+'\n\nThis sets several canary settings at once on the real-money canary. You can review and adjust the fields after.'))return;
fetch("api/canary/settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({preset:name})}).then(r=>r.json().then(j=>({ok:r.ok,j}))).then(({ok,j})=>{
if(!ok){$("cs-note").dataset.kept=1;$("cs-note").innerHTML='<p class="note neg">'+esc(j.error||"Rejected.")+'</p>';return}
const cur={};S.cset.fields.forEach(f=>cur[f.key]=j.values[f.key]);S.cset.current=cur;S.cset.overlay=true;
const rb=(j.restart_needed||[]);$("cs-note").dataset.kept=1;$("cs-note").innerHTML='<p class="note">Preset applied ('+Object.keys(j.changed||{}).length+' change(s)).'+(rb.length?" Restart for: "+rb.map(esc).join(", ")+".":"")+'</p>';csBuilt=false;render()})
.catch(e=>{$("cs-note").dataset.kept=1;$("cs-note").innerHTML='<p class="note neg">Could not apply: '+esc(""+e)+'</p>'})}
function canaryReset(){if(!confirm("Reset every canary setting to the deploy (fly.toml) values and clear the dashboard override?"))return;
fetch("api/canary/settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reset:true})}).then(r=>r.json()).then(j=>{const cur={};S.cset.fields.forEach(f=>cur[f.key]=j.values[f.key]);S.cset.current=cur;S.cset.overlay=false;$("cs-note").dataset.kept=1;$("cs-note").innerHTML='<p class="note">'+esc(j.msg||"Reset.")+'</p>';csBuilt=false;render()}).catch(e=>{$("cs-note").dataset.kept=1;$("cs-note").innerHTML='<p class="note neg">Could not reset: '+esc(""+e)+'</p>'})}
function flyDeploy(){const cmd=(S.cset&&S.cset.deploy_cmd)||null;
if(!cmd){alert("No deploy command on this machine. Deploy from your Mac with:\n\n  git pull && fly deploy");return}
if(!confirm("Deploy now?\n\nThis runs:\n  "+cmd+"\n\nThe machine will restart and this page will disconnect. Reload in a minute or two."))return;
const go=$("dep-go");go.disabled=true;$("dep-log").hidden=false;$("dep-log").textContent="Starting deploy…";
fetch("api/deploy",{method:"POST"}).then(r=>r.json()).then(d=>{if(d&&d.started){$("dep-log").textContent=(d.msg||"Started.")+"\n\n";pollDeploy()}else{go.disabled=false;$("dep-log").textContent=(d&&d.msg)||"Could not start the deploy."}}).catch(e=>{go.disabled=false;$("dep-log").textContent="Could not reach the deploy endpoint: "+e})}
function pollDeploy(){fetch("api/deploy/log").then(r=>r.json()).then(d=>{$("dep-log").textContent=(d.log||"(no output yet)")+(d.running?"\n\n… deploying …":d.returncode===0?"\n\n✓ Submitted. The machine is restarting — reload this page shortly.":"\n\n✗ Deploy exited with code "+d.returncode+".");const lg=$("dep-log");lg.scrollTop=lg.scrollHeight;if(d.running)setTimeout(pollDeploy,2000);else $("dep-go").disabled=false}).catch(()=>{$("dep-log").textContent+="\n\n(connection lost — the machine is likely restarting. Reload the page in a minute.)"})}
// --- passkeys (WebAuthn) ---
function b64urlToBuf(s){s=s.replace(/-/g,"+").replace(/_/g,"/");const pad=s.length%4?4-s.length%4:0;s+="=".repeat(pad);const bin=atob(s),b=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)b[i]=bin.charCodeAt(i);return b.buffer}
function bufToB64url(b){const bytes=new Uint8Array(b);let bin="";for(const x of bytes)bin+=String.fromCharCode(x);return btoa(bin).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"")}
function probePoll(){if(R.v!=="settings"){window._pbStarted=false;return}renderProbe();clearTimeout(window._pbT);window._pbT=setTimeout(probePoll,5000)}
function renderProbe(){const t=$("pb-tiles");if(!t)return;fetch("api/canary/probes").then(r=>r.json()).then(d=>{
t.innerHTML=[["Probed",d.probes],["Rugger",d.rugger],["Survivor",d.survivor],["Trip rate",d.trip_rate==null?"—":(d.trip_rate*100).toFixed(0)+"%"],["Blind revert",d.blind_revert_rate==null?"—":(d.blind_revert_rate*100).toFixed(0)+"%"],["Blind gap",d.blind_median_gap==null?"—":d.blind_median_gap],["Pending",d.pending]].map(([k,v])=>`<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
const rows=d.recent||[];$("pb-recent").innerHTML="<tr><th>Token</th><th>Wallet</th><th>Probe</th><th>Gap</th><th>Recovered SOL</th><th>Watch</th></tr>"+(rows.length?rows.map(r=>r.sent===false?`<tr><td><a href="https://dexscreener.com/solana/${esc(r.token||"")}" target="_blank" rel="noopener" class="mono">${esc((r.token||"").slice(0,8))}</a></td><td>${walletCell(r.wallet)}</td><td>${esc(r.probe||"")}</td><td colspan="3" class="sub" title="Caught by the free pre-buy simulation — no buy was broadcast, so $0 was at risk.">no money sent ($0 at risk)</td></tr>`:`<tr><td><a href="https://dexscreener.com/solana/${esc(r.token||"")}" target="_blank" rel="noopener" class="mono">${esc((r.token||"").slice(0,8))}</a></td><td>${walletCell(r.wallet)}</td><td>${esc(r.probe||"")}</td><td>${r.slot_gap==null?"—":r.slot_gap}</td><td>${r.recovered_sol==null?"—":r.recovered_sol}</td><td class="${r.watch==='rugger'?'neg':r.watch==='survivor'?'pos':''}">${esc(r.watch||"…")}</td></tr>`).join(""):'<tr><td colspan="6" class="sub">No probes yet. Enable the 1-slot probe preset and start the canary.</td></tr>')}).catch(()=>{})}
function probeOne(){const b=$("pb-one");b.disabled=true;$("pb-one-note").textContent="arming…";
fetch("api/canary/probe-one",{method:"POST"}).then(r=>r.json()).then(d=>{b.disabled=false;$("pb-one-note").textContent=d.msg||"armed."}).catch(e=>{b.disabled=false;$("pb-one-note").textContent="error: "+e})}
function probeAll(){if(!confirm("Probe every liquid pool continuously? This clears the kill switch and the one-at-a-time gate, so the canary probes each liquid candidate with real money (~$0.30 each)."))return;const b=$("pb-all");b.disabled=true;$("pb-one-note").textContent="starting…";
fetch("api/canary/probe-all",{method:"POST"}).then(r=>r.json()).then(d=>{b.disabled=false;$("pb-one-note").textContent=d.msg||"running."}).catch(e=>{b.disabled=false;$("pb-one-note").textContent="error: "+e})}
function probeSmoke(){const mint=$("pb-mint").value.trim();if(!mint){$("pb-smoke-note").textContent="enter a token mint";return}
const usd=parseFloat($("pb-usd").value)||1,pool=$("pb-pool").value.trim()||null;
if(!confirm("Run one REAL $"+usd+" probe on "+mint+"?\n\nIt buys and sells back in the next slot (~fee cost). Direct route only."))return;
const b=$("pb-smoke");b.disabled=true;$("pb-smoke-note").textContent="probing…";$("pb-smoke-out").hidden=false;$("pb-smoke-out").textContent="…";
fetch("api/canary/probe-smoke",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mint,usd,pool})}).then(r=>r.json().then(j=>({ok:r.ok,j}))).then(({ok,j})=>{b.disabled=false;$("pb-smoke-note").textContent=ok?"done":"failed";$("pb-smoke-out").textContent=JSON.stringify(j,null,2);renderProbe()}).catch(e=>{b.disabled=false;$("pb-smoke-note").textContent="error";$("pb-smoke-out").textContent=""+e})}
function renderPasskeys(){const t=$("pk-list");if(!t)return;fetch("api/auth/passkeys").then(r=>r.json()).then(d=>{const ks=d.passkeys||[];t.innerHTML="<tr><th>Device</th><th>Added</th><th>Last used</th><th></th></tr>"+(ks.length?ks.map(k=>`<tr><td>${esc(k.label)}</td><td class="sub">${k.created?new Date(k.created*1000).toLocaleDateString():"—"}</td><td class="sub">${k.last_used?new Date(k.last_used*1000).toLocaleString():"never"}</td><td><button onclick="removePasskey('${esc(k.id)}')" style="padding:3px 10px;border:2px solid var(--rule);background:var(--surface);color:var(--ink);cursor:pointer;font:inherit;font-size:12px">Remove</button></td></tr>`).join(""):'<tr><td colspan="4" class="sub">No passkeys yet. Add this device to sign in without the link.</td></tr>')}).catch(()=>{})}
function addPasskey(){const note=$("pk-note"),btn=$("pk-add");if(!window.PublicKeyCredential){note.textContent="This browser has no passkey support.";return}
const label=($("pk-label").value||"").trim();btn.disabled=true;note.textContent="Follow your device prompt…";
fetch("api/auth/register/options").then(r=>r.json()).then(o=>{o.challenge=b64urlToBuf(o.challenge);o.user.id=b64urlToBuf(o.user.id);(o.excludeCredentials||[]).forEach(c=>c.id=b64urlToBuf(c.id));return navigator.credentials.create({publicKey:o})}).then(cred=>{
const body={id:cred.id,rawId:bufToB64url(cred.rawId),type:cred.type,response:{clientDataJSON:bufToB64url(cred.response.clientDataJSON),attestationObject:bufToB64url(cred.response.attestationObject),transports:cred.response.getTransports?cred.response.getTransports():[]},authenticatorAttachment:cred.authenticatorAttachment,clientExtensionResults:cred.getClientExtensionResults?cred.getClientExtensionResults():{}};
return fetch("api/auth/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({credential:body,label})})}).then(r=>r.json().then(j=>({ok:r.ok,j}))).then(({ok,j})=>{btn.disabled=false;if(!ok){note.textContent=j.error||"Could not register.";return}note.textContent="Added.";$("pk-label").value="";renderPasskeys()}).catch(e=>{btn.disabled=false;note.textContent=(e&&e.name==="NotAllowedError")?"Cancelled.":("Error: "+e)})}
function removePasskey(id){if(!confirm("Remove this passkey? That device will no longer sign in."))return;
fetch("api/auth/passkeys/remove",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id})}).then(r=>r.json()).then(()=>renderPasskeys()).catch(()=>{})}
function signOut(){fetch("api/auth/logout",{method:"POST"}).then(()=>location.reload()).catch(()=>location.reload())}
function renderPools(){const P=S.pools,note=$("p-note");
if(!P){note.innerHTML='<p class="note">Loading pool reserves…</p>';["p-tiles","p-tbl"].forEach(i=>$(i).innerHTML="");return}
const s=P.summary||{},rows=P.rows||[];const when=P.fetched_ts?agoStr(Math.max(0,Math.round(Date.now()/1000-P.fetched_ts)))+" ago":"–";
note.innerHTML=P.stale?`<p class="note neg">Live reserves stale${P.error?": "+esc(P.error):""}. Showing the last read.</p>`:"";
tiles("p-tiles",[["Pools",String(s.pools||0),(P.sol_usd?"SOL $"+P.sol_usd.toFixed(0)+" · ":"")+"updated "+when],["Survivors",String(s.survivor||0),"alive at the ~48s watch — pools to enter"],["Ruggers",String(s.rugger||0),"pulled — we tripped and dodged"],["Survivor entry",s.survivor_entry_mean!=null?(s.survivor_entry_mean>=0?"+":"")+(s.survivor_entry_mean*100).toFixed(0)+"%":"–","enter at confirmation → m6, "+(s.survivor_entry_n||0)+" with a path (hindsight best "+(s.survivor_best_mean!=null?(s.survivor_best_mean>=0?"+":"")+(s.survivor_best_mean*100).toFixed(0)+"%":"–")+")"],["$25 would make",s.would_pnl25!=null?(s.would_pnl25>=0?"+":"")+"$"+s.would_pnl25.toFixed(0):"–","held to minute 6 across "+(s.with_outcome||0)+" finished"]]);
const hc=s.survivor_hold_curve||{},hk=Object.keys(hc);
if(hk.length){$("p-hold-card").style.display="";$("p-hold").innerHTML='<tr><th class="l">Hold to</th><th>n</th><th>Mean</th><th>Median</th><th>Win</th><th>+100%</th></tr>'+hk.map(k=>{const c=hc[k];return `<tr><td class="l">minute ${k}</td><td class="num">${c.n}</td><td class="num ${c.mean>0?"pos":c.mean<0?"neg":"muted"}">${(c.mean>=0?"+":"")+(c.mean*100).toFixed(0)}%</td><td class="num">${(c.median>=0?"+":"")+(c.median*100).toFixed(0)}%</td><td class="num">${(c.win*100).toFixed(0)}%</td><td class="num">${(c.big*100).toFixed(0)}%</td></tr>`}).join("")}else{$("p-hold-card").style.display="none"}
const f=($("p-filter")||{}).value||"all";$("p-filter").onchange=render;
const show=rows.filter(r=>f==="survivor"?r.watch_verdict==="survivor":f==="rugger"?r.watch_verdict==="rugger":f==="alive"?r.verdict==="alive":f==="pulled"?r.pulled:f==="live"?(r.pool_age_s!=null&&r.pool_age_s<1500&&!r.pulled):f==="win"?(r.pnl25!=null&&r.pnl25>0):true);
$("p-tbl").innerHTML=`<tr><th class="l">Pool</th><th>Wallet</th><th>Verdict</th><th>Age</th><th>Probe@</th><th>Liq</th><th>Live</th><th>Peak</th><th>Pulled@</th><th>+probe</th><th>$25 @ m6</th><th>Enter @conf</th><th>Best entry</th></tr>`+(show.length?show.map(poolRow).join(""):`<tr><td class="muted">No feinted pools in the window yet.</td></tr>`)}
function render(){renderHealth();renderFeints();if(!S.books.length)return;
if(R.v==="programs")renderPrograms();else if(R.v==="shorts")renderShorts();else if(R.v==="market")renderMarket();else if(R.v==="wallets")renderWallets();else if(R.v==="pools")renderPools();else if(R.v==="settings")renderSettings();else if(R.v==="ledger")renderLedger();else if(R.v==="books")renderFams();else if(R.v==="book")renderBook(R.arg);else if(R.v==="family")renderFamily(R.arg);else renderOverview();
$("built").textContent="Updated "+new Date().toLocaleTimeString([],{hour12:false})}
function schedule(){if(!pending){pending=true;setTimeout(()=>{pending=false;render()},300)}}
function fill(f,isNew){const li=document.createElement("li");if(isNew)li.className="new";
const t=new Date(f.ts*1000).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});
const side=f.side==="buy"?'<span class="side buy">BUY</span>':'<span class="side">SELL</span>';
const right=f.side==="sell"?`<span class="r ${cls(f.ret_pct)}">${f.ret_pct==null?"part":sgn(f.ret_pct)}</span>`:`<span class="r">$${fmt(f.usd,0)}</span>`;
const why=f.side==="sell"&&f.reason?` · ${esc(f.reason.replace(/_/g," "))}`:"";
li.innerHTML=`<span class="t mono">${t}</span><span>${side}<b>${esc(f.symbol)}</b><span class="bk"><a class="lk" href="#/book/${encodeURIComponent(f.book)}">${esc(f.book)}</a>${why}</span></span>${right}`;
const feed=$("feed");feed.prepend(li);while(feed.children.length>300)feed.lastChild.remove()}
document.querySelectorAll("#win button").forEach(bt=>bt.onclick=()=>{win=+bt.dataset.w;document.querySelectorAll("#win button").forEach(x=>x.classList.toggle("on",x===bt));render()});
$("lf-fam").onchange=()=>{fillSelects();LIM.ledger=200;renderLedger()};
for(const id of["lf-book","lf-status"])$(id).onchange=()=>{LIM.ledger=200;renderLedger()};
$("lf-q").oninput=()=>{LIM.ledger=200;renderLedger()};
$("ledger-more").onclick=()=>{LIM.ledger+=200;renderLedger()};$("b-more").onclick=()=>{LIM.b+=200;render()};$("f-more").onclick=()=>{LIM.f+=200;render()};
window.addEventListener("hashchange",route);window.addEventListener("resize",alignSub);if(document.fonts)document.fonts.ready.then(alignSub);
setInterval(render,30000);
function connect(){const es=new EventSource("api/stream");
es.onopen=()=>{$("conn").innerHTML='<span class="dot"></span>Live';$("conn").className="tag lime"};
es.onerror=()=>{$("conn").innerHTML='<span class="dot"></span>Reconnecting';$("conn").className="tag neg"};
es.addEventListener("init",e=>{const d=JSON.parse(e.data);S.books=d.books;S.fams=d.families||[];S.cash=d.cash;S.stuck=d.stuck_after_s;S.pc=d.pos_cols;S.lc=d.launch_cols;
S.shortsEnabled=!!d.shorts_enabled;S.marketEnabled=!!d.market_enabled;S.canary=!!d.canary;S.canaryHalted=!!d.canary_halted;S.pos.clear();S.launch.clear();S.health=d.health;merge(d);fillSelects();render()});
es.addEventListener("shorts",e=>{S.shorts=JSON.parse(e.data);schedule()});
es.addEventListener("market",e=>{S.market=JSON.parse(e.data);schedule()});
es.addEventListener("wallets",e=>{S.wallets=JSON.parse(e.data);schedule()});
es.addEventListener("pools",e=>{S.pools=JSON.parse(e.data);schedule()});
es.addEventListener("delta",e=>{merge(JSON.parse(e.data));schedule()});
es.addEventListener("summary",e=>{S.extra=JSON.parse(e.data);const d=S.extra.downtime;$("foot").textContent=d?`${d.outages} outages, ${fmt(d.hours_offline,1)} h offline since start`:"";schedule()});
es.addEventListener("backlog",e=>{$("feed").innerHTML="";JSON.parse(e.data).forEach(f=>fill(f,false))});
es.addEventListener("fill",e=>fill(JSON.parse(e.data),true));
es.addEventListener("feintlog",e=>{S.feints=JSON.parse(e.data);schedule()});
es.addEventListener("feint",e=>{S.feints.push(JSON.parse(e.data));if(S.feints.length>500)S.feints.shift();schedule()})}
route();connect();
</script></body></html>"""


LOGIN_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MZFq · Sign in</title>
<style>
:root{--ink:#0a0a0a;--surface:#fff;--rule:#0a0a0a;--rule-soft:#d8d8d8;--sub:#555;--muted:#888;--lime:#c9f24d;--sunken:#f5f5f3}
@media(prefers-color-scheme:dark){:root{--ink:#f2f2f2;--surface:#0e0e0e;--rule:#f2f2f2;--rule-soft:#2a2a2a;--sub:#a6a6a6;--muted:#777;--sunken:#161616}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:var(--surface);color:var(--ink);
font:500 15px/1.6 "Archivo",system-ui,-apple-system,sans-serif;display:flex;justify-content:center}
.wrap{width:100%;max-width:640px;padding:40px 20px 56px}
header{display:flex;align-items:center;gap:12px;border-bottom:2px solid var(--rule);padding-bottom:16px}
.mk{width:44px;height:44px;flex:none}.mk .sq{fill:var(--ink)}.mk .z{fill:var(--surface)}
.word{font-size:26px;font-weight:800;letter-spacing:-.02em;line-height:1}
.desc{font-size:11px;font-weight:700;letter-spacing:.16em;color:var(--muted);margin-top:3px}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:30px 0 6px}
h1{font-size:30px;line-height:1.1;margin:0 0 8px;font-weight:800;letter-spacing:-.02em}
.lead{font-size:16px;color:var(--sub);margin:0}
.progs{display:grid;gap:14px;margin-top:14px}
.prog{border:2px solid var(--rule);padding:16px 18px}
.prog h2{font-size:17px;margin:0 0 4px;font-weight:800}
.prog p{font-size:14px;color:var(--sub);margin:0}
.tag{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.04em;border:2px solid var(--rule-soft);padding:1px 7px;margin-left:8px;vertical-align:middle;color:var(--muted)}
.pts{list-style:none;padding:0;margin:10px 0 0;display:grid;gap:7px}
.pts li{font-size:13.5px;color:var(--sub);padding-left:18px;position:relative}
.pts li::before{content:"";position:absolute;left:0;top:8px;width:7px;height:7px;background:var(--lime);border:1.5px solid var(--ink)}
.signin{border:2px solid var(--rule);background:var(--sunken);padding:22px;margin-top:30px}
.signin h2{font-size:16px;margin:0 0 4px;font-weight:800}
.sub{color:var(--sub);font-size:13px;margin:0 0 16px}
button{width:100%;padding:13px;border:2px solid var(--ink);background:var(--lime);color:#000;cursor:pointer;
font:inherit;font-weight:800;font-size:15px}
button:disabled{opacity:.5;cursor:default}
.note{margin-top:14px;font-size:13px;color:var(--sub)}.neg{color:#c0392b}
footer{margin-top:28px;font-size:12px;color:var(--muted);border-top:1px solid var(--rule-soft);padding-top:14px}
code{font-family:ui-monospace,Menlo,monospace;font-size:12px}
</style></head><body>
<div class="wrap">
<header>
<svg class="mk" viewBox="0 0 100 100" aria-hidden="true"><rect class="sq" width="100" height="100"/><polygon class="z" points="24,26 76,26 76,39 38,61 76,61 76,74 24,74 24,61 62,39 24,39"/></svg>
<div><div class="word">MZFq</div><div class="desc">QUANT DESK</div></div>
</header>

<p class="eyebrow">Systematic trading</p>
<h1>Measured, risk-controlled strategies</h1>
<p class="lead">MZFq is the quant desk: each program holds its own strategies, run as paper books with the numbers tracked trade by trade. The edge comes from risk control and measurement, not hype.</p>

<p class="eyebrow">Programs</p>
<div class="progs">
<div class="prog">
<h2>Supermeme <span class="tag">on-chain</span></h2>
<p>Solana launch trading: pools are scored in their first seconds, filtered rug-vs-win and by pool size, then run as paper books. A tiny real-money canary probes live execution.</p>
<ul class="pts"><li>Second-4 launch scoring from on-chain data</li><li>Rug-vs-win and per-size-band filtering</li><li>Always-on meme market monitor</li></ul>
</div>
<div class="prog">
<h2>MZF Short Only <span class="tag">CEX</span></h2>
<p>Short-only strategies on centralised-exchange events. Strategy 1 shorts new listings into their post-listing decline, sized against open interest and funding.</p>
</div>
</div>

<div class="signin">
<h2>Private dashboard</h2>
<p class="sub">Sign in with your passkey — FaceID, TouchID, Windows Hello or a security key.</p>
<button id="go" onclick="login()">Sign in with passkey</button>
<p class="note" id="note"></p>
</div>

<footer>MZFq · internal quant desk. Everything here is research and paper trading; the canary trades minimal real size only to measure execution.</footer>
</div>
<script>
const $=i=>document.getElementById(i);
function b64urlToBuf(s){s=s.replace(/-/g,"+").replace(/_/g,"/");const pad=s.length%4?4-s.length%4:0;s+="=".repeat(pad);const bin=atob(s),b=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)b[i]=bin.charCodeAt(i);return b.buffer}
function bufToB64url(b){const bytes=new Uint8Array(b);let bin="";for(const x of bytes)bin+=String.fromCharCode(x);return btoa(bin).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"")}
fetch("api/auth/status").then(r=>r.json()).then(d=>{if(!d.passkeys){$("go").disabled=true;$("note").innerHTML=d.token?'No passkeys registered yet. Open this page once with <code>?key=&lt;DASHBOARD_TOKEN&gt;</code>, then add a passkey under Supermeme → Canary → Passkeys.':'Not set up yet.'}}).catch(()=>{});
function login(){const go=$("go"),note=$("note");if(!window.PublicKeyCredential){note.textContent="This browser has no passkey support.";return}
go.disabled=true;note.textContent="Follow your device prompt…";
fetch("api/auth/login/options").then(r=>r.json()).then(o=>{o.challenge=b64urlToBuf(o.challenge);(o.allowCredentials||[]).forEach(c=>c.id=b64urlToBuf(c.id));return navigator.credentials.get({publicKey:o})}).then(a=>{
const body={id:a.id,rawId:bufToB64url(a.rawId),type:a.type,response:{clientDataJSON:bufToB64url(a.response.clientDataJSON),authenticatorData:bufToB64url(a.response.authenticatorData),signature:bufToB64url(a.response.signature),userHandle:a.response.userHandle?bufToB64url(a.response.userHandle):null},clientExtensionResults:a.getClientExtensionResults?a.getClientExtensionResults():{}};
return fetch("api/auth/login/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({credential:body})})}).then(r=>r.json().then(j=>({ok:r.ok,j}))).then(({ok,j})=>{if(ok){location.reload()}else{go.disabled=false;$("note").className="note neg";$("note").textContent=j.error||"Sign-in failed."}}).catch(e=>{go.disabled=false;$("note").className="note neg";$("note").textContent=(e&&e.name==="NotAllowedError")?"Cancelled or timed out.":("Error: "+e)})}
</script></body></html>"""
