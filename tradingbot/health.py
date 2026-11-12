"""Liveness: heartbeat checks, a self-terminating watchdog and an optional /health endpoint.

The monitor thread writes `heartbeat` every pass and the scan loop writes
`scan_heartbeat` after every cycle. If either goes stale the watchdog exits the
process, so the supervisor (Fly.io restart policy, systemd, Docker) starts a fresh
one. Downtime is then recorded on start-up as usual.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tradingbot.config import Config
from tradingbot.journal import Journal

log = logging.getLogger(__name__)

WATCHDOG_INTERVAL_S = 30
ADDRESSBOOK_PATH = "/data/addressbook.json"  # key-free replica maintained by the ops keyvault; never holds secrets


def status(journal: Journal, cfg: Config, started_at: float | None = None) -> tuple[bool, dict]:
    """(healthy, details). A loop that hasn't reported since start-up is judged from start-up time."""
    now = time.time()
    out: dict = {}
    healthy = True
    for key, limit in (("heartbeat", cfg.watchdog_s), ("scan_heartbeat", cfg.scan_watchdog_s)):
        raw = journal.get_state(key)
        last = float(raw) if raw is not None else None
        if started_at is not None:
            last = max(last or 0.0, started_at)
        age = None if last is None else now - last
        ok = age is not None and age <= limit
        out[key] = {"age_s": None if age is None else round(age), "limit_s": limit, "ok": ok}
        healthy &= ok
    return healthy, out


def canary_wallets(cfg) -> dict:
    """Addresses only (never private keys) of the canary trading wallets the bot actually loaded, plus the
    treasury address. Served open at /api/wallets so the live set can be read without pasting anything."""
    out: dict = {
        "wallets": [],
        "treasury": None,
        "source": None,
        "rotate": bool(getattr(cfg, "launch_canary_rotate", False)),
        "route": getattr(cfg, "launch_canary_route", None),
        "wallets_path": getattr(cfg, "launch_canary_wallets_path", "") or None,
    }
    try:
        from tradingbot.execution.rotating import read_canary_wallets

        out["wallets"] = [str(k.pubkey()) for k in read_canary_wallets(cfg)]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"wallets: {e}"
    path = out["wallets_path"]
    if path and os.path.exists(path):
        out["source"] = path
    elif os.path.exists("/run/secrets/canary_wallets.json"):
        out["source"] = "/run/secrets/canary_wallets.json"
    elif os.environ.get("TB_CANARY_WALLETS"):
        out["source"] = "TB_CANARY_WALLETS"
    try:
        from solders.keypair import Keypair

        txt = None
        if os.path.exists("/data/new_treasury.json"):
            with open("/data/new_treasury.json") as fh:
                txt = fh.read()
        elif os.environ.get("TB_TREASURY_KEY"):
            txt = os.environ["TB_TREASURY_KEY"]
        if txt:
            txt = txt.strip().strip('"').strip("'").strip()
            kp = Keypair.from_bytes(bytes(json.loads(txt))) if txt.startswith("[") else Keypair.from_base58_string(txt)
            out["treasury"] = str(kp.pubkey())
    except Exception as e:  # noqa: BLE001
        out["treasury_error"] = str(e)
    return out


def address_book() -> dict:
    """The key-free address book (every treasury/wallet address ever recorded, with labels), read from the
    separate replica file so the web path never opens the file that holds keys. Addresses only, no secrets."""
    try:
        with open(ADDRESSBOOK_PATH) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {"addresses": {}, "count": 0, "note": "addressbook not synced yet (run keyvault.py sync/import)"}
    except Exception as e:  # noqa: BLE001
        return {"addresses": {}, "error": str(e)}


def start_watchdog(journal: Journal, cfg: Config, stop: threading.Event, exit_fn=os._exit) -> threading.Thread:
    started_at = time.time()

    def loop() -> None:
        while not stop.wait(WATCHDOG_INTERVAL_S):
            ok, detail = status(journal, cfg, started_at)
            if not ok:
                log.critical("watchdog: a loop has stalled, exiting for restart: %s", detail)
                logging.shutdown()
                exit_fn(1)

    t = threading.Thread(target=loop, name="watchdog", daemon=True)
    t.start()
    return t


def start_health_server(journal: Journal, cfg: Config, dashboard=None) -> ThreadingHTTPServer:
    """GET /health (open, for Fly's check); with a `dashboard.Dashboard` and the DASHBOARD_TOKEN secret also the
    live dashboard: / (page), /api/stream (server-sent events), /api/summary (JSON)."""
    from urllib.parse import parse_qs, urlsplit
    started_at = time.time()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                for one in (v if isinstance(v, (list, tuple)) else [v]):
                    self.send_header(k, one)
            self.end_headers()
            self.wfile.write(body)

        def _cookie(self, name: str, value: str, max_age: int) -> str:
            host = (self.headers.get("Host") or "").split(":")[0]
            secure = "" if host in ("localhost", "127.0.0.1") else " Secure;"
            if max_age == 0:
                return f"{name}=; Path=/; Max-Age=0; HttpOnly;{secure} SameSite=Strict"
            return f"{name}={value}; Path=/; Max-Age={max_age}; HttpOnly;{secure} SameSite=Strict"

        def _dash_on(self) -> bool:
            """The dashboard is served once a token has ever been set, or passkeys exist (the
            token can be removed after the first passkey is registered)."""
            from tradingbot import dashboard as dash
            from tradingbot import dashpasskey
            return dashboard is not None and (bool(dash.token()) or dashpasskey.has_passkeys(cfg))

        def do_GET(self):  # noqa: N802 (http.server API)
            url = urlsplit(self.path)
            path, query = url.path.rstrip("/") or "/", parse_qs(url.query)
            if path == "/health":
                ok, detail = status(journal, cfg, started_at)
                self._send(200 if ok else 503, json.dumps({"ok": ok, **detail}).encode(), "application/json")
                return
            if path == "/api/wallets":  # open, addresses only (never keys) so the live set is readable
                self._send(200, json.dumps(canary_wallets(cfg)).encode(), "application/json")
                return
            if path == "/api/addressbook":  # open, key-free replica: every address ever, with labels
                self._send(200, json.dumps(address_book()).encode(), "application/json")
                return
            from tradingbot import dashboard as dash
            from tradingbot import dashpasskey
            host = self.headers.get("Host")
            if not self._dash_on():
                self.send_error(404)
                return
            authed = dash.authorised(self.headers, query, cfg)
            # Open sign-in endpoints (no session needed, so you can sign in).
            if path == "/api/auth/status":
                self._send(200, json.dumps({
                    "authed": authed,
                    "token": bool(dash.token()),
                    "passkeys": dashpasskey.has_passkeys(cfg),
                }).encode(), "application/json")
                return
            if path == "/api/auth/login/options":
                try:
                    opts, chal = dashpasskey.authentication_options(cfg, host)
                except Exception as e:  # noqa: BLE001
                    self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
                    return
                self._send(200, opts.encode(), "application/json",
                           {"Cache-Control": "no-store",
                            "Set-Cookie": self._cookie(dashpasskey.CHAL_COOKIE, chal, dashpasskey.CHALLENGE_MAX_AGE_S)})
                return
            if path not in ("/", "/api/stream", "/api/summary", "/api/canary/rotate/log", "/api/canary/topup/log",
                            "/api/canary/settings", "/api/canary/probes", "/api/deploy/log", "/api/auth/passkeys",
                            "/api/auth/register/options"):
                self.send_error(404)
                return
            if query.get("key"):  # remember the break-glass token in a cookie, then drop it from the address bar
                cookie = self._cookie(dash.COOKIE, query["key"][0], 31536000)
                self._send(303, b"", "text/plain", {"Location": path, "Set-Cookie": cookie})
                return
            if path == "/" and not authed:  # show the sign-in page instead of a bare 401
                self._send(200, dash.LOGIN_PAGE.encode(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
                return
            if not authed:
                self._send(401, b"unauthorised", "text/plain; charset=utf-8")
                return
            # Authenticated endpoints.
            if path == "/api/auth/passkeys":
                self._send(200, json.dumps({"passkeys": dashpasskey.list_passkeys(cfg)}).encode(), "application/json")
                return
            if path == "/api/auth/register/options":
                try:
                    opts, chal = dashpasskey.registration_options(cfg, host)
                except Exception as e:  # noqa: BLE001
                    self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
                    return
                self._send(200, opts.encode(), "application/json",
                           {"Cache-Control": "no-store",
                            "Set-Cookie": self._cookie(dashpasskey.CHAL_COOKIE, chal, dashpasskey.CHALLENGE_MAX_AGE_S)})
                return
            if path == "/api/canary/rotate/log":  # progress of a running wallet rotation (addresses only, no keys)
                self._send(200, json.dumps(dash.rotate_log(cfg)).encode(), "application/json")
                return
            if path == "/api/canary/topup/log":  # progress of a running wallet top-up (addresses + amounts only)
                self._send(200, json.dumps(dash.topup_log(cfg)).encode(), "application/json")
                return
            if path == "/api/canary/settings":  # current canary settings + presets for the menu
                self._send(200, json.dumps(dash.canary_settings_menu(cfg)).encode(), "application/json")
                return
            if path == "/api/canary/probes":  # 1-slot probe measurement readout (trip rate, survivors, recent)
                self._send(200, json.dumps(dash.probe_summary(journal)).encode(), "application/json")
                return
            if path == "/api/deploy/log":  # progress of a running deploy
                self._send(200, json.dumps(dash.deploy_log(cfg)).encode(), "application/json")
                return
            if path == "/":
                self._send(200, dash.PAGE.encode(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
            elif path == "/api/summary":
                self._send(200, json.dumps(dashboard.current()[1]).encode(), "application/json")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                def write(data: bytes) -> None:
                    self.wfile.write(data)
                    self.wfile.flush()
                try:
                    dashboard.stream(write)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # the page was closed

        def do_POST(self):  # noqa: N802 (http.server API) -- the dashboard's canary kill-switch buttons
            url = urlsplit(self.path)
            path, query = url.path.rstrip("/") or "/", parse_qs(url.query)
            from tradingbot import dashboard as dash
            from tradingbot import dashpasskey
            host = self.headers.get("Host")
            if not self._dash_on():
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            # Sign-in verify is the one POST that must work without a session.
            if path == "/api/auth/login/verify":
                try:
                    data = json.loads(body or b"{}")
                    chal = dashpasskey.open_challenge(cfg, dash._cookies(self.headers).get(dashpasskey.CHAL_COOKIE, ""),
                                                      "login")
                    sess = dashpasskey.authentication_verify(cfg, host, data.get("credential"), chal) if chal else None
                except Exception as e:  # noqa: BLE001
                    self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
                    return
                if not sess:
                    self._send(401, json.dumps({"error": "passkey not recognised"}).encode(), "application/json")
                    return
                self._send(200, json.dumps({"ok": True}).encode(), "application/json", {"Set-Cookie": self._cookie(
                    dashpasskey.SESS_COOKIE, sess, dashpasskey.SESSION_MAX_AGE_S)})
                log.info("dashboard passkey sign-in")
                return
            if path == "/api/auth/logout":  # clear the passkey session AND the break-glass token cookie
                self._send(200, json.dumps({"ok": True}).encode(), "application/json",
                           {"Set-Cookie": [self._cookie(dashpasskey.SESS_COOKIE, "", 0),
                                           self._cookie(dash.COOKIE, "", 0)]})
                return
            if path not in ("/api/canary/stop", "/api/canary/start", "/api/canary/rotate", "/api/canary/topup",
                            "/api/canary/settings", "/api/canary/probe-smoke", "/api/canary/probe-one",
                            "/api/canary/probe-all", "/api/deploy",
                            "/api/auth/register/verify", "/api/auth/passkeys/remove"):
                self.send_error(404)
                return
            if not dash.authorised(self.headers, query, cfg):
                self._send(401, b"unauthorised", "text/plain; charset=utf-8")
                return
            try:
                if path == "/api/auth/register/verify":
                    data = json.loads(body or b"{}")
                    chal = dashpasskey.open_challenge(cfg, dash._cookies(self.headers).get(dashpasskey.CHAL_COOKIE, ""),
                                                      "reg")
                    if not chal:
                        raise ValueError("the registration window expired, try again")
                    res = dashpasskey.register_verify(cfg, host, data.get("credential"), chal, data.get("label", ""))
                    log.info("dashboard passkey registered: %s", data.get("label", ""))
                    self._send(200, json.dumps(res).encode(), "application/json",
                               {"Set-Cookie": self._cookie(dashpasskey.CHAL_COOKIE, "", 0)})
                    return
                if path == "/api/auth/passkeys/remove":
                    data = json.loads(body or b"{}")
                    ok = dashpasskey.remove_passkey(cfg, data.get("id", ""))
                    res = {"ok": ok, "passkeys": dashpasskey.list_passkeys(cfg)}
                elif path == "/api/deploy":
                    res = dash.fly_deploy(cfg)
                elif path == "/api/canary/settings":
                    res = dash.canary_settings_update(cfg, journal, json.loads(body or b"{}"))
                    log.info("dashboard canary settings: %s", {k: v["to"] for k, v in res.get("changed", {}).items()})
                elif path == "/api/canary/rotate":
                    data = json.loads(body or b"{}")
                    res = dash.canary_rotate(journal, cfg, data.get("wallets"), data.get("sol"))
                    log.info("dashboard wallet rotation: %s", res.get("msg"))
                elif path == "/api/canary/topup":
                    data = json.loads(body or b"{}")
                    res = dash.canary_topup(cfg, data.get("level"), data.get("wallets"))
                    log.info("dashboard wallet top-up: %s", res.get("msg"))
                elif path == "/api/canary/probe-one":  # arm exactly one probe (Execute one probe)
                    res = dash.canary_one(journal)
                    log.info("dashboard probe-one: %s", res.get("msg"))
                elif path == "/api/canary/probe-all":  # probe every liquid pool continuously (clear the one-shot gate + halt)
                    res = dash.canary_probe_all(journal, cfg)
                    log.info("dashboard probe-all: %s", res.get("msg"))
                elif path == "/api/canary/probe-smoke":  # one low-cost real probe to prove the N+1 sell on-chain
                    data = json.loads(body or b"{}")
                    res = dash.probe_smoke(cfg, data.get("mint"), data.get("usd"), data.get("pool"))
                    log.info("dashboard probe-smoke %s: %s", data.get("mint"),
                             (res.get("result") or {}).get("verdict"))
                else:
                    res = dash.canary_switch(journal, "stop" if path.endswith("stop") else "start")
                    log.info("dashboard canary switch: %s", res.get("msg"))
                self._send(200, json.dumps(res).encode(), "application/json")
            except ValueError as e:
                self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            except Exception as e:  # noqa: BLE001 (report the failure, never crash the server)
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")

        def log_message(self, *args):  # keep health probes and page views out of the logs
            pass

    server = ThreadingHTTPServer(("0.0.0.0", cfg.health_port), Handler)
    server.daemon_threads = True  # open streams must not keep the process alive
    threading.Thread(target=server.serve_forever, name="health", daemon=True).start()
    log.info("health endpoint on :%d/health%s", cfg.health_port, " + dashboard" if dashboard else "")
    return server
