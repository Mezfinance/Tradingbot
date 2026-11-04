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
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (http.server API)
            url = urlsplit(self.path)
            path, query = url.path.rstrip("/") or "/", parse_qs(url.query)
            if path == "/health":
                ok, detail = status(journal, cfg, started_at)
                self._send(200 if ok else 503, json.dumps({"ok": ok, **detail}).encode(), "application/json")
                return
            from tradingbot import dashboard as dash
            if dashboard is None or not dash.token() or path not in ("/", "/api/stream", "/api/summary"):
                self.send_error(404)
                return
            if not dash.authorised(self.headers, query):
                self._send(401, b"Open this page once with ?key=<DASHBOARD_TOKEN> at the end of the address.",
                           "text/plain; charset=utf-8")
                return
            if query.get("key"):  # remember the key in a cookie, then drop it from the address bar
                cookie = f"{dash.COOKIE}={query['key'][0]}; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Strict"
                self._send(303, b"", "text/plain", {"Location": path, "Set-Cookie": cookie})
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

        def log_message(self, *args):  # keep health probes and page views out of the logs
            pass

    server = ThreadingHTTPServer(("0.0.0.0", cfg.health_port), Handler)
    server.daemon_threads = True  # open streams must not keep the process alive
    threading.Thread(target=server.serve_forever, name="health", daemon=True).start()
    log.info("health endpoint on :%d/health%s", cfg.health_port, " + dashboard" if dashboard else "")
    return server
