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


def start_health_server(journal: Journal, cfg: Config) -> ThreadingHTTPServer:
    started_at = time.time()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            if self.path.rstrip("/") != "/health":
                self.send_error(404)
                return
            ok, detail = status(journal, cfg, started_at)
            body = json.dumps({"ok": ok, **detail}).encode()
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep health probes out of the logs
            pass

    server = ThreadingHTTPServer(("0.0.0.0", cfg.health_port), Handler)
    threading.Thread(target=server.serve_forever, name="health", daemon=True).start()
    log.info("health endpoint on :%d/health", cfg.health_port)
    return server
