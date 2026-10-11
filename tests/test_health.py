import json
import threading
import time
import urllib.request

from tradingbot import health
from tradingbot.journal import Journal


def test_status_flags_stale_loops(cfg):
    j = Journal(":memory:")
    ok, detail = health.status(j, cfg)
    assert not ok and detail["heartbeat"]["age_s"] is None  # never ran
    now = time.time()
    j.set_state("heartbeat", repr(now))
    j.set_state("scan_heartbeat", repr(now))
    assert health.status(j, cfg)[0]
    j.set_state("heartbeat", repr(now - cfg.watchdog_s - 5))
    ok, detail = health.status(j, cfg)
    assert not ok and not detail["heartbeat"]["ok"] and detail["scan_heartbeat"]["ok"]


def test_startup_grace(cfg):
    j = Journal(":memory:")
    j.set_state("heartbeat", repr(time.time() - 99999))  # stale from a previous run
    j.set_state("scan_heartbeat", repr(time.time() - 99999))
    assert health.status(j, cfg, started_at=time.time())[0]


def test_watchdog_exits_when_stalled(cfg, monkeypatch):
    monkeypatch.setattr(health, "WATCHDOG_INTERVAL_S", 0.01)
    cfg.watchdog_s = 0
    j = Journal(":memory:")
    exited = threading.Event()
    stop = threading.Event()
    health.start_watchdog(j, cfg, stop, exit_fn=lambda code: (exited.set(), stop.set()))
    assert exited.wait(2)


def test_health_endpoint(cfg):
    cfg.health_port = 18765
    j = Journal(":memory:")
    server = health.start_health_server(j, cfg)
    try:
        with urllib.request.urlopen("http://127.0.0.1:18765/health") as r:
            body = json.loads(r.read())
        assert r.status == 200 and body["ok"]
    finally:
        server.shutdown()


def test_backup_while_open(tmp_path):
    j = Journal(str(tmp_path / "live.db"))
    j.set_state("cash:claude", "123")
    j.backup(str(tmp_path / "copy.db"))
    assert Journal(str(tmp_path / "copy.db")).get_state("cash:claude") == "123"
