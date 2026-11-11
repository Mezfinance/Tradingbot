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


def test_canary_wallets_reports_loaded_addresses(cfg, tmp_path):
    from solders.keypair import Keypair

    kps = [Keypair(), Keypair()]
    f = tmp_path / "wallets.json"
    f.write_text(json.dumps([list(bytes(k)) for k in kps]))
    cfg.launch_canary_wallets_path = str(f)
    out = health.canary_wallets(cfg)
    assert out["wallets"] == [str(k.pubkey()) for k in kps]  # addresses, never keys
    assert out["source"] == str(f)
    assert all(isinstance(w, str) for w in out["wallets"])
    assert "secret" not in json.dumps(out).lower()


def test_wallets_endpoint_open(cfg, tmp_path):
    from solders.keypair import Keypair

    kp = Keypair()
    f = tmp_path / "wallets.json"
    f.write_text(json.dumps([list(bytes(kp))]))
    cfg.launch_canary_wallets_path = str(f)
    cfg.health_port = 18767
    j = Journal(":memory:")
    server = health.start_health_server(j, cfg)
    try:
        with urllib.request.urlopen("http://127.0.0.1:18767/api/wallets") as r:
            body = json.loads(r.read())
        assert r.status == 200 and body["wallets"] == [str(kp.pubkey())]
    finally:
        server.shutdown()


def test_addressbook_endpoint_open(cfg, tmp_path, monkeypatch):
    book = {"version": 1, "count": 1, "addresses": {"Abc123": {"label": "treasury-current", "created": "x", "notes": []}}}
    p = tmp_path / "addressbook.json"
    p.write_text(json.dumps(book))
    monkeypatch.setattr(health, "ADDRESSBOOK_PATH", str(p))
    assert health.address_book() == book
    cfg.health_port = 18768
    server = health.start_health_server(Journal(":memory:"), cfg)
    try:
        with urllib.request.urlopen("http://127.0.0.1:18768/api/addressbook") as r:
            got = json.loads(r.read())
        assert r.status == 200 and got["addresses"]["Abc123"]["label"] == "treasury-current"
    finally:
        server.shutdown()


def test_addressbook_missing_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "ADDRESSBOOK_PATH", str(tmp_path / "nope.json"))
    out = health.address_book()
    assert out["addresses"] == {} and "note" in out


def test_backup_while_open(tmp_path):
    j = Journal(str(tmp_path / "live.db"))
    j.set_state("cash:claude", "123")
    j.backup(str(tmp_path / "copy.db"))
    assert Journal(str(tmp_path / "copy.db")).get_state("cash:claude") == "123"
