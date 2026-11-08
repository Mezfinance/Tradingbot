"""Helius backup key: when the main key runs out of credits every default client moves to BACKUP_HELIUS_API_KEY,
and moves back once a probe of the main key answers again."""
import httpx
import pytest

from tradingbot.data import helius as hl


class _Http:
    """Answers per key: 'empty' keys get Helius's out-of-credits 429, others a result."""

    def __init__(self, empty):
        self.empty, self.keys = set(empty), []

    def post(self, url, params=None, json=None):
        key = params["api-key"]
        self.keys.append(key)
        req = httpx.Request("POST", url)
        if key in self.empty:
            return httpx.Response(429, json={"jsonrpc": "2.0", "error": {"code": -32429, "message": "max usage reached"}},
                                  request=req)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"key": key}}, request=req)


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setenv("HELIUS_API_KEY", "MAIN")
    monkeypatch.setenv(hl.BACKUP_ENV, "BACKUP")
    monkeypatch.setattr(hl, "KEYS", {"on_backup": False, "since": None, "switches": 0, "probe_at": 0.0,
                                     "last_reason": None})
    monkeypatch.setattr(hl, "MIN_INTERVAL_S", 0.0)


def test_switches_to_the_backup_when_the_main_is_out_of_credits_and_back_when_it_answers(monkeypatch):
    http = _Http(empty={"MAIN"})
    h = hl.Helius(client=http)
    h.min_interval_s = 0
    assert h._rpc("getSlot", [], 1) == {"key": "BACKUP"}  # the same call retried on the backup
    assert http.keys == ["MAIN", "BACKUP"] and hl.KEYS["on_backup"] and hl.current_key() == "BACKUP"
    other = hl.Helius(client=http)  # every default client follows
    other.min_interval_s = 0
    assert other.key() == "BACKUP" and other._rpc("getSlot", [], 1) == {"key": "BACKUP"}
    # an hour later one call probes the main key; still empty: stays on the backup
    monkeypatch.setattr(hl.time, "time", lambda: hl.KEYS["probe_at"] + 1)
    http.keys.clear()
    assert h._rpc("getSlot", [], 1) == {"key": "BACKUP"} and http.keys == ["MAIN", "BACKUP"]
    # credits back on the main: the next probe moves every client back
    http.empty.clear()
    monkeypatch.setattr(hl.time, "time", lambda: hl.KEYS["probe_at"] + 1)
    assert h._rpc("getSlot", [], 1) == {"key": "MAIN"}
    assert not hl.KEYS["on_backup"] and hl.KEYS["switches"] == 2 and other.key() == "MAIN"


def test_a_main_key_refusing_every_retry_moves_to_the_backup(monkeypatch):
    class _Busy(_Http):
        def post(self, url, params=None, json=None):
            if params["api-key"] == "MAIN":  # a bare 429 with no credits message, as seen on Fly 11-08
                self.keys.append("MAIN")
                return httpx.Response(429, text="Too Many Requests", request=httpx.Request("POST", url))
            return super().post(url, params, json)
    monkeypatch.setattr(hl.time, "sleep", lambda s: None)
    http = _Busy([])
    h = hl.Helius(client=http)
    h.min_interval_s = 0
    assert h._rpc("getSlot", [], 1) == {"key": "BACKUP"}
    assert http.keys == ["MAIN"] * (hl.RETRIES_429 + 1) + ["BACKUP"]
    assert hl.KEYS["on_backup"] and hl.KEYS["probe_at"] - hl.KEYS["since"] == pytest.approx(hl.RATE_PROBE_S, abs=1)


def test_explicit_keys_do_not_switch(monkeypatch):
    fixed = hl.Helius(api_key="OWN", client=_Http(empty={"OWN"}))  # the backfill's own account never falls back
    fixed.min_interval_s = 0
    monkeypatch.setattr(hl, "RETRIES_429", 0)
    assert fixed._rpc("getSlot", [], 1) is None and not hl.KEYS["on_backup"] and fixed.key() == "OWN"


def test_no_main_key_uses_the_backup(monkeypatch):
    monkeypatch.setenv("HELIUS_API_KEY", "")
    h = hl.Helius(client=_Http(empty=set()))
    h.min_interval_s = 0
    assert h.enabled and h.key() == "BACKUP" and hl.current_key() == "BACKUP"
    monkeypatch.delenv(hl.BACKUP_ENV)
    assert not hl.Helius(client=_Http(empty=set())).enabled
