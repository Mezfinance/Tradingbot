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
    h = hl.Helius(role=hl.LIVE, client=http)
    h.min_interval_s = 0
    assert h._rpc("getSlot", [], 1) == {"key": "BACKUP"}  # the same call retried on the backup
    assert http.keys == ["MAIN", "BACKUP"] and hl.KEYS["on_backup"] and hl.current_key() == "BACKUP"
    other = hl.Helius(role=hl.LIVE, client=http)  # every default client follows
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
    h = hl.Helius(role=hl.LIVE, client=http)
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
    h = hl.Helius(role=hl.LIVE, client=_Http(empty=set()))
    h.min_interval_s = 0
    assert h.enabled and h.key() == "BACKUP" and hl.current_key() == "BACKUP"
    monkeypatch.delenv(hl.BACKUP_ENV)
    assert not hl.Helius(role=hl.LIVE, client=_Http(empty=set())).enabled


def test_a_rate_limited_method_slows_down_alone_and_recovers(monkeypatch):
    monkeypatch.setattr(hl, "_method_pace", {})
    monkeypatch.setattr(hl.time, "sleep", lambda s: None)

    class _Limited(_Http):
        def __init__(self):
            super().__init__(empty=())
            self.busy = 2

        def post(self, url, params=None, json=None):
            if json["method"] == "getTransactionsForAddress" and self.busy:
                self.busy -= 1
                return httpx.Response(429, text="Too Many Requests", request=httpx.Request("POST", url))
            return super().post(url, params, json)

    http = _Limited()
    h = hl.Helius(role=hl.LIVE, client=http)
    h.min_interval_s = 0.15
    assert h._rpc("getTransactionsForAddress", [], 10) == {"key": "MAIN"}  # two 429s, then answered
    slowed = hl.pace_status()["getTransactionsForAddress"]
    assert slowed > 0.15 and "getSlot" not in hl.pace_status()  # other methods keep the base pace
    assert not hl.KEYS["on_backup"]  # a 429 that clears on retry doesn't switch keys
    for _ in range(100):  # successes narrow it back to the base
        h._rpc("getTransactionsForAddress", [], 10)
    assert "getTransactionsForAddress" not in hl.pace_status()
    for _ in range(40):
        hl._adapt("x", True, 0.15)
    assert hl.pace_status()["x"] == hl.MAX_INTERVAL_S
    assert "slowed_methods" in hl.key_status()


def test_a_cheap_call_answering_on_the_main_key_does_not_move_clients_back(monkeypatch):
    """Main key out of credits for history calls but still answering cheap ones: only a history call may probe."""
    class _Partial(_Http):
        def post(self, url, params=None, json=None):
            if params["api-key"] == "MAIN" and json["method"] == "getTransactionsForAddress":
                self.keys.append("MAIN")
                return httpx.Response(429, json={"error": {"message": "max usage reached"}},
                                      request=httpx.Request("POST", url))
            return super().post(url, params, json)

    http = _Partial(empty=())
    h = hl.Helius(role=hl.LIVE, client=http)
    h.min_interval_s = 0
    assert h._rpc("getTransactionsForAddress", [], 10) == {"key": "BACKUP"}
    assert hl.KEYS["on_backup"] and hl.key_status()["probe_method"] == "getTransactionsForAddress"
    monkeypatch.setattr(hl.time, "time", lambda: hl.KEYS["probe_at"] + 1)
    http.keys.clear()
    assert h._rpc("getSlot", [], 1) == {"key": "BACKUP"}  # a cheap call doesn't probe
    assert http.keys == ["BACKUP"] and hl.KEYS["on_backup"] and hl.KEYS["switches"] == 1
    assert h._rpc("getTransactionsForAddress", [], 10) == {"key": "BACKUP"}  # the probe: main still empty
    assert http.keys[-2:] == ["MAIN", "BACKUP"] and hl.KEYS["on_backup"] and hl.KEYS["switches"] == 1


def test_a_refused_websocket_moves_to_the_backup_then_the_public_node_and_probes_back(monkeypatch):
    """11-09: after a restart on the out-of-credits main key Helius refused the websocket (HTTP 429), no pool was
    seen, so no RPC call failed and nothing switched keys: the bot traded nothing until the websocket moved."""
    from tradingbot import launch_live as ll
    from tradingbot.config import Config
    from tradingbot.watcher import WatchStore
    live = ll.LaunchLive(Config(db_path=":memory:"), WatchStore(":memory:"), helius=hl.Helius(role=hl.LIVE))
    assert "api-key=MAIN" in live.ws_url()
    refused = Exception("server rejected WebSocket connection: HTTP 429")
    live._ws_refused(refused)
    assert hl.KEYS["on_backup"] and hl.KEYS["method"] == "websocket" and "api-key=BACKUP" in live.ws_url()
    live._ws_refused(Exception("connection reset"))  # an ordinary drop changes nothing
    assert "api-key=BACKUP" in live.ws_url()
    live._ws_refused(refused)  # the backup refused too
    assert live.ws_url() == ll.WS_PUBLIC
    live._ws_public_until = 0.0
    now = hl.KEYS["probe_at"] + 1
    monkeypatch.setattr(hl.time, "time", lambda: now)
    assert "api-key=MAIN" in live.ws_url() and live._ws_probing  # the probe tries the main key on the websocket
    live._ws_refused(refused)  # still refused: stays on the backup, next probe later
    assert hl.KEYS["on_backup"] and hl.KEYS["probe_at"] == now + hl.RATE_PROBE_S
