"""WS reserve feed helpers + the Consumer's on_pool / reserve-stamp hooks."""
import base64
import struct

from tradingbot import localshred_consumer as lc
from tradingbot import reserve_tracker as rt
from tradingbot import reserve_ws as rw
from tradingbot.data import pumpswap as ps

_ALPH = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58dec(s: str) -> bytes:
    n = 0
    for c in s:
        n = n * 58 + _ALPH.index(c)
    return n.to_bytes(32, "big")


def _pool_account(token_mint_raw=b"\x02" * 32, sol_vault_raw=b"\x03" * 32, token_vault_raw=b"\x04" * 32) -> bytes:
    """A minimal pumpswap pool account: WSOL as the base (sol_is_base), so sol_vault=base_vault@139,
    token_vault=quote_vault@171, token_mint=quote@75."""
    b = bytearray(243)
    b[43:75] = _b58dec(ps.WSOL)       # base = WSOL -> sol_is_base
    b[75:107] = token_mint_raw        # quote = the token
    b[139:171] = sol_vault_raw        # base_vault -> sol_vault
    b[171:203] = token_vault_raw      # quote_vault -> token_vault
    return bytes(b)


def test_pool_vaults_extracts_both_vaults_and_the_mint():
    acc = _pool_account()
    v = rw.pool_vaults(base64.b64encode(acc).decode())      # accepts a base64 str
    assert v == {"sol_vault": ps.b58encode(b"\x03" * 32),
                 "token_vault": ps.b58encode(b"\x04" * 32),
                 "token_mint": ps.b58encode(b"\x02" * 32)}
    assert rw.pool_vaults([base64.b64encode(acc).decode(), "base64"]) == v   # [b64, "base64"] form
    assert rw.pool_vaults(b"too short") is None
    assert rw.pool_vaults("!!!not base64") is None


def test_subscribe_confirm_and_route_a_push():
    subs = rw.SubManager()
    rid, req = subs.subscribe("POOL", "SOLV", now=1000.0)
    assert req["method"] == "accountSubscribe" and req["params"][0] == "SOLV"
    assert req["params"][1]["commitment"] == "processed"
    # the subscribe ack binds sub_id -> vault
    ev = rw.parse_message({"id": rid, "result": 77}, subs)
    assert ev == ("subscribed", rid, 77)
    subs.confirm(rid, 77)
    assert subs.vault_of(77) == "SOLV"
    # an accountNotification routes to the vault with its slot + data
    note = {"method": "accountNotification",
            "params": {"subscription": 77, "result": {"context": {"slot": 555}, "value": {"data": ["ZGF0YQ==", "base64"]}}}}
    assert rw.parse_message(note, subs) == ("push", "SOLV", ["ZGF0YQ==", "base64"], 555)
    # an unknown subscription id yields nothing
    note["params"]["subscription"] = 999
    assert rw.parse_message(note, subs) is None


def test_expire_and_drop_returns_sub_ids():
    subs = rw.SubManager()
    r1, _ = subs.subscribe("P", "SV", now=0.0)
    r2, _ = subs.subscribe("P", "TV", now=0.0)
    subs.confirm(r1, 1)
    subs.confirm(r2, 2)
    assert subs.expired(now=100.0, ttl=200.0) == []
    assert subs.expired(now=300.0, ttl=200.0) == ["P"]
    assert sorted(subs.drop("P")) == [1, 2]
    assert subs.vault_of(1) is None and subs.expired(now=300.0, ttl=200.0) == []
    assert rw.account_unsubscribe_request(9, 2) == {"jsonrpc": "2.0", "id": 9, "method": "accountUnsubscribe", "params": [2]}


# --- the Consumer's hooks ---

class _FakeRunner:
    def on_slot(self, rec):
        return {"signal_slot": rec["slot"], "eta": 1.0, "sym": "N", "m2": ".N", "tier": "1"}


def _vault_blob(amount):
    b = bytearray(72)
    b[64:72] = struct.pack("<Q", amount)
    return [base64.b64encode(bytes(b)).decode(), "base64"]


def test_on_pool_fires_once_per_new_handmade_pool():
    seen = []
    c = lc.Consumer({}, runner_factory=_FakeRunner, fresh_window_slots=0, on_pool=seen.append)
    c.on_swap(100, "POOL", "SomeMintNotPump", True, "W")
    c.on_swap(101, "POOL", "SomeMintNotPump", True, "W")     # same pool, advanced -> no second on_pool
    c.on_swap(102, "POOL2", "OtherMintNotPump", True, "W")
    assert seen == ["POOL", "POOL2"]
    # a pump.fun graduation is skipped entirely (handmade_only)
    c.on_swap(103, "GRAD", "endswithpump", True, "W")
    assert "GRAD" not in seen


def test_fire_is_stamped_with_the_box_reserve_view():
    fires = []
    tracker = rt.ReserveTracker()
    tracker.watch("POOL", "SOLV", "TOKV")
    tracker.on_push("SOLV", _vault_blob(300_000_000), slot=100)
    tracker.on_push("TOKV", _vault_blob(1_000_000), slot=100)
    c = lc.Consumer({}, runner_factory=_FakeRunner, fresh_window_slots=0, reserves=tracker, on_fire=fires.append)
    c.on_swap(100, "POOL", "SomeMintNotPump", True, "W")
    c.on_swap(101, "POOL", "SomeMintNotPump", True, "W")     # advancing the slot flushes slot 100 -> a fire
    assert len(fires) == 1
    assert fires[0]["sol"] == 300_000_000 and fires[0]["tok"] == 1_000_000 and fires[0]["reserve_slot"] == 100
    line = lc.fire_line(fires[0])
    assert line["sol"] == 300_000_000 and line["tok"] == 1_000_000 and line["reserve_slot"] == 100


def test_unstamped_fire_has_no_reserve_fields():
    fires = []
    c = lc.Consumer({}, runner_factory=_FakeRunner, fresh_window_slots=0, on_fire=fires.append)
    c.on_swap(100, "POOL", "SomeMintNotPump", True, "W")
    c.on_swap(101, "POOL", "SomeMintNotPump", True, "W")
    assert "sol" not in lc.fire_line(fires[0])
