"""The box's per-pool absolute reserve view (the WS feed's data half)."""
import base64
import struct

from tradingbot import reserve_tracker as rt


def _vault_bytes(amount: int) -> bytes:
    """A minimal SPL token-account blob: the u64 amount at byte 64 (pumpswap.token_amount reads [64:72])."""
    b = bytearray(72)
    b[64:72] = struct.pack("<Q", amount)
    return bytes(b)


def _push(amount: int):
    """An accountSubscribe-style data payload: [base64, "base64"]."""
    return [base64.b64encode(_vault_bytes(amount)).decode(), "base64"]


def test_decode_vault_amount_reads_the_u64_at_byte_64():
    assert rt.decode_vault_amount(_push(1_234_567)) == 1_234_567
    assert rt.decode_vault_amount(_vault_bytes(99)) == 99          # raw bytes also accepted
    assert rt.decode_vault_amount(None) is None
    assert rt.decode_vault_amount("not base64 !!!") is None


def test_get_needs_both_sides_and_stamps_the_older_slot():
    t = rt.ReserveTracker()
    t.watch("POOL", "SOLV", "TOKV")
    assert t.get("POOL") is None                                   # neither side seen
    t.on_push("SOLV", _push(300_000_000), slot=100)
    assert t.get("POOL") is None                                   # only one side
    t.on_push("TOKV", _push(1_000_000), slot=102)
    assert t.get("POOL") == (300_000_000, 1_000_000, 100)          # as_of = min(100, 102)
    # a later token push re-stamps its side; as_of tracks the older side
    t.on_push("TOKV", _push(900_000), slot=105)
    assert t.get("POOL") == (300_000_000, 900_000, 100)


def test_price_is_the_raw_reserve_ratio():
    t = rt.ReserveTracker()
    t.watch("POOL", "SOLV", "TOKV")
    assert t.price("POOL") is None
    t.on_push("SOLV", _push(300_000_000), slot=1)
    t.on_push("TOKV", _push(1_000_000), slot=1)
    assert t.price("POOL") == 300_000_000 / 1_000_000
    t.on_push("TOKV", _push(0), slot=2)                            # empty token side -> no price, not a crash
    assert t.price("POOL") is None


def test_unknown_vault_and_none_amount_are_ignored():
    t = rt.ReserveTracker()
    t.watch("POOL", "SOLV", "TOKV")
    t.on_vault("STRANGER", 5, slot=1)                              # not a watched vault
    t.on_vault("SOLV", None, slot=1)                               # a failed decode
    assert t.get("POOL") is None


def test_watch_stamp_and_unwatch():
    t = rt.ReserveTracker()
    t.watch("POOL", "SOLV", "TOKV")
    assert sorted(t.vaults_for("POOL")) == ["SOLV", "TOKV"]
    assert t.stamp("POOL") == {"sol": None, "tok": None, "reserve_slot": None}
    t.on_push("SOLV", _push(42), slot=7)
    t.on_push("TOKV", _push(21), slot=7)
    assert t.stamp("POOL") == {"sol": 42, "tok": 21, "reserve_slot": 7}
    t.unwatch("POOL", "SOLV", "TOKV")
    assert t.get("POOL") is None
    assert t.vaults_for("POOL") == []
