"""The Helius sec-4 arm gate: schedule -> delayed read -> is_armed, fail-closed, pumpswap.arm_ok by default."""
from tradingbot import arm_gate
from tradingbot.data import pumpswap as ps
from tests.test_pumpswap import HAND, GRAD      # real mainnet pool bytes


def test_arm_gate_arms_after_the_delay_and_reads_once():
    clk = {"t": 0.0}
    reads = []

    def fetch(pool):
        reads.append(pool)
        return b"ok"

    g = arm_gate.ArmGate(fetch, arm_ok=lambda raw: raw == b"ok", delay_s=4.0, clock=lambda: clk["t"])
    g.schedule("P")
    g.schedule("P")                                   # idempotent -> one read
    assert g.is_armed("P") is None                    # not read yet -> suppress
    assert g.worker_once(now=3.9) == []               # before the sec-4 delay, nothing due
    assert g.is_armed("P") is None
    assert g.worker_once(now=4.0) == ["P"]            # delay elapsed -> read and arm
    assert g.is_armed("P") is True
    assert reads == ["P"]
    g.schedule("P")                                   # already known -> never re-read
    g.worker_once(now=100.0)
    assert reads == ["P"]


def test_arm_gate_fails_closed_on_a_bad_or_empty_read():
    def fetch(pool):
        return None if pool == "EMPTY" else b"ok"     # empty account

    def boom(pool):
        raise RuntimeError("rpc down")

    g = arm_gate.ArmGate(fetch, arm_ok=lambda raw: True, delay_s=0.0, clock=lambda: 0.0)
    g.schedule("EMPTY")
    g.worker_once(now=0.0)
    assert g.is_armed("EMPTY") is False               # no bytes -> drop

    g2 = arm_gate.ArmGate(boom, arm_ok=lambda raw: True, delay_s=0.0, clock=lambda: 0.0)
    g2.schedule("ERR")
    g2.worker_once(now=0.0)
    assert g2.is_armed("ERR") is False                # fetch raised -> drop (never fire real money unconfirmed)


def test_arm_gate_defaults_to_pumpswap_arm_ok():
    g = arm_gate.ArmGate(lambda p: {"H": HAND, "G": GRAD}[p], delay_s=0.0, clock=lambda: 0.0)
    g.schedule("H")
    g.schedule("G")
    g.worker_once(now=0.0)
    assert g.is_armed("H") is True                     # hand-made SOL pair, LP live
    assert g.is_armed("G") is False                    # graduation (coin_creator set) -> not hand-made
    assert ps.arm_ok(HAND) is True and ps.arm_ok(GRAD) is False   # the gate used arm_ok, not a stub
