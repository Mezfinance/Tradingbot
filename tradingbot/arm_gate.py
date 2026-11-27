"""Helius sec-4 arm gate: confirm a shred-discovered pool is a hand-made SOL-pair pumpswap pool before the
front-run gate may fire on it.

ShredStream discovers every fresh pool whose token mint does not end 'pump' (localshred_consumer.is_handmade,
a heuristic over the mint string). This does ONE getAccountInfo on the pool account ~`delay_s` after first
sight and runs pumpswap.arm_ok on the bytes (coin_creator all zeros = hand-made, exactly one WSOL side, LP
supply > 0). A pool that cannot be CONFIRMED hand-made is dropped: is_armed returns False on a bad read as
well as on a non-hand-made pool, so real money never fires on an unconfirmed pool (fail-closed). A pool not
read yet returns None, which localshred_consumer treats as "suppress for now" -- the bank keeps advancing, so
a pool armed a few slots late still fires on its later slots with full history.

Non-blocking by design (the 2026-11-23 LUT lesson: a synchronous HTTP read on the gRPC consume thread stalls
the stream and the proxy drops it). schedule(pool) only enqueues; the read runs off-thread. The HTTP fetch is
injected (fetch(pool) -> account bytes | None) so the gate logic is unit-testable without a live endpoint; the
box passes a urllib Helius-RESEARCH getAccountInfo fetch (never the live execution RPC the executor uses).

This is the repo-tested core; the box wires it in run_localshred.py: on_pool(pool) -> gate.schedule(pool),
is_armed=gate.is_armed on lc.Consumer, and a daemon gate.run(...) draining the reads.
"""
import heapq
import threading
import time
from typing import Callable, Optional

try:                                  # in-package and standalone-on-the-box both work
    from .data import pumpswap
except ImportError:                   # noqa: E722 -- box-flat copy sits beside this file
    import pumpswap


class ArmGate:
    def __init__(self, fetch: Callable[[str], Optional[bytes]], arm_ok: Optional[Callable] = None,
                 delay_s: float = 4.0, clock: Optional[Callable[[], float]] = None):
        self.fetch = fetch
        self.arm_ok = arm_ok or pumpswap.arm_ok
        self.delay_s = float(delay_s)
        self._clock = clock or time.monotonic
        self._armed: dict = {}            # pool -> True (fire) / False (drop); absent = not read yet (suppress)
        self._pending: set = set()
        self._heap: list = []             # (due_ts, pool), min-heap by due time
        self._lock = threading.Lock()

    def is_armed(self, pool: str):
        """True = confirmed hand-made (fire); False = confirmed not / unreadable (drop); None = not read yet."""
        return self._armed.get(pool)

    def schedule(self, pool: str) -> None:
        """Enqueue a pool for its sec-4 read, once. Cheap and non-blocking (safe on the gRPC consume thread)."""
        with self._lock:
            if pool in self._armed or pool in self._pending:
                return
            self._pending.add(pool)
            heapq.heappush(self._heap, (self._clock() + self.delay_s, pool))

    def _pop_due(self, now: float) -> list:
        due = []
        with self._lock:
            while self._heap and self._heap[0][0] <= now:
                due.append(heapq.heappop(self._heap)[1])
        return due

    def _read(self, pool: str) -> bool:
        try:
            raw = self.fetch(pool)
            armed = bool(raw and self.arm_ok(raw))
        except Exception:
            armed = False                 # fail-closed: cannot confirm hand-made -> never fire real money on it
        with self._lock:
            self._armed[pool] = armed
            self._pending.discard(pool)
        return armed

    def worker_once(self, now: Optional[float] = None) -> list:
        """Read every pool whose sec-4 delay has elapsed by `now` (default: the clock). Synchronous; the unit
        tests drive the gate through this. Returns the pools read this call."""
        now = self._clock() if now is None else now
        read = self._pop_due(now)
        for pool in read:
            self._read(pool)
        return read

    def run(self, stop: Optional["threading.Event"] = None, workers: int = 6, poll_s: float = 0.1,
            on_result: Optional[Callable[[str, bool], None]] = None) -> None:
        """Box live loop: dispatch due reads across `workers` threads so the ~4s delay and the HTTP read never
        block each other or the consume thread. Runs until `stop` is set (or forever). Not used in tests."""
        import queue
        q: "queue.Queue" = queue.Queue()

        def _worker():
            while True:
                pool = q.get()
                if pool is None:
                    return
                armed = self._read(pool)
                if on_result:
                    try:
                        on_result(pool, armed)
                    except Exception:
                        pass
        threads = [threading.Thread(target=_worker, daemon=True) for _ in range(max(1, workers))]
        for t in threads:
            t.start()
        while stop is None or not stop.is_set():
            for pool in self._pop_due(self._clock()):
                q.put(pool)
            time.sleep(poll_s)
        for _ in threads:
            q.put(None)
