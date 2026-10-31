"""GeckoTerminal trending pools: discovery by actual trading activity. No key required.

DexScreener's discovery feeds are mostly paid listings (profiles, boosts), so they skew
towards fresh, self-promoted launches. GeckoTerminal ranks pools by recent trading, which
surfaces tokens with real liquidity and volume. Only token addresses are taken from here;
market data still comes from DexScreener so every candidate is measured the same way.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

log = logging.getLogger(__name__)

BASE = "https://api.geckoterminal.com/api/v2"
TRENDING_WINDOWS = ("1h", "6h")
# Pools quoted in these are fine; tokens that ARE these are not memecoins.
NOT_MEMECOINS = {
    "So11111111111111111111111111111111111111112",   # wrapped SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}


# The free API is advertised at ~30 calls/min per IP, but on 2026-10-27 it let through only ~7-10/min.
# Every client in the process (the pump watcher, its enricher, the engine's trending discovery) shares
# one gate: calls wait their turn in priority order, and the spacing widens on each 429 and narrows
# again as calls succeed, so the 5-minute snapshots the model trades on are never starved by extras.
MIN_INTERVAL_S = 2.2
MAX_INTERVAL_S = 12.0
MULTI_BATCH = 30  # max pools per /pools/multi call
DISCOVER_TTL_S = 600  # the engine's trending discovery is reused this long
# Lower goes first: snapshots feed the 5-minute model; candles make its training labels exact; token info
# only feeds 15-minute-and-later factors; trending lists, discovery and 1-minute candles are extras.
P_SNAPSHOT, P_CANDLES, P_INFO, P_EXTRA = 0, 1, 2, 3
# A call that has waited this long goes next whatever its priority: with the watcher's queue never empty,
# strict priority starved extras for good (10-31: the engine's trending discovery never returned, so its scan
# loop stalled and the watchdog restarted the bot every 30 minutes; 1-minute candles stopped on 10-27).
MAX_WAIT_S = 120.0


class _Gate:
    def __init__(self, max_wait_s: float = MAX_WAIT_S):
        self.cv = threading.Condition()
        self.waiting: list[tuple[int, int, float]] = []  # (priority, seq, enqueued at)
        self.seq = 0
        self.next_free = 0.0
        self.interval = MIN_INTERVAL_S
        self.max_wait_s = max_wait_s

    def _next(self) -> tuple[int, int, float]:
        now = time.monotonic()
        return min(self.waiting, key=lambda t: (-1 if now - t[2] >= self.max_wait_s else t[0], t[1]))

    def acquire(self, priority: int, min_interval: float) -> None:
        if min_interval <= 0:  # tests
            return
        with self.cv:
            self.seq += 1
            ticket = (priority, self.seq, time.monotonic())
            self.waiting.append(ticket)
            self.cv.notify_all()
            while True:
                if self._next() == ticket:
                    wait = self.next_free - time.monotonic()
                    if wait <= 0:
                        self.waiting.remove(ticket)
                        self.next_free = time.monotonic() + max(min_interval, self.interval)
                        self.cv.notify_all()
                        return
                    self.cv.wait(wait)
                else:
                    self.cv.wait(1.0)  # re-check: a waiting ticket may have aged past max_wait_s

    def throttled(self) -> None:
        with self.cv:
            self.interval = min(MAX_INTERVAL_S, self.interval * 1.5)
            self.next_free = max(self.next_free, time.monotonic() + self.interval)

    def ok(self) -> None:
        with self.cv:
            self.interval = max(MIN_INTERVAL_S, self.interval * 0.97)


_gate = _Gate()


def gecko_interval() -> float:
    """Current spacing between GeckoTerminal calls (rises while it rate-limits us)."""
    return _gate.interval


class GeckoTerminal:
    def __init__(self, client: httpx.Client | None = None, network: str = "solana",
                 min_interval_s: float = MIN_INTERVAL_S):
        self.http = client or httpx.Client(base_url=BASE, timeout=15, headers={"Accept": "application/json"})
        self.network = network
        self.min_interval_s = min_interval_s
        self._discovered: tuple[float, list[str]] | None = None
        self.discover_ttl_s = DISCOVER_TTL_S if min_interval_s > 0 else 0.0  # tests: no caching

    def _get(self, path: str, params: dict | None = None, priority: int = P_EXTRA) -> dict | None:
        """Gated GET; a 429 widens the shared spacing and is retried once. None on any failure
        (callers treat it as 'no data')."""
        for attempt in range(2):
            _gate.acquire(priority, self.min_interval_s)
            try:
                resp = self.http.get(path, params=params)
            except httpx.HTTPError as exc:
                log.warning("geckoterminal %s failed: %s", path, exc)
                return None
            if resp.status_code == 429:
                _gate.throttled()
                if attempt == 0:
                    continue
            if resp.status_code != 200:
                log.warning("geckoterminal %s -> HTTP %s", path, resp.status_code)
                return None
            _gate.ok()
            try:
                return resp.json()
            except ValueError:
                return None
        return None

    def trending_ranked(self, duration: str = "1h") -> list[str]:
        """Base-token mints of trending pools in rank order (no dedupe across pools of one token)."""
        data = self._get(f"/networks/{self.network}/trending_pools", {"duration": duration}, P_EXTRA)
        return [m for m in (base_token(p, self.network) for p in (data or {}).get("data") or [])
                if m and m not in NOT_MEMECOINS]

    def sol_price(self) -> float | None:
        sol = "So11111111111111111111111111111111111111112"
        data = self._get(f"/simple/networks/{self.network}/token_price/{sol}")
        try:
            return float(data["data"]["attributes"]["token_prices"][sol])
        except (TypeError, KeyError, ValueError):
            return None

    def token_prices(self, mints: list[str]) -> dict[str, float]:
        data = self._get(f"/simple/networks/{self.network}/token_price/{','.join(mints)}")
        try:
            raw = data["data"]["attributes"]["token_prices"]
        except (TypeError, KeyError):
            return {}
        return {m: float(v) for m, v in raw.items() if v is not None}

    def new_pools(self, page: int = 1) -> list[dict]:
        """Newest pools first (bonding-curve launches and graduations), with base_token and dex ids."""
        data = self._get(f"/networks/{self.network}/new_pools", {"page": page}, P_SNAPSHOT)
        return (data or {}).get("data") or []

    def pools(self, addresses: list[str]) -> dict[str, dict]:
        """Current attributes of up to many pools, keyed by pool address (batched)."""
        out: dict[str, dict] = {}
        for i in range(0, len(addresses), MULTI_BATCH):
            chunk = ",".join(addresses[i:i + MULTI_BATCH])
            data = self._get(f"/networks/{self.network}/pools/multi/{chunk}", priority=P_SNAPSHOT)
            for pool in (data or {}).get("data") or []:
                attrs = pool.get("attributes") or {}
                if attrs.get("address"):
                    out[attrs["address"]] = attrs
        return out

    def token_info(self, mint: str) -> dict | None:
        """Socials, developer wallet and holdings, launchpad progress, authorities, holders."""
        data = self._get(f"/networks/{self.network}/tokens/{mint}/info", priority=P_INFO)
        return ((data or {}).get("data") or {}).get("attributes")

    def ohlcv(self, pool: str, minutes: int = 5, limit: int = 300, before: int | None = None) -> list[list[float]] | None:
        """[ts, open, high, low, close, volume] candles, oldest first: the `limit` candles ending at `before`
        (unix seconds), or the latest ones without it."""
        params = {"aggregate": minutes, "limit": limit}
        if before is not None:
            params["before_timestamp"] = before
        data = self._get(f"/networks/{self.network}/pools/{pool}/ohlcv/minute", params,
                         P_CANDLES if minutes >= 5 else P_EXTRA)
        if data is None:
            return None
        rows = ((data.get("data") or {}).get("attributes") or {}).get("ohlcv_list") or []
        return sorted(rows, key=lambda r: r[0])

    def discover(self) -> list[str]:
        """Base-token addresses of trending pools, most active first, without duplicates (reused for
        DISCOVER_TTL_S: the engine asks every scan cycle, and the list barely moves in a minute)."""
        if self._discovered and time.monotonic() - self._discovered[0] < self.discover_ttl_s:
            return list(self._discovered[1])
        seen: dict[str, None] = {}
        for window in TRENDING_WINDOWS:
            data = self._get(f"/networks/{self.network}/trending_pools", {"duration": window}, P_EXTRA)
            if data is None:
                continue
            pools = data.get("data") or []
            for pool in pools:
                addr = base_token(pool, self.network)
                if addr and addr not in NOT_MEMECOINS:
                    seen.setdefault(addr, None)
        self._discovered = (time.monotonic(), list(seen))
        return list(seen)


def _rel_id(pool: dict, rel: str) -> str:
    return (((pool.get("relationships") or {}).get(rel) or {}).get("data") or {}).get("id", "")


def base_token(pool: dict, network: str = "solana") -> str | None:
    """'solana_<mint>' relationship id -> '<mint>'."""
    ident = _rel_id(pool, "base_token")
    prefix = f"{network}_"
    return ident[len(prefix):] if ident.startswith(prefix) else None


def dex_id(pool: dict) -> str:
    return _rel_id(pool, "dex")
