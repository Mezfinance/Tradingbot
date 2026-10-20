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


# The free API allows ~30 calls/min per IP. Every client in the process (trending discovery,
# the pump watcher) shares one pace so together they stay under it.
MIN_INTERVAL_S = 2.2
MULTI_BATCH = 30  # max pools per /pools/multi call
_pace = threading.Lock()
_last_call = [0.0]


class GeckoTerminal:
    def __init__(self, client: httpx.Client | None = None, network: str = "solana",
                 min_interval_s: float = MIN_INTERVAL_S):
        self.http = client or httpx.Client(base_url=BASE, timeout=15, headers={"Accept": "application/json"})
        self.network = network
        self.min_interval_s = min_interval_s

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        """Paced GET; one back-off retry on 429. None on any failure (callers treat it as 'no data')."""
        for attempt in range(2):
            with _pace:
                wait = _last_call[0] + self.min_interval_s - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                _last_call[0] = time.monotonic()
            try:
                resp = self.http.get(path, params=params)
            except httpx.HTTPError as exc:
                log.warning("geckoterminal %s failed: %s", path, exc)
                return None
            if resp.status_code == 429 and attempt == 0:
                time.sleep(5 * self.min_interval_s)
                continue
            if resp.status_code != 200:
                log.warning("geckoterminal %s -> HTTP %s", path, resp.status_code)
                return None
            try:
                return resp.json()
            except ValueError:
                return None
        return None

    def trending_ranked(self, duration: str = "1h") -> list[str]:
        """Base-token mints of trending pools in rank order (no dedupe across pools of one token)."""
        data = self._get(f"/networks/{self.network}/trending_pools", {"duration": duration})
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
        data = self._get(f"/networks/{self.network}/new_pools", {"page": page})
        return (data or {}).get("data") or []

    def pools(self, addresses: list[str]) -> dict[str, dict]:
        """Current attributes of up to many pools, keyed by pool address (batched)."""
        out: dict[str, dict] = {}
        for i in range(0, len(addresses), MULTI_BATCH):
            chunk = ",".join(addresses[i:i + MULTI_BATCH])
            data = self._get(f"/networks/{self.network}/pools/multi/{chunk}")
            for pool in (data or {}).get("data") or []:
                attrs = pool.get("attributes") or {}
                if attrs.get("address"):
                    out[attrs["address"]] = attrs
        return out

    def token_info(self, mint: str) -> dict | None:
        """Socials, developer wallet and holdings, launchpad progress, authorities, holders."""
        data = self._get(f"/networks/{self.network}/tokens/{mint}/info")
        return ((data or {}).get("data") or {}).get("attributes")

    def ohlcv(self, pool: str, minutes: int = 5, limit: int = 300, before: int | None = None) -> list[list[float]] | None:
        """[ts, open, high, low, close, volume] candles, oldest first: the `limit` candles ending at `before`
        (unix seconds), or the latest ones without it."""
        params = {"aggregate": minutes, "limit": limit}
        if before is not None:
            params["before_timestamp"] = before
        data = self._get(f"/networks/{self.network}/pools/{pool}/ohlcv/minute", params)
        if data is None:
            return None
        rows = ((data.get("data") or {}).get("attributes") or {}).get("ohlcv_list") or []
        return sorted(rows, key=lambda r: r[0])

    def discover(self) -> list[str]:
        """Base-token addresses of trending pools, most active first, without duplicates."""
        seen: dict[str, None] = {}
        for window in TRENDING_WINDOWS:
            data = self._get(f"/networks/{self.network}/trending_pools", {"duration": window})
            if data is None:
                continue
            pools = data.get("data") or []
            for pool in pools:
                addr = base_token(pool, self.network)
                if addr and addr not in NOT_MEMECOINS:
                    seen.setdefault(addr, None)
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
