"""GeckoTerminal trending pools: discovery by actual trading activity. No key required.

DexScreener's discovery feeds are mostly paid listings (profiles, boosts), so they skew
towards fresh, self-promoted launches. GeckoTerminal ranks pools by recent trading, which
surfaces tokens with real liquidity and volume. Only token addresses are taken from here;
market data still comes from DexScreener so every candidate is measured the same way.
"""

from __future__ import annotations

import logging

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


class GeckoTerminal:
    def __init__(self, client: httpx.Client | None = None, network: str = "solana"):
        self.http = client or httpx.Client(base_url=BASE, timeout=15, headers={"Accept": "application/json"})
        self.network = network

    def discover(self) -> list[str]:
        """Base-token addresses of trending pools, most active first, without duplicates."""
        seen: dict[str, None] = {}
        for window in TRENDING_WINDOWS:
            try:
                resp = self.http.get(f"/networks/{self.network}/trending_pools", params={"duration": window})
                resp.raise_for_status()
                pools = resp.json().get("data") or []
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("geckoterminal trending (%s) failed: %s", window, exc)
                continue
            for pool in pools:
                addr = base_token(pool, self.network)
                if addr and addr not in NOT_MEMECOINS:
                    seen.setdefault(addr, None)
        return list(seen)


def base_token(pool: dict, network: str = "solana") -> str | None:
    """'solana_<mint>' relationship id -> '<mint>'."""
    ident = (((pool.get("relationships") or {}).get("base_token") or {}).get("data") or {}).get("id", "")
    prefix = f"{network}_"
    return ident[len(prefix):] if ident.startswith(prefix) else None
