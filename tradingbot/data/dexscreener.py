"""DexScreener public API: discovery feeds and pair data. No key required."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import httpx

from tradingbot.models import PairSnapshot

log = logging.getLogger(__name__)

BASE = "https://api.dexscreener.com"
DISCOVERY_FEEDS = (
    "/token-profiles/latest/v1",
    "/token-boosts/latest/v1",
    "/token-boosts/top/v1",
)
BATCH = 30  # max addresses per /tokens/v1 call


class DexScreener:
    def __init__(self, client: httpx.Client | None = None, chain: str = "solana"):
        self.http = client or httpx.Client(base_url=BASE, timeout=15)
        self.chain = chain

    def discover(self) -> tuple[list[str], set[str]]:
        """Return (token addresses seen in discovery feeds, addresses that are paid-boosted)."""
        seen: dict[str, None] = {}
        boosted: set[str] = set()
        for path in DISCOVERY_FEEDS:
            try:
                resp = self.http.get(path)
                resp.raise_for_status()
                items = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("discovery feed %s failed: %s", path, exc)
                continue
            for item in items if isinstance(items, list) else []:
                if item.get("chainId") != self.chain:
                    continue
                addr = item.get("tokenAddress")
                if not addr:
                    continue
                seen.setdefault(addr, None)
                if "boost" in path:
                    boosted.add(addr)
        return list(seen), boosted

    def lists(self) -> dict[str, list[tuple[int, str, float]]]:
        """Paid-attention lists right now: feed -> [(rank, token, boost amount)] for this chain."""
        out: dict[str, list[tuple[int, str, float]]] = {}
        for path in DISCOVERY_FEEDS:
            try:
                resp = self.http.get(path)
                resp.raise_for_status()
                items = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                log.debug("list %s failed: %s", path, exc)
                continue
            name = path.strip("/").split("/")[0] + ":" + path.strip("/").split("/")[1]
            ranked = [i for i in (items if isinstance(items, list) else []) if i.get("chainId") == self.chain]
            out[name] = [(rank, i["tokenAddress"], float(i.get("totalAmount") or i.get("amount") or 0))
                         for rank, i in enumerate(ranked, 1) if i.get("tokenAddress")]
        return out

    def orders(self, token: str) -> list[dict] | None:
        """Paid DexScreener products for a token: profile, ads, community takeover, boosts."""
        try:
            resp = self.http.get(f"/orders/v1/{self.chain}/{token}")
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("orders %s failed: %s", token, exc)
            return None
        items = data.get("orders", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        boosts = data.get("boosts", []) if isinstance(data, dict) else []
        out = [{"type": o.get("type"), "status": o.get("status"), "paid_ts": (o.get("paymentTimestamp") or 0) / 1000}
               for o in items]
        out += [{"type": "boost", "status": "approved", "paid_ts": (b.get("paymentTimestamp") or 0) / 1000,
                 "amount": b.get("amount")} for b in boosts]
        return out

    def pairs(self, addresses: Iterable[str], boosted: set[str] = frozenset()) -> list[PairSnapshot]:
        """Best (deepest) pair for each token address."""
        addrs = list(addresses)
        best: dict[str, PairSnapshot] = {}
        for i in range(0, len(addrs), BATCH):
            chunk = ",".join(addrs[i : i + BATCH])
            try:
                resp = self.http.get(f"/tokens/v1/{self.chain}/{chunk}")
                resp.raise_for_status()
                raw_pairs = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("pair lookup failed: %s", exc)
                continue
            for raw in raw_pairs or []:
                snap = parse_pair(raw, boosted)
                if snap is None:
                    continue
                cur = best.get(snap.token_address)
                if cur is None or (snap.liquidity_usd or 0) > (cur.liquidity_usd or 0):
                    best[snap.token_address] = snap
        return list(best.values())

    def price(self, token_address: str) -> float | None:
        snaps = self.pairs([token_address])
        return snaps[0].price_usd if snaps else None


def _f(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_pair(raw: dict, boosted: set[str] = frozenset()) -> PairSnapshot | None:
    base = raw.get("baseToken") or {}
    addr = base.get("address")
    price = _f(raw.get("priceUsd"))
    if not addr or not price:
        return None
    txns = raw.get("txns") or {}
    info = raw.get("info") or {}
    socials = [s.get("url", "") for s in info.get("socials", []) if s.get("url")]
    socials += [w.get("url", "") for w in info.get("websites", []) if w.get("url")]
    return PairSnapshot(
        chain=raw.get("chainId", ""),
        token_address=addr,
        pair_address=raw.get("pairAddress", ""),
        symbol=base.get("symbol", "?"),
        name=base.get("name", "?"),
        dex=raw.get("dexId", ""),
        price_usd=price,
        liquidity_usd=_f((raw.get("liquidity") or {}).get("usd")),
        fdv_usd=_f(raw.get("fdv")),
        volume={k: _f(v) or 0.0 for k, v in (raw.get("volume") or {}).items()},
        price_change={k: _f(v) or 0.0 for k, v in (raw.get("priceChange") or {}).items()},
        buys={k: int(v.get("buys", 0)) for k, v in txns.items()},
        sells={k: int(v.get("sells", 0)) for k, v in txns.items()},
        created_at_ms=raw.get("pairCreatedAt"),
        socials=socials,
        boosted=addr in boosted or bool(raw.get("boosts")),
    )
