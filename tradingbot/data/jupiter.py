"""Jupiter public endpoints: swap quotes, USD prices and token decimals. No key required."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

BASE = "https://lite-api.jup.ag"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_DECIMALS = 6
PRICE_BATCH = 50
SOL_PRICE_TTL_S = 300
HELIUS_RPC = "https://mainnet.helius-rpc.com/"
PUBLIC_RPC = "https://api.mainnet-beta.solana.com"
MIN_REQUEST_INTERVAL_S = 0.5  # keeps the shared free tier comfortably under its rate limit
RETRIES_ON_429 = 3


@dataclass
class Quote:
    in_mint: str
    out_mint: str
    in_raw: int
    out_raw: int
    min_out_raw: int  # the swap reverts below this (slippage tolerance)
    price_impact_pct: float  # percent, e.g. 1.5 = 1.5%
    route: list[str]
    raw: dict  # full response, needed to build a live swap


@dataclass
class TokenPrice:
    usd: float
    liquidity_usd: float | None
    decimals: int | None


class Jupiter:
    def __init__(self, slippage_bps: int = 300, http: httpx.Client | None = None):
        self.http = http or httpx.Client(base_url=BASE, timeout=15)
        self.slippage_bps = slippage_bps
        self._decimals: dict[str, int] = {USDC_MINT: USDC_DECIMALS, SOL_MINT: 9}
        self._sol_price: tuple[float, float] | None = None  # (price, fetched_at)
        self._pace = threading.Lock()
        self._last_request = 0.0

    def _get(self, path: str, params: dict) -> httpx.Response:
        """GET with global pacing (both bot threads share this client) and back-off on 429 and 5xx."""
        for attempt in range(RETRIES_ON_429 + 1):
            with self._pace:
                wait = self._last_request + MIN_REQUEST_INTERVAL_S - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last_request = time.monotonic()
            resp = self.http.get(path, params=params)
            if not (resp.status_code == 429 or resp.status_code >= 500) or attempt == RETRIES_ON_429:
                return resp
            time.sleep(2 ** attempt)
        return resp

    def quote(self, in_mint: str, out_mint: str, in_raw: int) -> Quote | None:
        if in_raw <= 0:
            return None
        try:
            resp = self._get("/swap/v1/quote", {
                "inputMint": in_mint, "outputMint": out_mint,
                "amount": in_raw, "slippageBps": self.slippage_bps,
            })
        except httpx.HTTPError as exc:
            log.warning("quote request failed: %s", exc)
            return None
        if resp.status_code != 200:
            log.info("no quote %s->%s: %s", in_mint[:6], out_mint[:6], resp.text[:200])
            return None
        q = resp.json()
        return Quote(
            in_mint=in_mint, out_mint=out_mint,
            in_raw=int(q["inAmount"]), out_raw=int(q["outAmount"]),
            min_out_raw=int(q["otherAmountThreshold"]),
            # Jupiter reports impact as a fraction (0.09 = 9%).
            price_impact_pct=float(q.get("priceImpactPct") or 0) * 100,
            route=[r.get("swapInfo", {}).get("label", "?") for r in q.get("routePlan", [])],
            raw=q,
        )

    def prices(self, mints: list[str]) -> dict[str, TokenPrice]:
        out: dict[str, TokenPrice] = {}
        for i in range(0, len(mints), PRICE_BATCH):
            chunk = mints[i : i + PRICE_BATCH]
            try:
                resp = self._get("/price/v3", {"ids": ",".join(chunk)})
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("price lookup failed: %s", exc)
                continue
            for mint, info in (data or {}).items():
                if not info or info.get("usdPrice") is None:
                    continue
                dec = info.get("decimals")
                if dec is not None:
                    self._decimals[mint] = int(dec)
                out[mint] = TokenPrice(float(info["usdPrice"]), info.get("liquidity"), dec)
        return out

    def decimals(self, mint: str) -> int | None:
        """From the price API, else on-chain: a token minutes old is often routable before /price lists it."""
        if mint not in self._decimals:
            self.prices([mint])
        if mint not in self._decimals:
            dec = self._decimals_onchain(mint)
            if dec is not None:
                self._decimals[mint] = dec
        return self._decimals.get(mint)

    def _decimals_onchain(self, mint: str) -> int | None:
        key = os.environ.get("HELIUS_API_KEY")
        url = os.environ.get("SOLANA_RPC_URL") or (f"{HELIUS_RPC}?api-key={key}" if key else PUBLIC_RPC)
        try:
            resp = self.http.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "getTokenSupply", "params": [mint]})
            return int(resp.json()["result"]["value"]["decimals"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return None

    def quote_price(self, mint: str, usd: float = 20.0) -> float | None:
        """USD price implied by a small buy quote, for tokens the price API does not list yet."""
        dec = self.decimals(mint)
        q = self.quote(USDC_MINT, mint, int(usd * 10**USDC_DECIMALS)) if dec is not None else None
        if q is None or q.out_raw <= 0:
            return None
        return usd / (q.out_raw / 10**dec)

    def sell_price(self, mint: str, qty: float) -> float | None:
        """USD per token from a quote to sell `qty` back to USDC: what a holder could actually get.
        A buy quote on a drained pool implies a huge price while selling returns almost nothing."""
        dec = self.decimals(mint)
        raw = int(qty * 10**dec) if dec is not None else 0
        q = self.quote(mint, USDC_MINT, raw) if raw > 0 else None
        if q is None:
            return None
        return (q.out_raw / 10**USDC_DECIMALS) / qty

    def sol_price(self) -> float | None:
        if self._sol_price and time.time() - self._sol_price[1] < SOL_PRICE_TTL_S:
            return self._sol_price[0]
        p = self.prices([SOL_MINT]).get(SOL_MINT)
        if p:
            self._sol_price = (p.usd, time.time())
            return p.usd
        return self._sol_price[0] if self._sol_price else None

    def round_trip(self, token: str, usd: float) -> tuple[float, float] | None:
        """Quote buying `usd` of token and immediately selling it back.

        Returns (round_trip_loss_pct, buy_price_impact_pct), or None if either leg
        has no route. Honeypots, transfer taxes and fake liquidity all show up as a
        large round-trip loss.
        """
        buy = self.quote(USDC_MINT, token, int(usd * 10**USDC_DECIMALS))
        if buy is None:
            return None
        sell = self.quote(token, USDC_MINT, buy.out_raw)
        if sell is None:
            return None
        back_usd = sell.out_raw / 10**USDC_DECIMALS
        return (1 - back_usd / usd) * 100, buy.price_impact_pct
