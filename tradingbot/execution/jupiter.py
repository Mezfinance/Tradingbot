"""Live executor: USDC <-> token swaps through Jupiter, signed locally, sent via Solana RPC.

EXPERIMENTAL. Only enabled with TB_MODE=live plus the confirmation phrase.
The wallet should hold only USDC you can afford to lose entirely, plus a little SOL for fees.
"""

from __future__ import annotations

import base64
import json
import logging
import time

import httpx

from tradingbot.config import Config
from tradingbot.execution.base import Fill

log = logging.getLogger(__name__)

JUP = "https://lite-api.jup.ag/swap/v1"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDC_DECIMALS = 6
MAX_PRICE_IMPACT_PCT = 3.0
CONFIRM_TIMEOUT_S = 60


class JupiterExecutor:
    def __init__(self, cfg: Config, http: httpx.Client | None = None):
        from solders.keypair import Keypair  # optional dependency: pip install .[live]

        with open(cfg.keypair_path) as fh:
            self.keypair = Keypair.from_bytes(bytes(json.load(fh)))
        self.owner = str(self.keypair.pubkey())
        self.rpc_url = cfg.solana_rpc_url
        self.slippage_bps = cfg.slippage_bps
        self.http = http or httpx.Client(timeout=20)
        self._decimals: dict[str, int] = {USDC_MINT: USDC_DECIMALS}

    # --- public -------------------------------------------------------------
    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        raw_in = int(usd * 10**USDC_DECIMALS)
        result = self._swap(USDC_MINT, token, raw_in)
        if result is None:
            return None
        out_raw, sig = result
        qty = out_raw / 10 ** self._token_decimals(token)
        return Fill(qty=qty, price=usd / qty if qty else ref_price, usd=usd, fee_usd=0.0, tx=sig)

    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None:
        decimals = self._token_decimals(token)
        held = self._token_balance_raw(token)
        raw_in = min(int(qty * 10**decimals), held)
        if raw_in <= 0:
            log.warning("no %s balance to sell", token)
            return None
        result = self._swap(token, USDC_MINT, raw_in)
        if result is None:
            return None
        out_raw, sig = result
        usd = out_raw / 10**USDC_DECIMALS
        sold = raw_in / 10**decimals
        return Fill(qty=sold, price=usd / sold, usd=usd, fee_usd=0.0, tx=sig)

    # --- internals ------------------------------------------------------------
    def _swap(self, in_mint: str, out_mint: str, raw_amount: int) -> tuple[int, str] | None:
        from solders.transaction import VersionedTransaction

        quote = self.http.get(f"{JUP}/quote", params={
            "inputMint": in_mint, "outputMint": out_mint,
            "amount": raw_amount, "slippageBps": self.slippage_bps,
        })
        if quote.status_code != 200:
            log.error("quote failed: %s", quote.text[:300])
            return None
        q = quote.json()
        impact = float(q.get("priceImpactPct") or 0) * 100
        if impact > MAX_PRICE_IMPACT_PCT:
            log.warning("price impact %.2f%% too high, skipping", impact)
            return None

        swap = self.http.post(f"{JUP}/swap", json={
            "quoteResponse": q,
            "userPublicKey": self.owner,
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": "auto",
        })
        if swap.status_code != 200:
            log.error("swap build failed: %s", swap.text[:300])
            return None
        unsigned = VersionedTransaction.from_bytes(base64.b64decode(swap.json()["swapTransaction"]))
        signed = VersionedTransaction(unsigned.message, [self.keypair])
        sig = self._rpc("sendTransaction", [
            base64.b64encode(bytes(signed)).decode(),
            {"encoding": "base64", "skipPreflight": False, "maxRetries": 3},
        ])
        if not sig or not self._confirm(sig):
            log.error("swap not confirmed: %s", sig)
            return None
        # Quote output is the expected amount; actual may be lower within slippage.
        return int(q["otherAmountThreshold"]), sig

    def _confirm(self, sig: str) -> bool:
        deadline = time.time() + CONFIRM_TIMEOUT_S
        while time.time() < deadline:
            res = self._rpc("getSignatureStatuses", [[sig]]) or {}
            status = (res.get("value") or [None])[0]
            if status:
                if status.get("err"):
                    log.error("tx %s failed: %s", sig, status["err"])
                    return False
                if status.get("confirmationStatus") in ("confirmed", "finalized"):
                    return True
            time.sleep(2)
        return False

    def _token_decimals(self, mint: str) -> int:
        if mint not in self._decimals:
            res = self._rpc("getTokenSupply", [mint]) or {}
            self._decimals[mint] = int(res.get("value", {}).get("decimals", 6))
        return self._decimals[mint]

    def _token_balance_raw(self, mint: str) -> int:
        res = self._rpc("getTokenAccountsByOwner", [self.owner, {"mint": mint}, {"encoding": "jsonParsed"}]) or {}
        total = 0
        for acct in res.get("value", []):
            info = acct["account"]["data"]["parsed"]["info"]
            total += int(info["tokenAmount"]["amount"])
        return total

    def _rpc(self, method: str, params: list):
        try:
            resp = self.http.post(self.rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.error("rpc %s failed: %s", method, exc)
            return None
        if "error" in body:
            log.error("rpc %s error: %s", method, body["error"])
            return None
        return body.get("result")
