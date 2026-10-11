"""Live executor: USDC <-> token swaps through Jupiter, signed locally, sent via Solana RPC.

EXPERIMENTAL. Only enabled with live mode or canary trades, plus the confirmation phrase.
The wallet should hold only USDC you can afford to lose entirely, plus a little SOL for fees.

Fills are measured from wallet balance changes, not from the quote, so the journal
records what actually happened on-chain.
"""

from __future__ import annotations

import base64
import json
import logging
import time

import httpx

from tradingbot.config import Config
from tradingbot.data.jupiter import USDC_DECIMALS, USDC_MINT, Jupiter
from tradingbot.execution.base import Fill, failed

log = logging.getLogger(__name__)

CONFIRM_TIMEOUT_S = 60
BALANCE_SETTLE_S = 2


class JupiterExecutor:
    def __init__(self, cfg: Config, jupiter: Jupiter, http: httpx.Client | None = None):
        from solders.keypair import Keypair  # optional dependency: pip install .[live]

        with open(cfg.keypair_path) as fh:
            self.keypair = Keypair.from_bytes(bytes(json.load(fh)))
        self.owner = str(self.keypair.pubkey())
        self.cfg = cfg
        self.jup = jupiter
        self.rpc_url = cfg.solana_rpc_url
        self.http = http or httpx.Client(timeout=20)

    # --- public -------------------------------------------------------------
    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        decimals = self.jup.decimals(token)
        if decimals is None:
            return None
        before_tok, before_usdc = self._balance_raw(token), self._balance_raw(USDC_MINT)
        sent = self._swap(USDC_MINT, token, int(usd * 10**USDC_DECIMALS), cap_impact=True)
        if sent is None:
            return None
        sig, ok, note = sent
        fee = self._fee_usd(sig)
        if not ok:
            return failed(fee, note)
        time.sleep(BALANCE_SETTLE_S)
        got = self._balance_raw(token) - before_tok
        spent_usd = (before_usdc - self._balance_raw(USDC_MINT)) / 10**USDC_DECIMALS
        if got <= 0:
            return failed(fee, "confirmed but no tokens received")
        qty = got / 10**decimals
        return Fill(qty=qty, price=spent_usd / qty, usd=spent_usd + fee, fee_usd=fee, tx=sig, note="ok")

    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None:
        decimals = self.jup.decimals(token)
        if decimals is None:
            return None
        held = self._balance_raw(token)
        raw_in = min(int(qty * 10**decimals), held)
        if raw_in <= 0:
            log.warning("no %s balance to sell", token)
            return failed(0.0, "no balance")
        before_usdc = self._balance_raw(USDC_MINT)
        sent = self._swap(token, USDC_MINT, raw_in, cap_impact=False)
        if sent is None:
            return failed(0.0, "no route")
        sig, ok, note = sent
        fee = self._fee_usd(sig)
        if not ok:
            return failed(fee, note)
        time.sleep(BALANCE_SETTLE_S)
        proceeds = (self._balance_raw(USDC_MINT) - before_usdc) / 10**USDC_DECIMALS
        sold = raw_in / 10**decimals
        return Fill(qty=sold, price=proceeds / sold, usd=proceeds - fee, fee_usd=fee, tx=sig, note="ok")

    # --- internals ------------------------------------------------------------
    def _swap(self, in_mint: str, out_mint: str, raw_amount: int, cap_impact: bool) -> tuple[str, bool, str] | None:
        """Returns (signature, confirmed_ok, note), or None if nothing was sent."""
        from solders.transaction import VersionedTransaction

        q = self.jup.quote(in_mint, out_mint, raw_amount)
        if q is None:
            return None
        if cap_impact and q.price_impact_pct > self.cfg.max_price_impact_pct:
            log.warning("price impact %.2f%% too high, skipping", q.price_impact_pct)
            return None
        try:
            swap = self.jup.http.post("/swap/v1/swap", json={
                "quoteResponse": q.raw,
                "userPublicKey": self.owner,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": {"priorityLevelWithMaxLamports": {
                    "priorityLevel": "veryHigh",
                    "maxLamports": int(self.cfg.priority_fee_sol * 1e9),
                }},
            })
        except httpx.HTTPError as exc:
            log.error("swap build failed: %s", exc)
            return None
        if swap.status_code != 200:
            log.error("swap build failed: %s", swap.text[:300])
            return None
        unsigned = VersionedTransaction.from_bytes(base64.b64decode(swap.json()["swapTransaction"]))
        signed = VersionedTransaction(unsigned.message, [self.keypair])
        sig = self._rpc("sendTransaction", [
            base64.b64encode(bytes(signed)).decode(),
            {"encoding": "base64", "skipPreflight": False, "maxRetries": 3},
        ])
        if not sig:
            return None
        ok, note = self._confirm(sig)
        return sig, ok, note

    def _confirm(self, sig: str) -> tuple[bool, str]:
        deadline = time.time() + CONFIRM_TIMEOUT_S
        while time.time() < deadline:
            res = self._rpc("getSignatureStatuses", [[sig]]) or {}
            status = (res.get("value") or [None])[0]
            if status:
                if status.get("err"):
                    return False, f"reverted: {status['err']}"
                if status.get("confirmationStatus") in ("confirmed", "finalized"):
                    return True, "ok"
            time.sleep(2)
        return False, "not confirmed (expired)"

    def _fee_usd(self, sig: str) -> float:
        """Actual fee paid, read from the confirmed transaction."""
        tx = self._rpc("getTransaction", [sig, {"encoding": "json", "maxSupportedTransactionVersion": 0}])
        lamports = ((tx or {}).get("meta") or {}).get("fee")
        sol_price = self.jup.sol_price() or 0.0
        if lamports is None:
            return self.cfg.priority_fee_sol * sol_price
        return lamports / 1e9 * sol_price

    def _balance_raw(self, mint: str) -> int:
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
