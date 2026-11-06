"""Live executor: USDC <-> token swaps through Jupiter, signed locally, sent via Solana RPC.

Real money. Only built for live mode or a canary book, and only with the confirmation phrase.
The wallet should hold only USDC you can afford to lose entirely, plus a little SOL for fees.

Every fill is read from the confirmed transaction itself (the wallet's pre/post token balances and the
fee payer's SOL change), so trades running at the same time can't mix their balance changes, and the
journal records what happened on-chain. Built for the launch canary (11-05):
  - balances and transactions are read at 'confirmed' (RPC defaults to 'finalized', ~13 s behind),
  - confirmation is polled every CONFIRM_POLL_S, and a transaction not seen by CONFIRM_TIMEOUT_S is
    looked up once more before it counts as dropped (a blockhash lives ~60-90 s),
  - buys are refused before sending when the wallet's USDC or SOL is short,
  - Jupiter's impact figure is replaced by our own probe on seconds-old tokens (as the simulator),
  - sells use a wider slippage (an exit must go through), and an emptied token account is closed
    afterwards so its rent (~0.002 SOL) comes back; the reclaimed SOL counts in the sale's proceeds.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time

import httpx

from tradingbot.config import Config
from tradingbot.data.jupiter import HELIUS_RPC, RATE_LIMITED, USDC_DECIMALS, USDC_MINT, Jupiter
from tradingbot.execution.base import Fill, failed
from tradingbot.execution.sim import own_impact

log = logging.getLogger(__name__)

CONFIRM_TIMEOUT_S = 75
CONFIRM_POLL_S = 0.4
TX_FETCH_TRIES = 8  # getTransaction can lag the status by a moment
TX_FETCH_WAIT_S = 0.5
TX_VERSION = 1
MIN_SOL = 0.01  # keep this much SOL for fees and token-account rent, or refuse to buy
CLOSE_ACCOUNT = bytes([9])  # SPL Token / Token-2022 CloseAccount instruction
NOT_SENT = "not sent:"  # note prefix: nothing went out (preflight refused, build failed); retry soon
DEFAULT_PUBLIC_RPC = "https://api.mainnet-beta.solana.com"


def rpc_url(cfg: Config) -> str:
    """The configured RPC, or Helius when only the public default is set and a key exists (faster landing)."""
    key = os.environ.get("HELIUS_API_KEY")
    if cfg.solana_rpc_url == DEFAULT_PUBLIC_RPC and key:
        return f"{HELIUS_RPC}?api-key={key}"
    return cfg.solana_rpc_url


def load_keypair(text: str):
    """A Solana key in either common form: the JSON array of 64 numbers (solana-keygen's file) or the base58
    string a wallet app exports (Phantom, Solflare). Surrounding whitespace and quotes are ignored."""
    from solders.keypair import Keypair  # optional dependency: pip install .[live]

    text = text.strip().strip('"').strip("'").strip()
    if text.startswith("["):
        return Keypair.from_bytes(bytes(json.loads(text)))
    return Keypair.from_base58_string(text)


SECRET_KEYPAIR_PATH = "/run/secrets/keypair.json"  # written by deploy/entrypoint.sh, never on the volume


def read_keypair(cfg: Config):
    """The wallet key: the configured file, else the entrypoint's in-memory copy, else the TB_KEYPAIR_JSON secret
    itself. A `fly machine exec` command doesn't pass through the entrypoint, so it has neither TB_KEYPAIR_PATH
    nor (if the machine wasn't restarted after the secret was set) the file, but it does see the secret."""
    for path in (cfg.keypair_path, SECRET_KEYPAIR_PATH):
        if path and os.path.exists(path):
            with open(path) as fh:
                return load_keypair(fh.read())
    if os.environ.get("TB_KEYPAIR_JSON"):
        return load_keypair(os.environ["TB_KEYPAIR_JSON"])
    raise FileNotFoundError(f"no wallet key: {cfg.keypair_path} and {SECRET_KEYPAIR_PATH} are missing and the "
                            "TB_KEYPAIR_JSON secret is not set (fly secrets import -a supermeme)")


class JupiterExecutor:
    def __init__(self, cfg: Config, jupiter: Jupiter, http: httpx.Client | None = None, keypair=None,
                 priority_fee_sol: float | None = None, sell_slippage_bps: int | None = None,
                 sleep=time.sleep):
        if keypair is None:
            keypair = read_keypair(cfg)
        self.keypair = keypair
        self.owner = str(keypair.pubkey())
        self.cfg = cfg
        self.jup = jupiter
        self.rpc_url = rpc_url(cfg)
        self.http = http or httpx.Client(timeout=20)
        self.priority_fee_sol = cfg.priority_fee_sol if priority_fee_sol is None else priority_fee_sol
        self.sell_slippage_bps = sell_slippage_bps
        self.sleep = sleep
        self.last_note = ""

    # --- public -------------------------------------------------------------
    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        """None = declined before sending (reason in `last_note`); a Fill with ok=False = sent and failed."""
        decimals = self.jup.decimals(token)
        if decimals is None:
            self.last_note = "token decimals unknown"
            return None
        usdc = self.token_balance(USDC_MINT) / 10**USDC_DECIMALS
        if usdc < usd:
            self.last_note = f"wallet has ${usdc:.2f} USDC"
            return None
        sol = self.sol_balance()
        if sol is None or sol < MIN_SOL:
            self.last_note = f"wallet SOL low ({sol if sol is not None else '?'})"
            return None
        in_raw = int(usd * 10**USDC_DECIMALS)
        q = self.jup.quote(USDC_MINT, token, in_raw)
        if q is None:
            self.last_note = self._why_none("no route")
            return None
        impact = q.price_impact_pct
        if impact > self.cfg.max_price_impact_pct:
            def estimate(qq):
                if not ref_price or ref_price <= 0 or not qq.out_raw:
                    return None
                return max(0.0, (usd / (qq.out_raw / 10**decimals) / ref_price - 1) * 100)
            impact = own_impact(self.jup, USDC_MINT, token, in_raw, q, estimate)
        if impact > self.cfg.max_price_impact_pct:
            self.last_note = f"price impact {impact:.2f}%"
            return None
        sent = self._send_swap(q)
        if sent.startswith(NOT_SENT):
            self.last_note = sent
            return None
        res = self._settle(sent, token)
        if not res["ok"]:
            self.last_note = res["note"]
            return failed(res["fee_usd"], res["note"])
        got, spent = res["token"], -res["usdc"]
        if got <= 0 or spent <= 0:
            self.last_note = "confirmed but no tokens received"
            return failed(res["fee_usd"], self.last_note)
        qty = got / 10**decimals
        swap_usd = spent / 10**USDC_DECIMALS
        self.last_note = "ok"
        return Fill(qty=qty, price=swap_usd / qty, usd=swap_usd + res["fee_usd"], fee_usd=res["fee_usd"],
                    tx=sent, note="ok")

    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None:
        decimals = self.jup.decimals(token)
        if decimals is None:
            return failed(0.0, "token decimals unknown")
        held = self.token_balance(token)
        raw_in = min(int(qty * 10**decimals), held)
        if raw_in <= 0:
            log.warning("no %s balance to sell", token)
            return failed(0.0, "no balance")
        q = self.jup.quote(token, USDC_MINT, raw_in, self.sell_slippage_bps)
        if q is None:
            return failed(0.0, self._why_none("no route"))
        sent = self._send_swap(q)
        if sent.startswith(NOT_SENT):
            return failed(0.0, sent)  # nothing landed: the engine retries
        res = self._settle(sent, token)
        if not res["ok"]:
            return failed(res["fee_usd"], res["note"])
        sold_raw, proceeds_raw = -res["token"], res["usdc"]
        if sold_raw <= 0:
            return failed(res["fee_usd"], "confirmed but no tokens sold")
        sold = sold_raw / 10**decimals
        proceeds = proceeds_raw / 10**USDC_DECIMALS
        fee = res["fee_usd"]
        if raw_in >= held:  # sold everything: close the empty account and take its rent back
            fee -= self.close_empty(token)
        return Fill(qty=sold, price=proceeds / sold, usd=proceeds - fee, fee_usd=fee, tx=sent, note="ok")

    # --- wallet ---------------------------------------------------------------
    def token_accounts(self, mint: str) -> list[dict]:
        res = self._rpc("getTokenAccountsByOwner", [self.owner, {"mint": mint},
                                                    {"encoding": "jsonParsed", "commitment": "confirmed"}]) or {}
        return res.get("value") or []

    def token_balance(self, mint: str) -> int:
        total = 0
        for acct in self.token_accounts(mint):
            info = acct["account"]["data"]["parsed"]["info"]
            total += int(info["tokenAmount"]["amount"])
        return total

    def sol_balance(self) -> float | None:
        res = self._rpc("getBalance", [self.owner, {"commitment": "confirmed"}])
        return None if res is None else res.get("value", 0) / 1e9

    def close_empty(self, mint: str) -> float:
        """Close this wallet's empty token accounts for `mint`; returns the USD value of the SOL that came back
        (rent minus the fee), 0 if nothing was closed. Best effort: a failure only leaves the rent locked."""
        try:
            from solders.hash import Hash
            from solders.instruction import AccountMeta, Instruction
            from solders.message import MessageV0
            from solders.pubkey import Pubkey
            from solders.transaction import VersionedTransaction

            accounts = [a for a in self.token_accounts(mint)
                        if int(a["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"]) == 0]
            if not accounts:
                return 0.0
            me = self.keypair.pubkey()
            ixs = [Instruction(Pubkey.from_string(a["account"]["owner"]), CLOSE_ACCOUNT,
                               [AccountMeta(Pubkey.from_string(a["pubkey"]), False, True),
                                AccountMeta(me, False, True), AccountMeta(me, True, False)]) for a in accounts]
            bh = (self._rpc("getLatestBlockhash", [{"commitment": "confirmed"}]) or {}).get("value", {}).get("blockhash")
            if not bh:
                return 0.0
            msg = MessageV0.try_compile(me, ixs, [], Hash.from_string(bh))
            tx = VersionedTransaction(msg, [self.keypair])
            sig = self._rpc("sendTransaction", [base64.b64encode(bytes(tx)).decode(),
                                                {"encoding": "base64", "skipPreflight": False, "maxRetries": 3}])
            if not sig:
                return 0.0
            res = self._settle(sig, None)
            return -res["fee_usd"] if res["ok"] else 0.0  # negative cost = SOL came back
        except Exception:  # noqa: BLE001 (closing an account must never break a sale)
            log.warning("closing the empty %s account failed", mint, exc_info=True)
            return 0.0

    # --- internals ------------------------------------------------------------
    def _why_none(self, default: str) -> str:
        why = getattr(self.jup, "last_failure", lambda: default)()
        return why if why == RATE_LIMITED else default

    def _send_swap(self, q) -> str:
        """Build, sign and send the swap. Returns the signature, or a note (str prefixed 'not sent:') when
        nothing went out. Signatures are base58 and never contain a space, so the two can't be confused."""
        from solders.transaction import VersionedTransaction

        try:
            swap = self.jup.http.post("/swap/v1/swap", json={
                "quoteResponse": q.raw,
                "userPublicKey": self.owner,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": {"priorityLevelWithMaxLamports": {
                    "priorityLevel": "veryHigh",
                    "maxLamports": int(self.priority_fee_sol * 1e9),
                }},
            })
        except httpx.HTTPError as exc:
            log.error("swap build failed: %s", exc)
            return "not sent: swap build failed"
        if swap.status_code != 200:
            log.error("swap build failed: %s", swap.text[:300])
            return f"not sent: swap build HTTP {swap.status_code}"
        unsigned = VersionedTransaction.from_bytes(base64.b64decode(swap.json()["swapTransaction"]))
        signed = VersionedTransaction(unsigned.message, [self.keypair])
        sig, err = self._rpc_err("sendTransaction", [
            base64.b64encode(bytes(signed)).decode(),
            {"encoding": "base64", "skipPreflight": False, "maxRetries": 3},
        ])
        if not sig:  # preflight simulation refused it (e.g. slippage): nothing landed, nothing paid
            return f"not sent: {str(err)[:120] if err else 'send failed'}"
        return sig

    def _settle(self, sig: str, mint: str | None) -> dict:
        """Wait for the transaction and read what it did. {ok, note, token, usdc (raw deltas for this wallet),
        fee_usd (SOL the fee payer lost: fee + any new account rent, at the SOL price)}."""
        status = self._await(sig)
        tx = self._fetch_tx(sig) if status != "dropped" else None
        if tx is None:
            note = "not confirmed (dropped)" if status == "dropped" else "confirmed but transaction not readable"
            if status != "dropped":
                log.error("tx %s confirmed but unreadable: balances may be off", sig)
            return {"ok": False, "note": note, "token": 0, "usdc": 0,
                    "fee_usd": 0.0 if status == "dropped" else self._sol_usd(self.priority_fee_sol)}
        meta = tx.get("meta") or {}
        pre, post = meta.get("preBalances") or [0], meta.get("postBalances") or [0]
        fee_usd = self._sol_usd((pre[0] - post[0]) / 1e9)
        if meta.get("err"):
            return {"ok": False, "note": f"reverted: {meta['err']}", "token": 0, "usdc": 0, "fee_usd": fee_usd}
        return {"ok": True, "note": "ok", "fee_usd": fee_usd,
                "token": self._delta(meta, mint) if mint else 0, "usdc": self._delta(meta, USDC_MINT)}

    def _await(self, sig: str) -> str:
        """'landed' (seen with confirmed status or an error), or 'dropped' after the timeout and one last look."""
        deadline = time.time() + CONFIRM_TIMEOUT_S
        while True:
            res = self._rpc("getSignatureStatuses", [[sig], {"searchTransactionHistory": time.time() >= deadline}]) or {}
            status = (res.get("value") or [None])[0]
            if status and (status.get("err") or status.get("confirmationStatus") in ("confirmed", "finalized")):
                return "landed"
            if time.time() >= deadline:
                return "dropped"
            self.sleep(CONFIRM_POLL_S)

    def _fetch_tx(self, sig: str) -> dict | None:
        for _ in range(TX_FETCH_TRIES):
            tx = self._rpc("getTransaction", [sig, {"encoding": "jsonParsed", "commitment": "confirmed",
                                                    "maxSupportedTransactionVersion": TX_VERSION}])
            if tx:
                return tx
            self.sleep(TX_FETCH_WAIT_S)
        return None

    def _delta(self, meta: dict, mint: str) -> int:
        """This wallet's raw balance change of `mint` in one transaction."""
        def total(rows):
            return sum(int((r.get("uiTokenAmount") or {}).get("amount") or 0) for r in rows or []
                       if r.get("mint") == mint and r.get("owner") == self.owner)
        return total(meta.get("postTokenBalances")) - total(meta.get("preTokenBalances"))

    def _sol_usd(self, sol: float) -> float:
        price = self.jup.sol_price() or 0.0
        return sol * price

    def _rpc(self, method: str, params: list):
        return self._rpc_err(method, params)[0]

    def _rpc_err(self, method: str, params: list):
        try:
            resp = self.http.post(self.rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.error("rpc %s failed: %s", method, exc)
            return None, str(exc)
        if "error" in body:
            log.error("rpc %s error: %s", method, body["error"])
            return None, (body["error"] or {}).get("message", body["error"])
        return body.get("result"), None
