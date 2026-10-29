"""Helius RPC (needs HELIUS_API_KEY): the wallet-level picture of a launch.

  early trades       getTransactionsForAddress(mint, oldest first, full) -> who bought in
                     the launch slot (bundles), the first seconds (snipers), buy sizes,
                     whether the developer sold
  holder concentration  getTokenLargestAccounts + owners (pool/curve accounts excluded later)
  wallet origin      the wallet's own first transaction -> wallet age and who funded it
  pool operator      a pool's first transaction (who created it, SOL seeded) and its latest ones (a
                     liquidity withdrawal: when the operator pulled it)

Credits are counted with Helius's published prices (full history: 10 per 100 transactions,
10 minimum; standard RPC: 1) so the caller can enforce a daily budget.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

log = logging.getLogger(__name__)

RPC_URL = "https://mainnet.helius-rpc.com/"
# Jito's public tip accounts: SOL sent here buys a place at the front of the block.
JITO_TIP_ACCOUNTS = {
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5", "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY", "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh", "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL", "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
}
HISTORY_CREDITS_PER_100 = 10
RPC_CREDITS = 1
SOL_MINT = "So11111111111111111111111111111111111111112"
# Program log lines of a liquidity withdrawal (pumpswap `Withdraw`, Meteora/Raydium remove-liquidity variants).
WITHDRAW_RE = re.compile(r"Instruction: (Withdraw|RemoveLiquidity|RemoveAllLiquidity|RemoveBalanceLiquidity|"
                         r"remove_liquidity|withdraw)\b")


class Helius:
    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("HELIUS_API_KEY", "")
        self.http = client or httpx.Client(timeout=30)
        self.credits_used = 0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _rpc(self, method: str, params: list, credits: int):
        if not self.enabled:
            return None
        try:
            resp = self.http.post(RPC_URL, params={"api-key": self.api_key},
                                  json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        except httpx.HTTPError as exc:
            log.warning("helius %s failed: %s", method, exc)
            return None
        if resp.status_code != 200:
            log.warning("helius %s -> HTTP %s", method, resp.status_code)
            return None
        self.credits_used += credits
        body = resp.json()
        if body.get("error"):
            log.warning("helius %s error: %s", method, body["error"])
            return None
        return body.get("result")

    def history(self, address: str, limit: int = 100) -> list[dict]:
        """Oldest-first full transactions touching `address`."""
        res = self._rpc("getTransactionsForAddress",
                        [address, {"transactionDetails": "full", "sortOrder": "asc", "limit": limit,
                                   "maxSupportedTransactionVersion": 0, "encoding": "jsonParsed"}],
                        max(HISTORY_CREDITS_PER_100, HISTORY_CREDITS_PER_100 * limit // 100))
        return (res or {}).get("data") or []

    def recent(self, address: str, limit: int = 50) -> list[dict]:
        """Newest-first full transactions touching `address`."""
        res = self._rpc("getTransactionsForAddress",
                        [address, {"transactionDetails": "full", "sortOrder": "desc", "limit": limit,
                                   "maxSupportedTransactionVersion": 0, "encoding": "jsonParsed"}],
                        max(HISTORY_CREDITS_PER_100, HISTORY_CREDITS_PER_100 * limit // 100))
        return (res or {}).get("data") or []

    def largest_accounts(self, mint: str) -> list[dict]:
        res = self._rpc("getTokenLargestAccounts", [mint], RPC_CREDITS)
        return (res or {}).get("value") or []

    def token_supply(self, mint: str) -> float | None:
        res = self._rpc("getTokenSupply", [mint], RPC_CREDITS)
        try:
            return float(res["value"]["uiAmountString"])
        except (TypeError, KeyError, ValueError):
            return None

    def owners(self, accounts: list[str]) -> dict[str, str]:
        res = self._rpc("getMultipleAccounts", [accounts, {"encoding": "jsonParsed"}], RPC_CREDITS)
        out = {}
        for acct, info in zip(accounts, (res or {}).get("value") or []):
            try:
                out[acct] = info["data"]["parsed"]["info"]["owner"]
            except (TypeError, KeyError):
                continue
        return out

    def balances(self, wallets: list[str]) -> dict[str, float]:
        """SOL balance of each wallet (100 per call)."""
        out: dict[str, float] = {}
        for i in range(0, len(wallets), 100):
            chunk = wallets[i:i + 100]
            res = self._rpc("getMultipleAccounts", [chunk, {"encoding": "base64", "dataSlice": {"offset": 0, "length": 0}}],
                            RPC_CREDITS)
            for w, info in zip(chunk, (res or {}).get("value") or []):
                out[w] = (info or {}).get("lamports", 0) / 1e9
        return out

    def wallet_origin(self, wallet: str) -> dict | None:
        """{"first_ts", "funder"} from the wallet's first transaction; None if unknown."""
        txs = self.history(wallet, limit=1)
        if not txs:
            return None
        tx = txs[0]
        return {"first_ts": tx.get("blockTime"), "funder": funder_of(tx, wallet)}


def _keys(tx: dict) -> list[str]:
    keys = ((tx.get("transaction") or {}).get("message") or {}).get("accountKeys") or []
    return [k["pubkey"] if isinstance(k, dict) else k for k in keys]


def funder_of(tx: dict, wallet: str) -> str | None:
    """The account whose SOL dropped the most while `wallet` received SOL (its funder)."""
    keys, meta = _keys(tx), tx.get("meta") or {}
    pre, post = meta.get("preBalances") or [], meta.get("postBalances") or []
    deltas = {k: post[i] - pre[i] for i, k in enumerate(keys) if i < len(pre) and i < len(post)}
    if deltas.get(wallet, 0) <= 0:
        return keys[0] if keys and keys[0] != wallet else None  # fee payer created it
    senders = [(d, k) for k, d in deltas.items() if k != wallet and d < 0]
    return min(senders)[1] if senders else None


def trades(txs: list[dict], mint: str) -> list[list]:
    """Compact per-owner token balance changes:
    [blockTime, slot, owner, delta_tokens, fee_payer, fee_lamports, jito_tip_lamports]."""
    out = []
    for tx in txs:
        meta = tx.get("meta") or {}
        if meta.get("err"):
            continue
        change: dict[str, float] = {}
        for sign, side in ((-1, "preTokenBalances"), (1, "postTokenBalances")):
            for b in meta.get(side) or []:
                if b.get("mint") != mint or not b.get("owner"):
                    continue
                amt = float((b.get("uiTokenAmount") or {}).get("uiAmount") or 0)
                change[b["owner"]] = change.get(b["owner"], 0.0) + sign * amt
        keys = _keys(tx)
        payer = keys[0] if keys else None
        pre, post = meta.get("preBalances") or [], meta.get("postBalances") or []
        tip = sum(post[i] - pre[i] for i, k in enumerate(keys)
                  if k in JITO_TIP_ACCOUNTS and i < len(pre) and i < len(post))
        for owner, delta in change.items():
            if abs(delta) > 0:
                out.append([tx.get("blockTime"), tx.get("slot"), owner, delta, payer, meta.get("fee") or 0, max(tip, 0)])
    return out


def instructions(tx: dict) -> list[str]:
    """Instruction names from the program logs, in order, without repeats."""
    out = []
    for line in (tx.get("meta") or {}).get("logMessages") or []:
        if "Instruction: " in line:
            name = line.split("Instruction: ", 1)[1].strip()
            if name not in out:
                out.append(name)
    return out


def _pool_sol(tx: dict, pool: str, side: str) -> float | None:
    for b in (tx.get("meta") or {}).get(side) or []:
        if b.get("owner") == pool and b.get("mint") == SOL_MINT:
            return float((b.get("uiTokenAmount") or {}).get("uiAmount") or 0)
    return None


def pool_creation(tx: dict, pool: str) -> dict:
    """From a pool's first transaction: who created it (fee payer), when, and the SOL it was seeded with."""
    keys = _keys(tx)
    return {"creator": keys[0] if keys else None, "create_ts": tx.get("blockTime"),
            "seed_sol": _pool_sol(tx, pool, "postTokenBalances"), "create_ix": instructions(tx)[:6]}


def find_withdraw(txs: list[dict], pool: str) -> dict | None:
    """The earliest successful liquidity withdrawal among `txs`: {"ts", "by", "sol_out"}; None if none."""
    hits = []
    for tx in txs:
        meta = tx.get("meta") or {}
        if meta.get("err") or not any(WITHDRAW_RE.search(line) for line in meta.get("logMessages") or []):
            continue
        pre, post = _pool_sol(tx, pool, "preTokenBalances"), _pool_sol(tx, pool, "postTokenBalances")
        keys = _keys(tx)
        hits.append({"ts": tx.get("blockTime"), "by": keys[0] if keys else None,
                     "sol_out": round(pre - post, 4) if pre is not None and post is not None else None})
    return min(hits, key=lambda h: h["ts"] or 0) if hits else None
