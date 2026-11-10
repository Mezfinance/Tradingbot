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

import base64
import logging
import os
import re
import time
import threading

import httpx

log = logging.getLogger(__name__)

RPC_URL = "https://mainnet.helius-rpc.com/"
PUBLIC_RPC = "https://api.mainnet-beta.solana.com"
PUMPSWAP_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
POOL_CREATOR_OFFSET = 11  # pumpswap Pool account: 8-byte discriminator, bump (1), index (2), then creator (32)
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
# Jito's public tip accounts: SOL sent here buys a place at the front of the block.
JITO_TIP_ACCOUNTS = {
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5", "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY", "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh", "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL", "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
}
HISTORY_CREDITS_PER_100 = 10
# Solana has version-1 transactions since late 2026; asking for at most 0 makes Helius reject any history that
# holds one (error -32015), which left pools with no launch data.
TX_VERSION = 1
# One process-wide pace for Helius (the free plan allows ~10 requests/s) and a few retries on 429.
MIN_INTERVAL_S = 0.15
RETRIES_429 = 4
RPC_CREDITS = 1
SOL_MINT = "So11111111111111111111111111111111111111112"
# Program log lines of a liquidity withdrawal (pumpswap `Withdraw`, Meteora/Raydium remove-liquidity variants).
WITHDRAW_RE = re.compile(r"Instruction: (Withdraw|RemoveLiquidity|RemoveAllLiquidity|RemoveBalanceLiquidity|"
                         r"remove_liquidity|withdraw)\b")


# Backup key (11-08): when the main key's credits run out Helius answers HTTP 429 "max usage reached"
# (helius.dev billing FAQ); every client on the default key then moves to BACKUP_HELIUS_API_KEY at once, and one
# call per PROBE_S tries the main key again, so the bot moves back when the main account has credits again.
# Clients given an explicit key (the history backfill's own account) never switch.
BACKUP_ENV = "BACKUP_HELIUS_API_KEY"
PROBE_S = 3600
RATE_PROBE_S = 600  # after a switch on plain 429s (no credits message), try the main again sooner
OUT_OF_CREDITS = re.compile(r"max usage|out of credits|credit limit|insufficient credits|usage limit|quota",
                            re.IGNORECASE)
KEYS = {"on_backup": False, "since": None, "switches": 0, "probe_at": 0.0, "last_reason": None}
_keys_lock = threading.Lock()


def main_key() -> str:
    return os.environ.get("HELIUS_API_KEY", "")


def backup_key() -> str:
    return os.environ.get(BACKUP_ENV, "")


def research_key() -> str:
    """The key of every research client (enricher, operator/creator reads, studies, paper-book helpers): the backup
    account; the main key only when no backup is set (local runs, tests). 11-10 (user): the main key is the paid
    live account, touched only by live trading and the scorer."""
    return backup_key() or main_key()


def current_key() -> str:
    """The key every default client uses now: the backup while the main is out of credits (or missing)."""
    if (KEYS["on_backup"] or not main_key()) and backup_key():
        return backup_key()
    return main_key()


def out_of_credits(resp) -> bool:
    """Helius's 'no credits left' answer (429 / 402 / 403 with a usage message), not an ordinary rate limit."""
    if resp.status_code not in (402, 403, 429):
        return False
    try:
        text = resp.text or ""
    except Exception:  # noqa: BLE001
        text = ""
    return resp.status_code == 402 or bool(OUT_OF_CREDITS.search(text))


def _switch(to_backup: bool, reason: str, probe_s: float = PROBE_S, method: str | None = None) -> None:
    with _keys_lock:
        if KEYS["on_backup"] == to_backup:
            return
        KEYS["on_backup"] = to_backup
        # 11-08: the probe must use the call type that failed. On Fly the main key, out of credits for
        # getTransactionsForAddress, still answered cheap calls, so any call's 200 moved every client back and the
        # next history call moved them away again (13 switches, a websocket reconnect each, ~25% of pools lost)
        KEYS["method"] = method if to_backup else None
        KEYS["since"] = time.time()
        KEYS["switches"] += 1
        KEYS["last_reason"] = reason
        KEYS["probe_at"] = time.time() + probe_s
    if to_backup:
        log.warning("helius: main key out of credits (%s): switched to %s", reason, BACKUP_ENV)
    else:
        log.warning("helius: main key answers again: switched back from %s", BACKUP_ENV)


def key_status() -> dict:
    return {"on_backup": KEYS["on_backup"], "backup_set": bool(backup_key()), "switches": KEYS["switches"],
            "since": KEYS["since"], "last_reason": KEYS["last_reason"], "probe_method": KEYS.get("method"),
            "slowed_methods": pace_status()}


_pace_lock = threading.Lock()
_next_free = [0.0]
# Process-wide call counts, so slow live decisions can be traced to Helius refusing calls (429s are retried
# silently with back-off). `wait_s` = time spent in the pace queue plus 429 back-off.
STATS = {"calls": 0, "rate_limited": 0, "failed": 0, "wait_s": 0.0}


# Adaptive pace per method (11-08): on Fly both free-plan keys answered plain 429 'Too Many Requests' to
# getTransactionsForAddress while other methods were fine, so each method gets its own spacing on top of the
# process-wide one: x ADAPT_UP after a 429 (from at least the base interval, up to MAX_INTERVAL_S), x ADAPT_DOWN
# after a success, back to the base (0 = only the process-wide pace) once it falls below it.
ADAPT_UP = 1.5
ADAPT_DOWN = 0.97
MAX_INTERVAL_S = 3.0
_method_pace: dict[str, list[float]] = {}  # method -> [interval, next free (monotonic)]


def _pace(interval: float, method: str | None = None) -> None:
    """Space Helius calls across every client and thread in the process (and per method when one is slowed)."""
    if interval <= 0:
        return
    with _pace_lock:
        now = time.monotonic()
        start = max(now, _next_free[0])
        gate = _method_pace.get(method) if method else None
        if gate and gate[0] > 0:
            start = max(start, gate[1])
            gate[1] = start + gate[0]
        _next_free[0] = start + interval
        wait = start - now
    if wait > 0:
        STATS["wait_s"] += wait
        time.sleep(wait)


def _adapt(method: str, limited: bool, base: float) -> None:
    """Widen a method's spacing after a 429, narrow it after a success."""
    if base <= 0:
        return
    with _pace_lock:
        gate = _method_pace.setdefault(method, [0.0, 0.0])
        if limited:
            gate[0] = min(MAX_INTERVAL_S, max(gate[0], base) * ADAPT_UP)
        elif gate[0] > 0:
            gate[0] *= ADAPT_DOWN
            if gate[0] < base:
                gate[0] = 0.0


def pace_status() -> dict:
    """Methods currently slowed below the process-wide pace: seconds between their calls."""
    return {m: round(g[0], 2) for m, g in _method_pace.items() if g[0] > 0}


LIVE, RESEARCH = "live", "research"


class Helius:
    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None, role: str = RESEARCH):
        # role LIVE (the scorer): the main key, falling back to the backup when the main is refused; role RESEARCH
        # (the default, every other client): the backup key only, never the main (research_key)
        self.role = role
        if api_key is None and role == RESEARCH and backup_key():
            api_key = backup_key()
        self.fixed_key = api_key is not None  # an explicit key never falls back to the backup
        self.api_key = api_key if api_key is not None else os.environ.get("HELIUS_API_KEY", "")
        self.http = client or httpx.Client(timeout=30)
        self.credits_used = 0
        self._credit_lock = threading.Lock()  # the history backfill calls from several threads
        self.min_interval_s = MIN_INTERVAL_S

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) or (not self.fixed_key and bool(backup_key()))

    def key(self) -> str:
        """The key for the next call (also the websocket's)."""
        if self.fixed_key:
            return self.api_key
        if (KEYS["on_backup"] or not self.api_key) and backup_key():
            return backup_key()
        return self.api_key

    def _rpc(self, method: str, params: list, credits: int):
        if not self.enabled:
            return None
        probing = False
        if (not self.fixed_key and KEYS["on_backup"] and self.api_key and time.time() >= KEYS["probe_at"]
                and KEYS.get("method") in (None, method)):
            with _keys_lock:  # one call per PROBE_S tries the main key again
                if time.time() >= KEYS["probe_at"]:
                    KEYS["probe_at"] = time.time() + PROBE_S
                    probing = True
        for attempt in range(RETRIES_429 + 1):
            _pace(self.min_interval_s, method)
            STATS["calls"] += 1
            key = self.api_key if probing else self.key()
            try:
                resp = self.http.post(RPC_URL, params={"api-key": key},
                                      json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
            except httpx.HTTPError as exc:
                STATS["failed"] += 1
                log.warning("helius %s failed: %s", method, exc)
                return None
            if not self.fixed_key and backup_key() and key != backup_key() and out_of_credits(resp):
                probing = False
                _switch(True, f"HTTP {resp.status_code}", method=method)
                continue  # the same call again on the backup key
            if probing and resp.status_code == 200:
                _switch(False, "probe answered")
                probing = False
            if resp.status_code != 429:
                if resp.status_code == 200:
                    _adapt(method, False, self.min_interval_s)
                break
            STATS["rate_limited"] += 1
            if not out_of_credits(resp):
                _adapt(method, True, self.min_interval_s)
            if attempt < RETRIES_429:
                back_off = min(8.0, 1.0 * 2 ** attempt)
                STATS["wait_s"] += back_off
                time.sleep(back_off)
        if resp.status_code != 200:
            try:
                text = (resp.text or "")[:200]
            except Exception:  # noqa: BLE001
                text = ""
            log.warning("helius %s -> HTTP %s%s (key: %s)", method, resp.status_code, f": {text}" if text else "",
                        "backup" if key == backup_key() and backup_key() else "main")
            if (resp.status_code == 429 and not self.fixed_key and backup_key() and key != backup_key()):
                # 11-08: on Fly the main key answered 429 to every retry for minutes; whether its reply carries the
                # 'max usage' text or not, a key that refuses every retry is no use to the scorer: move to the backup
                # and probe the main again sooner than after a credits reply
                _switch(True, "HTTP 429 through every retry", probe_s=RATE_PROBE_S, method=method)
                return self._rpc(method, params, credits)
            return None
        with self._credit_lock:
            self.credits_used += credits
        body = resp.json()
        if body.get("error"):
            log.warning("helius %s error: %s", method, body["error"])
            return None
        return body.get("result")

    def history_until(self, address: str, until_ts: float, max_pages: int = 4) -> tuple[list[dict], bool, bool]:
        """Oldest-first full transactions touching `address`, page by page (100 per page, 10 credits each) until one
        is later than `until_ts`, the history ends or `max_pages` is reached. Returns (txs, reached the end or
        until_ts, every call succeeded)."""
        txs: list[dict] = []
        token = None
        for _ in range(max_pages):
            opts = {"transactionDetails": "full", "sortOrder": "asc", "limit": 100,
                    "maxSupportedTransactionVersion": TX_VERSION, "encoding": "jsonParsed"}
            if token:
                opts["paginationToken"] = token
            res = self._rpc("getTransactionsForAddress", [address, opts], HISTORY_CREDITS_PER_100)
            if res is None:
                return txs, False, False
            page = res.get("data") or []
            txs += page
            token = res.get("paginationToken")
            if not page or not token or (page[-1].get("blockTime") or 0) > until_ts:
                return txs, True, True
        return txs, False, True

    def slot(self) -> int | None:
        res = self._rpc("getSlot", [{"commitment": "confirmed"}], RPC_CREDITS)
        return int(res) if res is not None else None

    def program_accounts_changed(self, program: str, filters: list, data_slice: dict, since_slot: int,
                                 limit: int = 10000, max_pages: int = 5000, progress=None):
        """Helius getProgramAccountsV2: accounts of `program` matching `filters` that changed since
        `since_slot`, page by page (10 credits per call, counted as such). Filters apply per page, so a page can
        come back empty with a key to the next one: paging stops only when no key is returned. Returns
        [(pubkey, bytes)] or None when the first call failed."""
        out, key, seen_keys = [], None, set()
        self.last_pages = 0
        for page in range(max_pages):
            self.last_pages = page + 1
            if progress and page % 25 == 0:
                progress(page, len(out))
            opts = {"encoding": "base64", "filters": filters, "dataSlice": data_slice, "limit": limit,
                    "changedSinceSlot": since_slot}
            if key:
                opts["paginationKey"] = key
            res = self._rpc("getProgramAccountsV2", [program, opts], HISTORY_CREDITS_PER_100)
            if res is None:
                return out if page else None
            accounts = res.get("accounts") if isinstance(res, dict) else res
            if accounts is None and isinstance(res, dict):
                accounts = res.get("value")
            for a in accounts or []:
                data = (a.get("account") or {}).get("data") or [""]
                out.append((a["pubkey"], base64.b64decode(data[0] if isinstance(data, list) else data)))
            key = res.get("paginationKey") if isinstance(res, dict) else None
            if not key or key in seen_keys:
                break
            seen_keys.add(key)
        return out

    def first_time(self, address: str) -> float | None:
        """blockTime of the oldest of an address's last 1,000 signatures (1 credit): an LP mint's creation."""
        res = self._rpc("getSignaturesForAddress", [address, {"limit": 1000}], RPC_CREDITS)
        times = [s.get("blockTime") for s in res or [] if s.get("blockTime")]
        return float(min(times)) if times else None

    def lp_activity(self, lp_mint: str) -> dict | None:
        """An LP mint's successful signatures (1 credit, last 1,000): creation = the oldest; the LP only moves on
        create / deposit / withdraw, so the newest one after creation is when a full pull happened (if the supply
        is 0 now). None if the call failed or there is no history."""
        res = self._rpc("getSignaturesForAddress", [lp_mint, {"limit": 1000}], RPC_CREDITS)
        times = sorted(float(s["blockTime"]) for s in res or [] if s.get("blockTime") and not s.get("err"))
        return {"first": times[0], "last": times[-1], "count": len(times)} if times else None

    def pull_time(self, lp_mint: str, activity: dict) -> tuple[float | None, bool]:
        """(when the LP supply went to 0, or None if it hasn't; checked). One supply read (1 credit), only when the
        LP moved after creation. A supply read that fails returns checked=False, so the pool is tried again."""
        if activity["count"] <= 1:
            return None, True
        supply = self.token_supply(lp_mint)
        if supply is None:
            return None, False
        return (activity["last"] if supply == 0 else None), True

    def history_from_start(self, address: str, span_s: float, max_pages: int = 6) -> tuple[list[dict], bool, bool]:
        """Oldest-first transactions from the first one until `span_s` seconds after it (or the end of the
        history, or `max_pages` pages of 100). Returns (txs, reached span or end, every call succeeded)."""
        txs: list[dict] = []
        token, t0 = None, None
        for _ in range(max_pages):
            opts = {"transactionDetails": "full", "sortOrder": "asc", "limit": 100,
                    "maxSupportedTransactionVersion": TX_VERSION, "encoding": "jsonParsed"}
            if token:
                opts["paginationToken"] = token
            res = self._rpc("getTransactionsForAddress", [address, opts], HISTORY_CREDITS_PER_100)
            if res is None:
                return txs, False, False
            page = res.get("data") or []
            txs += page
            if t0 is None and page:
                t0 = page[0].get("blockTime") or 0
            token = res.get("paginationToken")
            if not page or not token or (page[-1].get("blockTime") or 0) > (t0 or 0) + span_s:
                return txs, True, True
        return txs, False, True

    def last_before(self, address: str, ts: float, limit: int = 10) -> dict | None | bool:
        """The newest successful transaction touching `address` at or before `ts` (one call, 10 credits), using
        getTransactionsForAddress's blockTime filter. Returns the tx, None if there is none, False if the call
        failed (e.g. the filter isn't supported)."""
        res = self._rpc("getTransactionsForAddress",
                        [address, {"transactionDetails": "full", "sortOrder": "desc", "limit": limit,
                                   "maxSupportedTransactionVersion": TX_VERSION, "encoding": "jsonParsed",
                                   "filters": {"blockTime": {"lte": int(ts)}}}], HISTORY_CREDITS_PER_100)
        if res is None:
            return False
        for tx in res.get("data") or []:
            if (tx.get("meta") or {}).get("err") is None and (tx.get("blockTime") or 0) <= ts:
                return tx
        return None

    def first_tx_time(self, address: str) -> float | None | bool:
        """blockTime of the oldest transaction touching `address` (one call, 10 credits): a token mint's creation.
        None if it has none, False if the call failed."""
        res = self._rpc("getTransactionsForAddress",
                        [address, {"transactionDetails": "full", "sortOrder": "asc", "limit": 1,
                                   "maxSupportedTransactionVersion": TX_VERSION, "encoding": "jsonParsed"}],
                        HISTORY_CREDITS_PER_100)
        if res is None:
            return False
        times = [tx.get("blockTime") for tx in res.get("data") or [] if tx.get("blockTime")]
        return float(min(times)) if times else None

    def history(self, address: str, limit: int = 100) -> list[dict]:
        """Oldest-first full transactions touching `address`."""
        res = self._rpc("getTransactionsForAddress",
                        [address, {"transactionDetails": "full", "sortOrder": "asc", "limit": limit,
                                   "maxSupportedTransactionVersion": TX_VERSION, "encoding": "jsonParsed"}],
                        max(HISTORY_CREDITS_PER_100, HISTORY_CREDITS_PER_100 * limit // 100))
        return (res or {}).get("data") or []

    def pool_creators(self, pools: list[str]) -> dict[str, str | None]:
        """Creator wallet of each pumpswap pool, read from the pool account (100 per call; 1 credit each on
        Helius, the public RPC without a key). None = not a pumpswap pool; pools whose read failed are left out."""
        out: dict[str, str | None] = {}
        for i in range(0, len(pools), 100):
            chunk = pools[i:i + 100]
            params = [chunk, {"encoding": "base64", "dataSlice": {"offset": POOL_CREATOR_OFFSET, "length": 32}}]
            res = self._rpc("getMultipleAccounts", params, RPC_CREDITS) if self.enabled else self._public(
                "getMultipleAccounts", params)
            values = (res or {}).get("value")
            if values is None:
                continue
            for pool, acct in zip(chunk, values):
                ok = acct and acct.get("owner") == PUMPSWAP_PROGRAM
                out[pool] = b58encode(base64.b64decode(acct["data"][0])) if ok else None
        return out

    def accounts(self, addresses: list[str], commitment: str = "confirmed") -> dict[str, tuple[str, bytes] | None] | None:
        """Raw accounts (owner, data bytes), up to 100 per call (1 credit; the public RPC without a key).
        None for an address that doesn't exist; returns None when the call failed."""
        out: dict[str, tuple[str, bytes] | None] = {}
        for i in range(0, len(addresses), 100):
            chunk = addresses[i:i + 100]
            params = [chunk, {"encoding": "base64", "commitment": commitment}]
            res = self._rpc("getMultipleAccounts", params, RPC_CREDITS) if self.enabled else self._public(
                "getMultipleAccounts", params)
            values = (res or {}).get("value")
            if values is None:
                return None
            for a, acct in zip(chunk, values):
                out[a] = (acct["owner"], base64.b64decode(acct["data"][0])) if acct else None
        return out

    def _public(self, method: str, params: list):
        try:
            body = self.http.post(PUBLIC_RPC, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("public rpc %s failed: %s", method, exc)
            return None
        return body.get("result")

    def recent(self, address: str, limit: int = 50) -> list[dict]:
        """Newest-first full transactions touching `address`."""
        res = self._rpc("getTransactionsForAddress",
                        [address, {"transactionDetails": "full", "sortOrder": "desc", "limit": limit,
                                   "maxSupportedTransactionVersion": TX_VERSION, "encoding": "jsonParsed"}],
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


def _vault(tx: dict, pool: str, mint: str, side: str) -> float:
    for b in (tx.get("meta") or {}).get(side) or []:
        if b.get("owner") == pool and b.get("mint") == mint:
            return float((b.get("uiTokenAmount") or {}).get("uiAmount") or 0)
    return 0.0


def launch_record(txs: list[dict], pool: str, base_mint: str, window_s: float = 180,
                  exhausted: bool | None = None) -> dict:
    """A pool's first `window_s` seconds from its oldest-first transactions: one row per successful tx,
    [seconds after creation, slot, fee payer, pool's token change, pool's SOL change, tokens after, SOL after].
    Positive token change with negative SOL change = someone sold to the pool; the reverse = a buy."""
    if not txs:
        return {"error": "no transactions"}
    t0 = txs[0].get("blockTime") or 0
    rows, failed, complete = [], 0, (len(txs) < 100 if exhausted is None else exhausted)
    for tx in txs:
        bt = tx.get("blockTime") or 0
        if bt > t0 + window_s:
            complete = True
            break
        if (tx.get("meta") or {}).get("err") is not None:
            failed += 1
            continue
        b0, b1 = _vault(tx, pool, base_mint, "preTokenBalances"), _vault(tx, pool, base_mint, "postTokenBalances")
        q0, q1 = _vault(tx, pool, SOL_MINT, "preTokenBalances"), _vault(tx, pool, SOL_MINT, "postTokenBalances")
        keys = _keys(tx)
        rows.append([bt - t0, tx.get("slot"), keys[0] if keys else None, round(b1 - b0, 6), round(q1 - q0, 9),
                     round(b1, 6), round(q1, 9)])
    seed = next((r for r in rows if r[4] > 0 and r[6] == r[4]), None)  # the tx that funded the empty SOL vault
    creator = seed[2] if seed else (_keys(txs[0]) or [None])[0]
    return {"create_ts": t0, "creator": creator, "seed_sol": seed[6] if seed else None, "window_s": window_s,
            "txs": rows,
            "failed": failed, "complete": complete, "fetched": len(txs)}


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


def b58encode(raw: bytes) -> str:
    n = int.from_bytes(raw, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = B58[r] + out
    return "1" * (len(raw) - len(raw.lstrip(b"\0"))) + out
