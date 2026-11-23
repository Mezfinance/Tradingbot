"""WS reserve feed for the box's execution path -- the pure, testable helpers (I/O stays on the box).

The box firing from shreds has no reserves (shreds carry no post-execution balances). To price fills and size
orders, the box runs a Solana `accountSubscribe` WS on each active pool's two vaults at `processed` commitment:
every swap pushes the vault's new absolute balance, which feeds reserve_tracker.ReserveTracker (<=1 slot stale,
no accumulation drift). The gate still fires from shreds; this only supplies reserves for the executor + the NET.

This module is the box-local-I/O-free half:
  * pool_vaults(account)          -- the two vault addresses + token mint from a pumpswap pool account (one RPC
                                     read per pool: getAccountInfo/getMultipleAccounts, base64).
  * account_subscribe_request     -- the JSON-RPC accountSubscribe frame (base64, processed).
  * account_unsubscribe_request   -- the accountUnsubscribe frame (cleanup after a pool ages out).
  * parse_message(msg, subs)      -- a parsed WS frame -> ("subscribed", req_id, sub_id) | ("push", vault, data,
                                     slot) | None, resolving a subscription id to its vault.
  * SubManager                    -- req_id/sub_id/pool bookkeeping + a TTL so a pool's subs drop ~180s after first
                                     sight (the fresh window).

Box run loop (run_localshred.py, validated on the box, not here): open a websocket to the Helius WS endpoint; on
the Consumer's on_pool(pool) hook, read the pool account (RPC) -> pool_vaults -> SubManager.subscribe each vault
-> ws.send; on each ws frame, parse_message -> confirm a "subscribed" or feed a "push" to ReserveTracker.on_push;
periodically SubManager.expired(now, ttl) -> account_unsubscribe_request + tracker.unwatch. Fires are stamped by
passing the tracker to Consumer(reserves=tracker).
"""
import base64

from .data import pumpswap


def _to_bytes(account):
    if isinstance(account, (list, tuple)) and account:
        account = account[0]
    if isinstance(account, (bytes, bytearray)):
        return bytes(account)
    if isinstance(account, str):
        try:
            return base64.b64decode(account)
        except Exception:
            return None
    return None


def pool_vaults(account) -> dict | None:
    """The pool's (sol_vault, token_vault, token_mint) from its account data (base64 str, [b64,'base64'], or raw
    bytes). None if it is not a readable pumpswap SOL-pair pool."""
    raw = _to_bytes(account)
    if raw is None:
        return None
    p = pumpswap.parse_pool(raw)
    if not p:
        return None
    return {"sol_vault": p["sol_vault"], "token_vault": p["token_vault"], "token_mint": p["token_mint"]}


def account_subscribe_request(req_id: int, vault: str, commitment: str = "processed") -> dict:
    """processed: the freshest post-execution balance, ~<=1 slot behind the in-flight slot."""
    return {"jsonrpc": "2.0", "id": int(req_id), "method": "accountSubscribe",
            "params": [vault, {"encoding": "base64", "commitment": commitment}]}


def account_unsubscribe_request(req_id: int, sub_id: int) -> dict:
    return {"jsonrpc": "2.0", "id": int(req_id), "method": "accountUnsubscribe", "params": [int(sub_id)]}


def parse_message(msg: dict, subs: "SubManager"):
    """A parsed WS JSON frame -> an event:
       accountNotification        -> ("push", vault, data, slot)   (data = value.data, slot = context.slot)
       subscribe ack {id, result} -> ("subscribed", req_id, sub_id)
       anything else              -> None"""
    if not isinstance(msg, dict):
        return None
    if msg.get("method") == "accountNotification":
        p = msg.get("params") or {}
        vault = subs.vault_of(p.get("subscription"))
        res = p.get("result") or {}
        slot = (res.get("context") or {}).get("slot")
        if vault is not None and slot is not None:
            return ("push", vault, (res.get("value") or {}).get("data"), int(slot))
        return None
    if "id" in msg and isinstance(msg.get("result"), int):
        return ("subscribed", msg["id"], msg["result"])
    return None


class SubManager:
    """req_id/sub_id/pool bookkeeping for the vault subscriptions, plus a first-seen TTL for cleanup. Pure -- the
    box drives it (allocates a req_id, sends the frame, confirms the ack, resolves pushes, expires old pools)."""

    def __init__(self) -> None:
        self._pending: dict[int, tuple[str, str]] = {}   # req_id -> (pool, vault), awaiting the subscribe ack
        self._by_sub: dict[int, str] = {}                # sub_id -> vault, for routing pushes
        self._pool_subs: dict[str, list[int]] = {}       # pool -> [sub_id, ...]
        self._seen: dict[str, float] = {}                # pool -> first-seen wall time
        self._next = 1

    def subscribe(self, pool: str, vault: str, now: float) -> tuple[int, dict]:
        """Allocate a req_id for a (pool, vault) subscription; returns (req_id, the request frame to send)."""
        rid = self._next
        self._next += 1
        self._pending[rid] = (pool, vault)
        self._seen.setdefault(pool, now)
        return rid, account_subscribe_request(rid, vault)

    def confirm(self, req_id: int, sub_id: int):
        """Resolve a subscribe ack: binds sub_id -> vault. Returns (pool, vault) or None if the req_id is unknown."""
        pv = self._pending.pop(req_id, None)
        if pv is None:
            return None
        pool, vault = pv
        self._by_sub[sub_id] = vault
        self._pool_subs.setdefault(pool, []).append(sub_id)
        return pool, vault

    def vault_of(self, sub_id):
        return self._by_sub.get(sub_id)

    def expired(self, now: float, ttl: float) -> list[str]:
        """Pools first seen more than `ttl` seconds ago -- the box unsubscribes and unwatches them."""
        return [p for p, t in self._seen.items() if now - t > ttl]

    def drop(self, pool: str) -> list[int]:
        """Forget a pool; returns its sub_ids so the box can send accountUnsubscribe for each."""
        sids = self._pool_subs.pop(pool, [])
        for sid in sids:
            self._by_sub.pop(sid, None)
        self._seen.pop(pool, None)
        return sids
