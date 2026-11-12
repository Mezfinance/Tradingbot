"""Wallet rotation for the real-money launch canary (11-11).

The operators appear to pull once they recognise our wallet: a brand-new wallet's first trade
survives, then the next trades on that same wallet rug. Rotation spreads each trade across up to
``MAX_WALLETS`` wallets and never lets a wallet hold more than one open trade at a time, so every
buy enters on the least-recently-used (freshest) free wallet. Built for the direct route only
(the LP route needs the same wallet to withdraw its own liquidity, which rotation would complicate,
and Jupiter routing is a separate experiment).

``RotatingExecutor`` wraps one direct executor per wallet and routes by token mint (the canary holds
one position per mint): a buy is assigned the least-recently-used free wallet; the sell goes back to
the wallet that holds the tokens; the assignment is persisted so a restart still sells from the right
wallet. A wallet frees on a successful sell, or is reclaimed after ``reclaim_s`` if its trade stuck
(a rugged pool whose sale never lands), so a wallet is never lost to rotation for good.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

from tradingbot.execution.base import Fill

log = logging.getLogger(__name__)

MAX_WALLETS = 10
RECLAIM_S = 1200.0  # free a wallet whose trade has been open this long (a stuck/rugged position, written off at $0)
STATE_FILE = "canary_rotation.json"  # next to the journal: mint -> wallet, so a restart sells from the right wallet


def read_canary_wallets(cfg) -> list:
    """Up to ``MAX_WALLETS`` rotation keys, from the configured file, the entrypoint's secret copy, or the
    ``TB_CANARY_WALLETS`` secret itself (a ``fly machine exec`` command sees only the last). The value is a JSON
    array whose elements are each a solana-keygen int array or a base58 string, or base58 keys one per line or
    comma-separated. Duplicate pubkeys are dropped and the first ``MAX_WALLETS`` are kept."""
    from tradingbot.execution.jupiter import load_keypair

    text = None
    for path in (getattr(cfg, "launch_canary_wallets_path", "") or "", "/run/secrets/canary_wallets.json"):
        if path and os.path.exists(path):
            with open(path) as fh:
                text = fh.read()
            break
    if text is None:
        text = os.environ.get("TB_CANARY_WALLETS")
    if not text or not text.strip():
        return []
    text = text.strip()
    keys = []
    if text.startswith("["):
        for el in json.loads(text):
            if isinstance(el, list):
                from solders.keypair import Keypair

                keys.append(Keypair.from_bytes(bytes(el)))
            else:
                keys.append(load_keypair(str(el)))
    else:
        for part in text.replace(",", "\n").split("\n"):
            if part.strip():
                keys.append(load_keypair(part))
    seen: set[str] = set()
    out = []
    for k in keys:
        pk = str(k.pubkey())
        if pk in seen:
            continue
        seen.add(pk)
        out.append(k)
        if len(out) >= MAX_WALLETS:
            break
    return out


class RotatingExecutor:
    """Routes canary trades across sub-executors (one per wallet), one open trade per wallet."""

    def __init__(self, subs: list, state_path: str | None = None, reclaim_s: float = RECLAIM_S, clock=time.time):
        if not subs:
            raise ValueError("RotatingExecutor needs at least one wallet")
        self.subs = subs
        self.route = getattr(subs[0], "route", "direct")
        self.reclaim_s = reclaim_s
        self.clock = clock
        self.last_note = ""
        self._lock = threading.RLock()
        self._assign: dict[str, int] = {}   # mint -> sub index holding its open trade
        self._opened: dict[str, float] = {}  # mint -> when the trade was assigned
        self._used: dict[int, float] = {}    # sub index -> last time it opened a trade (missing = never used, freshest)
        self._pools: dict[str, str] = {}
        self._state_path = state_path
        self._load()
        for mint, idx in self._assign.items():  # a reloaded trade must have its pool ready on its wallet to sell
            self._apply_pool(idx, mint)

    # ---- persistence ----
    def _load(self) -> None:
        if not self._state_path or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as fh:
                d = json.load(fh)
            n = len(self.subs)
            self._assign = {m: int(i) for m, i in d.get("assign", {}).items() if 0 <= int(i) < n}
            self._opened = {m: float(t) for m, t in d.get("opened", {}).items() if m in self._assign}
            self._used = {int(i): float(t) for i, t in d.get("used", {}).items() if 0 <= int(i) < n}
            self._pools = dict(d.get("pools", {}))
        except (OSError, ValueError):
            log.warning("could not read %s", self._state_path, exc_info=True)

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"assign": self._assign, "opened": self._opened,
                           "used": {str(i): t for i, t in self._used.items()},
                           "pools": dict(list(self._pools.items())[-500:])}, fh)
            os.replace(tmp, self._state_path)
        except OSError:
            log.warning("could not save %s", self._state_path, exc_info=True)

    # ---- wallet selection ----
    def _free_indices(self) -> list[int]:
        n = len(self.subs)
        free = [i for i in range(n) if i not in set(self._assign.values())]
        if free:
            return free
        now = self.clock()  # none free: reclaim wallets whose trade stuck past reclaim_s (a rug that never sold)
        for m, t in list(self._opened.items()):
            if now - t >= self.reclaim_s:
                self._assign.pop(m, None)
                self._opened.pop(m, None)
        return [i for i in range(n) if i not in set(self._assign.values())]

    def _holder(self, token: str) -> int:
        idx = self._assign.get(token)
        if idx is not None:
            return idx
        for i, sub in enumerate(self.subs):  # assignment lost (old restart): find the wallet that holds the tokens
            try:
                if sub.token_balance(token) > 0:
                    return i
            except Exception:  # noqa: BLE001
                continue
        return 0

    def _apply_pool(self, idx: int, token: str) -> None:
        pool = self._pools.get(token)
        if pool and hasattr(self.subs[idx], "remember_pool"):
            self.subs[idx].remember_pool(token, pool)

    # ---- executor surface ----
    def remember_pool(self, token: str, pool: str) -> None:
        with self._lock:
            self._pools[token] = pool
            idx = self._assign.get(token)
            self._save()
        if idx is not None:
            self._apply_pool(idx, token)

    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        with self._lock:
            idx = self._assign.get(token)
            if idx is None:
                free = self._free_indices()
                if not free:
                    self.last_note = f"no free wallet (all {len(self.subs)} busy)"
                    return None
                idx = min(free, key=lambda i: self._used.get(i, 0.0))  # the freshest free wallet
                self._assign[token] = idx
                self._opened[token] = self.clock()
                self._save()
            self._apply_pool(idx, token)
        fill = self.subs[idx].buy(token, usd, ref_price)
        with self._lock:
            if fill is None or not getattr(fill, "ok", True):  # the buy never opened: release the wallet, keep it fresh
                self._assign.pop(token, None)
                self._opened.pop(token, None)
                self._save()
                self.last_note = getattr(self.subs[idx], "last_note", "") or (getattr(fill, "note", "") if fill else "")
            else:
                self._used[idx] = self.clock()
                self.last_note = ""
                self._save()
        return fill

    def feint(self, token: str, usd: float, commit: bool = True) -> str:
        """Probe the pool from the wallet that will hold the commit, so the feint and the real buy come from one
        wallet. Keeps the wallet assigned on 'alive' so buy() reuses it; releases it otherwise -- and always when
        `commit` is False (feint-only measurement runs make no commit, so the wallet must not stay assigned)."""
        with self._lock:
            idx = self._assign.get(token)
            if idx is None:
                free = self._free_indices()
                if not free:
                    self.last_note = f"no free wallet (all {len(self.subs)} busy)"
                    return "inconclusive"
                idx = min(free, key=lambda i: self._used.get(i, 0.0))
                self._assign[token] = idx
                self._opened[token] = self.clock()
                self._save()
            self._apply_pool(idx, token)
        sub = self.subs[idx]
        verdict = sub.feint(token, usd) if hasattr(sub, "feint") else "alive"
        self.last_feint = getattr(sub, "last_feint", {})
        self.last_note = getattr(sub, "last_note", "")
        with self._lock:
            if verdict != "alive" or not commit:  # no commit coming (pulled, inconclusive, or feint-only): free it
                self._assign.pop(token, None)
                self._opened.pop(token, None)
                self._save()
        return verdict

    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None:
        idx = self._holder(token)
        self._apply_pool(idx, token)
        fill = self.subs[idx].sell(token, qty, ref_price)
        with self._lock:
            if fill is not None and getattr(fill, "ok", False):  # a successful (full) exit frees the wallet
                self._assign.pop(token, None)
                self._opened.pop(token, None)
                self._save()
        return fill

    def sweep(self, keep=frozenset(), apply: bool = True) -> dict:
        merged: dict = {"closed": 0, "usd_back": 0.0, "held": []}
        for sub in self.subs:
            if not hasattr(sub, "sweep"):
                continue
            try:
                r = sub.sweep(keep, apply)
            except TypeError:
                try:
                    r = sub.sweep(keep)
                except Exception:  # noqa: BLE001
                    continue
            except Exception:  # noqa: BLE001
                continue
            for k, v in r.items():
                if isinstance(v, bool):
                    merged[k] = v
                elif isinstance(v, (int, float)):
                    merged[k] = merged.get(k, 0) + v
                elif isinstance(v, list):
                    merged[k] = merged.get(k, []) + v
                else:
                    merged[k] = v
        return merged

    def token_balance(self, mint: str) -> int:
        return self.subs[self._holder(mint)].token_balance(mint)
