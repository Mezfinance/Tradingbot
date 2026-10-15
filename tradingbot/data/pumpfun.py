"""pump.fun's own (unofficial) frontend API: livestream status, all-time-high market cap,
security verdict, and the creator's full launch history. No key; can change without notice,
so every call fails soft (None)."""

from __future__ import annotations

import logging
import threading
import time

import httpx

log = logging.getLogger(__name__)

BASE = "https://frontend-api-v3.pump.fun"
HEADERS = {"User-Agent": "Mozilla/5.0", "Origin": "https://pump.fun", "Accept": "application/json"}
MIN_INTERVAL_S = 0.5


class PumpFun:
    def __init__(self, client: httpx.Client | None = None, min_interval_s: float = MIN_INTERVAL_S):
        self.http = client or httpx.Client(base_url=BASE, timeout=15, headers=HEADERS)
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        with self._lock:
            wait = self._last + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        try:
            resp = self.http.get(path, params=params)
        except httpx.HTTPError as exc:
            log.debug("pump.fun %s failed: %s", path, exc)
            return None
        if resp.status_code != 200:
            log.debug("pump.fun %s -> HTTP %s", path, resp.status_code)
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    def coin(self, mint: str) -> dict | None:
        return self._get(f"/coins-v2/{mint}")

    def creator_coins(self, creator: str, limit: int = 50) -> dict | None:
        """{"count": total launches by this wallet, "coins": [...newest first...]}."""
        return self._get(f"/coins-v2/user-created-coins/{creator}", {"offset": 0, "limit": limit})


def compact_coin(c: dict | None) -> dict:
    if not c:
        return {}
    return {
        "creator": c.get("creator"), "bonding_curve": c.get("bonding_curve"),
        "is_live": bool(c.get("is_currently_live")), "complete": bool(c.get("complete")),
        "usd_market_cap": c.get("usd_market_cap"), "ath_market_cap_usd": _ath_usd(c),
        "volume_1h_usd": c.get("volume_1h_usd"), "boost_mode": c.get("boost_mode"),
        "verified": bool(c.get("verified")), "nsfw": bool(c.get("nsfw")),
        "twitter": c.get("twitter") or "", "website": c.get("website") or "", "telegram": c.get("telegram") or "",
        "image_uri": c.get("image_uri") or "",
        "transfer_fee_bps": c.get("transfer_fee_bps") or 0, "transfer_hook": bool(c.get("transfer_hook_program")),
        "security_verdict": (c.get("security_verdict") or {}).get("verdict"),
        "is_cashback": bool(c.get("is_cashback_enabled")), "tokenized_agent": bool(c.get("tokenized_agent")),
    }


def _ath_usd(c: dict) -> float | None:
    """All-time-high market cap in USD (checked live: TAILER ath 5,677 vs usd_market_cap 2,921)."""
    try:
        v = float(c["ath_market_cap"])
    except (KeyError, TypeError, ValueError):
        return None
    return v if v > 0 else None


def compact_creator(d: dict | None, before_ms: float) -> dict:
    """The creator's track record from coins launched BEFORE `before_ms` (the token being judged)."""
    if not d:
        return {}
    # The list carries no market caps (all 0), but launch times and graduation flags are real.
    prior = [c for c in d.get("coins") or [] if 0 < (c.get("created_timestamp") or 0) < before_ms]
    day_ms = 86_400_000
    return {
        "creator_launches": d.get("count"),  # all-time total (may include a few later launches)
        "creator_recent_prior": len(prior),  # of the newest `limit` coins
        "creator_prior_graduated": sum(bool(c.get("complete")) for c in prior),
        "creator_launches_24h": sum(before_ms - c["created_timestamp"] <= day_ms for c in prior),
        "creator_last_launch_gap_min": ((before_ms - max(c["created_timestamp"] for c in prior)) / 60_000
                                        if prior else None),
        "creator_prior_names": [(c.get("symbol") or "")[:20] for c in prior[:20]],
    }
