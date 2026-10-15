"""X (Twitter) API v2 (needs X_BEARER_TOKEN): who is posting a token's contract address.

X bills pay-per-use (~$0.005 per post read in 2026), so the caller enforces a daily dollar
budget with the estimates below (`cost_usd`). Only the contract address is searched: tickers
and names collide with unrelated posts.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

BASE = "https://api.x.com/2"
KOL_FOLLOWERS = 50_000


class XClient:
    def __init__(self, bearer: str | None = None, client: httpx.Client | None = None,
                 cost_per_post: float = 0.005, cost_per_count_call: float = 0.005):
        self.bearer = bearer if bearer is not None else os.environ.get("X_BEARER_TOKEN", "")
        self.http = client or httpx.Client(base_url=BASE, timeout=20)
        self.cost_per_post, self.cost_per_count_call = cost_per_post, cost_per_count_call
        self.cost_usd = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.bearer)

    def _get(self, path: str, params: dict) -> dict | None:
        if not self.enabled:
            return None
        try:
            resp = self.http.get(path, params=params, headers={"Authorization": f"Bearer {self.bearer}"})
        except httpx.HTTPError as exc:
            log.warning("x api %s failed: %s", path, exc)
            return None
        if resp.status_code != 200:
            log.warning("x api %s -> HTTP %s: %s", path, resp.status_code, resp.text[:200])
            return None
        return resp.json()

    def count(self, query: str) -> int | None:
        """Posts matching `query` over the last 7 days (counts endpoint)."""
        data = self._get("/tweets/counts/recent", {"query": query, "granularity": "day"})
        if data is None:
            return None
        self.cost_usd += self.cost_per_count_call
        return int((data.get("meta") or {}).get("total_tweet_count") or 0)

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        data = self._get("/tweets/search/recent", {
            "query": query, "max_results": max(10, min(100, max_results)),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id", "user.fields": "public_metrics,created_at,verified",
        })
        if data is None:
            return []
        posts = data.get("data") or []
        self.cost_usd += self.cost_per_post * len(posts)
        users = {u["id"]: u for u in (data.get("includes") or {}).get("users") or []}
        out = []
        for p in posts:
            u = users.get(p.get("author_id"), {})
            pm, um = p.get("public_metrics") or {}, u.get("public_metrics") or {}
            out.append({
                "ts": _ts(p.get("created_at")), "author": p.get("author_id"),
                "followers": um.get("followers_count"), "author_created": _ts(u.get("created_at")),
                "verified": bool(u.get("verified")),
                "likes": pm.get("like_count"), "reposts": pm.get("retweet_count"), "replies": pm.get("reply_count"),
            })
        return out


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def mint_query(mint: str) -> str:
    return f'"{mint}" -is:retweet'
