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
                 cost_per_post: float = 0.005, cost_per_count_call: float = 0.005, cost_per_user: float = 0.01):
        self.bearer = bearer if bearer is not None else os.environ.get("X_BEARER_TOKEN", "")
        self.http = client or httpx.Client(base_url=BASE, timeout=20)
        self.cost_per_post, self.cost_per_count_call = cost_per_post, cost_per_count_call
        self.cost_per_user = cost_per_user
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

    def user(self, username: str) -> dict | None:
        """An account's age and reach (one user read)."""
        data = self._get(f"/users/by/username/{username}", {"user.fields": "created_at,public_metrics,verified"})
        if data is None:
            return None
        self.cost_usd += self.cost_per_user
        u = data.get("data") or {}
        pm = u.get("public_metrics") or {}
        return {"id": u.get("id"), "created": _ts(u.get("created_at")), "followers": pm.get("followers_count"),
                "following": pm.get("following_count"), "posts": pm.get("tweet_count"), "verified": bool(u.get("verified"))}

    def post(self, post_id: str) -> dict | None:
        """One post's reach (the viral post a token was launched off)."""
        data = self._get(f"/tweets/{post_id}", {"tweet.fields": "created_at,public_metrics,author_id",
                                                "expansions": "author_id", "user.fields": "public_metrics"})
        if data is None:
            return None
        self.cost_usd += self.cost_per_post
        t = data.get("data") or {}
        pm = t.get("public_metrics") or {}
        author = ((data.get("includes") or {}).get("users") or [{}])[0]
        return {"ts": _ts(t.get("created_at")), "likes": pm.get("like_count"), "reposts": pm.get("retweet_count"),
                "quotes": pm.get("quote_count"), "replies": pm.get("reply_count"),
                "views": pm.get("impression_count"),
                "author_followers": (author.get("public_metrics") or {}).get("followers_count")}

    def timeline(self, user_id: str, max_results: int = 5) -> list[dict]:
        data = self._get(f"/users/{user_id}/tweets", {"max_results": max(5, min(100, max_results)),
                                                      "tweet.fields": "created_at"})
        posts = (data or {}).get("data") or []
        self.cost_usd += self.cost_per_post * len(posts)
        return [{"ts": _ts(p.get("created_at")), "text": (p.get("text") or "")[:400]} for p in posts]

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        data = self._get("/tweets/search/recent", {
            "query": query, "max_results": max(10, min(100, max_results)),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id", "user.fields": "public_metrics,created_at,verified,username",
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
                "username": (u.get("username") or "").lower(), "text": (p.get("text") or "")[:280],
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


def cashtag_query(symbol: str) -> str:
    return f"${symbol} -is:retweet"


def post_id(link: str) -> str | None:
    """'https://x.com/someone/status/123?s=20' -> '123'."""
    if "/status/" not in (link or ""):
        return None
    tail = link.split("/status/", 1)[1]
    digits = "".join(ch for ch in tail.split("?")[0].split("/")[0] if ch.isdigit())
    return digits or None


def handle(link: str) -> str | None:
    """'https://x.com/someone' or 'someone' -> 'someone' (None for a post link)."""
    if not link or "/status/" in link:
        return None
    h = link.rstrip("/").rsplit("/", 1)[-1].lstrip("@").split("?")[0]
    return h if h and h.replace("_", "").isalnum() else None
