"""Public Telegram channel previews (t.me/s/<channel>): contract addresses posted by calling
channels, with post times. No key; only public channels; configured via `tg_channels`."""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

BASE = "https://t.me/s/"
MINT_RE = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")
POST_RE = re.compile(r'data-post="([^"]+)"(.*?)(?=data-post="|\Z)', re.S)
TIME_RE = re.compile(r'<time[^>]+datetime="([^"]+)"')
TAG_RE = re.compile(r"<[^>]+>")


def fetch_channel(name: str, client: httpx.Client | None = None) -> list[dict]:
    """[{"post": "chan/123", "ts": epoch, "mints": [...]}] for the latest posts on the preview page."""
    http = client or httpx.Client(timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = http.get(f"{BASE}{name}")
    except httpx.HTTPError as exc:
        log.debug("telegram %s failed: %s", name, exc)
        return []
    if resp.status_code != 200:
        return []
    return parse_page(resp.text)


def parse_page(page: str) -> list[dict]:
    out = []
    for post, body in POST_RE.findall(page):
        t = TIME_RE.search(body)
        ts = None
        if t:
            try:
                ts = datetime.fromisoformat(t.group(1)).timestamp()
            except ValueError:
                pass
        text = html.unescape(TAG_RE.sub(" ", body))
        mints = sorted(set(MINT_RE.findall(text)))
        if mints:
            out.append({"post": post, "ts": ts, "mints": mints})
    return out
