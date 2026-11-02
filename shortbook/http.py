"""Small fail-soft JSON client: per-host pacing, retries on 429/5xx/network errors, last error per host."""
from __future__ import annotations
import threading, time
from urllib.parse import urlparse
import httpx


class Http:
    def __init__(self, min_interval_s: float = 0.25, retries: int = 3, timeout: float = 15.0, client=None):
        self.c = client or httpx.Client(timeout=timeout, headers={"User-Agent": "shortbook/0.1"}, follow_redirects=True)
        self.min_interval_s, self.retries = min_interval_s, retries
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self.errors: dict[str, str] = {}   # host -> last error ('' after a success)

    def _pace(self, host: str) -> None:
        with self._lock:
            wait = self._last.get(host, 0) + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def get(self, url: str, params: dict | None = None):
        host = urlparse(url).netloc
        for attempt in range(self.retries + 1):
            self._pace(host)
            try:
                r = self.c.get(url, params=params)
            except Exception as e:  # network error
                self.errors[host] = f"{type(e).__name__}: {e}"[:200]
                time.sleep(min(2 ** attempt, 8) if self.min_interval_s else 0)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                self.errors[host] = f"HTTP {r.status_code}"
                time.sleep(min(2 ** attempt, 8) if self.min_interval_s else 0)
                continue
            if r.status_code != 200:
                self.errors[host] = f"HTTP {r.status_code}: {r.text[:120]}"
                return None
            self.errors[host] = ""
            try:
                return r.json()
            except Exception:
                self.errors[host] = "bad JSON"
                return None
        return None
