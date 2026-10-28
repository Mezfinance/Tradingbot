"""Free public sources for the long tail of pump factors (FACTORS.md). No keys; every call
fails soft (None or empty). Checked live from a server on 2026-10-15."""

from __future__ import annotations

import html
import logging
import re
import threading
import time
from datetime import datetime

import httpx

from tradingbot.data import jupiter

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0"}
MINT_RE = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")
MEMECOIN_INDEX = {  # big Solana memecoins: their move is the memecoin market's mood
    "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "POPCAT": "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
}


class Paced:
    """Shared minimum gap between calls to one host."""

    def __init__(self, interval_s: float):
        self.interval_s, self._lock, self._last = interval_s, threading.Lock(), 0.0

    def wait(self) -> None:
        with self._lock:
            gap = self._last + self.interval_s - time.monotonic()
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()


class Extras:
    def __init__(self, client: httpx.Client | None = None, pace_s: float = 0.4):
        self.http = client or httpx.Client(timeout=15, headers=UA, follow_redirects=True)
        self.pace = {h: Paced(pace_s) for h in ("dex", "misc")}
        self.pace["jup"] = jupiter.GATE  # Jupiter's limit is per key/IP: share the trading client's pace
        self.pace["rug"] = Paced(2.0)  # RugCheck rate-limits hard; fresh tokens may also not be indexed yet

    def _get(self, host: str, url: str, **kw):
        self.pace[host].wait()
        try:
            resp = self.http.get(url, **kw)
        except httpx.HTTPError as exc:
            log.debug("%s failed: %s", url, exc)
            return None
        return resp if resp.status_code == 200 else None

    def _json(self, host: str, url: str, **kw):
        resp = self._get(host, url, **kw)
        try:
            return resp.json() if resp is not None else None
        except ValueError:
            return None

    # --- per token -----------------------------------------------------------------
    def jupiter_token(self, mint: str) -> dict | None:
        data = self._json("jup", f"{jupiter.base_url()}/tokens/v2/search", params={"query": mint},
                          headers=jupiter.headers())
        return next((t for t in data or [] if t.get("id") == mint), None)

    def jupiter_trending(self, interval: str = "1h", limit: int = 100) -> list[str]:
        data = self._json("jup", f"{jupiter.base_url()}/tokens/v2/toptrending/{interval}", params={"limit": limit},
                          headers=jupiter.headers())
        return [t["id"] for t in data or [] if t.get("id")]

    def rugcheck(self, mint: str) -> dict | None:
        return self._json("rug", f"https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary")

    def dex_search(self, query: str) -> list[dict]:
        data = self._json("dex", "https://api.dexscreener.com/latest/dex/search", params={"q": query})
        return (data or {}).get("pairs") or []

    def discord_members(self, invite_url: str) -> int | None:
        code = invite_url.rstrip("/").rsplit("/", 1)[-1]
        data = self._json("misc", f"https://discord.com/api/v10/invites/{code}", params={"with_counts": "true"})
        return (data or {}).get("approximate_member_count")

    def telegram_members(self, handle: str) -> int | None:
        handle = handle.rstrip("/").rsplit("/", 1)[-1].lstrip("@")
        resp = self._get("misc", f"https://t.me/{handle}")
        if resp is None:
            return None
        m = re.search(r'tgme_page_extra">\s*([\d][\d ,\xa0]*)\s+(members|subscribers)', html.unescape(resp.text))
        return int(re.sub(r"\D", "", m.group(1))) if m and re.sub(r"\D", "", m.group(1)) else None

    def website(self, url: str) -> dict:
        if not url.startswith("http"):
            url = "https://" + url
        resp = self._get("misc", url)
        if resp is None:
            return {"reachable": False}
        page = resp.text[:300_000].lower()
        host = str(resp.url.host or "")
        return {"reachable": True, "bytes": len(resp.content), "links": page.count("<a "),
                "scripts": page.count("<script"), "images": page.count("<img"),
                "template_host": any(t in host for t in ("linktr.ee", "carrd.co", "vercel.app", "netlify.app",
                                                         "github.io", "wixsite", "webflow.io", "framer"))}

    def domain_registered(self, domain: str) -> float | None:
        data = self._json("misc", f"https://rdap.org/domain/{domain}")
        for ev in (data or {}).get("events") or []:
            if ev.get("eventAction") == "registration":
                try:
                    return datetime.fromisoformat(ev["eventDate"].replace("Z", "+00:00")).timestamp()
                except (KeyError, ValueError):
                    return None
        return None

    # --- market-wide ---------------------------------------------------------------
    def coinbase_spot(self, pair: str) -> float | None:
        data = self._json("misc", f"https://api.coinbase.com/v2/prices/{pair}/spot")
        try:
            return float(data["data"]["amount"])
        except (TypeError, KeyError, ValueError):
            return None

    def fear_greed(self) -> float | None:
        data = self._json("misc", "https://api.alternative.me/fng/", params={"limit": 1})
        try:
            return float(data["data"][0]["value"])
        except (TypeError, KeyError, IndexError, ValueError):
            return None

    def hyperliquid(self) -> dict:
        """SOL funding and open interest, and every coin with a Hyperliquid perpetual."""
        self.pace["misc"].wait()
        try:
            resp = self.http.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"})
            meta, ctxs = resp.json()
        except (httpx.HTTPError, ValueError, TypeError):
            return {}
        names = [u["name"] for u in meta.get("universe", [])]
        out = {"perp_symbols": names}
        if "SOL" in names:
            ctx = ctxs[names.index("SOL")]
            out.update({"sol_funding": float(ctx.get("funding") or 0), "sol_oi": float(ctx.get("openInterest") or 0)})
        return out

    def solana_stables(self) -> tuple[float | None, float | None]:
        """Solana stablecoin supply now and 7 days ago (DefiLlama, daily)."""
        data = self._json("misc", "https://stablecoins.llama.fi/stablecoincharts/Solana")
        try:
            now, week = data[-1], data[-8]
            return (float(now["totalCirculatingUSD"]["peggedUSD"]), float(week["totalCirculatingUSD"]["peggedUSD"]))
        except (TypeError, KeyError, IndexError, ValueError):
            return None, None

    def solana_load(self) -> dict:
        """Transactions per second and median priority fee from the public RPC."""
        out = {}
        for method, params in (("getRecentPerformanceSamples", [5]), ("getRecentPrioritizationFees", [])):
            self.pace["misc"].wait()
            try:
                resp = self.http.post("https://api.mainnet-beta.solana.com",
                                      json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
                res = resp.json().get("result") or []
            except (httpx.HTTPError, ValueError):
                continue
            if method == "getRecentPerformanceSamples" and res:
                out["tps"] = sum(r["numTransactions"] for r in res) / sum(r["samplePeriodSecs"] for r in res)
            if method == "getRecentPrioritizationFees" and res:
                fees = sorted(r.get("prioritizationFee", 0) for r in res)
                out["priority_fee_median"] = fees[len(fees) // 2]
        return out

    def biz_catalog(self) -> list[dict]:
        """4chan /biz/ threads mentioning contract addresses: [{"post", "ts", "mints", "text"}]."""
        data = self._json("misc", "https://a.4cdn.org/biz/catalog.json")
        out = []
        for page in data or []:
            for th in page.get("threads", []):
                posts = [th] + (th.get("last_replies") or [])
                for p in posts:
                    text = html.unescape(re.sub(r"<[^>]+>", " ", (p.get("sub") or "") + " " + (p.get("com") or "")))
                    mints = sorted(set(MINT_RE.findall(text)))
                    if mints:
                        out.append({"post": f"biz/{p.get('no')}", "ts": p.get("time"), "mints": mints})
        return out
