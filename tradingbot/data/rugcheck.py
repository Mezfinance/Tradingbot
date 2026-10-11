"""RugCheck token safety summary (Solana). No key required for the summary endpoint."""

from __future__ import annotations

import logging

import httpx

from tradingbot.models import SafetyReport

log = logging.getLogger(__name__)

BASE = "https://api.rugcheck.xyz/v1"


class RugCheck:
    def __init__(self, client: httpx.Client | None = None):
        self.http = client or httpx.Client(base_url=BASE, timeout=15)

    def report(self, mint: str) -> SafetyReport | None:
        """None means the check could not be completed, which the caller must treat as unsafe."""
        try:
            resp = self.http.get(f"/tokens/{mint}/report/summary")
            resp.raise_for_status()
            return parse_summary(resp.json())
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("rugcheck failed for %s: %s", mint, exc)
            return None


def parse_summary(data: dict) -> SafetyReport:
    risks = data.get("risks") or []
    return SafetyReport(
        score=data.get("score"),
        lp_locked_pct=data.get("lpLockedPct"),
        risks=[f"{r.get('level', '?')}: {r.get('name', '?')}" for r in risks],
        danger=any(r.get("level") == "danger" for r in risks),
    )
