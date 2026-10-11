"""Plain data types shared across the bot."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class PairSnapshot:
    """One DEX pair at a point in time, normalised from DexScreener."""

    chain: str
    token_address: str
    pair_address: str
    symbol: str
    name: str
    dex: str
    price_usd: float
    liquidity_usd: float | None  # None on pump.fun bonding-curve pairs
    fdv_usd: float | None
    volume: dict[str, float]  # keys m5, h1, h6, h24
    price_change: dict[str, float]
    buys: dict[str, int]
    sells: dict[str, int]
    created_at_ms: int | None
    socials: list[str] = field(default_factory=list)
    boosted: bool = False
    fetched_at: float = field(default_factory=time.time)
    sources: list[str] = field(default_factory=list)  # discovery feeds that surfaced it

    @property
    def age_minutes(self) -> float | None:
        if not self.created_at_ms:
            return None
        return (self.fetched_at * 1000 - self.created_at_ms) / 60_000


@dataclass
class SafetyReport:
    score: int | None  # rugcheck raw score; lower is safer
    lp_locked_pct: float | None
    risks: list[str]
    danger: bool  # any risk flagged "danger" by rugcheck


@dataclass
class FactorResult:
    passed: bool
    score: float  # 0..1 composite
    components: dict[str, float]
    rejections: list[str]
    profile: str = "fresh"  # "fresh" or "established" (see factors.py)


class Verdict(BaseModel):
    """Claude's structured decision. The schema is enforced by the API."""

    action: Literal["buy", "pass"]
    conviction: float = Field(description="0.0-1.0 probability-weighted confidence")
    thesis: str = Field(description="Why this could run, in 1-3 sentences")
    red_flags: list[str] = Field(description="Concrete risks found during research")
    catalysts: list[str] = Field(description="Concrete, dated catalysts found, if any")
    social_signal: Literal["none", "organic", "paid_or_botted", "mixed"]
    social_score: float = Field(default=0.0, description="0.0-1.0 strength of organic social demand")
    mentions_found: int = Field(default=0, description="Distinct relevant posts/threads found")
    notable_accounts: list[str] = Field(default_factory=list, description="Influential accounts involved")
    buzz_recency: Literal["last_hour", "today", "older", "none"] = "none"
    social_evidence: str = Field(default="", description="What the social score is based on")
    sources: list[str] = Field(description="URLs relied on")


@dataclass
class Position:
    id: int | None
    token_address: str
    symbol: str
    entry_price: float
    qty: float  # token units still held
    initial_qty: float
    cost_usd: float
    opened_at: float
    stop_price: float
    peak_price: float
    tp_hits: int = 0
    realized_usd: float = 0.0
    closed_at: float | None = None
    exit_reason: str | None = None
    strategy: str = "claude"  # which book owns it (claude, or a shadow baseline)
