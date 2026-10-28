import time

import pytest

from tradingbot.config import Config
from tradingbot.data import jupiter
from tradingbot.models import PairSnapshot, SafetyReport


@pytest.fixture(autouse=True)
def _no_jupiter_pace(monkeypatch):
    monkeypatch.setattr(jupiter.GATE, "interval_s", 0.0)


@pytest.fixture
def cfg():
    return Config(db_path=":memory:")


@pytest.fixture
def good_pair():
    now = time.time()
    return PairSnapshot(
        chain="solana", token_address="MINT1", pair_address="PAIR1", symbol="GOOD", name="Good",
        dex="raydium", price_usd=0.001, liquidity_usd=60_000, fdv_usd=800_000,
        volume={"m5": 12_000, "h1": 90_000, "h6": 200_000, "h24": 300_000},
        price_change={"m5": 4, "h1": 35, "h6": 60, "h24": 60},
        buys={"h1": 900}, sells={"h1": 450},
        created_at_ms=int((now - 3 * 3600) * 1000), socials=["https://x.com/good"], fetched_at=now,
    )


@pytest.fixture
def safe():
    return SafetyReport(score=300, lp_locked_pct=100, risks=[], danger=False)
