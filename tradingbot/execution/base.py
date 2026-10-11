from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Fill:
    qty: float  # token units
    price: float  # effective USD per token, after slippage
    usd: float  # USD spent (buy) or received (sell), after fees
    fee_usd: float
    tx: str | None = None


class Executor(Protocol):
    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None: ...
    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None: ...
