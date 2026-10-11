from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Fill:
    """Result of an attempted swap.

    `usd` is the net cash movement: spent on a buy (swap + fees), received on a
    sell (proceeds - fees). A failed transaction has ok=False, qty=0, and still
    costs its network fee, as on-chain.
    """

    qty: float  # token units
    price: float  # effective USD per token, after slippage
    usd: float
    fee_usd: float
    tx: str | None = None
    ok: bool = True
    note: str = ""


class Executor(Protocol):
    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None: ...
    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None: ...


def failed(fee_usd: float, note: str) -> Fill:
    return Fill(qty=0.0, price=0.0, usd=fee_usd, fee_usd=fee_usd, tx=None, ok=False, note=note)
