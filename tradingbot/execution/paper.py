"""Paper executor: fills at the reference price with pessimistic slippage and fees."""

from __future__ import annotations

from tradingbot.config import Config
from tradingbot.execution.base import Fill


class PaperExecutor:
    def __init__(self, cfg: Config):
        self.slip = cfg.slippage_bps / 10_000
        self.fee = cfg.fee_bps / 10_000

    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        if usd <= 0 or ref_price <= 0:
            return None
        fee = usd * self.fee
        price = ref_price * (1 + self.slip)
        return Fill(qty=(usd - fee) / price, price=price, usd=usd, fee_usd=fee, tx="paper")

    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None:
        if qty <= 0 or ref_price <= 0:
            return None
        price = ref_price * (1 - self.slip)
        gross = qty * price
        fee = gross * self.fee
        return Fill(qty=qty, price=price, usd=gross - fee, fee_usd=fee, tx="paper")
