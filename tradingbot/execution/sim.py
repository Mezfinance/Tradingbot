"""Simulated executor that fills against live Jupiter quotes.

For every trade it:
  1. quotes the exact size (real route, pool fees and price impact; no route = no trade),
  2. waits `sim_latency_s` to mimic the time a transaction takes to land,
  3. re-quotes; if the new output is below the first quote's slippage floor the
     transaction "reverts" and only the network fee is charged,
  4. otherwise fills between the fresh quote and the slippage floor according to
     `sim_fill_mode` (midpoint approximates sandwich/MEV leakage),
  5. charges the priority fee, plus token-account rent on the first buy of a token.
"""

from __future__ import annotations

import logging
import time

from tradingbot.config import Config
from tradingbot.data.jupiter import USDC_DECIMALS, USDC_MINT, Jupiter, Quote
from tradingbot.execution.base import Fill, failed

log = logging.getLogger(__name__)


class SimExecutor:
    def __init__(self, cfg: Config, jupiter: Jupiter, latency_s: float | None = None, sleep=time.sleep):
        self.cfg = cfg
        self.jup = jupiter
        self.latency_s = cfg.sim_latency_s if latency_s is None else latency_s
        self.sleep = sleep

    # --- costs ---------------------------------------------------------------
    def _sol_usd(self, sol: float) -> float:
        price = self.jup.sol_price()
        return sol * price if price else 0.0

    def tx_fee_usd(self) -> float:
        return self._sol_usd(self.cfg.priority_fee_sol)

    # --- fills ----------------------------------------------------------------
    def _execute(self, in_mint: str, out_mint: str, in_raw: int, cap_impact: bool) -> tuple[int | None, str]:
        """Returns (filled output raw amount, note). None output = reverted / no route.

        Exits pass cap_impact=False: a stop must go through even on a thin pool.
        """
        first = self.jup.quote(in_mint, out_mint, in_raw)
        if first is None:
            return None, "no route"
        if cap_impact and first.price_impact_pct > self.cfg.max_price_impact_pct:
            return None, f"price impact {first.price_impact_pct:.2f}%"
        if self.latency_s > 0:
            self.sleep(self.latency_s)
        landed = self.jup.quote(in_mint, out_mint, in_raw)
        if landed is None:
            return None, "route vanished"
        if landed.out_raw < first.min_out_raw:
            return None, "slippage exceeded (reverted)"
        return self._fill_amount(first, landed), "ok"

    def _fill_amount(self, first: Quote, landed: Quote) -> int:
        mode = self.cfg.sim_fill_mode
        if mode == "expected":
            return landed.out_raw
        if mode == "pessimistic":
            return first.min_out_raw
        return (landed.out_raw + first.min_out_raw) // 2

    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        decimals = self.jup.decimals(token)
        if decimals is None:
            return None
        fee = self.tx_fee_usd() + self._sol_usd(self.cfg.ata_rent_sol)
        swap_usd = usd - fee
        if swap_usd <= 0:
            return None
        out_raw, note = self._execute(USDC_MINT, token, int(swap_usd * 10**USDC_DECIMALS), cap_impact=True)
        if out_raw is None:
            # Pre-flight rejections cost nothing; a reverted transaction costs its fee.
            return failed(self.tx_fee_usd(), note) if "reverted" in note else None
        qty = out_raw / 10**decimals
        return Fill(qty=qty, price=swap_usd / qty, usd=usd, fee_usd=fee, tx="sim", note=note)

    def sell(self, token: str, qty: float, ref_price: float) -> Fill | None:
        decimals = self.jup.decimals(token)
        if decimals is None or qty <= 0:
            return None
        fee = self.tx_fee_usd()
        out_raw, note = self._execute(token, USDC_MINT, int(qty * 10**decimals), cap_impact=False)
        if out_raw is None:
            # A reverted exit burns its fee. With no route nothing is sent, but the
            # position is stuck, which is exactly what happens when liquidity is pulled.
            return failed(fee if "reverted" in note else 0.0, note)
        proceeds = out_raw / 10**USDC_DECIMALS
        return Fill(qty=qty, price=proceeds / qty, usd=proceeds - fee, fee_usd=fee, tx="sim", note=note)
