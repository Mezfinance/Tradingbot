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
from tradingbot.data.jupiter import RATE_LIMITED, USDC_DECIMALS, USDC_MINT, Jupiter, Quote
from tradingbot.execution.base import Fill, failed

UNKNOWN_IMPACT_PCT = 99.9  # Jupiter's 'can't value this token' answer (priceImpactPct 1)
PROBE_SHARE = 100  # own-impact check: the same route quoted for 1% of the size

log = logging.getLogger(__name__)


class SimExecutor:
    def __init__(self, cfg: Config, jupiter: Jupiter, latency_s: float | None = None, sleep=time.sleep):
        self.cfg = cfg
        self.jup = jupiter
        self.latency_s = cfg.sim_latency_s if latency_s is None else latency_s
        self.sleep = sleep
        self.sell_slippage_bps: int | None = None  # None = Jupiter's default (cfg.slippage_bps); launch books 1000

    # --- costs ---------------------------------------------------------------
    def _sol_usd(self, sol: float) -> float:
        price = self.jup.sol_price()
        return sol * price if price else 0.0

    def tx_fee_usd(self) -> float:
        return self._sol_usd(self.cfg.priority_fee_sol)

    # --- fills ----------------------------------------------------------------
    def _execute(self, in_mint: str, out_mint: str, in_raw: int, cap_impact: bool,
                 estimate_impact=None, slippage_bps: int | None = None) -> tuple[int | None, str]:
        """Returns (filled output raw amount, note). None output = reverted / no route.

        Exits pass cap_impact=False: a stop must go through even on a thin pool. Jupiter reports 100% impact
        when it can't value a token yet (seconds-old pools; measured 11-02: reported 100%, real 1.5-1.8%);
        then `estimate_impact(quote)` gives the impact from the quote's price vs the trigger price.
        """
        first = self._quote(in_mint, out_mint, in_raw, slippage_bps)
        if first is None:
            return None, self._why_none("no route")
        impact = first.price_impact_pct
        if cap_impact and impact > self.cfg.max_price_impact_pct:
            impact = self._own_impact(in_mint, out_mint, in_raw, first, estimate_impact)
        if cap_impact and impact > self.cfg.max_price_impact_pct:
            return None, f"price impact {impact:.2f}%"
        if self.latency_s > 0:
            self.sleep(self.latency_s)
        landed = self._quote(in_mint, out_mint, in_raw, slippage_bps)
        if landed is None:
            return None, self._why_none("route vanished")
        if landed.out_raw < first.min_out_raw:
            return None, "slippage exceeded (reverted)"
        return self._fill_amount(first, landed, wide=slippage_bps is not None), "ok"

    def _quote(self, in_mint: str, out_mint: str, in_raw: int, slippage_bps: int | None):
        if slippage_bps is None:
            return self.jup.quote(in_mint, out_mint, in_raw)
        return self.jup.quote(in_mint, out_mint, in_raw, slippage_bps=slippage_bps)

    def _own_impact(self, in_mint: str, out_mint: str, in_raw: int, first: Quote, estimate_impact) -> float:
        return own_impact(self.jup, in_mint, out_mint, in_raw, first, estimate_impact)

    def _why_none(self, default: str) -> str:
        """A refused (429/5xx) quote says nothing about the route: report it as rate-limited."""
        why = getattr(self.jup, "last_failure", lambda: default)()
        return why if why == RATE_LIMITED else default

    def _fill_amount(self, first: Quote, landed: Quote, wide: bool = False) -> int:
        """`midpoint` leaks half the room down to the slippage floor (sandwich/MEV). With a wider tolerance that only
        stops reverts (launch-book sells), the leak is still measured against the default `slippage_bps` from the
        landed quote, so widening the tolerance never adds cost of its own (11-06: at 10% it took ~5% a sale)."""
        floor = first.min_out_raw
        if wide:
            floor = max(floor, int(landed.out_raw * (1 - self.cfg.slippage_bps / 10_000)))
        mode = self.cfg.sim_fill_mode
        if mode == "expected":
            return landed.out_raw
        if mode == "pessimistic":
            return min(floor, landed.out_raw)
        return (landed.out_raw + min(floor, landed.out_raw)) // 2

    def buy(self, token: str, usd: float, ref_price: float) -> Fill | None:
        """None = declined before sending (reason in `last_note`); a Fill with ok=False = reverted."""
        decimals = self.jup.decimals(token)
        if decimals is None:
            self.last_note = "token decimals unknown"
            return None
        fee = self.tx_fee_usd() + self._sol_usd(self.cfg.ata_rent_sol)
        swap_usd = usd - fee
        if swap_usd <= 0:
            self.last_note = "size below fees"
            return None
        def estimate(q):  # the quote's price per token vs the trigger price, in percent
            if not ref_price or ref_price <= 0 or not q.out_raw:
                return None
            return max(0.0, (swap_usd / (q.out_raw / 10**decimals) / ref_price - 1) * 100)
        out_raw, note = self._execute(USDC_MINT, token, int(swap_usd * 10**USDC_DECIMALS), cap_impact=True,
                                      estimate_impact=estimate)
        self.last_note = note
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
        out_raw, note = self._execute(token, USDC_MINT, int(qty * 10**decimals), cap_impact=False,
                                      slippage_bps=self.sell_slippage_bps)
        if out_raw is None:
            # A reverted exit burns its fee. With no route nothing is sent, but the
            # position is stuck, which is exactly what happens when liquidity is pulled.
            return failed(fee if "reverted" in note else 0.0, note)
        proceeds = out_raw / 10**USDC_DECIMALS
        return Fill(qty=qty, price=proceeds / qty, usd=proceeds - fee, fee_usd=fee, tx="sim", note=note)


def own_impact(jup: Jupiter, in_mint: str, out_mint: str, in_raw: int, first: Quote, estimate_impact=None) -> float:
    """Jupiter's priceImpactPct is unusable on seconds-old tokens (it can't value them: 100% on 11-02, ~94% on
    the 11-04 low-decile buys that a check showed were the books' winners). Measure our own impact instead: the
    same route quoted for 1/PROBE_SHARE of the size, and the price gap between the two. Falls back to the
    trigger-price estimate (which also counts any move since the trigger) only for Jupiter's 100% answer."""
    reported = first.price_impact_pct
    probe_raw = max(1, in_raw // PROBE_SHARE)
    probe = jup.quote(in_mint, out_mint, probe_raw) if in_raw >= PROBE_SHARE else None
    if probe is not None and probe.out_raw and first.out_raw:
        return max(0.0, (1 - (first.out_raw / in_raw) / (probe.out_raw / probe_raw)) * 100)
    if reported >= UNKNOWN_IMPACT_PCT and estimate_impact is not None:
        est = estimate_impact(first)
        return est if est is not None else reported
    return reported
