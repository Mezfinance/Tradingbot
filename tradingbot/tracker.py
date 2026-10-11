"""Price-path tracker for every candidate token, bought or not.

Each tracked decision gets a price sample roughly every scan cycle for 24 hours.
That gives:
  * 1h / 6h / 24h forward returns and the best/worst point reached, for an unbiased
    buy-vs-pass comparison and to judge whether stops and targets fit the market;
  * recorded price paths the replay tool (replay.py) re-runs risk rules against.

A token that both price sources stop quoting while others still quote is recorded
at 0: on these markets that almost always means liquidity was pulled.
"""

from __future__ import annotations

import logging
import time

from tradingbot.data.dexscreener import DexScreener
from tradingbot.data.jupiter import Jupiter
from tradingbot.journal import Journal

log = logging.getLogger(__name__)

TRACK_FOR_S = 24 * 3600
HORIZONS = (("p1h", 3600), ("p6h", 6 * 3600), ("p24h", 24 * 3600))


class Tracker:
    def __init__(self, journal: Journal, dex: DexScreener, jupiter: Jupiter | None = None):
        self.journal, self.dex, self.jupiter = journal, dex, jupiter

    def track(self, decision_id: int, token: str, price: float) -> None:
        self.journal.execute(
            """INSERT OR IGNORE INTO tracking(decision_id, token, start_ts, start_price, max_price, min_price)
               VALUES(?,?,?,?,?,?)""",
            (decision_id, token, time.time(), price, price, price),
        )

    def _prices(self, tokens: set[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        if self.jupiter is not None:
            prices = {m: p.usd for m, p in self.jupiter.prices(sorted(tokens)).items()}
        missing = tokens - prices.keys()
        if missing:
            prices.update({s.token_address: s.price_usd for s in self.dex.pairs(missing)})
        return prices

    def poll(self, now: float | None = None) -> int:
        """Sample prices for every active path. Returns the number of tokens sampled."""
        now = now or time.time()
        active = self.journal.query("SELECT * FROM tracking WHERE done=0")
        if not active:
            return 0
        tokens = {r["token"] for r in active}
        prices = self._prices(tokens)
        if not prices:
            log.warning("tracker: no prices returned; skipping this sample")
            return 0
        self.journal.executemany(
            "INSERT INTO prices(token, ts, price) VALUES(?,?,?)",
            [(t, now, p) for t, p in prices.items()],
        )
        for r in active:
            price = prices.get(r["token"], 0.0)  # gone while others quote = liquidity pulled
            age = now - r["start_ts"]
            updates = {
                "max_price": max(r["max_price"], price),
                "min_price": min(r["min_price"], price),
            }
            for col, horizon in HORIZONS:
                if r[col] is None and age >= horizon:
                    updates[col] = price
            if age >= TRACK_FOR_S:
                updates["done"] = 1
            sets = ", ".join(f"{k}=?" for k in updates)
            self.journal.execute(
                f"UPDATE tracking SET {sets} WHERE decision_id=?", (*updates.values(), r["decision_id"])
            )
            if "p6h" in updates:  # keep the legacy 6h column in decisions in sync
                self.journal.execute(
                    "UPDATE decisions SET outcome_price=?, outcome_ts=? WHERE id=?",
                    (updates["p6h"], now, r["decision_id"]),
                )
        return len(prices)
