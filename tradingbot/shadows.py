"""Baseline ("shadow") strategies that answer: is Claude worth paying for?

Every baseline trades its own simulated book with the same filters, risk rules,
fill model and costs as the Claude book. Only the selection rule differs:

  filters_only  buy every token that clears the deterministic gates (no Claude).
                If Claude cannot beat this, the API spend is wasted.
  random        among the tokens Claude looked at (triage included), buy at random with Claude's
                own historical buy rate. If Claude cannot beat this, its picks
                carry no information beyond the filters.
"""

from __future__ import annotations

import random

from tradingbot.config import Config
from tradingbot.journal import Journal

KNOWN = ("filters_only", "random")
PRIOR_BUY_RATE = 0.25
PRIOR_WEIGHT = 20  # pseudo-observations behind the prior


def parse_shadows(spec: str) -> list[str]:
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in KNOWN]
    if unknown:
        raise ValueError(f"unknown shadow strategies {unknown}; choose from {KNOWN}")
    return names


def claude_buy_rate(journal: Journal, cfg: Config) -> float:
    """Share of candidates Claude chose to buy (triage skips count as passes), shrunk towards a prior."""
    row = journal.query(
        """SELECT COUNT(*) AS n,
                  COALESCE(SUM(action='buy' AND conviction>=?), 0) AS buys
           FROM decisions WHERE stage IN ('analysed', 'triaged_out')""",
        (cfg.min_conviction,),
    )[0]
    return (row["buys"] + PRIOR_BUY_RATE * PRIOR_WEIGHT) / (row["n"] + PRIOR_WEIGHT)


def picks(name: str, survivors: list, analysed: list, buy_rate: float, rng: random.Random) -> list:
    """Survivors (snap, factors, safety) a baseline wants to buy this scan, in priority order."""
    if name == "filters_only":
        return list(survivors)
    if name == "random":
        return [s for s, _action in analysed if rng.random() < buy_rate]
    raise ValueError(name)
