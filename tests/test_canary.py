from dataclasses import replace
from types import SimpleNamespace

import pytest

from tradingbot import report
from tradingbot.analyst import Analyst
from tradingbot.config import LIVE_CONFIRM_PHRASE, Config
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tests.test_analyst_engine import BUY_JSON, FakeClient, FakeDex, resp


def test_canary_needs_confirmation_and_paper_mode():
    with pytest.raises(ValueError, match="real money"):
        Config(canary_every_n=3).validate()
    with pytest.raises(ValueError, match="paper mode"):
        Config(canary_every_n=3, mode="live", live_confirm=LIVE_CONFIRM_PHRASE).validate()
    Config(canary_every_n=3, live_confirm=LIVE_CONFIRM_PHRASE).validate()


class MultiDex:
    """Serves several tokens; each tick exposes one new token for discovery."""

    def __init__(self, pairs):
        self.by_token = {p.token_address: p for p in pairs}
        self.visible = []

    def discover(self):
        return list(self.visible), set()

    def pairs(self, addrs, boosted=frozenset()):
        return [self.by_token[a] for a in addrs if a in self.by_token]


def test_every_nth_claude_buy_is_mirrored_by_a_fixed_size_canary(cfg, good_pair, safe):
    cfg.canary_every_n = 2
    cfg.canary_usd = 5.0
    cfg.canary_budget_usd = 100.0
    cfg.max_open_positions = 10
    pairs = [replace(good_pair, token_address=f"T{i}", symbol=f"T{i}") for i in range(4)]
    dex = MultiDex(pairs)
    real = PaperExecutor(cfg)
    real.slip = 0.05  # "real" fills are worse than the sim's 3%
    j = Journal(":memory:")
    eng = Engine(cfg, dex, SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)] * 4)), PaperExecutor(cfg), j,
                 canary_executor=real)
    for p in pairs:
        dex.visible = [p.token_address]
        eng.tick()
    assert len(j.open_positions("claude")) == 4
    canaries = j.open_positions("canary")
    assert [c.token_address for c in canaries] == ["T1", "T3"]
    assert all(c.cost_usd == pytest.approx(5.0) for c in canaries)
    assert eng.books["canary"].cash == pytest.approx(90.0)

    # Close everything at +0% and check calibration sees the extra real slippage.
    for p in pairs:
        dex.by_token[p.token_address] = replace(p, price_usd=p.price_usd * 0.5)
    eng.manage_positions()
    cal = report.build(j, cfg)["calibration"]
    assert cal["matched_trades"] == 2
    assert cal["median_real_minus_sim_pct"] < 0  # real fills worse than simulated
