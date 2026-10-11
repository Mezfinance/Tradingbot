import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tradingbot import report
from tradingbot.analyst import Analyst
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tradingbot.tracker import Tracker
from tests.test_analyst_engine import BUY_JSON, FakeClient, FakeDex, resp


class PriceDex:
    def __init__(self, prices):
        self.prices = prices

    def pairs(self, tokens, boosted=frozenset()):
        return [SimpleNamespace(token_address=t, price_usd=self.prices[t]) for t in tokens if t in self.prices]


def test_tracker_horizons_extremes_and_gone_tokens():
    j = Journal(":memory:")
    dex = PriceDex({"A": 1.0, "B": 1.0})
    tr = Tracker(j, dex)
    j.execute("INSERT INTO decisions(id, ts, token, stage) VALUES(1, 0, 'A', 'analysed')")
    j.execute("INSERT INTO decisions(id, ts, token, stage) VALUES(2, 0, 'B', 'analysed')")
    tr.track(1, "A", 1.0)
    tr.track(2, "B", 1.0)
    t0 = j.query("SELECT start_ts FROM tracking WHERE decision_id=1")[0]["start_ts"]

    dex.prices = {"A": 3.0, "B": 0.5}
    tr.poll(t0 + 1800)
    dex.prices = {"A": 2.0}  # B vanished while A still quotes -> recorded as 0
    tr.poll(t0 + 3601)
    a, b = (j.query("SELECT * FROM tracking WHERE decision_id=?", (i,))[0] for i in (1, 2))
    assert a["p1h"] == 2.0 and a["max_price"] == 3.0 and a["p6h"] is None
    assert b["p1h"] == 0.0 and b["min_price"] == 0.0

    dex.prices = {"A": 4.0}
    tr.poll(t0 + 24 * 3600 + 1)
    a = j.query("SELECT * FROM tracking WHERE decision_id=1")[0]
    assert a["p6h"] == 4.0 and a["p24h"] == 4.0 and a["done"] == 1
    assert j.query("SELECT outcome_price FROM decisions WHERE id=1")[0]["outcome_price"] == 4.0
    assert len(j.query("SELECT * FROM prices WHERE token='A'")) == 3


def test_tracker_skips_sample_when_price_feeds_are_down():
    j = Journal(":memory:")
    tr = Tracker(j, PriceDex({}))
    j.execute("INSERT INTO decisions(id, ts, token, stage) VALUES(1, 0, 'A', 'analysed')")
    tr.track(1, "A", 1.0)
    assert tr.poll(time.time() + 7200) == 0
    assert j.query("SELECT p1h FROM tracking")[0]["p1h"] is None  # not mistaken for a rug


def test_restart_keeps_cooldowns_budget_and_logs_downtime(cfg, good_pair, safe, tmp_path):
    db = str(tmp_path / "bot.db")
    cfg.max_analyses_per_hour = 1
    j = Journal(db)
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)])), PaperExecutor(cfg), j)
    eng.tick()
    eng._cool("BADTOKEN")
    j.set_state("heartbeat", repr(time.time() - 3600))  # pretend we died an hour ago

    j2 = Journal(db)
    analyst2 = Analyst(cfg, FakeClient([]))
    eng2 = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe), analyst2,
                  PaperExecutor(cfg), j2)
    assert "BADTOKEN" in eng2._cooldown
    assert not analyst2.budget_ok()  # the one call this hour was remembered
    [row] = j2.query("SELECT * FROM downtime")
    assert row["open_positions"] == 1
    assert report.build(j2, cfg)["downtime"]["outages"] == 1


def test_unanalysed_survivors_are_tracked_once(cfg, good_pair, safe):
    cfg.max_analyses_per_hour = 0  # Claude never gets to them
    j = Journal(":memory:")
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, FakeClient([])), PaperExecutor(cfg), j)
    eng.tick()
    eng.tick()
    rows = j.query("SELECT stage FROM decisions")
    assert [r["stage"] for r in rows] == ["survivor"]
    assert len(j.query("SELECT * FROM tracking")) == 1
    sel = report.build(j, cfg)["claude_selection"]
    assert sel["not analysed"]["n"] == 1
