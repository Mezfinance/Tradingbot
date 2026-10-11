import pytest

from tradingbot import replay
from tradingbot.config import Config
from tradingbot.journal import Journal

DAY = 86400.0


def seed(j, decision_id, token, start_ts, path, stage="analysed", action="buy", conviction=0.9):
    j.execute("INSERT INTO decisions(id, ts, token, stage, action, conviction, factor_score) VALUES(?,?,?,?,?,?,0.7)",
              (decision_id, start_ts, token, stage, action, conviction))
    j.execute("INSERT INTO tracking(decision_id, token, start_ts, start_price, max_price, min_price) VALUES(?,?,?,?,?,?)",
              (decision_id, token, start_ts, 1.0, 1.0, 1.0))
    j.executemany("INSERT INTO prices(token, ts, price) VALUES(?,?,?)",
                  [(token, start_ts + 60 * (i + 1), p) for i, p in enumerate(path)])


@pytest.fixture
def journal():
    j = Journal(":memory:")
    seed(j, 1, "WIN", 1 * DAY, [1.2, 1.6, 2.6, 5.5])       # runs through every target
    seed(j, 2, "LOSE", 1 * DAY, [0.9, 0.7])                 # stopped out
    seed(j, 3, "DIP", 3 * DAY, [0.78, 1.6, 2.6, 5.5])       # survives a -22% dip, then runs
    seed(j, 4, "PASS", 1 * DAY, [3.0], action="pass")
    return j


def no_costs():
    return replay.CostModel(0, 0, 0, 0, "test")


def test_groups(journal):
    cfg = Config()
    assert [c["token"] for c in replay.candidates(journal, "bought", cfg)] == ["WIN", "LOSE", "DIP"]
    assert [c["token"] for c in replay.candidates(journal, "passed", cfg)] == ["PASS"]


def test_simulated_exits_match_risk_rules(journal):
    trades = {t.decision_id: t for t in replay.run(journal, Config(), "bought", 100, no_costs())}
    # 40% at +50, 30% at +150, 30% at +400 (sold at the path price that crossed each target)
    assert trades[1].pnl_usd == pytest.approx(40 * 0.6 + 30 * 1.6 + 30 * 4.5)
    assert trades[2].exit_reason == "stop_loss" and trades[2].pnl_usd == pytest.approx(-30)
    assert trades[3].exit_reason == "take_profit_3"


def test_tighter_stop_changes_outcome(journal):
    tight = Config().with_overrides({"stop_loss_pct": "20"})
    trades = {t.decision_id: t for t in replay.run(journal, tight, "bought", 100, no_costs())}
    assert trades[3].exit_reason == "stop_loss"  # the -22% dip now stops it out


def test_report_split_and_overrides(journal):
    out = replay.report(journal, Config(), {"stop_loss_pct": "20"}, "bought", 100, "1970-01-03")
    assert "current settings" in out and "stop_loss_pct=20" in out
    assert "before 1970-01-03 (tune)" in out and "after 1970-01-03 (check)" in out
    assert "defaults" in out  # no fills recorded yet


def test_cost_model_uses_observed_fills():
    j = Journal(":memory:")
    for i in range(5):
        j.execute("INSERT INTO positions(id, token, strategy) VALUES(?, 'T', 'claude')", (i,))
        j.execute("INSERT INTO fills(position_id, side, price, trigger_price, fee_usd) VALUES(?, 'buy', 1.02, 1.0, 0.3)", (i,))
        j.execute("INSERT INTO fills(position_id, side, price, trigger_price, fee_usd) VALUES(?, 'sell', 0.97, 1.0, 0.1)", (i,))
    c = replay.cost_model(j)
    assert c.entry_slip_pct == pytest.approx(2.0) and c.exit_slip_pct == pytest.approx(3.0)
    assert "claude fills" in c.source
