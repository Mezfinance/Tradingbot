import json
import time
from types import SimpleNamespace

import pytest

from tradingbot.config import Config
from tradingbot.journal import Journal
from tradingbot.learning import optimizer as opt
from tradingbot.learning import params, versions
from tests.test_analyst_engine import FakeClient, resp
from tests.test_learning_postmortem import tool_use

WIN = [1.6, 2.6, 5.5]           # runs through every take-profit
LOSE = [0.85, 0.40, 0.30]       # dips 15%, then collapses


def seed(j, n_win, n_lose, start_ts, first_id=1, liquidity=50_000):
    """Interleaved winners and losers, one minute apart, each with a recorded price path."""
    kinds = ["w"] * n_win + ["l"] * n_lose
    kinds = [k for pair in zip(kinds[::2], kinds[1::2]) for k in pair] + kinds[len(kinds) // 2 * 2:]
    for i, k in enumerate(kinds):
        did = first_id + i
        ts = start_ts + i * 60
        token = f"T{did}"
        snap = json.dumps({"liquidity_usd": liquidity, "fdv_usd": 1e6, "volume": {"h1": 90_000},
                           "created_at_ms": int((ts - 3 * 3600) * 1000), "fetched_at": ts})
        j.execute("""INSERT INTO decisions(id, ts, token, stage, action, conviction, factor_score, liquidity, snapshot)
                     VALUES(?,?,?,'analysed','buy',0.8,0.7,?,?)""", (did, ts, token, liquidity, snap))
        j.execute("INSERT INTO tracking(decision_id, token, start_ts, start_price, max_price, min_price) "
                  "VALUES(?,?,?,1.0,1.0,1.0)", (did, token, ts))
        path = WIN if k == "w" else LOSE
        j.executemany("INSERT INTO prices(token, ts, price) VALUES(?,?,?)",
                      [(token, ts + 60 * (s + 1), p) for s, p in enumerate(path)])


@pytest.fixture
def setup():
    cfg = Config(db_path=":memory:")
    j = Journal(":memory:")
    versions.ensure(j)
    return cfg, j


def test_passes_rechecks_filters_from_snapshot(setup):
    cfg, j = setup
    seed(j, 1, 1, 1000, liquidity=20_000)
    rows = opt.history(j)
    assert all(opt.passes(r, cfg) for r in rows)
    strict = opt.with_changes(cfg, {"min_liquidity_usd": 30_000})
    assert not any(opt.passes(r, strict) for r in rows)
    assert not any(opt.passes(r, opt.with_changes(cfg, {"min_conviction": 0.9})) for r in rows)


def test_tighter_stop_is_accepted_on_this_history(setup):
    cfg, j = setup
    seed(j, 15, 15, 1000)
    ev = opt.evaluate(j, cfg, {"stop_loss_pct": 12}, cfg.optimizer_holdout_frac)
    ok, why = opt.acceptance(ev, cfg)
    assert ok, why
    assert ev["holdout"]["proposed"]["total_usd"] > ev["holdout"]["current"]["total_usd"]


def test_acceptance_guards(setup):
    cfg, j = setup
    seed(j, 4, 4, 1000)
    ok, why = opt.acceptance(opt.evaluate(j, cfg, {"stop_loss_pct": 12}, 0.3), cfg)
    assert not ok and "candidates" in why  # too little history
    j2 = Journal(":memory:")
    seed(j2, 15, 15, 1000)
    ok, why = opt.acceptance(opt.evaluate(j2, cfg, {"min_conviction": 0.95}, 0.3), cfg)
    assert not ok  # "improves" only by not trading


def test_optimizer_round_applies_validated_change_and_stores_proposals(setup):
    cfg, j = setup
    seed(j, 15, 15, 1000)
    final = {"changes": [{"parameter": "stop_loss_pct", "value": 12},
                         {"parameter": "risk_per_trade_pct", "value": 5}],  # out of bounds: dropped
             "rationale": "Losers dip 15% before collapsing; a 12% stop exits earlier.",
             "code_proposals": [{"title": "Track holder concentration", "problem": "rugs", "evidence": "4 rugs",
                                 "idea": "use Helius", "priority": 2}]}
    client = FakeClient([tool_use("test_settings", {"changes": {"stop_loss_pct": 12}}),
                         resp("end_turn", json.dumps(final))])
    o = opt.Optimizer(j, cfg)
    res = o.run(client)
    assert res.outcome == "ok"
    tool_result = res.transcript[2]["content"][0]["content"]
    assert json.loads(tool_result)["would_be_accepted"] is True
    assert cfg.stop_loss_pct == 12 and cfg.risk_per_trade_pct == 2.0
    [v2, v1] = versions.history(j)
    assert v2["status"] == "active" and json.loads(v2["changes"]) == {"stop_loss_pct": 12.0}
    assert v1["reason"] == "baseline" and v1["status"] == "superseded"
    assert j.query("SELECT title FROM proposals")[0]["title"] == "Track holder concentration"

    restarted = Config(db_path=":memory:")
    versions.apply_active(j, restarted)
    assert restarted.stop_loss_pct == 12


def test_rejected_proposal_changes_nothing(setup):
    cfg, j = setup
    seed(j, 15, 15, 1000)
    final = {"changes": [{"parameter": "stop_loss_pct", "value": 40}], "rationale": "wider", "code_proposals": []}
    opt.Optimizer(j, cfg).run(FakeClient([resp("end_turn", json.dumps(final))]))
    assert cfg.stop_loss_pct == 25
    assert j.query("SELECT kind FROM events")[0]["kind"] == "optimizer_rejected"


def test_live_review_rolls_back_when_parent_would_have_done_better(setup):
    cfg, j = setup
    seed(j, 15, 15, 1000)
    o = opt.Optimizer(j, cfg)
    o._decide({"stop_loss_pct": 12}, "test")
    assert cfg.stop_loss_pct == 12
    v = versions.active(j)
    # After going live, the market changes: tokens dip 15% and then run. The tight stop now hurts.
    later = v["ts"] + 10
    for i in range(12):
        did, token, ts = 1000 + i, f"N{i}", later + i * 60
        j.execute("INSERT INTO decisions(id, ts, token, stage, action, conviction, factor_score, liquidity) "
                  "VALUES(?,?,?,'analysed','buy',0.8,0.7,50000)", (did, ts, token))
        j.execute("INSERT INTO tracking(decision_id, token, start_ts, start_price, max_price, min_price) "
                  "VALUES(?,?,?,1.0,1.0,1.0)", (did, token, ts))
        j.executemany("INSERT INTO prices(token, ts, price) VALUES(?,?,?)",
                      [(token, ts + 60 * (s + 1), p) for s, p in enumerate([0.85, 1.6, 2.6, 5.5])])
        j.execute("INSERT INTO positions(token, strategy, opened_at, closed_at, cost_usd, realized_usd) "
                  "VALUES(?, 'claude', ?, ?, 50, 40)", (token, ts, ts + 600))
    why = o.review_active()
    assert why and "would have made" in why
    assert cfg.stop_loss_pct == 25
    assert versions.active(j)["reason"] == "baseline"
