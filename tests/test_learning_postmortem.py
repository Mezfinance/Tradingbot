import json
import sqlite3
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tradingbot.analyst import Analyst
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tradingbot.learning import dossier
from tradingbot.learning.learner import Learner
from tradingbot.llm import FAILED, OK, UNAVAILABLE, structured_call, tool_loop
from tests.test_analyst_engine import BUY_JSON, FakeClient, FakeDex, resp
from tests.test_api_errors import ScriptedClient, billing


def pm_json(category="thesis_wrong", tag="avoid_paid_shills", statement="Skip tokens whose buzz is mostly paid callers."):
    return json.dumps({
        "category": category, "decision_quality": "poor", "what_happened": "Bought into paid hype; it dumped.",
        "contributing_factors": [{"area": "social_and_narrative", "factor": "paid callers", "effect": "hurt",
                                  "evidence": "3 paid calls", "confidence": 0.7}],
        "lesson": {"tag": tag, "statement": statement, "applies_when": "social_signal is mixed", "confidence": 0.6},
        "supports_lessons": [], "contradicts_lessons": [],
        "parameter_hints": [{"parameter": "min_conviction", "direction": "increase", "reason": "weak thesis"}],
        "code_ideas": [], "keep_as_is": False,
    })


# --- journal migration -----------------------------------------------------------

def test_old_database_gains_new_columns(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY, ts REAL NOT NULL, token TEXT NOT NULL, symbol TEXT,"
                " stage TEXT NOT NULL, price REAL, liquidity REAL, fdv REAL, factor_score REAL, components TEXT,"
                " rejections TEXT, verdict TEXT, action TEXT, conviction REAL, outcome_price REAL, outcome_ts REAL)")
    con.commit()
    con.close()
    j = Journal(str(path))
    cols = {r["name"] for r in j.query("PRAGMA table_info(decisions)")}
    assert "snapshot" in cols


# --- llm helpers -----------------------------------------------------------------

def test_structured_call_resumes_and_parses(cfg):
    client = FakeClient([resp("pause_turn"), resp("end_turn", '{"x": 1}')])
    out = structured_call(client, cfg, system="s", user="u", schema={"type": "object"}, web_searches=2)
    assert out.outcome == OK and out.data == {"x": 1} and len(client.calls) == 2
    assert client.calls[0]["tools"][0]["max_uses"] == 2


def test_structured_call_outcomes(cfg):
    assert structured_call(ScriptedClient([billing()]), cfg, system="s", user="u", schema={}).outcome == UNAVAILABLE
    assert structured_call(FakeClient([resp("refusal")]), cfg, system="s", user="u", schema={}).outcome == FAILED
    assert structured_call(FakeClient([resp("end_turn", "nope")]), cfg, system="s", user="u", schema={}).outcome == FAILED


def tool_use(name, inp, id_="t1"):
    return SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                           content=[SimpleNamespace(type="tool_use", name=name, input=inp, id=id_)])


def test_tool_loop_runs_tools_and_reports_errors(cfg):
    seen = []
    client = FakeClient([tool_use("add", {"a": 2}), tool_use("boom", {}, "t2"), resp("end_turn", '{"done": true}')])
    out = tool_loop(client, cfg, system="s", user="u", tools=[], schema={"type": "object"},
                    handlers={"add": lambda i: seen.append(i) or i["a"] + 1})
    assert out.outcome == OK and out.data == {"done": True} and seen == [{"a": 2}]
    results = [m for m in out.transcript if m["role"] == "user" and isinstance(m["content"], list)]
    assert results[0]["content"][0]["content"] == "3"
    assert results[1]["content"][0]["is_error"]  # unknown tool reported back, not raised


# --- dossier and learner -------------------------------------------------------------

@pytest.fixture
def closed_trade(cfg, good_pair, safe):
    j = Journal(":memory:")
    dex = FakeDex(good_pair)
    eng = Engine(cfg, dex, SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, FakeClient([resp("end_turn", BUY_JSON)])), PaperExecutor(cfg), j)
    eng.tick()
    [pos] = j.open_positions()
    t0 = pos.opened_at
    j.executemany("INSERT INTO prices(token, ts, price) VALUES(?,?,?)",
                  [(good_pair.token_address, t0 + 60 * i, good_pair.price_usd * (1 - 0.05 * i)) for i in range(1, 7)])
    dex.pair = replace(good_pair, price_usd=good_pair.price_usd * 0.5)
    eng.manage_positions()
    return j, pos


def test_trade_dossier_has_the_evidence(closed_trade):
    j, pos = closed_trade
    d = dossier.trade_dossier(j, pos.id)
    assert d["result"]["exit_reason"] == "stop_loss" and d["result"]["return_pct"] < 0
    assert d["entry_decision"]["claude_verdict"]["action"] == "buy"
    assert d["entry_decision"]["market_snapshot"]["safety"]["lp_locked_pct"] == 100
    assert [f["side"] for f in d["fills"]] == ["buy", "sell"]
    assert "market_context" in d


def test_learner_waits_then_reviews_and_stores(cfg, closed_trade):
    j, pos = closed_trade
    client = FakeClient([resp("end_turn", pm_json())])
    learner = Learner(j, cfg, client)
    assert learner.step() == 0  # too soon: waits to see what happened after the exit
    j.execute("UPDATE positions SET closed_at = closed_at - ?", (cfg.postmortem_delay_h * 3600 + 1,))
    assert learner.step() == 1
    [row] = j.query("SELECT * FROM postmortems")
    assert row["category"] == "thesis_wrong" and row["position_id"] == pos.id
    assert learner.step() == 0  # never reviewed twice
    assert j.query("SELECT kind FROM costs WHERE kind='postmortem'")


def test_learner_respects_budget_and_pauses_on_billing(cfg, closed_trade):
    j, _ = closed_trade
    j.execute("UPDATE positions SET closed_at = closed_at - 99999")
    cfg.learning_budget_usd_per_day = 1.0
    j.log_cost("optimizer", 1.5)
    assert Learner(j, cfg, FakeClient([])).step() == 0  # over budget: no calls

    j.execute("DELETE FROM costs")
    learner = Learner(j, cfg, ScriptedClient([billing()]))
    assert learner.step() == 0
    assert learner.paused_until > time.time() and not learner.can_spend()


def test_missed_winners_are_found(cfg):
    j = Journal(":memory:")
    j.execute("INSERT INTO decisions(id, ts, token, symbol, stage, action) VALUES(1, 0, 'MOON', 'MOON', 'analysed', 'pass')")
    j.execute("INSERT INTO decisions(id, ts, token, symbol, stage, action) VALUES(2, 0, 'MEH', 'MEH', 'analysed', 'pass')")
    j.execute("INSERT INTO tracking(decision_id, token, start_ts, start_price, max_price, min_price, done) "
              "VALUES(1, 'MOON', 0, 1.0, 3.5, 0.9, 1), (2, 'MEH', 0, 1.0, 1.3, 0.8, 1)")
    learner = Learner(j, cfg, FakeClient([resp("end_turn", pm_json("missed_opportunity"))]))
    assert [r["token"] for r in learner.pending_missed()] == ["MOON"]
    assert learner.step() == 1
    assert j.query("SELECT kind, return_pct FROM postmortems")[0]["return_pct"] == 250.0
