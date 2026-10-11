import json
from types import SimpleNamespace

from tradingbot import report
from tradingbot.analyst import SYSTEM_PROMPT, VERDICT_SCHEMA, Analyst, parse_verdict
from tradingbot.cli import main
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tradingbot.learning import optimizer
from tests.test_analyst_engine import FakeClient, FakeDex, resp


def verdict(social, action="buy"):
    return json.dumps({"action": action, "conviction": 0.8, "thesis": "t", "red_flags": [], "catalysts": [],
                       "social_signal": "organic", "social_score": social, "mentions_found": 12,
                       "notable_accounts": ["@bigcaller"], "buzz_recency": "last_hour",
                       "social_evidence": "12 distinct posts", "sources": []})


def test_schema_and_prompt_ask_for_the_social_score():
    for field in ("social_score", "mentions_found", "notable_accounts", "buzz_recency", "social_evidence"):
        assert field in VERDICT_SCHEMA["required"]
    assert "social_score (0.0-1.0)" in SYSTEM_PROMPT


def test_social_score_is_clamped():
    v = parse_verdict([SimpleNamespace(type="text", text=verdict(1.7))])
    assert v.social_score == 1.0 and v.notable_accounts == ["@bigcaller"] and v.buzz_recency == "last_hour"


def test_social_gate_and_storage(cfg, good_pair, safe):
    cfg.min_social_score = 0.5
    j = Journal(":memory:")
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: safe),
                 Analyst(cfg, FakeClient([resp("end_turn", verdict(0.2))])), PaperExecutor(cfg), j)
    eng.tick()
    assert j.open_positions() == []  # Claude said buy, but the social score is below the gate
    assert j.query("SELECT social_score FROM decisions WHERE stage='analysed'")[0]["social_score"] == 0.2

    row = {"action": "buy", "conviction": 0.8, "factor_score": 0.7, "liquidity": 50_000, "fdv": 1e6,
           "snapshot": None, "social_score": 0.2}
    assert not optimizer.passes(row, cfg)
    assert optimizer.passes({**row, "social_score": 0.7}, cfg)
    assert optimizer.passes({**row, "social_score": None}, cfg)  # decisions from before the score existed


def test_report_compares_high_and_low_social(cfg):
    j = Journal(":memory:")
    for i, (s, p6) in enumerate([(0.8, 1.5), (0.7, 1.3), (0.1, 0.6)], start=1):
        j.execute("INSERT INTO decisions(id, ts, token, stage, social_score) VALUES(?,0,?,'analysed',?)",
                  (i, f"T{i}", s))
        j.execute("INSERT INTO tracking(decision_id, token, start_ts, start_price, p6h, max_price, min_price) "
                  "VALUES(?,?,0,1.0,?,?,0.5)", (i, f"T{i}", p6, p6))
    rep = report.build(j, cfg)
    assert rep["social"]["high"]["n"] == 2 and rep["social"]["high"]["median_6h_pct"] == 40.0
    assert rep["social"]["low"]["median_6h_pct"] == -40.0
    assert "SOCIAL SCORE vs OUTCOME" in report.render(rep)


def test_propose_queues_work_for_the_engineer(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "bot.db"))
    main(["propose", "Mention velocity factor", "--idea", "Count Reddit mentions per hour", "--priority", "2"])
    assert "queued proposal #1" in capsys.readouterr().out
    main(["proposals"])
    out = capsys.readouterr().out
    assert "#1 [open] p2 (user) Mention velocity factor" in out and "Count Reddit mentions" in out
    row = Journal(str(tmp_path / "bot.db")).query("SELECT * FROM proposals")[0]
    assert row["source"] == "user" and row["status"] == "open"
