import json
import random
from types import SimpleNamespace

import anthropic
import httpx

from tradingbot import report
from tradingbot.analyst import Analyst
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tradingbot.triage import Triage
from tests.test_analyst_engine import BUY_JSON, FakeDex, resp


def tri(decision, reason="copycat name, no socials"):
    return resp("end_turn", json.dumps({"decision": decision, "reason": reason, "red_flags": []}))


class RoutingClient:
    """Answers triage (Haiku) and full-analysis (Opus) calls from separate scripts."""

    def __init__(self, triage, full):
        self.scripts = {"triage": list(triage), "full": list(full)}
        self.calls = {"triage": [], "full": []}
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        kind = "triage" if kw["model"].startswith("claude-haiku") else "full"
        self.calls[kind].append(kw)
        item = self.scripts[kind].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(cfg, pair, safe, triage, full, audit_pct=0.0):
    cfg.triage_audit_pct = audit_pct
    client = RoutingClient(triage, full)
    j = Journal(":memory:")
    eng = Engine(cfg, FakeDex(pair), SimpleNamespace(report=lambda m: safe), Analyst(cfg, client),
                 PaperExecutor(cfg), j, triage=Triage(cfg, client, random.Random(0)))
    return eng, j, client


def test_skip_never_reaches_the_full_analysis_but_is_tracked(cfg, good_pair, safe):
    eng, j, client = make(cfg, good_pair, safe, [tri("skip")], [])
    eng.tick()
    assert client.calls["full"] == []
    [d] = j.query("SELECT * FROM decisions WHERE stage='triaged_out'")
    assert json.loads(d["triage"])["decision"] == "skip"
    assert j.query("SELECT COUNT(*) AS n FROM tracking")[0]["n"] == 1
    assert j.query("SELECT kind FROM costs")[0]["kind"] == "triage"  # counted in the claude book's API bill
    eng.tick()  # not triaged again within the re-analysis window
    assert len(client.calls["triage"]) == 1


def test_analyse_goes_on_and_buys(cfg, good_pair, safe):
    eng, j, client = make(cfg, good_pair, safe, [tri("analyse", "distinct meme")], [resp("end_turn", BUY_JSON)])
    eng.tick()
    assert len(client.calls["full"]) == 1 and len(j.open_positions()) == 1
    d = j.query("SELECT * FROM decisions WHERE stage='analysed'")[0]
    assert json.loads(d["triage"])["decision"] == "analyse"


def test_audited_skip_still_gets_the_full_analysis(cfg, good_pair, safe):
    eng, j, client = make(cfg, good_pair, safe, [tri("skip")], [resp("end_turn", BUY_JSON)], audit_pct=100)
    eng.tick()
    assert len(client.calls["full"]) == 1
    t = json.loads(j.query("SELECT triage FROM decisions WHERE stage='analysed'")[0]["triage"])
    assert t["decision"] == "skip" and t["audit"] is True
    rep = report.build(j, cfg)
    assert rep["triage"]["audited_skips"] == 1 and rep["triage"]["audited_skips_opus_would_buy"] == 1


def test_broken_triage_fails_open(cfg, good_pair, safe):
    eng, j, client = make(cfg, good_pair, safe, [resp("end_turn", "not json")], [resp("end_turn", BUY_JSON)])
    eng.tick()
    assert len(client.calls["full"]) == 1


def test_api_down_leaves_the_token_queued(cfg, good_pair, safe):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)
    eng, j, client = make(cfg, good_pair, safe, [err], [])
    eng.tick()
    assert client.calls["full"] == []
    assert [r["stage"] for r in j.query("SELECT stage FROM decisions")] == ["survivor"]  # stays queued


def test_triage_request_uses_the_cheap_model_without_fallbacks(cfg, good_pair, safe):
    eng, j, client = make(cfg, good_pair, safe, [tri("skip")], [])
    eng.tick()
    [kw] = client.calls["triage"]
    assert kw["model"] == cfg.triage_model == "claude-haiku-5-5"
    assert "fallbacks" not in kw and "tools" not in kw
    assert "untrusted" in kw["system"].lower()


def test_report_shows_triaged_out_outcomes(cfg, good_pair, safe):
    eng, j, _ = make(cfg, good_pair, safe, [tri("skip")], [])
    eng.tick()
    rep = report.build(j, cfg)
    assert rep["claude_selection"]["triaged out"]["n"] == 1
    assert "TRIAGE: 1 checked" in report.render(rep)
