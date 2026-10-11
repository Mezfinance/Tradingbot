"""Fixes from the first live logs: long Claude calls stream, the engineer survives crashes,
and Jupiter retries server errors."""

import json
from types import SimpleNamespace

import httpx
import pytest

from tradingbot import llm
from tradingbot.config import Config
from tradingbot.data import jupiter as jupiter_mod
from tradingbot.data.jupiter import Jupiter
from tradingbot.journal import Journal
from tradingbot.learning import engineer, versions
from tests.test_analyst_engine import resp


class StreamingClient:
    """Like the real SDK: create() refuses large max_tokens, stream() works."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.streamed = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create, stream=self._stream))

    def _create(self, **kwargs):
        if kwargs["max_tokens"] > llm.STREAM_ABOVE_TOKENS:
            raise ValueError("Streaming is required for operations that may take longer than 10 minutes.")
        return self.responses.pop(0)

    def _stream(self, **kwargs):
        self.streamed += 1
        final = self.responses.pop(0)

        class Ctx:
            def __enter__(self):
                return SimpleNamespace(get_final_message=lambda: final)

            def __exit__(self, *exc):
                return False
        return Ctx()


def test_tool_loop_streams_large_requests(cfg):
    client = StreamingClient([resp("end_turn", json.dumps({"ok": True}))])
    res = llm.tool_loop(client, cfg, system="s", user="u", tools=[], handlers={})
    assert res.outcome == llm.OK and client.streamed == 1


def test_small_requests_do_not_stream(cfg):
    client = StreamingClient([resp("end_turn", json.dumps({"ok": True}))])
    res = llm.structured_call(client, cfg, system="s", user="u", schema={"type": "object"})
    assert res.outcome == llm.OK and client.streamed == 0


@pytest.fixture
def eng(tmp_path):
    cfg = Config(db_path=str(tmp_path / "bot.db"), code_dir=str(tmp_path / "code"))
    j = Journal(cfg.db_path)
    versions.ensure(j)
    j.execute("INSERT INTO proposals(ts, source, title, problem, evidence, idea, priority) "
              "VALUES(0, 'user', 'Idea', 'p', 'e', 'i', 1)")
    return engineer.Engineer(j, cfg), j


def test_engineer_crash_requeues_the_proposal(eng, monkeypatch):
    e, j = eng

    def boom(*a, **k):
        raise ValueError("Streaming is required")
    monkeypatch.setattr(e, "_build", boom)
    assert e.attempt(object()) is None
    assert j.query("SELECT status FROM proposals")[0]["status"] == "open"
    assert j.query("SELECT kind FROM events")[0]["kind"] == "engineer_crashed"


def test_stuck_building_proposal_reopens_on_start(eng):
    e, j = eng
    j.execute("UPDATE proposals SET status='building'")
    engineer.Engineer(j, e.cfg)
    assert j.query("SELECT status FROM proposals")[0]["status"] == "open"


def test_jupiter_retries_server_errors(monkeypatch):
    monkeypatch.setattr(jupiter_mod.time, "sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"inAmount": "1", "outAmount": "1000", "otherAmountThreshold": "990",
                                         "priceImpactPct": "0", "routePlan": []})

    jup = Jupiter(http=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x"))
    assert jup.quote(jupiter_mod.USDC_MINT, "TOK", 1_000_000) is not None
    assert len(calls) == 3
