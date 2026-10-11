from types import SimpleNamespace

import anthropic
import httpx

from tradingbot.analyst import FAILED, UNAVAILABLE, Analyst, is_billing_error
from tradingbot.engine import Engine
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tests.test_analyst_engine import BUY_JSON, FakeDex, resp

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
BILLING_BODY = {"type": "error", "error": {
    "type": "invalid_request_error",
    "message": "Your credit balance is too low to access the Anthropic API. "
               "Please go to Plans & Billing to upgrade or purchase credits."}}


def api_error(cls, status, body):
    return cls(body["error"]["message"], response=httpx.Response(status, request=REQ), body=body)


def billing():
    return api_error(anthropic.BadRequestError, 400, BILLING_BODY)


class ScriptedClient:
    """Raises or returns each scripted item in turn."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(cfg, pair, safe, script):
    client = ScriptedClient(script)
    j = Journal(":memory:")
    eng = Engine(cfg, FakeDex(pair), SimpleNamespace(report=lambda m: safe), Analyst(cfg, client),
                 PaperExecutor(cfg), j)
    return eng, j, client


def test_detects_billing_error_from_real_message():
    assert is_billing_error(billing())
    other = api_error(anthropic.BadRequestError, 400,
                      {"type": "error", "error": {"type": "invalid_request_error", "message": "bad schema"}})
    assert not is_billing_error(other)


def test_billing_error_pauses_and_keeps_token_queued(cfg, good_pair, safe):
    eng, j, client = make(cfg, good_pair, safe, [billing(), resp("end_turn", BUY_JSON)])
    eng.tick()
    assert eng.analyst.last_outcome == UNAVAILABLE
    assert good_pair.token_address not in eng._cooldown  # not set aside
    assert len(eng.analyst._calls) == 0  # failed call not counted against the hourly cap
    assert not eng.analyst.budget_ok()  # paused

    eng.tick()
    assert client.calls == 1  # no calls while paused

    eng.analyst.paused_until = 0  # pause expires (e.g. credit added)
    eng.tick()
    assert client.calls == 2
    assert len(j.open_positions()) == 1  # same token analysed and bought


def test_auth_error_pauses_for_account_duration(cfg, good_pair, safe):
    err = api_error(anthropic.AuthenticationError, 401,
                    {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
    eng, _, _ = make(cfg, good_pair, safe, [err])
    eng.tick()
    remaining = eng.analyst.paused_until - __import__("time").time()
    assert cfg.api_account_pause_s - 5 < remaining <= cfg.api_account_pause_s


def test_rate_limit_pauses_briefly(cfg, good_pair, safe):
    err = api_error(anthropic.RateLimitError, 429,
                    {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    eng, _, _ = make(cfg, good_pair, safe, [err])
    eng.tick()
    remaining = eng.analyst.paused_until - __import__("time").time()
    assert 0 < remaining <= cfg.api_transient_pause_s
    assert good_pair.token_address not in eng._cooldown


def test_token_specific_failure_still_sets_token_aside(cfg, good_pair, safe):
    eng, _, _ = make(cfg, good_pair, safe, [resp("refusal")])
    eng.tick()
    assert eng.analyst.last_outcome == FAILED
    assert good_pair.token_address in eng._cooldown
    assert eng.analyst.budget_ok()  # not paused
