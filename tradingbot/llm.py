"""Shared Claude calls for the learning system (post-mortems, playbook, optimiser, engineer).

`structured_call` returns schema-validated JSON. `tool_loop` runs a manual tool-use
loop with our own client-side tools. Both use the same model settings as the analyst
and report an outcome the caller can act on:

  ok           usable result
  failed       this request failed (refusal, truncation, bad output); try other work
  unavailable  the API is unusable (billing, auth, rate limit, outage); back off
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from tradingbot.analyst import is_billing_error
from tradingbot.config import Config
from tradingbot.costs import usage_cost_usd

log = logging.getLogger(__name__)

OK, FAILED, UNAVAILABLE = "ok", "failed", "unavailable"
MAX_PAUSE_RESUMES = 4
STREAM_ABOVE_TOKENS = 16000


@dataclass
class LLMResult:
    data: dict | None = None
    cost_usd: float = 0.0
    outcome: str = FAILED
    error: str | None = None
    transcript: list = field(default_factory=list)


def _request(client, cfg: Config, *, system: str, messages: list, tools: list, schema: dict | None,
             effort: str, max_tokens: int, model: str | None = None):
    model = model or cfg.model
    output_config: dict = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    kwargs = dict(
        model=model, max_tokens=max_tokens, system=system, messages=messages,
        thinking={"type": "adaptive"}, output_config=output_config,
    )
    if not model.startswith("claude-haiku"):  # Haiku has no server-side refusal fallback
        kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if tools:
        kwargs["tools"] = tools
    # The SDK refuses non-streaming requests whose max_tokens could run past 10 minutes.
    stream = getattr(client.beta.messages, "stream", None)
    if max_tokens > STREAM_ABOVE_TOKENS and stream is not None:
        with stream(**kwargs) as s:
            return s.get_final_message()
    return client.beta.messages.create(**kwargs)


def _classify(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return UNAVAILABLE, f"auth {exc.status_code}"
    if isinstance(exc, (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)):
        return UNAVAILABLE, type(exc).__name__
    if isinstance(exc, anthropic.APIStatusError) and is_billing_error(exc):
        return UNAVAILABLE, "billing"
    return FAILED, f"{type(exc).__name__}: {exc}"


def _final_json(content) -> dict | None:
    texts = [b.text for b in content if getattr(b, "type", None) == "text"]
    if not texts:
        return None
    try:
        data = json.loads(texts[-1])
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def structured_call(client, cfg: Config, *, system: str, user: str, schema: dict, web_searches: int = 0,
                    effort: str = "medium", max_tokens: int = 16000, model: str | None = None) -> LLMResult:
    tools = ([{"type": "web_search_20260209", "name": "web_search", "max_uses": web_searches}]
             if web_searches > 0 else [])
    messages: list = [{"role": "user", "content": user}]
    result = LLMResult()
    try:
        for _ in range(MAX_PAUSE_RESUMES + 1):
            resp = _request(client, cfg, system=system, messages=messages, tools=tools, schema=schema,
                            effort=effort, max_tokens=max_tokens, model=model)
            result.cost_usd += usage_cost_usd(getattr(resp, "model", None) or model or cfg.model, resp.usage)
            if resp.stop_reason != "pause_turn":
                break
            messages = [messages[0], {"role": "assistant", "content": resp.content}]
        else:
            result.error = "still paused"
            return result
    except anthropic.APIError as exc:
        result.outcome, result.error = _classify(exc)
        return result
    if resp.stop_reason in ("refusal", "max_tokens"):
        result.error = resp.stop_reason
        return result
    result.data = _final_json(resp.content)
    result.outcome = OK if result.data is not None else FAILED
    if result.data is None:
        result.error = "no valid JSON in response"
    return result


def tool_loop(client, cfg: Config, *, system: str, user: str, tools: list[dict],
              handlers: dict[str, Callable[[dict], str]], schema: dict | None = None,
              effort: str = "high", max_turns: int = 30, max_tokens: int = 32000,
              budget_usd: float | None = None) -> LLMResult:
    """Let Claude call our tools until it finishes; the final message must match `schema` if given.

    Tool handler errors are returned to Claude as error results rather than raised.
    Stops early (outcome failed) if `budget_usd` would be exceeded.
    """
    messages: list = [{"role": "user", "content": user}]
    result = LLMResult(transcript=messages)
    for _ in range(max_turns):
        try:
            resp = _request(client, cfg, system=system, messages=messages, tools=tools, schema=schema,
                            effort=effort, max_tokens=max_tokens)
        except anthropic.APIError as exc:
            result.outcome, result.error = _classify(exc)
            return result
        result.cost_usd += usage_cost_usd(getattr(resp, "model", cfg.model), resp.usage)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "pause_turn":
            continue
        if resp.stop_reason != "tool_use":
            if resp.stop_reason in ("refusal", "max_tokens"):
                result.error = resp.stop_reason
                return result
            result.data = _final_json(resp.content) if schema else {"text": "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text")}
            result.outcome = OK if result.data is not None else FAILED
            return result
        if budget_usd is not None and result.cost_usd > budget_usd:
            result.error = f"budget ${budget_usd:.2f} exhausted"
            return result
        results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            handler = handlers.get(block.name)
            try:
                if handler is None:
                    raise ValueError(f"unknown tool {block.name}")
                output = handler(dict(block.input))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)[:50_000]})
            except Exception as exc:  # noqa: BLE001 (report tool errors to Claude)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"error: {type(exc).__name__}: {exc}", "is_error": True})
        messages.append({"role": "user", "content": results})
    result.error = "too many turns"
    return result
