"""Claude API cost from a response's usage block (list prices, USD)."""

from __future__ import annotations

# model: (input, output, cache read, 5-min cache write) per million tokens
PRICES_PER_MTOK = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": (5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
WEB_SEARCH_USD = 10.00 / 1000


def usage_cost_usd(model: str, usage) -> float:
    """Cost of one response. A server-side fallback is priced at the model that served it."""
    inp, out, cache_read, cache_write = PRICES_PER_MTOK.get(model, PRICES_PER_MTOK["claude-opus-5-5"])
    tokens = (
        (getattr(usage, "input_tokens", 0) or 0) * inp
        + (getattr(usage, "output_tokens", 0) or 0) * out
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * cache_read
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * cache_write
    ) / 1_000_000
    server = getattr(usage, "server_tool_use", None)
    searches = (getattr(server, "web_search_requests", 0) or 0) if server else 0
    return tokens + searches * WEB_SEARCH_USD
