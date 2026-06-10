"""Per-model pricing for LLM call cost computation.

Prices are USD per 1M tokens, as (input, output). Keep this catalogue close
to the published vendor pricing pages — when a price changes, edit here.

Unknown models map to (0, 0) and produce a 0$ cost; the call still emits a
metric so we can spot missing entries via Grafana.
"""

from __future__ import annotations

from typing import Mapping

# (input_per_1m_usd, output_per_1m_usd)
_PRICING: Mapping[str, tuple[float, float]] = {
    # Gemini / Vertex
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-flash-preview": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.10, 0.40),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.4-mini": (0.25, 2.00),
    # Mistral
    "mistral-medium-latest": (2.70, 8.10),
    "mistral-large-latest": (3.00, 9.00),
    "pixtral-large-latest": (2.00, 6.00),
}


def compute_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """USD cost for one call. Returns 0.0 for unknown models."""
    if not model:
        return 0.0
    # Strip the priority/location suffix the router uses internally
    # (e.g. "gemini-3-flash-preview@global+priority").
    base = model.split("@", 1)[0]
    rates = _PRICING.get(base)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0


def is_known(model: str) -> bool:
    return model.split("@", 1)[0] in _PRICING
