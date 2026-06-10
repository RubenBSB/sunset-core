"""Helpers for OTel instrumentation of LLM calls.

Centralizes:
 - the mapping from model name → provider name (used as `gen_ai.system`)
 - the conversion of a finished attempt into span attrs + metric points

Kept separate from the router so the router stays readable. OpenTelemetry is
optional: when the `observability` extra is not installed, `tracer` and
`obs_metrics` degrade to no-ops so the router works unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Optional

from .pricing import compute_cost


def provider_for(model: str) -> str:
    """Map a model name to a `gen_ai.system` value (OTel semconv)."""
    if model.startswith("gemini"):
        return "vertex_ai"
    if model.startswith("mistral") or model.startswith("pixtral"):
        return "mistral"
    return "openai"


def record_attempt(
    span: Any,
    *,
    model: str,
    location: str,
    priority: bool,
    channel: str,
    response: Optional[Mapping[str, Any]],
    latency_s: float,
    outcome: str,
    error: Optional[BaseException] = None,
    extra_attrs: Optional[Mapping[str, Any]] = None,
) -> float:
    """Stamp the span and emit metrics for one attempt. Returns the cost USD.

    outcome ∈ {"success", "empty", "error", "timeout", "skipped_breaker"}.
    """
    provider = provider_for(model)
    attrs: dict[str, Any] = {
        "gen_ai.system": provider,
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model,
        "llm.location": location,
        "llm.priority": priority,
        "llm.channel": channel,
        "llm.outcome": outcome,
    }

    usage = (response or {}).get("usage") or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    if input_tokens:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens:
        attrs["gen_ai.usage.output_tokens"] = output_tokens

    cost = compute_cost(model, input_tokens, output_tokens)
    if cost:
        attrs["llm.cost.usd"] = cost

    if extra_attrs:
        for k, v in extra_attrs.items():
            attrs[k] = v

    for k, v in attrs.items():
        if v is not None:
            span.set_attribute(k, v)

    if error is not None:
        span.record_exception(error)

    metric_attrs = {
        "model": model,
        "provider": provider,
        "channel": channel,
        "priority": str(priority).lower(),
    }
    obs_metrics.llm_request_total.add(1, attributes={**metric_attrs, "status": outcome})
    if outcome == "success":
        obs_metrics.llm_request_duration.record(latency_s, attributes=metric_attrs)
        if input_tokens:
            obs_metrics.llm_tokens.add(
                input_tokens,
                attributes={**metric_attrs, "token_type": "input"},
            )
        if output_tokens:
            obs_metrics.llm_tokens.add(
                output_tokens,
                attributes={**metric_attrs, "token_type": "output"},
            )
        thinking = int(usage.get("thinking_tokens", 0) or 0)
        if thinking:
            obs_metrics.llm_tokens.add(
                thinking,
                attributes={**metric_attrs, "token_type": "thinking"},
            )
        cached = int(usage.get("cached_tokens", 0) or 0)
        if cached:
            obs_metrics.llm_tokens.add(
                cached,
                attributes={**metric_attrs, "token_type": "cached"},
            )
        if cost:
            obs_metrics.llm_cost_usd.add(cost, attributes=metric_attrs)

    return cost


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str, attributes: Any = None):
        yield _NoopSpan()


class _NoopInstrument:
    def add(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NoopMetrics:
    def __getattr__(self, name: str) -> _NoopInstrument:
        return _NoopInstrument()


# Real OTel when the observability extra is installed, no-ops otherwise —
# must come after the no-op classes are defined.
try:
    from opentelemetry import trace as _trace

    from sunset.services.observability import metrics as obs_metrics

    tracer = _trace.get_tracer("sunset.llm")
except ImportError:
    obs_metrics = _NoopMetrics()
    tracer = _NoopTracer()
