"""OTel metric definitions for the LLM gateway and conversation activity.

Counters, histograms, and one observable gauge that reads from Redis. All
metric names follow the `llm.*` / `conversations.*` namespaces so they group
naturally in Grafana.

Attributes follow OTel GenAI semantic conventions (`gen_ai.system`,
`gen_ai.request.model`) where applicable, plus `llm.*` for gateway-specific
dims (channel, fallback step, breaker key).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable, Optional

from opentelemetry import metrics

logger = logging.getLogger(__name__)

_meter = metrics.get_meter("sunset.llm")

llm_request_total = _meter.create_counter(
    name="llm.request.total",
    unit="1",
    description="LLM request count (each fallback attempt counts once).",
)

llm_request_duration = _meter.create_histogram(
    name="llm.request.duration",
    unit="s",
    description="Per-attempt LLM call duration.",
)

llm_tokens = _meter.create_counter(
    name="llm.tokens",
    unit="1",
    description="LLM token usage. token_type ∈ {input, output, total, thinking, cached}.",
)

llm_cost_usd = _meter.create_counter(
    name="llm.cost.usd",
    unit="USD",
    description="LLM call cost in USD, computed from token usage and the pricing table.",
)

llm_fallback_advanced = _meter.create_counter(
    name="llm.fallback.advanced",
    unit="1",
    description="Number of times the fallback chain advanced past a step.",
)

llm_breaker_opened = _meter.create_counter(
    name="llm.breaker.opened",
    unit="1",
    description="Circuit breaker transitions to open state.",
)

llm_breaker_skipped = _meter.create_counter(
    name="llm.breaker.skipped",
    unit="1",
    description="Steps skipped because the breaker was open.",
)

llm_throttled = _meter.create_counter(
    name="llm.throttled",
    unit="1",
    description=(
        "Steps skipped because the local per-provider concurrency semaphore "
        "could not be acquired within the timeout."
    ),
)


# ── conversations.active observable gauge ─────────────────────────────────
#
# Tracks the rolling count of distinct conversation_ids that exchanged a
# message in the last ~60s. We keep 6 buckets of 10s each in Redis as
# regular sets, expiring 70s after creation; the gauge callback unions them.

_ACTIVE_BUCKETS = 6
_BUCKET_WIDTH_S = 10
_BUCKET_TTL_S = 70  # > buckets * width so the youngest never expires mid-read
_POLL_INTERVAL_S = 10
_redis = None
_active_count = 0
_poller_task: Optional["asyncio.Task[None]"] = None


def register_redis(redis, start_poller: bool = True) -> None:
    """Bind the app's redis client and optionally start the background poller.

    Must be called from inside the app's asyncio loop (e.g. FastAPI lifespan)
    so the poller task lives on the same loop as the redis client's
    connection pool. Without that, every read crosses event loops and fails.

    start_poller=False binds redis for mark_conversation_active() writes only.
    Use it in services that may run with CPU throttling (Cloud Run) — a
    background poller would starve between requests there.
    """
    global _redis, _poller_task
    _redis = redis
    if not start_poller:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _poller_task is None or _poller_task.done():
        _poller_task = loop.create_task(_poll_active_conversations())


async def _poll_active_conversations() -> None:
    """Background task: refresh _active_count every _POLL_INTERVAL_S seconds.

    The OTel gauge callback runs on the metric reader thread (no loop), so we
    can't await redis from there. Instead, we poll on the app loop and the
    callback just reads the cached int.
    """
    global _active_count
    while True:
        try:
            members = await _redis.client.sunion(*_bucket_keys())
            _active_count = len(members or [])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("active conversations poll failed", exc_info=True)
        await asyncio.sleep(_POLL_INTERVAL_S)


def _bucket_keys(now: Optional[float] = None) -> list[str]:
    t = int(now if now is not None else time.time())
    base = t // _BUCKET_WIDTH_S
    return [
        f"conversations:active:{(base - i) % (_ACTIVE_BUCKETS * 2)}"
        for i in range(_ACTIVE_BUCKETS)
    ]


async def mark_conversation_active(conversation_id: str) -> None:
    """SADD the conv id into the current bucket. Call from request handlers."""
    if _redis is None or not conversation_id:
        return
    try:
        key = _bucket_keys()[0]
        pipe = _redis.client.pipeline()
        pipe.sadd(key, conversation_id)
        pipe.expire(key, _BUCKET_TTL_S)
        await pipe.execute()
    except Exception:
        logger.warning("mark_conversation_active failed", exc_info=True)


def _active_conversations_callback(
    options: metrics.CallbackOptions,
) -> Iterable[metrics.Observation]:
    # Only processes running the poller export the gauge — a non-polling
    # process would report a frozen/zero value and pollute the metric.
    if _poller_task is None:
        return ()
    return (metrics.Observation(int(_active_count)),)


_meter.create_observable_gauge(
    name="conversations.active",
    callbacks=[_active_conversations_callback],
    unit="1",
    description="Distinct conversations with activity in the last ~60s.",
)
