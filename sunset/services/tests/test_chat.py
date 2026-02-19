"""Stress test for ChatService — verifies async concurrency under load."""

import asyncio
import time

import pytest

from sunset.services.chat import ChatService, ConversationContext
from sunset.services.llm import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMULATED_LLM_LATENCY = 0.5  # 500ms per LLM call
SIMULATED_DB_LATENCY = 0.02  # 20ms per DB hook
CONCURRENT_REQUESTS = 50


class FakeLLM:
    """Mock LLM that simulates network latency and tracks calls."""

    def __init__(self, latency: float = SIMULATED_LLM_LATENCY):
        self.latency = latency
        self.call_count = 0

    async def generate_response(self, **kwargs) -> LLMResponse:
        self.call_count += 1
        await asyncio.sleep(self.latency)
        return LLMResponse(text="mock response", cited_chunks=None, tool_calls=None)


async def make_db_hook(store: list, latency: float = SIMULATED_DB_LATENCY):
    """Return before_generate / on_response hooks backed by a shared list."""

    async def before_generate(ctx: ConversationContext):
        await asyncio.sleep(latency)
        # Simulate loading history from DB
        ctx.history = list(ctx.history)

    async def on_response(ctx: ConversationContext, response: LLMResponse):
        await asyncio.sleep(latency)
        store.append(
            {
                "conversation_meta": ctx.meta,
                "last_message": ctx.last_message,
                "response": response["text"],
            }
        )

    return before_generate, on_response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_conversations():
    """Many independent conversations running in parallel should complete
    faster than sequential execution, proving true async concurrency."""

    saved: list = []
    fake_llm = FakeLLM()
    before_gen, on_resp = await make_db_hook(saved)

    service = ChatService(
        llm=fake_llm,
        system_prompt="You are a test bot.",
        model="test-model",
        before_generate=before_gen,
        on_response=on_resp,
    )

    async def run_conversation(i: int) -> LLMResponse:
        conv = service.conversation(meta={"user_id": f"user_{i}"})
        return await conv.send(f"message from user {i}")

    start = time.monotonic()
    results = await asyncio.gather(
        *(run_conversation(i) for i in range(CONCURRENT_REQUESTS))
    )
    elapsed = time.monotonic() - start

    # Every request got a response
    assert len(results) == CONCURRENT_REQUESTS
    assert all(r["text"] == "mock response" for r in results)

    # LLM was called exactly once per request
    assert fake_llm.call_count == CONCURRENT_REQUESTS

    # on_response hook recorded every response
    assert len(saved) == CONCURRENT_REQUESTS

    # Concurrency check: sequential would take at least
    # N * (llm_latency + 2*db_latency). Parallel should be a fraction of that.
    sequential_estimate = CONCURRENT_REQUESTS * (
        SIMULATED_LLM_LATENCY + 2 * SIMULATED_DB_LATENCY
    )
    assert elapsed < sequential_estimate * 0.5, (
        f"Took {elapsed:.2f}s — expected well under {sequential_estimate:.2f}s "
        f"(sequential estimate). Async concurrency may not be working."
    )


@pytest.mark.asyncio
async def test_concurrent_messages_same_conversation():
    """Multiple messages sent concurrently on the *same* conversation object.
    Each send() appends to shared history — verify all complete and the
    history length is correct."""

    fake_llm = FakeLLM()
    service = ChatService(
        llm=fake_llm,
        system_prompt="You are a test bot.",
        model="test-model",
    )

    conv = service.conversation(meta={"user_id": "shared"})
    n = 30

    results = await asyncio.gather(*(conv.send(f"msg {i}") for i in range(n)))

    assert len(results) == n
    assert fake_llm.call_count == n
    # Each send appends a user msg + assistant msg = 2 entries per call
    assert len(conv.history) == n * 2


@pytest.mark.asyncio
async def test_slow_hook_does_not_block_other_conversations():
    """A slow before_generate hook on one conversation must not block others."""

    fake_llm = FakeLLM(latency=0.01)

    slow_delay = 0.3

    async def slow_before_generate(ctx: ConversationContext):
        if ctx.meta.get("slow"):
            await asyncio.sleep(slow_delay)

    service = ChatService(
        llm=fake_llm,
        system_prompt="test",
        model="test-model",
        before_generate=slow_before_generate,
    )

    async def fast_conv(i: int):
        c = service.conversation(meta={"slow": False, "id": i})
        return await c.send("fast")

    async def slow_conv():
        c = service.conversation(meta={"slow": True, "id": "slow"})
        return await c.send("slow")

    start = time.monotonic()
    results = await asyncio.gather(slow_conv(), *(fast_conv(i) for i in range(20)))
    elapsed = time.monotonic() - start

    assert len(results) == 21
    # Should finish roughly around the slow_delay, not slow_delay + 20*fast
    assert elapsed < slow_delay + 0.15, (
        f"Took {elapsed:.2f}s — fast conversations were blocked by the slow one"
    )


@pytest.mark.asyncio
async def test_hook_error_propagates():
    """If a hook raises, the error should propagate to the caller,
    and other concurrent conversations should be unaffected."""

    fake_llm = FakeLLM(latency=0.01)

    async def failing_hook(ctx: ConversationContext):
        if ctx.meta.get("fail"):
            raise ValueError("db connection lost")

    service = ChatService(
        llm=fake_llm,
        system_prompt="test",
        model="test-model",
        before_generate=failing_hook,
    )

    async def good_conv(i: int):
        c = service.conversation(meta={"fail": False})
        return await c.send(f"ok {i}")

    async def bad_conv():
        c = service.conversation(meta={"fail": True})
        return await c.send("boom")

    tasks = [asyncio.ensure_future(bad_conv())] + [
        asyncio.ensure_future(good_conv(i)) for i in range(10)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # First result is the failing one
    assert isinstance(results[0], ValueError)
    assert str(results[0]) == "db connection lost"

    # The rest succeeded
    for r in results[1:]:
        assert not isinstance(r, Exception)
        assert r["text"] == "mock response"
