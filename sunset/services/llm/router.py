import asyncio
import logging
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from .base import (
    _FILE_SEARCH,
    BreakerProtocol,
    FallbackStep,
    LLMFallbackChainExhausted,
    LLMResponse,
    ToolExecutor,
    _FileSearchSentinel,
)
from .gemini import GeminiService
from .mistral import MistralService
from .openai import OpenAIService
from .store import FileStore
from .vertex import VertexAIGeminiService

logger = logging.getLogger(__name__)


# (step, outcome, error, attempt) → None | Awaitable[None]
AttemptHook = Callable[
    [FallbackStep, str, Optional[BaseException], int],
    Union[None, Awaitable[None]],
]
ScopeFactory = Callable[[FallbackStep], AbstractContextManager[Any]]


class LLMServiceRouter:
    """
    Unified LLM service that routes requests across multiple providers and
    runs cross-provider fallback chains with optional per-step circuit breaking.

    Model routing in generate_response():
    - 'gemini*' -> GeminiService or VertexAIGeminiService (primary location)
    - 'mistral*' -> MistralService (if mistral_api_key provided)
    - everything else -> OpenAIService

    Fallback runner (generate_with_fallback):
    - Caller passes an explicit chain of FallbackStep(model, location, priority).
    - For Vertex Gemini, multiple location-pinned clients are managed internally
      (see vertex_locations). location is the GCP region (or "global").
    - For OpenAI / Mistral, location is just a marker ("openai" / "mistral").
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        openai_file_store_id: Optional[str] = None,
        gemini_file_store_id: Optional[str] = None,
        file_store_name: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        # Vertex AI options
        use_vertex_ai: bool = False,
        vertex_project: Optional[str] = None,
        vertex_project_number: Optional[str] = None,
        vertex_location: str = "europe-west1",
        # Additional Vertex location-pinned clients spun up for fallback
        # (e.g. ["global"]). Keyed by location string.
        vertex_extra_locations: Optional[List[str]] = None,
        search_engine_id: Optional[str] = None,
        search_data_store_ids: Optional[List[str]] = None,
        # Monitoring
        monitoring_project: Optional[str] = None,
        # Retrieval (pgvector RAG) — propagated to all providers
        retrieval: Optional[Any] = None,
    ):
        self._openai: Optional[OpenAIService] = None
        if openai_api_key:
            self._openai = OpenAIService(
                api_key=openai_api_key,
                file_store_id=openai_file_store_id,
                file_store_name=file_store_name,
                retrieval=retrieval,
            )

        self._mistral: Optional[MistralService] = None
        if mistral_api_key:
            self._mistral = MistralService(
                api_key=mistral_api_key,
                retrieval=retrieval,
            )

        self.use_vertex_ai = use_vertex_ai
        self.vertex_location = vertex_location
        self._vertex_clients: Dict[str, VertexAIGeminiService] = {}
        self._gemini: Optional[Union[GeminiService, VertexAIGeminiService]] = None

        monitoring = None
        effective_project = monitoring_project or vertex_project
        if effective_project:
            from sunset.services.monitoring import MonitoringService

            monitoring = MonitoringService(project=effective_project)

        if use_vertex_ai:
            if not vertex_project:
                raise ValueError("vertex_project is required when use_vertex_ai=True")
            # vertex_project_number is only used by the Discovery Engine
            # grounded-generation path. For pgvector-RAG / standard chat,
            # it can be empty.
            if search_engine_id and not vertex_project_number:
                raise ValueError(
                    "vertex_project_number is required when search_engine_id is set"
                )
            primary = VertexAIGeminiService(
                project=vertex_project,
                project_number=vertex_project_number,
                location=vertex_location,
                search_engine_id=search_engine_id,
                search_data_store_ids=search_data_store_ids,
                monitoring=monitoring,
                retrieval=retrieval,
            )
            self._vertex_clients[vertex_location] = primary
            self._gemini = primary
            for loc in vertex_extra_locations or []:
                if loc == vertex_location:
                    continue
                self._vertex_clients[loc] = VertexAIGeminiService(
                    project=vertex_project,
                    project_number=vertex_project_number,
                    location=loc,
                    search_engine_id=search_engine_id,
                    search_data_store_ids=search_data_store_ids,
                    monitoring=monitoring,
                    retrieval=retrieval,
                )
            logger.info(
                f"Using Vertex AI Gemini (project={vertex_project}, "
                f"locations={list(self._vertex_clients)}, "
                f"search_engine={search_engine_id})"
            )
        elif gemini_api_key:
            self._gemini = GeminiService(
                api_key=gemini_api_key,
                file_store_id=gemini_file_store_id,
                file_store_name=file_store_name,
                monitoring=monitoring,
                retrieval=retrieval,
            )

        self.default_model = default_model

    # ── retrieval propagation ─────────────────────────────────────────────

    @property
    def retrieval(self):
        if self._gemini is not None:
            return getattr(self._gemini, "retrieval", None)
        if self._openai is not None:
            return getattr(self._openai, "retrieval", None)
        return None

    @retrieval.setter
    def retrieval(self, value: Any) -> None:
        for svc in (
            *self._vertex_clients.values(),
            self._openai,
            self._mistral,
        ):
            if svc is not None:
                svc.retrieval = value
        if (
            self._gemini is not None
            and self._gemini not in self._vertex_clients.values()
        ):
            self._gemini.retrieval = value

    # ── stores / metadata accessors ───────────────────────────────────────

    @property
    def openai_file_store_id(self) -> Optional[str]:
        if self._openai is None or self._openai.store is None:
            return None
        return self._openai.store._store_id

    @property
    def gemini_file_store_id(self) -> Optional[str]:
        if self._gemini is None:
            return None
        store = self._gemini.store
        return getattr(store, "_store_name", None) if store else None

    @property
    def search_engine_id(self) -> Optional[str]:
        if self.use_vertex_ai and self._gemini is not None:
            return getattr(self._gemini, "search_engine_id", None)
        return None

    @property
    def search_data_store_ids(self) -> List[str]:
        if self.use_vertex_ai and self._gemini is not None:
            return getattr(self._gemini, "search_data_store_ids", []) or []
        return []

    @property
    def store(self) -> Optional[FileStore]:
        return self._gemini.store if self._gemini is not None else None

    # ── routing ───────────────────────────────────────────────────────────

    def _get_service(self, model: str):
        if model.startswith("gemini"):
            if self._gemini is None:
                raise RuntimeError("No Gemini provider configured")
            return self._gemini
        if model.startswith("mistral") or model.startswith("pixtral"):
            if self._mistral is None:
                raise RuntimeError("No Mistral provider configured")
            return self._mistral
        if self._openai is None:
            raise RuntimeError("No OpenAI provider configured")
        return self._openai

    def _service_for_step(self, step: FallbackStep):
        if step.model.startswith("gemini"):
            client = self._vertex_clients.get(step.location)
            if client is not None:
                return client
            # Fall through: use the primary if no exact location match.
            if self._gemini is not None:
                return self._gemini
            raise RuntimeError(f"No Vertex client for location={step.location}")
        if step.model.startswith("mistral") or step.model.startswith("pixtral"):
            if self._mistral is None:
                raise RuntimeError("No Mistral provider configured")
            return self._mistral
        if self._openai is None:
            raise RuntimeError("No OpenAI provider configured")
        return self._openai

    # ── chain builders ────────────────────────────────────────────────────

    def default_fallback_chain(self, model: str) -> List[FallbackStep]:
        """Two-phase chain matching the legacy gemini-only behavior, then a
        cross-provider safety net (OpenAI, Mistral) appended if those
        providers are configured.

        For gemini preview models we mirror the legacy preference order:
        prefer same-family preview, then stable gemini-2.5-flash dual-homed,
        then a Priority Paygo phase that re-tries the preferred models on a
        separate quota pool. Cross-provider fallbacks come last."""
        chain: List[FallbackStep] = []

        if model == "gemini-3.1-flash-lite-preview":
            chain.extend(
                [
                    FallbackStep(model, "global", False),
                    FallbackStep("gemini-3-flash-preview", "global", False),
                    FallbackStep("gemini-2.5-flash", "europe-west1", False),
                    FallbackStep("gemini-2.5-flash", "global", False),
                    FallbackStep(model, "global", True),
                    FallbackStep("gemini-3-flash-preview", "global", True),
                    FallbackStep("gemini-2.5-flash", "global", True),
                ]
            )
        elif model == "gemini-3-flash-preview":
            chain.extend(
                [
                    FallbackStep(model, "global", False),
                    FallbackStep("gemini-3.1-flash-lite-preview", "global", False),
                    FallbackStep("gemini-2.5-flash", "europe-west1", False),
                    FallbackStep("gemini-2.5-flash", "global", False),
                    FallbackStep(model, "global", True),
                    FallbackStep("gemini-3.1-flash-lite-preview", "global", True),
                    FallbackStep("gemini-2.5-flash", "global", True),
                ]
            )
        elif model.startswith("gemini"):
            # Stable models: try local, then global as a free retry.
            chain.append(FallbackStep(model, self.vertex_location, False))
            if "global" in self._vertex_clients and self.vertex_location != "global":
                chain.append(FallbackStep(model, "global", False))
        else:
            chain.append(FallbackStep(model, "openai", False))

        # Cross-provider safety net — only when those providers are configured
        # and the requested model isn't already from that provider.
        if self._openai is not None and not chain[0].model.startswith("gpt"):
            chain.append(FallbackStep("gpt-5.4-mini", "openai", False))
        if self._mistral is not None and not chain[0].model.startswith("mistral"):
            chain.append(FallbackStep("mistral-medium-latest", "mistral", False))

        return chain

    # ── single-shot generation (back-compat) ──────────────────────────────

    async def generate_response(
        self,
        input: str | List[Dict[str, Any]],
        model: Optional[str] = None,
        function_tools: Optional[List[Any]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        text_format: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        metric_tag: str = "",
        priority: bool = False,
        # Deprecated — use function_tools=[file_search] instead
        file_search: bool = False,
    ) -> LLMResponse:
        model = model or self.default_model

        if file_search:
            function_tools = list(function_tools or [])
            if not any(isinstance(t, _FileSearchSentinel) for t in function_tools):
                function_tools.insert(0, _FILE_SEARCH)

        service = self._get_service(model)
        return await service.generate_response(
            input,
            model,
            function_tools=function_tools,
            tool_executor=tool_executor,
            text_format=text_format,
            temperature=temperature,
            metric_tag=metric_tag,
            priority=priority,
        )

    # ── fallback runner ───────────────────────────────────────────────────

    async def generate_with_fallback(
        self,
        input: str | List[Dict[str, Any]],
        chain: List[FallbackStep],
        function_tools: Optional[List[Any]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        tool_executor_factory: Optional[Callable[[], ToolExecutor]] = None,
        text_format: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        metric_tag: str = "",
        breaker: Optional[BreakerProtocol] = None,
        on_attempt: Optional[AttemptHook] = None,
        attempt_scope: Optional[ScopeFactory] = None,
        attempt_timeout_s: float = 10.0,
        max_empty_attempts: int = 3,
        slow_threshold_s: Optional[float] = None,
    ) -> LLMResponse:
        """Run a chain of LLM attempts, advancing on exception/timeout/empty.

        - breaker: optional per-step circuit breaker. A step is skipped if
          its breaker is open; 429s record into it. The last step in the chain
          is never skipped — it's the safety net.
        - on_attempt(step, outcome, error, attempt): callback invoked with
          outcome in {"start", "skipped_breaker", "success", "empty",
          "error", "timeout"}.
        - attempt_scope: optional context manager factory wrapping each step
          (for setting Sentry tags etc.).
        - attempt_timeout_s: per-step timeout. The empty-response retry loop
          for a given step shares this single budget.
        - max_empty_attempts: how many times to re-call the same step if the
          model returns no text (only function_call, etc.).
        - tool_executor_factory: when provided, a fresh executor is built per
          attempt. Use this for executors that accumulate side state (e.g.
          a cited_chunks list) so retries don't compound duplicates.
        - slow_threshold_s: if set and breaker is provided, any successful
          step that took longer than this threshold counts towards the
          breaker's fail count (record_slow). Catches soft-throttling that
          surfaces as latency rather than 429s.
        """
        if not chain:
            raise ValueError("Empty fallback chain")

        async def _emit(
            step: FallbackStep,
            outcome: str,
            error: Optional[BaseException],
            attempt: int,
        ) -> None:
            if on_attempt is None:
                return
            try:
                result = on_attempt(step, outcome, error, attempt)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.warning("on_attempt hook raised", exc_info=True)

        last_error: Optional[BaseException] = None
        last_response: Optional[LLMResponse] = None

        for chain_idx, step in enumerate(chain):
            is_last = chain_idx == len(chain) - 1

            if not is_last and breaker is not None:
                try:
                    if await breaker.is_open(step.breaker_key):
                        await _emit(step, "skipped_breaker", None, 0)
                        continue
                except Exception:
                    logger.warning("breaker.is_open failed", exc_info=True)

            service = self._service_for_step(step)
            scope_cm = (
                attempt_scope(step) if attempt_scope is not None else nullcontext()
            )

            with scope_cm:
                try:
                    async with asyncio.timeout(attempt_timeout_s):
                        response: Optional[LLMResponse] = None
                        for attempt in range(1, max_empty_attempts + 1):
                            await _emit(step, "start", None, attempt)
                            executor = (
                                tool_executor_factory()
                                if tool_executor_factory is not None
                                else tool_executor
                            )
                            call_started = asyncio.get_event_loop().time()
                            response = await service.generate_response(
                                input=input,
                                model=step.model,
                                function_tools=function_tools,
                                tool_executor=executor,
                                text_format=text_format,
                                temperature=temperature,
                                metric_tag=metric_tag,
                                priority=step.priority,
                            )
                            elapsed = asyncio.get_event_loop().time() - call_started
                            # Surface executor-accumulated citations (e.g.
                            # smart-tool flows) on the response so callers
                            # don't need to know about the executor's state.
                            if (
                                executor is not None
                                and not response.get("cited_chunks")
                                and hasattr(executor, "cited_chunks")
                            ):
                                response["cited_chunks"] = list(executor.cited_chunks)
                            if response.get("text"):
                                # Soft-throttle signal: a successful-but-slow
                                # call counts towards the breaker. A step
                                # degrading into latency gets shed before
                                # users feel multiple slow replies.
                                if (
                                    breaker is not None
                                    and slow_threshold_s is not None
                                    and elapsed >= slow_threshold_s
                                ):
                                    try:
                                        await breaker.record_slow(step.breaker_key)
                                    except Exception:
                                        logger.warning(
                                            "breaker.record_slow failed",
                                            exc_info=True,
                                        )
                                await _emit(step, "success", None, attempt)
                                return response
                            await _emit(step, "empty", None, attempt)
                        last_response = response
                except (TimeoutError, asyncio.TimeoutError) as e:
                    last_error = e
                    await _emit(step, "timeout", e, 0)
                    continue
                except Exception as e:
                    last_error = e
                    is_429 = type(e).__name__ in (
                        "ClientError",
                        "RateLimitError",
                    ) and (
                        getattr(e, "code", None) == 429
                        or getattr(e, "status_code", None) == 429
                        or getattr(e, "status", None) == 429
                    )
                    if is_429 and breaker is not None:
                        try:
                            await breaker.record_429(step.breaker_key)
                        except Exception:
                            logger.warning("breaker.record_429 failed", exc_info=True)
                    await _emit(step, "error", e, 0)
                    continue

        if last_response and last_response.get("text"):
            return last_response

        raise LLMFallbackChainExhausted(
            f"All fallback steps exhausted (chain={chain})"
        ) from last_error

    # ── JSON ──────────────────────────────────────────────────────────────

    async def generate_json(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        text_format: Optional[Type[BaseModel]] = None,
        metric_tag: str = "",
    ) -> Dict[str, Any]:
        """Generate structured JSON using the appropriate service based on model."""
        model = model or self.default_model
        service = self._get_service(model)
        return await service.generate_json(
            messages, model, temperature, text_format, metric_tag=metric_tag
        )
