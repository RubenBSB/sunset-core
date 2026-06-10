"""Shared foundation for the LLM package — types, protocols, sentinels, and
the abstract base class every provider implements.

Submodules (gemini, openai, vertex, mistral, router) import from this file
to avoid circular dependencies through __init__.py.
"""

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypedDict,
)

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type for tool executor callback
ToolExecutor = Any  # Callable[[str, Dict[str, Any]], Any]


def _split_tools(
    function_tools: Optional[List[Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Separate file_search sentinel from regular tool dicts."""
    if not function_tools:
        return False, []
    has_fs = False
    regular: List[Dict[str, Any]] = []
    for t in function_tools:
        if isinstance(t, _FileSearchSentinel):
            has_fs = True
        elif isinstance(t, dict):
            regular.append(t)
        else:
            logger.warning(f"Ignoring non-dict tool entry: {t!r}")
    return has_fs, regular


class SourceChunk(TypedDict):
    file_id: str
    filename: str
    text: str
    score: float


class ToolCall(TypedDict):
    name: str
    arguments: Dict[str, Any]
    id: str


class Usage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    thinking_tokens: int
    cached_tokens: int


class LLMResponse(TypedDict, total=False):
    text: str
    cited_chunks: Optional[List[SourceChunk]]
    tool_calls: Optional[List[ToolCall]]
    usage: Optional[Usage]
    # Why a completion may be empty: finish_reason, native_finish_reason,
    # has_reasoning, refusal, iterations, hit_max_iterations, provider.
    diagnostics: Optional[Dict[str, Any]]


class FallbackStep(NamedTuple):
    """A single attempt in a fallback chain.

    location is provider-specific:
      - Vertex Gemini: GCP region or "global"
      - OpenAI: "openai"
      - Mistral: "mistral"
    priority maps to Vertex AI Priority Paygo (no-op for other providers).
    """

    model: str
    location: str
    priority: bool = False

    @property
    def breaker_key(self) -> str:
        return f"{self.model}@{self.location}{'+priority' if self.priority else ''}"


class BreakerProtocol(Protocol):
    """Per-key circuit breaker. The fallback runner skips a step when its
    breaker is open. It records 429s and slow-but-successful calls — both
    contribute to the same fail count, since a saturated step degrades into
    latency before throwing."""

    async def is_open(self, key: str) -> bool: ...
    async def record_429(self, key: str) -> bool: ...
    async def record_slow(self, key: str) -> bool: ...


class LLMFallbackChainExhausted(Exception):
    """Every step in the fallback chain failed (or was skipped by the breaker)."""


class _FileSearchSentinel:
    """Sentinel to enable file search / RAG in generate_response.

    Usage:
        from sunset.services.llm import file_search
        await llm.generate_response(input=msgs, model="gemini-2.5-flash", function_tools=[file_search])
    """

    def __repr__(self):
        return "file_search"


class LLMService(ABC):
    """
    Abstract base class for LLM providers (OpenAI, Gemini, etc.).

    Provides unified interface for text generation with optional file search (RAG).
    Subclasses must implement client initialization, file store management, and generation.
    """

    def __init__(self):
        self.client = self.get_client()

    @abstractmethod
    def get_client(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def generate_response(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        function_tools: Optional[List[Any]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        text_format: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        metric_tag: str = "",
        priority: bool = False,
    ) -> LLMResponse:
        """
        Generate a response, optionally with tool calling and/or file search.

        Args:
            input: String or List of message dicts with 'role' and 'content'.
            model: Model identifier.
            function_tools: List of tool dicts and/or the file_search sentinel.
            tool_executor: Async callback (name, args) -> result for custom tools.
            text_format: Optional Pydantic model for structured output.
            temperature: Sampling temperature.
            metric_tag: Custom tag for metric tracking.

        Returns:
            LLMResponse with text, optional sources, and optional tool_calls.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def generate_json(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        text_format: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response (for evaluation, judging, test generation).

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model identifier.
            temperature: Sampling temperature.
            text_format: Optional Pydantic model for structured output schema.

        Returns:
            Parsed JSON dict from the response.
        """
        raise NotImplementedError("Subclasses must implement this method")


# Sentinel singleton — must come after _FileSearchSentinel is defined.
file_search = _FileSearchSentinel()
_FILE_SEARCH = file_search  # Internal ref to avoid shadowing by parameter names
