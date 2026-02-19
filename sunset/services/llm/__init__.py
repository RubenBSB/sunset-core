import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)


class SourceChunk(TypedDict):
    file_id: str
    filename: str
    text: str
    score: float


class ToolCall(TypedDict):
    name: str
    arguments: Dict[str, Any]
    id: str


class LLMResponse(TypedDict):
    text: str
    cited_chunks: Optional[List[SourceChunk]]
    tool_calls: Optional[List[ToolCall]]


# Deprecated alias — use LLMResponse instead
ToolCallResponse = LLMResponse

# Type for tool executor callback
ToolExecutor = Any  # Callable[[str, Dict[str, Any]], Any]


class _FileSearchSentinel:
    """Sentinel to enable file search / RAG in generate_response.

    Usage:
        from sunset.services.llm import file_search
        await llm.generate_response(input=msgs, model="gemini-2.5-flash", function_tools=[file_search])
    """

    def __repr__(self):
        return "file_search"


file_search = _FileSearchSentinel()
_FILE_SEARCH = file_search  # Internal ref to avoid shadowing by parameter names


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
        else:
            regular.append(t)
    return has_fs, regular


# Re-export classes so all existing imports continue to work
from .base import LLMService  # noqa: E402
from .gemini import GeminiService  # noqa: E402
from .openai import OpenAIService  # noqa: E402
from .router import LLMServiceRouter  # noqa: E402
from .store import FileInfo, FileStore, StoreInfo  # noqa: E402
from .vertex import VertexAIGeminiService  # noqa: E402

__all__ = [
    "SourceChunk",
    "ToolCall",
    "LLMResponse",
    "ToolCallResponse",
    "ToolExecutor",
    "file_search",
    "_FileSearchSentinel",
    "_FILE_SEARCH",
    "_split_tools",
    "LLMService",
    "OpenAIService",
    "GeminiService",
    "VertexAIGeminiService",
    "LLMServiceRouter",
    "FileStore",
    "FileInfo",
    "StoreInfo",
]
