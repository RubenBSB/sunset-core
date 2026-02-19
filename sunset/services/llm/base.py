from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from . import LLMResponse, ToolExecutor


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
