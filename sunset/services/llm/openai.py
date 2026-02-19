import json
import logging
from typing import Any, Dict, List, Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from . import LLMResponse, SourceChunk, ToolCall, ToolExecutor, _split_tools
from .base import LLMService
from .store import OpenAIFileStore

logger = logging.getLogger(__name__)


class OpenAIService(LLMService):
    """OpenAI Responses API implementation with vector store file search."""

    def __init__(
        self,
        api_key: str,
        file_store_id: Optional[str] = None,
        file_store_name: Optional[str] = None,
        monitoring: Optional[Any] = None,
        retrieval: Optional[Any] = None,
    ):
        self.api_key = api_key
        self.monitoring = monitoring
        self.retrieval = retrieval
        super().__init__()
        self.file_store_id = file_store_id
        self.file_store_name = file_store_name or "knowledge-base"
        self._file_store = None  # Lazy-loaded
        self._store: Optional[OpenAIFileStore] = None

    async def get_file_store(self):
        """
        Lazily retrieve or create and cache the file store.

        If file_store_id is provided, retrieves the existing store.
        If not provided, creates a new store with file_store_name.
        """
        if self._file_store:
            return self._file_store

        if self.file_store_id:
            # Retrieve existing store
            self._file_store = await self.client.vector_stores.retrieve(
                vector_store_id=self.file_store_id
            )
        else:
            # Create new store lazily
            self._file_store = await self.client.vector_stores.create(
                name=self.file_store_name
            )
            self.file_store_id = self._file_store.id
            logger.info(
                f"Created OpenAI vector store: {self._file_store.id} "
                f"(name: {self.file_store_name}). "
                "Save this ID to OPENAI_FILE_STORE_ID to reuse."
            )

        if not self._store:
            self._store = OpenAIFileStore(self.client, self._file_store.id)

        return self._file_store

    @property
    def store(self) -> Optional[OpenAIFileStore]:
        return self._store

    def get_client(self):
        if not hasattr(self, "client"):
            return AsyncOpenAI(api_key=self.api_key)
        return self.client

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
        has_file_search, regular_tools = _split_tools(function_tools)

        tools = []
        reasoning = None
        include = []

        # File search setup
        store = await self.get_file_store() if has_file_search else None
        if has_file_search and store:
            tools.append(
                {
                    "type": "file_search",
                    "vector_store_ids": [store.id],
                }
            )
            reasoning = (
                {"effort": "low"}
                if model in ["gpt-5-mini", "gpt-5-nano", "gpt-5.1"]
                else None
            )
            include = ["output[*].file_search_call.search_results"]

        # Add custom function tools
        tools.extend(regular_tools)

        # Tool-calling loop when custom tools are present
        if regular_tools and tool_executor:
            conversation = (
                input
                if isinstance(input, list)
                else [{"role": "user", "content": input}]
            )
            tool_calls_made: List[ToolCall] = []
            max_iterations = 3

            for _ in range(max_iterations):
                response = await self.client.responses.create(
                    model=model,
                    input=conversation,
                    tools=tools if tools else None,
                    include=include if include else None,
                )

                function_calls = [
                    item
                    for item in response.output
                    if getattr(item, "type", None) == "function_call"
                ]

                if not function_calls:
                    break

                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = (
                        json.loads(fc.arguments)
                        if isinstance(fc.arguments, str)
                        else fc.arguments
                    )
                    tool_id = fc.call_id

                    tool_calls_made.append(
                        ToolCall(name=tool_name, arguments=tool_args, id=tool_id)
                    )

                    try:
                        result = await tool_executor(tool_name, tool_args)
                        result_str = (
                            json.dumps(result)
                            if not isinstance(result, str)
                            else result
                        )
                    except Exception as e:
                        logger.error(f"Tool execution error for {tool_name}: {e}")
                        result_str = json.dumps({"error": str(e)})

                    conversation = response.output + [
                        {
                            "type": "function_call_output",
                            "call_id": tool_id,
                            "output": result_str,
                        }
                    ]

            cited_chunks = (
                self._extract_file_search_sources(response) if has_file_search else []
            )
            return LLMResponse(
                text=response.output_text,
                cited_chunks=cited_chunks if cited_chunks else None,
                tool_calls=tool_calls_made if tool_calls_made else None,
            )

        # Simple generation (possibly with file_search only)
        if text_format:
            response = await self.client.responses.parse(
                model=model,
                input=input,
                tools=tools,
                reasoning=reasoning,
                include=include if include else None,
                tool_choice="required" if tools else None,
                text_format=text_format,
            )
        else:
            response = await self.client.responses.create(
                model=model,
                input=input,
                tools=tools,
                reasoning=reasoning,
                include=include if include else None,
                tool_choice="required" if tools else None,
            )

        # Extract cited chunks from file search results
        cited_chunks: List[SourceChunk] = []
        if has_file_search:
            cited_chunks = self._extract_file_search_sources(response)

        if (
            text_format
            and hasattr(response, "output_parsed")
            and response.output_parsed
        ):
            output_text = response.output_parsed.model_dump_json()
        else:
            output_text = response.output_text

        return LLMResponse(
            text=output_text,
            cited_chunks=cited_chunks if cited_chunks else None,
            tool_calls=None,
        )

    def _extract_file_search_sources(self, response) -> List[SourceChunk]:
        """Extract cited chunks from OpenAI file search results."""
        try:
            cited_file_ids = set()
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content_part in item.content:
                        if (
                            hasattr(content_part, "annotations")
                            and content_part.annotations
                        ):
                            for annotation in content_part.annotations:
                                cited_file_ids.add(annotation.file_id)

            all_chunks: List[SourceChunk] = []
            for item in response.output:
                if getattr(item, "type", None) == "file_search_call":
                    results = getattr(item, "results", None)
                    if results:
                        for chunk in results:
                            if chunk.file_id in cited_file_ids:
                                all_chunks.append(
                                    SourceChunk(
                                        file_id=chunk.file_id,
                                        filename=chunk.filename,
                                        text=chunk.text,
                                        score=chunk.score,
                                    )
                                )

            all_chunks.sort(key=lambda x: x["score"], reverse=True)
            return all_chunks[:5]
        except Exception as e:
            logger.debug(f"Could not extract sources from response: {e}")
            return []

    async def generate_json(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        text_format: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON response using OpenAI chat completions."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI JSON generation error: {e}")
            return {}
