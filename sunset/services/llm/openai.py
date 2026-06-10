import json
import logging
from typing import Any, Dict, List, Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from .base import (
    LLMResponse,
    LLMService,
    SourceChunk,
    ToolCall,
    ToolExecutor,
    Usage,
    _split_tools,
)
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
        self.file_store_name = file_store_name or "knowledge-base"
        self._store: Optional[OpenAIFileStore] = None
        if file_store_id:
            self._store = OpenAIFileStore(self.client, file_store_id)

    @property
    def store(self) -> Optional[OpenAIFileStore]:
        return self._store

    async def _ensure_store(self) -> Optional[OpenAIFileStore]:
        """Lazily create or return the file store."""
        if self._store:
            return self._store
        self._store = await OpenAIFileStore.create(
            self.client, name=self.file_store_name
        )
        return self._store

    def get_client(self):
        if not hasattr(self, "client"):
            return AsyncOpenAI(api_key=self.api_key)
        return self.client

    @staticmethod
    def _extract_usage(response) -> Usage:
        # Responses API exposes input_tokens/output_tokens; Chat Completions
        # exposes prompt_tokens/completion_tokens. Support both shapes.
        usage = getattr(response, "usage", None)
        if not usage:
            return {}
        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", 0)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", 0)
            or 0
        )
        total_tokens = getattr(usage, "total_tokens", 0) or (
            input_tokens + output_tokens
        )
        cached = 0
        details = getattr(usage, "input_tokens_details", None) or getattr(
            usage, "prompt_tokens_details", None
        )
        if details:
            cached = getattr(details, "cached_tokens", 0) or 0
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "cached_tokens": int(cached),
        }

    @staticmethod
    def _reasoning_for(model: str) -> Optional[Dict[str, Any]]:
        """Reasoning config per model. Returns None for non-reasoning models."""
        if model in ("gpt-5-mini", "gpt-5-nano", "gpt-5.1"):
            return {"effort": "low"}
        if model == "gpt-5.4-mini":
            return {"effort": "medium"}
        return None

    @staticmethod
    def _to_responses_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Chat-Completions-style function tool to the flatter
        Responses-API shape OpenAI expects.

        Chat Completions:  {"type": "function", "function": {"name", "description", "parameters", ...}}
        Responses:         {"type": "function", "name", "description", "parameters", ...}

        Pass through any other tool type (e.g. file_search) and any tool
        already in the flat shape unchanged.
        """
        if tool.get("type") != "function" or "function" not in tool:
            return tool
        fn = tool["function"] or {}
        out = {"type": "function"}
        for key in ("name", "description", "parameters", "strict"):
            if key in fn:
                out[key] = fn[key]
        return out

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
        has_file_search, regular_tools = _split_tools(function_tools)

        # If file_search is requested but no vector store is configured AND a
        # pgvector retrieval service is available, route through the retrieval
        # path so RAG works without uploading files to OpenAI.
        if has_file_search and self._store is None and self.retrieval is not None:
            return await self._retrieval_generate(
                input, model, regular_tools, tool_executor, metric_tag, temperature
            )

        tools = []
        include = []
        reasoning = self._reasoning_for(model)

        # File search setup
        file_store = await self._ensure_store() if has_file_search else None
        if has_file_search and file_store:
            tools.append(
                {
                    "type": "file_search",
                    "vector_store_ids": [file_store._store_id],
                }
            )
            include = ["output[*].file_search_call.search_results"]

        # Add custom function tools — convert each from Chat-Completions shape
        # to the flat Responses-API shape OpenAI requires.
        tools.extend(self._to_responses_tool(t) for t in regular_tools)

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
                    reasoning=reasoning,
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
                usage=self._extract_usage(response),
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
            usage=self._extract_usage(response),
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

    async def _retrieval_generate(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        extra_executor: Optional[ToolExecutor] = None,
        metric_tag: str = "",
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """RAG via pgvector. Exposes a search_knowledge function tool to the
        model and runs the standard tool loop. Mirrors VertexAIGeminiService.
        """
        search_tool = {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search the knowledge base for relevant documents and information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response_language": {
                            "type": "string",
                            "description": (
                                "The dominant language of the conversation so far, as an ISO "
                                "639-1 code (e.g. 'en', 'fr', 'es', 'de'). Base this on the full "
                                "conversation, not just the last message — short or ambiguous "
                                "turns (e.g. 'in toulouse', 'ok', proper nouns) do NOT change "
                                "the language. Only change languages if the user clearly and "
                                "deliberately switched. Your final reply MUST be written in this "
                                "language, regardless of the language of the retrieved content."
                            ),
                        },
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["response_language", "query"],
                },
            },
        }

        retrieved_chunks: List[Dict[str, Any]] = []

        async def composite_executor(name: str, args: dict):
            if name == "search_knowledge":
                response_language = (
                    args.get("response_language") or "the user's language"
                )
                chunks = await self.retrieval.query(args["query"], top_k=5)
                retrieved_chunks.extend(chunks)
                return {
                    "language_reminder": (
                        f"The retrieved content may be in a different language than the user. "
                        f"Your final reply MUST be written in '{response_language}'. "
                        f"Translate any content inline as needed — never switch languages on "
                        f"the user, and never mix languages in a single reply."
                    ),
                    "results": [
                        {
                            "content": c["content"],
                            "source": c["source_file"],
                            "score": c["score"],
                        }
                        for c in chunks
                    ],
                }
            if extra_executor:
                return await extra_executor(name, args)
            return {"error": f"Unknown tool: {name}"}

        all_tools = [search_tool] + (extra_tools or [])

        response = await self.generate_response(
            input=input,
            model=model,
            function_tools=all_tools,
            tool_executor=composite_executor,
            temperature=temperature,
            metric_tag=metric_tag,
        )

        cited_chunks: List[SourceChunk] = [
            SourceChunk(
                file_id=chunk["id"],
                filename=chunk["source_file"],
                text=chunk["content"],
                score=chunk["score"],
            )
            for chunk in retrieved_chunks
        ]
        return LLMResponse(
            text=response["text"],
            cited_chunks=cited_chunks[:5] if cited_chunks else None,
            tool_calls=response.get("tool_calls"),
            usage=response.get("usage"),
        )

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
