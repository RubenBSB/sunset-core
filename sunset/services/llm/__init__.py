import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict

from google import genai
from google.genai import types
from openai import AsyncOpenAI
from pydantic import BaseModel

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
    def create_file_store(self, initial_files: Optional[List[str]] = None):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def upload_files(self, file_descriptions: List[Dict[str, Any]]):
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

        return self._file_store

    def get_client(self):
        if not hasattr(self, "client"):
            return AsyncOpenAI(api_key=self.api_key)
        return self.client

    def create_file_store(
        self, store_name: str, initial_files: Optional[List[Dict[str, Any]]] = None
    ):
        self._file_store = self.client.vector_stores.create(name=store_name)

        if initial_files:
            self.upload_files(initial_files)

    def upload_files(self, file_descriptions: List[Dict[str, Any]]):
        for description in file_descriptions:
            with open(description["path"], "rb") as fileb:
                file = self.client.files.create(
                    file=(description["name"], fileb), purpose="user_data"
                )
                description["file_id"] = file.id

        store_file_batch = self.client.vector_stores.file_batches.create(
            vector_store_id=self.file_store.id,
            files=[
                {
                    "file_id": description["file_id"],
                    "attributes": {**description.get("attributes", {})},
                }
                for description in file_descriptions
            ],
        )
        logger.info(store_file_batch)

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


class GeminiService(LLMService):
    """Google Gemini API implementation with file search store grounding."""

    def __init__(
        self,
        api_key: str,
        file_store_id: Optional[str] = None,
        file_store_name: Optional[str] = None,
        monitoring: Optional[Any] = None,
    ):
        self.api_key = api_key
        self.monitoring = monitoring
        super().__init__()
        self.file_store_id = file_store_id
        self.file_store_name = file_store_name or "knowledge-base"
        self._file_store = None  # Lazy-loaded

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
            try:
                self._file_store = await self.client.aio.file_search_stores.get(
                    name=self.file_store_id
                )
            except Exception as e:
                logger.error(f"Failed to retrieve Gemini file store: {e}")
                return None
        else:
            # Create new store lazily
            try:
                self._file_store = self.client.file_search_stores.create(
                    config={"display_name": self.file_store_name}
                )
                self.file_store_id = self._file_store.name
                logger.info(
                    f"Created Gemini file search store: {self._file_store.name} "
                    f"(display_name: {self.file_store_name}). "
                    "Save this ID to GEMINI_FILE_STORE_ID to reuse."
                )
            except Exception as e:
                logger.error(f"Failed to create Gemini file store: {e}")
                return None

        return self._file_store

    def get_client(self):
        if not hasattr(self, "client"):
            return genai.Client(api_key=self.api_key)
        return self.client

    def _track_tokens(self, response, model: str, method: str, metric_tag: str = ""):
        """Extract usage_metadata from response and push to Cloud Monitoring."""
        if not self.monitoring:
            return
        try:
            usage = getattr(response, "usage_metadata", None)
            if usage:
                self.monitoring.write_token_metric(
                    model=model,
                    method=method,
                    prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                    total_tokens=getattr(usage, "total_token_count", 0) or 0,
                    thinking_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
                    cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
                    tag=metric_tag,
                )
        except Exception as e:
            logger.warning(f"Failed to track tokens: {e}")

    def create_file_store(
        self, store_name: str, initial_files: Optional[List[Dict[str, Any]]] = None
    ):
        self._file_store = self.client.file_search_stores.create(
            config={"display_name": store_name}
        )

        if initial_files:
            self.upload_files(initial_files)

    def upload_files(self, file_descriptions: List[Dict[str, Any]]):
        operations = []
        if not self.file_store:
            logger.error("No file store available for upload")
            return

        for description in file_descriptions:
            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=description["path"],
                file_search_store_name=self.file_store.name,
                config={"display_name": description["name"]},
            )
            operations.append(operation)

        while not all(operation.done for operation in operations):
            time.sleep(5)
            operations = [
                self.client.operations.get(operation) for operation in operations
            ]

        logger.info(operations)

    async def upload_file_async(
        self, file_path: str, display_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Upload a single file to the file search store asynchronously.

        Returns file metadata dict with name, display_name, size_bytes, create_time, state.
        """
        import os
        from datetime import datetime, timezone

        store = await self.get_file_store()
        if not store:
            logger.error("No file store available for upload")
            return None

        try:
            # Get file size before upload
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

            # Start the upload operation
            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=store.name,
                config={"display_name": display_name},
            )

            # Poll until complete (with async sleep)
            import asyncio

            while not operation.done:
                await asyncio.sleep(2)
                operation = self.client.operations.get(operation)

            # Operation completed - try to extract document info from response
            # The response structure varies, so we try multiple attributes
            doc_name = ""
            doc_state = "ACTIVE"

            # Try to get the document name from the operation
            if hasattr(operation, "response") and operation.response:
                doc_name = getattr(operation.response, "name", "")
                doc_state = str(getattr(operation.response, "state", "ACTIVE"))
            elif hasattr(operation, "name"):
                # The operation name often contains the document reference
                doc_name = operation.name

            logger.info(f"Upload completed for {display_name}, operation: {operation}")

            return {
                "name": doc_name,
                "display_name": display_name,
                "size_bytes": file_size,
                "create_time": datetime.now(timezone.utc).isoformat(),
                "state": doc_state,
            }
        except Exception as e:
            logger.exception(f"Failed to upload file to Gemini: {e}")
            return None

    async def list_files(self) -> List[Dict[str, Any]]:
        """
        List all documents in the file search store.

        Returns list of file metadata dicts.
        """
        store = await self.get_file_store()
        if not store:
            logger.error("No file store available")
            return []

        try:
            # List documents in the file search store using the documents.list method
            documents = self.client.file_search_stores.documents.list(parent=store.name)
            result = []
            for doc in documents:
                result.append(
                    {
                        "name": getattr(doc, "name", ""),
                        "display_name": getattr(doc, "display_name", ""),
                        "size_bytes": getattr(doc, "size_bytes", 0),
                        "create_time": getattr(doc, "create_time", None),
                        "state": str(getattr(doc, "state", "ACTIVE")),
                    }
                )
            return result
        except Exception as e:
            logger.exception(f"Failed to list files from Gemini: {e}")
            return []

    async def delete_file(self, file_name: str) -> bool:
        """
        Delete a document from the file search store.

        Args:
            file_name: The full resource name of the document

        Returns True if deleted successfully.
        """
        try:
            # Use config={'force': True} to delete document and all its chunks
            self.client.file_search_stores.documents.delete(
                name=file_name, config={"force": True}
            )
            logger.info(f"Deleted file: {file_name}")
            return True
        except Exception as e:
            logger.exception(f"Failed to delete file {file_name}: {e}")
            return False

    def _parse_data_url(self, data_url: str) -> tuple[bytes, str]:
        """Parse a data URL and return (bytes, mime_type)."""
        match = re.match(r"data:([^;]+);base64,(.+)", data_url)
        if match:
            mime_type = match.group(1)
            b64_data = match.group(2)
            return base64.b64decode(b64_data), mime_type
        raise ValueError("Invalid data URL format")

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

        # pgvector retrieval (tool-based, composes with custom tools)
        if has_file_search and self.retrieval:
            return await self._retrieval_generate(
                input, model, regular_tools, tool_executor, metric_tag
            )

        tools = []
        store = await self.get_file_store() if has_file_search else None

        if has_file_search and store:
            tools.append(
                types.Tool(
                    file_search=types.FileSearch(file_search_store_names=[store.name])
                )
            )

        system_instruction = None
        gemini_messages = []

        if isinstance(input, str):
            gemini_messages.append(
                types.Content(role="user", parts=[types.Part(text=input)])
            )
        else:
            for msg in input:
                role = msg.get("role")
                content = msg.get("content")

                if role in ("system", "developer"):
                    # Extract system/developer prompt - append to system_instruction
                    text = None
                    if isinstance(content, str):
                        text = content
                    elif (
                        isinstance(content, list)
                        and len(content) > 0
                        and content[0].get("type") == "text"
                    ):
                        text = content[0].get("text")

                    if text:
                        system_instruction = (
                            f"{system_instruction}\n\n{text}"
                            if system_instruction
                            else text
                        )
                    continue

                parts = []
                if isinstance(content, str):
                    parts.append(types.Part(text=content))
                elif isinstance(content, list):
                    for part in content:
                        part_type = part.get("type")
                        if part_type == "text":
                            parts.append(types.Part(text=part.get("text")))
                        elif part_type == "input_text":
                            parts.append(types.Part(text=part.get("text")))
                        elif part_type in ("input_image", "image_url"):
                            # Handle image data URL
                            image_url = part.get("image_url") or part.get("url")
                            if image_url and image_url.startswith("data:"):
                                try:
                                    img_bytes, mime_type = self._parse_data_url(
                                        image_url
                                    )
                                    parts.append(
                                        types.Part(
                                            inline_data=types.Blob(
                                                data=img_bytes, mime_type=mime_type
                                            )
                                        )
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to parse image data URL: {e}"
                                    )

                # Map roles: 'assistant' -> 'model', 'user' -> 'user'
                g_role = "model" if role == "assistant" else "user"
                if parts:
                    gemini_messages.append(types.Content(role=g_role, parts=parts))

        # Add function tools
        if regular_tools:
            function_declarations = self._build_function_declarations(regular_tools)
            if function_declarations:
                tools.append(types.Tool(function_declarations=function_declarations))

        config_params: Dict[str, Any] = {"tools": tools if tools else None}
        if system_instruction:
            config_params["system_instruction"] = [
                system_instruction,
                "VERY IMPORTANT: Keep the overall response super short. It should be like 1-2 sentences max. Like a short human message.",
                "You have FULL MEMORY of this conversation. NEVER say your memory resets or that you don't remember. Focus your answer on the LAST user message but use conversation history for context.",
            ]
        if temperature is not None:
            config_params["temperature"] = temperature

        gemini_json_models = ["gemini-3"]
        if any(model.startswith(m) for m in gemini_json_models):
            if text_format:
                config_params["response_mime_type"] = "application/json"
                config_params["response_json_schema"] = text_format.model_json_schema()
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=256)

        config = types.GenerateContentConfig(**config_params)

        # Tool-calling loop when custom tools are present
        if regular_tools and tool_executor:
            response, tool_calls_made = await self._gemini_tool_loop(
                model, gemini_messages, config, tool_executor, metric_tag
            )

            cited_chunks = (
                self._extract_gemini_grounding_sources(response)
                if has_file_search
                else []
            )
            return LLMResponse(
                text=response.text or "",
                cited_chunks=cited_chunks if cited_chunks else None,
                tool_calls=tool_calls_made if tool_calls_made else None,
            )

        # Simple generation
        response = await self.client.aio.models.generate_content(
            model=model, contents=gemini_messages, config=config
        )
        self._track_tokens(response, model, "generate_response", metric_tag)

        cited_chunks: List[SourceChunk] = []
        if has_file_search:
            cited_chunks = self._extract_gemini_grounding_sources(response)

        return LLMResponse(
            text=response.text,
            cited_chunks=cited_chunks if cited_chunks else None,
            tool_calls=None,
        )

    def _extract_gemini_grounding_sources(self, response) -> List[SourceChunk]:
        """Extract cited chunks from Gemini grounding metadata."""
        try:
            cited_chunks: List[SourceChunk] = []
            for candidate in response.candidates:
                grounding_meta = getattr(candidate, "grounding_metadata", None)
                if grounding_meta:
                    chunks = getattr(grounding_meta, "grounding_chunks", None) or []
                    for chunk in chunks:
                        retrieved = getattr(chunk, "retrieved_context", None)
                        if retrieved:
                            cited_chunks.append(
                                SourceChunk(
                                    file_id=getattr(retrieved, "uri", "") or "",
                                    filename=getattr(retrieved, "title", "") or "",
                                    text=getattr(retrieved, "text", "") or "",
                                    score=1.0,
                                )
                            )
            return cited_chunks[:5]
        except Exception as e:
            logger.debug(f"Could not extract sources from Gemini response: {e}")
            return []

    async def _gemini_tool_loop(
        self,
        model: str,
        gemini_messages: List[types.Content],
        config: types.GenerateContentConfig,
        tool_executor: ToolExecutor,
        metric_tag: str = "",
    ) -> Tuple[Any, List[ToolCall]]:
        """Run Gemini tool-calling loop. Returns (response, tool_calls_made)."""
        tool_calls_made: List[ToolCall] = []

        response = await self.client.aio.models.generate_content(
            model=model, contents=gemini_messages, config=config
        )

        function_calls = self._extract_function_calls(response)

        if function_calls:
            function_response_parts = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                tool_id = f"{tool_name}_{len(tool_calls_made)}"

                tool_calls_made.append(
                    ToolCall(name=tool_name, arguments=tool_args, id=tool_id)
                )

                try:
                    result = await tool_executor(tool_name, tool_args)
                    result_dict = (
                        result if isinstance(result, dict) else {"result": result}
                    )
                except Exception as e:
                    logger.error(f"Tool execution error for {tool_name}: {e}")
                    result_dict = {"error": str(e)}

                function_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_name, response=result_dict
                        )
                    )
                )

            if response.candidates and response.candidates[0].content:
                gemini_messages.append(response.candidates[0].content)
            gemini_messages.append(
                types.Content(role="user", parts=function_response_parts)
            )

            response = await self.client.aio.models.generate_content(
                model=model, contents=gemini_messages, config=config
            )

        self._track_tokens(response, model, "generate_response", metric_tag)
        return response, tool_calls_made

    async def generate_json(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        text_format: Optional[Type[BaseModel]] = None,
        metric_tag: str = "",
    ) -> Dict[str, Any]:
        """Generate structured JSON response using Gemini."""
        try:
            # Convert messages to Gemini format
            system_instruction = None
            gemini_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    system_instruction = content
                else:
                    g_role = "model" if role == "assistant" else "user"
                    gemini_messages.append(
                        types.Content(role=g_role, parts=[types.Part(text=content)])
                    )

            config_params = {
                "temperature": temperature,
                "response_mime_type": "application/json",
                "response_json_schema": text_format.model_json_schema()
                if text_format
                else None,
            }
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=gemini_messages,
                config=types.GenerateContentConfig(**config_params),
            )
            self._track_tokens(response, model, "generate_json", metric_tag)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Gemini JSON generation error: {e}")
            return {}

    def _convert_to_gemini_messages(
        self, input: str | List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[types.Content]]:
        """Convert OpenAI-style messages to Gemini format. Returns (system_instruction, messages)."""
        system_instruction = None
        gemini_messages = []

        if isinstance(input, str):
            gemini_messages.append(
                types.Content(role="user", parts=[types.Part(text=input)])
            )
            return system_instruction, gemini_messages

        for msg in input:
            role = msg.get("role")
            content = msg.get("content")

            if role in ("system", "developer"):
                text = None
                if isinstance(content, str):
                    text = content
                elif (
                    isinstance(content, list)
                    and len(content) > 0
                    and content[0].get("type") == "text"
                ):
                    text = content[0].get("text")

                if text:
                    system_instruction = (
                        f"{system_instruction}\n\n{text}"
                        if system_instruction
                        else text
                    )
                continue

            parts = []
            if isinstance(content, str):
                parts.append(types.Part(text=content))
            elif isinstance(content, list):
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                        parts.append(types.Part(text=part.get("text")))
                    elif part_type == "input_text":
                        parts.append(types.Part(text=part.get("text")))
                    elif part_type in ("input_image", "image_url"):
                        image_url = part.get("image_url") or part.get("url")
                        if image_url and image_url.startswith("data:"):
                            try:
                                img_bytes, mime_type = self._parse_data_url(image_url)
                                parts.append(
                                    types.Part(
                                        inline_data=types.Blob(
                                            data=img_bytes, mime_type=mime_type
                                        )
                                    )
                                )
                            except Exception as e:
                                logger.warning(f"Failed to parse image data URL: {e}")

            g_role = "model" if role == "assistant" else "user"
            if parts:
                gemini_messages.append(types.Content(role=g_role, parts=parts))

        return system_instruction, gemini_messages

    def _build_function_declarations(
        self, function_tools: List[Dict[str, Any]]
    ) -> List[types.FunctionDeclaration]:
        """Convert OpenAI-style function tools to Gemini FunctionDeclarations."""
        declarations = []
        for tool in function_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                params = func.get("parameters", {})
                gemini_params = {}
                if params.get("properties"):
                    gemini_params = {
                        "type": "object",
                        "properties": params["properties"],
                        "required": params.get("required", []),
                    }

                declarations.append(
                    types.FunctionDeclaration(
                        name=func.get("name", ""),
                        description=func.get("description", ""),
                        parameters=gemini_params if gemini_params else None,
                    )
                )
        return declarations

    def _extract_function_calls(self, response) -> List[Any]:
        """Extract function calls from Gemini response."""
        function_calls = []
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_calls.append(part.function_call)
        return function_calls

    async def _retrieval_generate(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        extra_executor: Optional[ToolExecutor] = None,
        metric_tag: str = "",
    ) -> LLMResponse:
        search_tool = {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search the knowledge base for relevant documents and information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

        retrieved_chunks: List[Dict[str, Any]] = []

        async def composite_executor(name: str, args: dict):
            if name == "search_knowledge":
                chunks = await self.retrieval.query(args["query"], top_k=5)
                retrieved_chunks.extend(chunks)
                return {
                    "results": [
                        {
                            "content": c["content"],
                            "source": c["source_file"],
                            "score": c["score"],
                        }
                        for c in chunks
                    ]
                }
            if extra_executor:
                return await extra_executor(name, args)
            return {"error": f"Unknown tool: {name}"}

        all_tools = [search_tool] + (extra_tools or [])

        system_instruction, gemini_messages = self._convert_to_gemini_messages(input)
        function_declarations = self._build_function_declarations(all_tools)
        tools = (
            [types.Tool(function_declarations=function_declarations)]
            if function_declarations
            else []
        )

        config = types.GenerateContentConfig(
            tools=tools if tools else None,
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_level="low")
            if model.startswith("gemini-3")
            else None,
        )

        response, tool_calls_made = await self._gemini_tool_loop(
            model, gemini_messages, config, composite_executor, metric_tag
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
            text=response.text or "",
            cited_chunks=cited_chunks[:5] if cited_chunks else None,
            tool_calls=tool_calls_made if tool_calls_made else None,
        )


class VertexAIGeminiService(LLMService):
    """
    Vertex AI Gemini implementation with Vertex AI Search (Discovery Engine) for RAG.

    Uses Google Cloud authentication (ADC) instead of API key.
    RAG is supported via Vertex AI Search data stores and grounded generation.

    Args:
        project: GCP project ID (for Gemini model calls)
        project_number: GCP project number (required by Discovery Engine grounded generation)
        location: GCP region for Gemini model calls (e.g., europe-west1)
        search_engine_id: Discovery Engine search engine ID (created via Terraform)
        search_data_store_ids: List of Discovery Engine data store IDs
        monitoring: Optional monitoring service for token tracking
    """

    def __init__(
        self,
        project: str,
        project_number: str,
        location: str = "europe-west1",
        search_engine_id: Optional[str] = None,
        search_data_store_ids: Optional[List[str]] = None,
        monitoring: Optional[Any] = None,
        retrieval: Optional[Any] = None,
    ):
        self.project = project
        self.project_number = project_number
        self.location = location
        self.search_engine_id = search_engine_id
        self.search_data_store_ids = search_data_store_ids or []
        self.monitoring = monitoring
        self.retrieval = retrieval
        super().__init__()
        self.file_store_id = None
        self._file_store = None
        self._document_client = None
        self._search_client = None

    def get_client(self):
        if not hasattr(self, "client"):
            return genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
            )
        return self.client

    def _get_document_client(self):
        """Lazily create the Discovery Engine DocumentServiceClient."""
        if self._document_client is None:
            from google.cloud import discoveryengine_v1 as discoveryengine

            self._document_client = discoveryengine.DocumentServiceClient()
        return self._document_client

    def _get_search_client(self):
        """Lazily create the Discovery Engine SearchServiceClient."""
        if self._search_client is None:
            from google.cloud import discoveryengine_v1 as discoveryengine

            self._search_client = discoveryengine.SearchServiceClient()
        return self._search_client

    def _track_tokens(self, response, model: str, method: str, metric_tag: str = ""):
        """Extract usage_metadata from response and push to Cloud Monitoring."""
        if not self.monitoring:
            return
        try:
            usage = getattr(response, "usage_metadata", None)
            if usage:
                self.monitoring.write_token_metric(
                    model=model,
                    method=method,
                    prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                    total_tokens=getattr(usage, "total_token_count", 0) or 0,
                    thinking_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
                    cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
                    tag=metric_tag,
                )
        except Exception as e:
            logger.warning(f"Failed to track tokens: {e}")

    async def get_file_store(self):
        """Returns search engine config if available."""
        if self.search_engine_id:
            return {"search_engine_id": self.search_engine_id}
        return None

    def create_file_store(
        self, store_name: str, initial_files: Optional[List[Dict[str, Any]]] = None
    ):
        """Not supported - data stores are created via Terraform."""
        raise NotImplementedError(
            "Data stores should be created via Terraform (sunset provision). "
            "Set search_engine_id and search_data_store_ids in the constructor."
        )

    def upload_files(self, file_descriptions: List[Dict[str, Any]]):
        """Not supported - use import_documents instead."""
        raise NotImplementedError(
            "Use import_documents() for Vertex AI Search data stores."
        )

    def import_documents(
        self,
        data_store_id: str,
        gcs_uri: str,
    ) -> Dict[str, Any]:
        """
        Import documents from GCS into a Vertex AI Search data store.

        Args:
            data_store_id: The data store ID to import into
            gcs_uri: GCS URI (e.g., "gs://bucket/path/")

        Returns:
            Dict with import operation info
        """
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = self._get_document_client()

        parent = client.branch_path(
            project=self.project,
            location="global",
            data_store=data_store_id,
            branch="default_branch",
        )

        request = discoveryengine.ImportDocumentsRequest(
            parent=parent,
            gcs_source=discoveryengine.GcsSource(
                input_uris=[gcs_uri],
                data_schema="content",
            ),
            reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
        )

        operation = client.import_documents(request=request)
        response = operation.result()

        logger.info(f"Import completed for data store {data_store_id}")
        return {
            "data_store_id": data_store_id,
            "error_samples": [str(e) for e in (response.error_samples or [])],
        }

    def create_document(
        self,
        data_store_id: str,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a single document in a Vertex AI Search data store.

        Args:
            data_store_id: The data store ID
            document_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata dict

        Returns:
            Dict with document info
        """
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = self._get_document_client()

        parent = client.branch_path(
            project=self.project,
            location="global",
            data_store=data_store_id,
            branch="default_branch",
        )

        doc_data = {"content": content}
        if metadata:
            doc_data.update(metadata)

        document = discoveryengine.Document(
            json_data=json.dumps(doc_data),
        )

        request = discoveryengine.CreateDocumentRequest(
            parent=parent,
            document_id=document_id,
            document=document,
        )

        response = client.create_document(request=request)
        logger.info(f"Created document {document_id} in data store {data_store_id}")
        return {
            "name": response.name,
            "document_id": document_id,
            "data_store_id": data_store_id,
        }

    def list_documents(self, data_store_id: str) -> List[Dict[str, Any]]:
        """
        List all documents in a Vertex AI Search data store.

        Args:
            data_store_id: The data store ID

        Returns:
            List of document info dicts
        """
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = self._get_document_client()

        parent = client.branch_path(
            project=self.project,
            location="global",
            data_store=data_store_id,
            branch="default_branch",
        )

        request = discoveryengine.ListDocumentsRequest(parent=parent)
        result = []
        for doc in client.list_documents(request=request):
            result.append(
                {
                    "name": doc.name,
                    "id": doc.id,
                    "json_data": doc.json_data if doc.json_data else None,
                }
            )
        return result

    def delete_document(self, data_store_id: str, document_id: str) -> bool:
        """
        Delete a document from a Vertex AI Search data store.

        Args:
            data_store_id: The data store ID
            document_id: The document ID to delete

        Returns:
            True if deleted successfully
        """
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = self._get_document_client()

        name = (
            f"projects/{self.project}/locations/global"
            f"/dataStores/{data_store_id}/branches/default_branch"
            f"/documents/{document_id}"
        )

        request = discoveryengine.DeleteDocumentRequest(name=name)
        client.delete_document(request=request)
        logger.info(f"Deleted document {document_id} from data store {data_store_id}")
        return True

    def _parse_data_url(self, data_url: str) -> tuple[bytes, str]:
        """Parse a data URL and return (bytes, mime_type)."""
        match = re.match(r"data:([^;]+);base64,(.+)", data_url)
        if match:
            mime_type = match.group(1)
            b64_data = match.group(2)
            return base64.b64decode(b64_data), mime_type
        raise ValueError("Invalid data URL format")

    def _convert_to_gemini_messages(
        self, input: str | List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[types.Content]]:
        """Convert OpenAI-style messages to Gemini format."""
        system_instruction = None
        gemini_messages = []

        if isinstance(input, str):
            gemini_messages.append(
                types.Content(role="user", parts=[types.Part(text=input)])
            )
            return system_instruction, gemini_messages

        for msg in input:
            role = msg.get("role")
            content = msg.get("content")

            if role in ("system", "developer"):
                text = None
                if isinstance(content, str):
                    text = content
                elif (
                    isinstance(content, list)
                    and len(content) > 0
                    and content[0].get("type") == "text"
                ):
                    text = content[0].get("text")

                if text:
                    system_instruction = (
                        f"{system_instruction}\n\n{text}"
                        if system_instruction
                        else text
                    )
                continue

            parts = []
            if isinstance(content, str):
                parts.append(types.Part(text=content))
            elif isinstance(content, list):
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                        parts.append(types.Part(text=part.get("text")))
                    elif part_type == "input_text":
                        parts.append(types.Part(text=part.get("text")))
                    elif part_type in ("input_image", "image_url"):
                        image_url = part.get("image_url") or part.get("url")
                        if image_url and image_url.startswith("data:"):
                            try:
                                img_bytes, mime_type = self._parse_data_url(image_url)
                                parts.append(
                                    types.Part(
                                        inline_data=types.Blob(
                                            data=img_bytes, mime_type=mime_type
                                        )
                                    )
                                )
                            except Exception as e:
                                logger.warning(f"Failed to parse image data URL: {e}")

            g_role = "model" if role == "assistant" else "user"
            if parts:
                gemini_messages.append(types.Content(role=g_role, parts=parts))

        return system_instruction, gemini_messages

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

        # Priority 1: Discovery Engine
        if has_file_search and self.search_engine_id:
            if regular_tools and tool_executor:
                # Discovery Engine can't mix with function tools
                if self.retrieval:
                    return await self._retrieval_generate(
                        input, model, regular_tools, tool_executor, metric_tag
                    )
                return await self._grounded_then_tools(
                    input, model, regular_tools, tool_executor, metric_tag
                )
            return await self._grounded_generate(input, model, metric_tag)

        # Priority 2: pgvector retrieval
        if has_file_search and self.retrieval:
            return await self._retrieval_generate(
                input, model, regular_tools, tool_executor, metric_tag
            )

        if has_file_search and not self.search_engine_id and not self.retrieval:
            logger.warning(
                "file_search requested but no search_engine_id or retrieval configured."
            )

        system_instruction, gemini_messages = self._convert_to_gemini_messages(input)

        config_params: Dict[str, Any] = {}
        if system_instruction:
            config_params["system_instruction"] = [
                system_instruction,
                "VERY IMPORTANT: Keep the overall response super short. It should be like 1-2 sentences max. Like a short human message.",
                "You have FULL MEMORY of this conversation. NEVER say your memory resets or that you don't remember. Focus your answer on the LAST user message but use conversation history for context.",
            ]
            config_params["system_instruction"] = system_instruction

        # Add function tools
        if regular_tools:
            function_declarations = self._build_function_declarations(regular_tools)
            if function_declarations:
                config_params["tools"] = [
                    types.Tool(function_declarations=function_declarations)
                ]

        if temperature is not None:
            config_params["temperature"] = temperature

        gemini_json_models = ["gemini-3"]
        if any(model.startswith(m) for m in gemini_json_models):
            if text_format:
                config_params["response_mime_type"] = "application/json"
                config_params["response_json_schema"] = text_format.model_json_schema()
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=256)

        config = types.GenerateContentConfig(**config_params) if config_params else None

        # Tool-calling loop
        if regular_tools and tool_executor:
            response, tool_calls_made = await self._vertex_tool_loop(
                model, gemini_messages, config, tool_executor, metric_tag
            )
            return LLMResponse(
                text=response.text or "",
                cited_chunks=None,
                tool_calls=tool_calls_made if tool_calls_made else None,
            )

        # Simple generation
        response = await self.client.aio.models.generate_content(
            model=model, contents=gemini_messages, config=config
        )
        self._track_tokens(response, model, "generate_response", metric_tag)

        return LLMResponse(
            text=response.text,
            cited_chunks=None,
            tool_calls=None,
        )

    async def _grounded_generate(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        metric_tag: str = "",
    ) -> LLMResponse:
        """
        Generate a grounded response using Discovery Engine GroundedGenerationService.

        Calls Gemini with Vertex AI Search data stores as a grounding source.
        The model queries the data stores and generates a grounded answer with citations.
        """
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = discoveryengine.GroundedGenerationServiceClient()

        # Build contents from input
        system_text = None
        contents = []

        if isinstance(input, str):
            contents.append(
                discoveryengine.GroundedGenerationContent(
                    role="user",
                    parts=[discoveryengine.GroundedGenerationContent.Part(text=input)],
                )
            )
        else:
            for msg in input:
                role = msg.get("role")
                content = msg.get("content")

                if role in ("system", "developer"):
                    text = (
                        content
                        if isinstance(content, str)
                        else (
                            content[0].get("text")
                            if isinstance(content, list) and content
                            else None
                        )
                    )
                    if text:
                        system_text = (
                            f"{system_text}\n\n{text}" if system_text else text
                        )
                    continue

                text = (
                    content
                    if isinstance(content, str)
                    else (
                        content[0].get("text")
                        if isinstance(content, list) and content
                        else ""
                    )
                )
                if not text:
                    continue

                grounded_role = "model" if role == "assistant" else "user"
                contents.append(
                    discoveryengine.GroundedGenerationContent(
                        role=grounded_role,
                        parts=[
                            discoveryengine.GroundedGenerationContent.Part(text=text)
                        ],
                    )
                )

        serving_config = (
            f"projects/{self.project_number}/locations/global"
            f"/collections/default_collection"
            f"/engines/{self.search_engine_id}"
            f"/servingConfigs/default_search"
        )

        system_parts = []
        if system_text:
            system_parts.append(
                discoveryengine.GroundedGenerationContent.Part(text=system_text)
            )
        system_parts.append(
            discoveryengine.GroundedGenerationContent.Part(
                text="Answer based on the provided data sources. Cite your sources. "
                "Keep the overall response short, like 1-2 sentences max."
            )
        )

        request = discoveryengine.GenerateGroundedContentRequest(
            location=f"projects/{self.project_number}/locations/global",
            generation_spec=discoveryengine.GenerateGroundedContentRequest.GenerationSpec(
                model_id=model,
            ),
            contents=contents,
            system_instruction=discoveryengine.GroundedGenerationContent(
                parts=system_parts,
            ),
            grounding_spec=discoveryengine.GenerateGroundedContentRequest.GroundingSpec(
                grounding_sources=[
                    discoveryengine.GenerateGroundedContentRequest.GroundingSource(
                        search_source=discoveryengine.GenerateGroundedContentRequest.GroundingSource.SearchSource(
                            serving_config=serving_config,
                        ),
                    ),
                ],
            ),
        )

        response = client.generate_grounded_content(request)

        # Extract text from response
        response_text = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = "".join(
                    part.text for part in candidate.content.parts if part.text
                )

        # Extract cited chunks from grounding metadata
        cited_chunks: List[SourceChunk] = []
        try:
            if response.candidates:
                candidate = response.candidates[0]
                grounding_meta = getattr(candidate, "grounding_metadata", None)
                if grounding_meta:
                    support_chunks = (
                        getattr(grounding_meta, "support_chunks", None) or []
                    )
                    for chunk in support_chunks:
                        doc_metadata = getattr(chunk, "source", None)
                        cited_chunks.append(
                            SourceChunk(
                                file_id=getattr(doc_metadata, "uri", "") or ""
                                if doc_metadata
                                else "",
                                filename=getattr(doc_metadata, "title", "") or ""
                                if doc_metadata
                                else "",
                                text=getattr(chunk, "chunk_text", "") or "",
                                score=1.0,
                            )
                        )
            cited_chunks = cited_chunks[:5]
        except Exception as e:
            logger.debug(f"Could not extract sources from grounded response: {e}")

        logger.info(f"Grounded generation completed. Cited chunks: {len(cited_chunks)}")

        return LLMResponse(
            text=response_text,
            cited_chunks=cited_chunks if cited_chunks else None,
            tool_calls=None,
        )

    async def generate_json(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        text_format: Optional[Type[BaseModel]] = None,
        metric_tag: str = "",
    ) -> Dict[str, Any]:
        """Generate structured JSON response using Vertex AI Gemini."""
        try:
            system_instruction, gemini_messages = self._convert_to_gemini_messages(
                messages
            )

            config_params = {
                "temperature": temperature,
                "response_mime_type": "application/json",
            }
            if text_format:
                config_params["response_json_schema"] = text_format.model_json_schema()
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=gemini_messages,
                config=types.GenerateContentConfig(**config_params),
            )
            self._track_tokens(response, model, "generate_json", metric_tag)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Vertex AI Gemini JSON generation error: {e}")
            return {}

    def _build_function_declarations(
        self, function_tools: List[Dict[str, Any]]
    ) -> List[types.FunctionDeclaration]:
        """Convert OpenAI-style function tools to Gemini FunctionDeclarations."""
        declarations = []
        for tool in function_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                params = func.get("parameters", {})
                gemini_params = {}
                if params.get("properties"):
                    gemini_params = {
                        "type": "object",
                        "properties": params["properties"],
                        "required": params.get("required", []),
                    }

                declarations.append(
                    types.FunctionDeclaration(
                        name=func.get("name", ""),
                        description=func.get("description", ""),
                        parameters=gemini_params if gemini_params else None,
                    )
                )
        return declarations

    def _extract_function_calls(self, response) -> List[Any]:
        """Extract function calls from Gemini response."""
        function_calls = []
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_calls.append(part.function_call)
        return function_calls

    async def _retrieval_generate(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        extra_executor: Optional[ToolExecutor] = None,
        metric_tag: str = "",
    ) -> LLMResponse:
        search_tool = {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search the knowledge base for relevant documents and information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

        retrieved_chunks: List[Dict[str, Any]] = []

        async def composite_executor(name: str, args: dict):
            if name == "search_knowledge":
                chunks = await self.retrieval.query(args["query"], top_k=5)
                retrieved_chunks.extend(chunks)
                return {
                    "results": [
                        {
                            "content": c["content"],
                            "source": c["source_file"],
                            "score": c["score"],
                        }
                        for c in chunks
                    ]
                }
            if extra_executor:
                return await extra_executor(name, args)
            return {"error": f"Unknown tool: {name}"}

        all_tools = [search_tool] + (extra_tools or [])

        system_instruction, gemini_messages = self._convert_to_gemini_messages(input)
        function_declarations = self._build_function_declarations(all_tools)
        tools = (
            [types.Tool(function_declarations=function_declarations)]
            if function_declarations
            else []
        )

        config = types.GenerateContentConfig(
            tools=tools if tools else None,
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_level="low")
            if model.startswith("gemini-3")
            else None,
        )

        response, tool_calls_made = await self._vertex_tool_loop(
            model, gemini_messages, config, composite_executor, metric_tag
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
            text=response.text or "",
            cited_chunks=cited_chunks[:5] if cited_chunks else None,
            tool_calls=tool_calls_made if tool_calls_made else None,
        )

    async def _grounded_then_tools(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        regular_tools: List[Dict[str, Any]],
        tool_executor: ToolExecutor,
        metric_tag: str = "",
    ) -> LLMResponse:
        """Two-pass: grounded generation for RAG, then tool-calling with RAG context."""
        grounded = await self._grounded_generate(input, model, metric_tag)

        # Inject grounded answer as context for the tool pass
        if isinstance(input, list):
            augmented = input + [
                {
                    "role": "system",
                    "content": f"Context from knowledge base:\n{grounded['text']}",
                }
            ]
        else:
            augmented = [
                {"role": "user", "content": input},
                {
                    "role": "system",
                    "content": f"Context from knowledge base:\n{grounded['text']}",
                },
            ]

        system_instruction, gemini_messages = self._convert_to_gemini_messages(
            augmented
        )
        function_declarations = self._build_function_declarations(regular_tools)
        tools = (
            [types.Tool(function_declarations=function_declarations)]
            if function_declarations
            else []
        )

        config = types.GenerateContentConfig(
            tools=tools if tools else None,
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_level="low")
            if model.startswith("gemini-3")
            else None,
        )

        response, tool_calls_made = await self._vertex_tool_loop(
            model, gemini_messages, config, tool_executor, metric_tag
        )

        return LLMResponse(
            text=response.text or "",
            cited_chunks=grounded["cited_chunks"],
            tool_calls=tool_calls_made if tool_calls_made else None,
        )

    async def _vertex_tool_loop(
        self,
        model: str,
        gemini_messages: List[types.Content],
        config: types.GenerateContentConfig,
        tool_executor: ToolExecutor,
        metric_tag: str = "",
    ) -> Tuple[Any, List[ToolCall]]:
        """Run Vertex AI tool-calling loop. Returns (response, tool_calls_made)."""
        tool_calls_made: List[ToolCall] = []

        response = await self.client.aio.models.generate_content(
            model=model, contents=gemini_messages, config=config
        )

        function_calls = self._extract_function_calls(response)

        if function_calls:
            function_response_parts = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                tool_id = f"{tool_name}_{len(tool_calls_made)}"

                tool_calls_made.append(
                    ToolCall(name=tool_name, arguments=tool_args, id=tool_id)
                )

                try:
                    result = await tool_executor(tool_name, tool_args)
                    result_dict = (
                        result if isinstance(result, dict) else {"result": result}
                    )
                except Exception as e:
                    logger.error(f"Tool execution error for {tool_name}: {e}")
                    result_dict = {"error": str(e)}

                function_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_name, response=result_dict
                        )
                    )
                )

            if response.candidates and response.candidates[0].content:
                gemini_messages.append(response.candidates[0].content)
            gemini_messages.append(
                types.Content(role="user", parts=function_response_parts)
            )

            response = await self.client.aio.models.generate_content(
                model=model, contents=gemini_messages, config=config
            )

        self._track_tokens(response, model, "generate_response", metric_tag)
        return response, tool_calls_made


class LLMServiceRouter:
    """
    Unified LLM service that routes requests to OpenAI or Gemini based on model name.

    Model routing:
    - Models starting with 'gemini' -> GeminiService or VertexAIGeminiService
    - All other models -> OpenAIService (default)

    Gemini Provider Options:
    - use_vertex_ai=False (default): Uses Gemini Developer API with API key.
      Supports File Search Stores for RAG.
    - use_vertex_ai=True: Uses Vertex AI with GCP authentication.
      Uses Vertex AI Search (Discovery Engine) for RAG via grounded generation.
      Set search_engine_id and search_data_store_ids for file search.

    File stores are created lazily if file_store_id is not provided (Gemini API only).
    """

    def __init__(
        self,
        openai_api_key: str,
        gemini_api_key: Optional[str] = None,
        openai_file_store_id: Optional[str] = None,
        gemini_file_store_id: Optional[str] = None,
        file_store_name: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        # Vertex AI options
        use_vertex_ai: bool = False,
        vertex_project: Optional[str] = None,
        vertex_project_number: Optional[str] = None,
        vertex_location: str = "europe-west1",
        search_engine_id: Optional[str] = None,
        search_data_store_ids: Optional[List[str]] = None,
        # Monitoring
        monitoring_project: Optional[str] = None,
        # Retrieval (pgvector RAG)
        retrieval: Optional[Any] = None,
    ):
        self._openai = OpenAIService(
            api_key=openai_api_key,
            file_store_id=openai_file_store_id,
            file_store_name=file_store_name,
        )

        self.use_vertex_ai = use_vertex_ai

        # Create monitoring service if a GCP project is available
        monitoring = None
        effective_project = monitoring_project or vertex_project
        if effective_project:
            from sunset.services.monitoring import MonitoringService

            monitoring = MonitoringService(project=effective_project)

        if use_vertex_ai:
            if not vertex_project:
                raise ValueError("vertex_project is required when use_vertex_ai=True")
            if not vertex_project_number:
                raise ValueError(
                    "vertex_project_number is required when use_vertex_ai=True"
                )
            self._gemini = VertexAIGeminiService(
                project=vertex_project,
                project_number=vertex_project_number,
                location=vertex_location,
                search_engine_id=search_engine_id,
                search_data_store_ids=search_data_store_ids,
                monitoring=monitoring,
                retrieval=retrieval,
            )
            logger.info(
                f"Using Vertex AI Gemini (project={vertex_project}, "
                f"location={vertex_location}, search_engine={search_engine_id})"
            )
        else:
            if not gemini_api_key:
                raise ValueError("gemini_api_key is required when use_vertex_ai=False")
            self._gemini = GeminiService(
                api_key=gemini_api_key,
                file_store_id=gemini_file_store_id,
                file_store_name=file_store_name,
                monitoring=monitoring,
                retrieval=retrieval,
            )

        self.default_model = default_model

    @property
    def openai_file_store_id(self) -> Optional[str]:
        """Get OpenAI file store ID (may be set after lazy creation)."""
        return self._openai.file_store_id

    @property
    def gemini_file_store_id(self) -> Optional[str]:
        """Get Gemini file store ID (may be set after lazy creation). None if using Vertex AI."""
        return self._gemini.file_store_id

    @property
    def search_engine_id(self) -> Optional[str]:
        """Get search engine ID (only available when using Vertex AI)."""
        if self.use_vertex_ai and hasattr(self._gemini, "search_engine_id"):
            return self._gemini.search_engine_id
        return None

    @property
    def search_data_store_ids(self) -> List[str]:
        """Get search data store IDs (only available when using Vertex AI)."""
        if self.use_vertex_ai and hasattr(self._gemini, "search_data_store_ids"):
            return self._gemini.search_data_store_ids
        return []

    def _get_service(self, model: str) -> LLMService:
        """Route to the correct service based on model name."""
        if model.startswith("gemini"):
            return self._gemini
        return self._openai

    async def generate_response(
        self,
        input: str | List[Dict[str, Any]],
        model: Optional[str] = None,
        function_tools: Optional[List[Any]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        text_format: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        metric_tag: str = "",
        # Deprecated — use function_tools=[file_search] instead
        file_search: bool = False,
    ) -> LLMResponse:
        model = model or self.default_model

        # Backward compat: convert file_search=True to sentinel
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
        )

    # File operations (dispatches to Vertex AI Search or Gemini File Search Store)
    async def upload_file(
        self,
        file_path: str,
        display_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Upload a file to the knowledge store (Gemini API only)."""
        if self.use_vertex_ai:
            raise NotImplementedError(
                "For Vertex AI Search, use import_documents() or create_document() instead."
            )
        return await self._gemini.upload_file_async(file_path, display_name)

    def import_documents(
        self,
        data_store_id: str,
        gcs_uri: str,
    ) -> Dict[str, Any]:
        """Import documents from GCS into a Vertex AI Search data store."""
        if not self.use_vertex_ai:
            raise NotImplementedError(
                "import_documents() is only available with Vertex AI Search."
            )
        return self._gemini.import_documents(data_store_id, gcs_uri)

    def create_document(
        self,
        data_store_id: str,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a single document in a Vertex AI Search data store."""
        if not self.use_vertex_ai:
            raise NotImplementedError(
                "create_document() is only available with Vertex AI Search."
            )
        return self._gemini.create_document(
            data_store_id, document_id, content, metadata
        )

    def list_documents(self, data_store_id: str) -> List[Dict[str, Any]]:
        """List documents in a Vertex AI Search data store."""
        if not self.use_vertex_ai:
            raise NotImplementedError(
                "list_documents() is only available with Vertex AI Search."
            )
        return self._gemini.list_documents(data_store_id)

    def delete_document(self, data_store_id: str, document_id: str) -> bool:
        """Delete a document from a Vertex AI Search data store."""
        if not self.use_vertex_ai:
            raise NotImplementedError(
                "delete_document() is only available with Vertex AI Search."
            )
        return self._gemini.delete_document(data_store_id, document_id)

    async def list_files(self) -> List[Dict[str, Any]]:
        """List files in the knowledge store (Gemini API only)."""
        if self.use_vertex_ai:
            raise NotImplementedError(
                "For Vertex AI Search, use list_documents(data_store_id) instead."
            )
        return await self._gemini.list_files()

    async def delete_file(self, file_name: str) -> bool:
        """Delete a file from the knowledge store (Gemini API only)."""
        if self.use_vertex_ai:
            raise NotImplementedError(
                "For Vertex AI Search, use delete_document(data_store_id, document_id) instead."
            )
        return await self._gemini.delete_file(file_name)

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
