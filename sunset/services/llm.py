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


class LLMResponse(TypedDict):
    text: str
    sources: Optional[
        List[SourceChunk]
    ]  # Top 3 distinct files for display (score > 60%)
    cited_chunks: Optional[
        List[SourceChunk]
    ]  # Chunks with text from actually cited files (for evaluation)


class ToolCall(TypedDict):
    name: str
    arguments: Dict[str, Any]
    id: str


class ToolCallResponse(TypedDict):
    text: str
    sources: Optional[List[SourceChunk]]
    cited_chunks: Optional[
        List[SourceChunk]
    ]  # Chunks with text from actually cited files (for evaluation)
    tool_calls: Optional[List[ToolCall]]  # Tool calls that were made (for logging)


# Type for tool executor callback
ToolExecutor = Any  # Callable[[str, Dict[str, Any]], Any]


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
        file_search: bool = False,
        text_format: Optional[Type[BaseModel]] = None,
    ) -> LLMResponse:
        """
        Generate a response based on the conversation history or input string.

        Args:
            input: String or List of message dicts with 'role' and 'content'.
            model: Model identifier.
            file_search: Whether to enable file search capabilities.
            text_format: Optional Pydantic model for structured output (OpenAI only).

        Returns:
            LLMResponse with text and optional sources list.
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
        file_search: bool = False,
        text_format: Optional[Type[BaseModel]] = None,
    ) -> LLMResponse:
        tools = []
        reasoning = None
        include = []

        # Lazily retrieve file store if file_search is requested
        store = await self.get_file_store() if file_search else None

        if file_search and store:
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

        # Use responses.parse for structured output, responses.create otherwise
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

        # Extract sources and cited chunks from file search results
        sources: List[SourceChunk] = []
        cited_chunks: List[SourceChunk] = []
        if file_search:
            try:
                # 1. Extract cited file_ids from annotations
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

                # 2. Collect all file search results
                file_scores: dict[str, SourceChunk] = {}
                all_chunks: List[SourceChunk] = []
                for item in response.output:
                    if getattr(item, "type", None) == "file_search_call":
                        results = getattr(item, "results", None)
                        if results:
                            for chunk in results:
                                # Collect cited chunks with text for evaluation
                                if chunk.file_id in cited_file_ids:
                                    all_chunks.append(
                                        SourceChunk(
                                            file_id=chunk.file_id,
                                            filename=chunk.filename,
                                            text=chunk.text,
                                            score=chunk.score,
                                        )
                                    )

                                # Skip files with score < 60% for display sources
                                if chunk.score < 0.6:
                                    continue
                                # Keep best score per file for display
                                if (
                                    chunk.filename not in file_scores
                                    or chunk.score
                                    > file_scores[chunk.filename]["score"]
                                ):
                                    file_scores[chunk.filename] = SourceChunk(
                                        file_id=chunk.file_id,
                                        filename=chunk.filename,
                                        text="",
                                        score=chunk.score,
                                    )

                # 3. Sort and limit results
                sources = sorted(
                    file_scores.values(), key=lambda x: x["score"], reverse=True
                )[:3]
                all_chunks.sort(key=lambda x: x["score"], reverse=True)
                cited_chunks = all_chunks[:5]  # Top 5 cited chunks for evaluation
            except Exception as e:
                logger.debug(f"Could not extract sources from response: {e}")

        # When using text_format, serialize the parsed output to JSON
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
            sources=sources if sources else None,
            cited_chunks=cited_chunks if cited_chunks else None,
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

    async def generate_response_with_tools(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        function_tools: List[Dict[str, Any]],
        tool_executor: ToolExecutor,
    ) -> ToolCallResponse:
        """
        Generate a response with function calling support.

        Handles function tools only. For file search with citations, use generate_response first.
        """
        tools = list(function_tools)  # Copy to avoid mutation

        # Build conversation for multi-turn tool calling
        conversation = (
            input if isinstance(input, list) else [{"role": "user", "content": input}]
        )
        tool_calls_made: List[ToolCall] = []
        max_iterations = 3  # Prevent infinite loops

        for _ in range(max_iterations):
            response = await self.client.responses.create(
                model=model,
                input=conversation,
                tools=tools if tools else None,
            )

            # Check for function calls in output
            function_calls = [
                item
                for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]

            if not function_calls:
                # No more tool calls, we have the final response
                break

            # Process each function call
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

                # Execute the tool (supports async executors)
                try:
                    result = await tool_executor(tool_name, tool_args)
                    result_str = (
                        json.dumps(result) if not isinstance(result, str) else result
                    )
                except Exception as e:
                    logger.error(f"Tool execution error for {tool_name}: {e}")
                    result_str = json.dumps({"error": str(e)})

                # Add to conversation for next iteration
                conversation = response.output + [
                    {
                        "type": "function_call_output",
                        "call_id": tool_id,
                        "output": result_str,
                    }
                ]

        return ToolCallResponse(
            text=response.output_text,
            sources=None,
            cited_chunks=None,
            tool_calls=tool_calls_made if tool_calls_made else None,
        )


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
        file_search: bool = False,
        text_format: Optional[Type[BaseModel]] = None,
        metric_tag: str = "",
    ) -> LLMResponse:
        tools = []
        store = await self.get_file_store() if file_search else None

        if file_search and store:
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

        config_params = {"tools": tools}
        if system_instruction:
            config_params["system_instruction"] = [
                system_instruction,
                "VERY IMPORTANT: Keep the overall response super short. It should be like 1-2 sentences max. Like a short human message.",
                "You have FULL MEMORY of this conversation. NEVER say your memory resets or that you don't remember. Focus your answer on the LAST user message but use conversation history for context.",
            ]

        # Add structured output config for Gemini when text_format is provided
        # Only supported on certain models (e.g., gemini-2.5-pro-preview)
        gemini_json_models = ["gemini-3"]
        if any(model.startswith(m) for m in gemini_json_models):
            if text_format:
                config_params["response_mime_type"] = "application/json"
                config_params["response_json_schema"] = text_format.model_json_schema()
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=256)

        config = types.GenerateContentConfig(**config_params)

        response = await self.client.aio.models.generate_content(
            model=model, contents=gemini_messages, config=config
        )
        self._track_tokens(response, model, "generate_response", metric_tag)

        # Extract sources and cited chunks from grounding metadata
        sources: List[SourceChunk] = []
        cited_chunks: List[SourceChunk] = []
        if file_search:
            try:
                seen_files: set[str] = set()
                for candidate in response.candidates:
                    grounding_meta = getattr(candidate, "grounding_metadata", None)
                    if grounding_meta:
                        chunks = getattr(grounding_meta, "grounding_chunks", None) or []
                        for chunk in chunks:
                            retrieved = getattr(chunk, "retrieved_context", None)
                            if retrieved:
                                filename = getattr(retrieved, "title", "") or ""
                                # Text is in retrieved_context.text
                                text = getattr(retrieved, "text", "") or ""
                                file_id = getattr(retrieved, "uri", "") or ""

                                # All chunks for evaluation (with text)
                                cited_chunks.append(
                                    SourceChunk(
                                        file_id=file_id,
                                        filename=filename,
                                        text=text,
                                        score=1.0,
                                    )
                                )

                                # Distinct files for display
                                if filename and filename not in seen_files:
                                    seen_files.add(filename)
                                    sources.append(
                                        SourceChunk(
                                            file_id=file_id,
                                            filename=filename,
                                            text="",
                                            score=1.0,
                                        )
                                    )
                sources = sources[:3]
                cited_chunks = cited_chunks[:5]
            except Exception as e:
                logger.debug(f"Could not extract sources from Gemini response: {e}")

        return LLMResponse(
            text=response.text,
            sources=sources if sources else None,
            cited_chunks=cited_chunks if cited_chunks else None,
        )

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

    async def generate_response_with_tools(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        function_tools: List[Dict[str, Any]],
        tool_executor: ToolExecutor,
        temperature: Optional[float] = None,
        metric_tag: str = "",
    ) -> ToolCallResponse:
        """
        Generate a response with function calling support for Gemini.

        Handles function tools only. For file search with citations, use generate_response first.
        Simple flow: call model -> if tool requested, execute and get final response.
        """
        # Convert input to Gemini format
        system_instruction, gemini_messages = self._convert_to_gemini_messages(input)

        # Build function tools
        function_declarations = self._build_function_declarations(function_tools)
        tools = (
            [types.Tool(function_declarations=function_declarations)]
            if function_declarations
            else []
        )

        # Build system instruction, filtering out None
        # sys_instructions = [
        #     system_instruction,
        #     "VERY IMPORTANT: Keep the overall response super short. It should be like 1-2 sentences max. Like a short human message.",
        #     "You have FULL MEMORY of this conversation. NEVER say your memory resets or that you don't remember.",
        #     "If you end up recommending a product, you should also include the exact product URL in the response (doesn't count towards the max length).",
        # ]
        # sys_instructions = [s for s in sys_instructions if s]

        config = types.GenerateContentConfig(
            tools=tools if tools else None,
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_level="low")
            if model.startswith("gemini-3")
            else None,
            temperature=temperature,
        )

        tool_calls_made: List[ToolCall] = []

        # First call
        response = await self.client.aio.models.generate_content(
            model=model, contents=gemini_messages, config=config
        )

        logger.info(
            f"[tools] First response text: {response.text[:200] if response.text else 'None'}"
        )

        # Check for function calls
        function_calls = self._extract_function_calls(response)
        logger.info(
            f"[tools] Function calls detected: {[fc.name for fc in function_calls] if function_calls else 'None'}"
        )

        if function_calls:
            # Execute tools and build response parts
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

            # Add model response and tool results to conversation
            if response.candidates and response.candidates[0].content:
                gemini_messages.append(response.candidates[0].content)
            gemini_messages.append(
                types.Content(role="user", parts=function_response_parts)
            )

            # Get final response
            response = await self.client.aio.models.generate_content(
                model=model, contents=gemini_messages, config=config
            )
            logger.info(
                f"[tools] Final response after tool call - text: {response.text[:200] if response.text else 'None'}"
            )

        self._track_tokens(response, model, "generate_response_with_tools", metric_tag)

        # Log if response is empty (helps debug thinking-only responses)
        if not response.text:
            logger.warning(f"Empty response text. Candidates: {response.candidates}")

        return ToolCallResponse(
            text=response.text or "",
            sources=None,
            cited_chunks=None,
            tool_calls=tool_calls_made if tool_calls_made else None,
        )


class VertexAIGeminiService(LLMService):
    """
    Vertex AI Gemini implementation for GDPR-compliant, scalable deployments.

    Uses Google Cloud authentication (ADC) instead of API key.
    Data stays within the specified GCP region (e.g., europe-west1 for EU).

    RAG is supported via Vertex AI RAG Engine when rag_corpus_name is provided.
    Use file_search=True with rag_filter to query the RAG corpus.
    """

    def __init__(
        self,
        project: str,
        location: str = "europe-west1",
        rag_corpus_name: Optional[str] = None,
        monitoring: Optional[Any] = None,
    ):
        self.project = project
        self.location = location
        self.rag_corpus_name = rag_corpus_name
        self.monitoring = monitoring
        super().__init__()
        # Vertex AI doesn't support File Search Stores (uses RAG Engine instead)
        self.file_store_id = None
        self._file_store = None
        self._vertexai_initialized = False

    def get_client(self):
        if not hasattr(self, "client"):
            return genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
            )
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

    def _ensure_vertexai_init(self):
        """Initialize vertexai SDK if not already done."""
        if not self._vertexai_initialized:
            try:
                import vertexai

                vertexai.init(project=self.project, location=self.location)
                self._vertexai_initialized = True
            except ImportError:
                raise ImportError(
                    "google-cloud-aiplatform is required for RAG functions. "
                    "Install with: pip install google-cloud-aiplatform"
                )

    async def get_file_store(self):
        """Vertex AI uses RAG Engine instead of File Search Stores."""
        if self.rag_corpus_name:
            return {"rag_corpus_name": self.rag_corpus_name}
        return None

    def create_file_store(
        self, store_name: str, initial_files: Optional[List[Dict[str, Any]]] = None
    ):
        """Not supported - RAG corpus is created via Terraform."""
        raise NotImplementedError(
            "RAG corpus should be created via Terraform (sunset provision). "
            "Set rag_corpus_name in the constructor to use an existing corpus."
        )

    def upload_files(self, file_descriptions: List[Dict[str, Any]]):
        """Not supported - use upload_rag_file instead."""
        raise NotImplementedError("Use upload_rag_file() for Vertex AI RAG Engine.")

    def upload_rag_file(
        self,
        file_path: str,
        display_name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Upload a file to the RAG corpus with optional metadata.

        Args:
            file_path: Local path to the file
            display_name: Display name for the file in the corpus
            metadata: Optional metadata dict (e.g., {"doctor_id": "uuid"})

        Returns:
            Dict with file info including 'name' and 'display_name'
        """
        if not self.rag_corpus_name:
            raise ValueError("rag_corpus_name is required for RAG file operations")

        self._ensure_vertexai_init()
        from vertexai import rag

        try:
            rag_file = rag.upload_file(
                corpus_name=self.rag_corpus_name,
                path=file_path,
                display_name=display_name,
            )
            logger.info(f"Uploaded RAG file: {rag_file.name}")
            return {
                "name": rag_file.name,
                "display_name": display_name,
                "metadata": metadata,
            }
        except Exception as e:
            logger.error(f"Failed to upload RAG file: {e}")
            raise

    def list_rag_files(self) -> List[Dict[str, Any]]:
        """
        List all files in the RAG corpus.

        Returns:
            List of file info dicts
        """
        if not self.rag_corpus_name:
            raise ValueError("rag_corpus_name is required for RAG file operations")

        self._ensure_vertexai_init()
        from vertexai import rag

        try:
            files = list(rag.list_files(corpus_name=self.rag_corpus_name))
            return [
                {
                    "name": f.name,
                    "display_name": getattr(f, "display_name", ""),
                    "size_bytes": getattr(f, "size_bytes", 0),
                    "create_time": str(getattr(f, "create_time", "")),
                }
                for f in files
            ]
        except Exception as e:
            logger.error(f"Failed to list RAG files: {e}")
            raise

    def delete_rag_file(self, file_name: str) -> bool:
        """
        Delete a file from the RAG corpus.

        Args:
            file_name: Full resource name of the file

        Returns:
            True if deleted successfully
        """
        if not self.rag_corpus_name:
            raise ValueError("rag_corpus_name is required for RAG file operations")

        self._ensure_vertexai_init()
        from vertexai import rag

        try:
            rag.delete_file(name=file_name)
            logger.info(f"Deleted RAG file: {file_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete RAG file: {e}")
            raise

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
        file_search: bool = False,
        text_format: Optional[Type[BaseModel]] = None,
        rag_filter: Optional[Dict[str, str]] = None,
        metric_tag: str = "",
    ) -> LLMResponse:
        """
        Generate a response using Vertex AI Gemini.

        Args:
            input: String or conversation messages
            model: Model identifier
            file_search: If True and rag_corpus_name is set, enables RAG retrieval
            text_format: Optional Pydantic model for structured output
            rag_filter: Optional metadata filter for RAG (e.g., {"doctor_id": "uuid"})

        Returns:
            LLMResponse with text and optional sources
        """
        system_instruction, gemini_messages = self._convert_to_gemini_messages(input)

        config_params = {}
        tools = []

        # RAG via Vertex AI RAG Engine
        if file_search and self.rag_corpus_name:
            rag_resource = types.VertexRagStoreRagResource(
                rag_corpus=self.rag_corpus_name
            )
            rag_store_config = types.VertexRagStore(rag_resources=[rag_resource])

            # Add metadata filter if provided
            if rag_filter:
                # Build filter conditions for metadata
                rag_store_config.rag_retrieval_config = types.RagRetrievalConfig(
                    top_k=5,
                    filter=types.MetadataFilter(
                        key=list(rag_filter.keys())[0],
                        value=list(rag_filter.values())[0],
                    )
                    if len(rag_filter) == 1
                    else None,
                )

            tools.append(
                types.Tool(retrieval=types.Retrieval(vertex_rag_store=rag_store_config))
            )
            logger.info(f"RAG enabled with corpus: {self.rag_corpus_name}")
        elif file_search:
            logger.warning(
                "file_search=True but no rag_corpus_name configured. "
                "RAG retrieval disabled."
            )

        if system_instruction:
            config_params["system_instruction"] = [
                system_instruction,
                "VERY IMPORTANT: Keep the overall response super short. It should be like 1-2 sentences max. Like a short human message.",
                "You have FULL MEMORY of this conversation. NEVER say your memory resets or that you don't remember. Focus your answer on the LAST user message but use conversation history for context.",
            ]

        if tools:
            config_params["tools"] = tools

        # Add structured output config for supported models
        gemini_json_models = ["gemini-3"]
        if any(model.startswith(m) for m in gemini_json_models):
            if text_format:
                config_params["response_mime_type"] = "application/json"
                config_params["response_json_schema"] = text_format.model_json_schema()
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=256)

        config = types.GenerateContentConfig(**config_params) if config_params else None

        logger.info(f"Generate config {config}")
        logger.info(f"Model {model}")
        response = await self.client.aio.models.generate_content(
            model=model, contents=gemini_messages, config=config
        )
        self._track_tokens(response, model, "generate_response", metric_tag)

        # Extract sources from grounding metadata if RAG was used
        sources: List[SourceChunk] = []
        cited_chunks: List[SourceChunk] = []
        if file_search and self.rag_corpus_name:
            try:
                for candidate in response.candidates:
                    grounding_meta = getattr(candidate, "grounding_metadata", None)
                    if grounding_meta:
                        chunks = getattr(grounding_meta, "grounding_chunks", None) or []
                        seen_files: set[str] = set()
                        for chunk in chunks:
                            retrieved = getattr(chunk, "retrieved_context", None)
                            if retrieved:
                                filename = getattr(retrieved, "title", "") or ""
                                text = getattr(retrieved, "text", "") or ""
                                file_id = getattr(retrieved, "uri", "") or ""

                                cited_chunks.append(
                                    SourceChunk(
                                        file_id=file_id,
                                        filename=filename,
                                        text=text,
                                        score=1.0,
                                    )
                                )

                                if filename and filename not in seen_files:
                                    seen_files.add(filename)
                                    sources.append(
                                        SourceChunk(
                                            file_id=file_id,
                                            filename=filename,
                                            text="",
                                            score=1.0,
                                        )
                                    )
                sources = sources[:3]
                cited_chunks = cited_chunks[:5]
            except Exception as e:
                logger.debug(f"Could not extract sources from RAG response: {e}")

        return LLMResponse(
            text=response.text,
            sources=sources if sources else None,
            cited_chunks=cited_chunks if cited_chunks else None,
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

    async def generate_response_with_tools(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        function_tools: List[Dict[str, Any]],
        tool_executor: Any,
        temperature: Optional[float] = None,
        metric_tag: str = "",
    ) -> ToolCallResponse:
        """Generate response with function calling on Vertex AI."""
        system_instruction, gemini_messages = self._convert_to_gemini_messages(input)

        function_declarations = self._build_function_declarations(function_tools)
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
            temperature=temperature,
        )

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

        self._track_tokens(response, model, "generate_response_with_tools", metric_tag)

        return ToolCallResponse(
            text=response.text or "",
            sources=None,
            cited_chunks=None,
            tool_calls=tool_calls_made if tool_calls_made else None,
        )


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
      More GDPR-compliant (data stays in region), better for production.
      Supports Vertex AI RAG Engine for RAG (set rag_corpus_name).

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
        vertex_location: str = "europe-west1",
        rag_corpus_name: Optional[str] = None,
        # Monitoring
        monitoring_project: Optional[str] = None,
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
            self._gemini = VertexAIGeminiService(
                project=vertex_project,
                location=vertex_location,
                rag_corpus_name=rag_corpus_name,
                monitoring=monitoring,
            )
            logger.info(
                f"Using Vertex AI Gemini (project={vertex_project}, "
                f"location={vertex_location}, rag_corpus={rag_corpus_name})"
            )
        else:
            if not gemini_api_key:
                raise ValueError("gemini_api_key is required when use_vertex_ai=False")
            self._gemini = GeminiService(
                api_key=gemini_api_key,
                file_store_id=gemini_file_store_id,
                file_store_name=file_store_name,
                monitoring=monitoring,
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
    def rag_corpus_name(self) -> Optional[str]:
        """Get RAG corpus name (only available when using Vertex AI)."""
        if self.use_vertex_ai and hasattr(self._gemini, "rag_corpus_name"):
            return self._gemini.rag_corpus_name
        return None

    def _get_service(self, model: str) -> LLMService:
        """Route to the correct service based on model name."""
        if model.startswith("gemini"):
            return self._gemini
        return self._openai

    async def generate_response(
        self,
        input: str | List[Dict[str, Any]],
        model: Optional[str] = None,
        file_search: bool = False,
        text_format: Optional[Type[BaseModel]] = None,
        rag_filter: Optional[Dict[str, str]] = None,
        metric_tag: str = "",
    ) -> LLMResponse:
        """
        Generate a response using the appropriate service based on model.

        Args:
            input: String or conversation messages
            model: Model to use (defaults to default_model)
            file_search: Enable RAG retrieval
            text_format: Pydantic model for structured output
            rag_filter: Metadata filter for RAG (Vertex AI only, e.g. {"doctor_id": "uuid"})
            metric_tag: Custom tag for metric tracking
        """
        model = model or self.default_model
        service = self._get_service(model)

        # Pass rag_filter to Vertex AI service if available
        if self.use_vertex_ai and model.startswith("gemini") and rag_filter:
            return await service.generate_response(
                input,
                model,
                file_search,
                text_format,
                rag_filter=rag_filter,
                metric_tag=metric_tag,
            )
        return await service.generate_response(
            input, model, file_search, text_format, metric_tag=metric_tag
        )

    # File operations (dispatches to Vertex AI RAG or Gemini File Search Store)
    async def upload_file(
        self,
        file_path: str,
        display_name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Upload a file to the knowledge store."""
        if self.use_vertex_ai:
            return self._gemini.upload_rag_file(file_path, display_name, metadata)
        return await self._gemini.upload_file_async(file_path, display_name)

    async def list_files(self) -> List[Dict[str, Any]]:
        """List files in the knowledge store."""
        if self.use_vertex_ai:
            return self._gemini.list_rag_files()
        return await self._gemini.list_files()

    async def delete_file(self, file_name: str) -> bool:
        """Delete a file from the knowledge store."""
        if self.use_vertex_ai:
            return self._gemini.delete_rag_file(file_name)
        return await self._gemini.delete_file(file_name)

    # Keep old names as sync wrappers for backwards compatibility
    def upload_rag_file(
        self,
        file_path: str,
        display_name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Upload a file (sync). Prefer upload_file() for async code."""
        if self.use_vertex_ai:
            return self._gemini.upload_rag_file(file_path, display_name, metadata)
        import asyncio

        return asyncio.run(self._gemini.upload_file_async(file_path, display_name))

    def list_rag_files(self) -> List[Dict[str, Any]]:
        """List files (sync). Prefer list_files() for async code."""
        if self.use_vertex_ai:
            return self._gemini.list_rag_files()
        import asyncio

        return asyncio.run(self._gemini.list_files())

    def delete_rag_file(self, file_name: str) -> bool:
        """Delete a file (sync). Prefer delete_file() for async code."""
        if self.use_vertex_ai:
            return self._gemini.delete_rag_file(file_name)
        import asyncio

        return asyncio.run(self._gemini.delete_file(file_name))

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

    async def generate_response_with_tools(
        self,
        input: str | List[Dict[str, Any]],
        model: Optional[str] = None,
        function_tools: Optional[List[Dict[str, Any]]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        temperature: Optional[float] = None,
        metric_tag: str = "",
    ) -> ToolCallResponse:
        """
        Generate a response with function calling support.

        For file search with citations, use generate_response first.
        Routes to the appropriate service based on model.
        """
        model = model or self.default_model
        service = self._get_service(model)
        return await service.generate_response_with_tools(
            input,
            model,
            function_tools or [],
            tool_executor,
            temperature,
            metric_tag=metric_tag,
        )
