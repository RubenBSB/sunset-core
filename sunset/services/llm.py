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
    ):
        self.project = project
        self.project_number = project_number
        self.location = location
        self.search_engine_id = search_engine_id
        self.search_data_store_ids = search_data_store_ids or []
        self.monitoring = monitoring
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
        file_search: bool = False,
        text_format: Optional[Type[BaseModel]] = None,
        metric_tag: str = "",
    ) -> LLMResponse:
        """
        Generate a response using Vertex AI Gemini with optional Vertex AI Search grounding.

        When file_search=True and search_engine_id is configured, uses Discovery Engine
        grounded generation to ground the response in your search data stores.

        Args:
            input: String or conversation messages
            model: Model identifier
            file_search: If True and search_engine_id is set, enables grounded generation
            text_format: Optional Pydantic model for structured output

        Returns:
            LLMResponse with text and optional sources
        """
        # If file_search is enabled and we have a search engine, use grounded generation
        if file_search and self.search_engine_id:
            return await self._grounded_generate(input, model, metric_tag)

        if file_search and not self.search_engine_id:
            logger.warning(
                "file_search=True but no search_engine_id configured. "
                "Grounded generation disabled."
            )

        system_instruction, gemini_messages = self._convert_to_gemini_messages(input)

        config_params = {}

        if system_instruction:
            config_params["system_instruction"] = [
                system_instruction,
                "VERY IMPORTANT: Keep the overall response super short. It should be like 1-2 sentences max. Like a short human message.",
                "You have FULL MEMORY of this conversation. NEVER say your memory resets or that you don't remember. Focus your answer on the LAST user message but use conversation history for context.",
            ]

        gemini_json_models = ["gemini-3"]
        if any(model.startswith(m) for m in gemini_json_models):
            if text_format:
                config_params["response_mime_type"] = "application/json"
                config_params["response_json_schema"] = text_format.model_json_schema()
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=256)

        config = types.GenerateContentConfig(**config_params) if config_params else None

        response = await self.client.aio.models.generate_content(
            model=model, contents=gemini_messages, config=config
        )
        self._track_tokens(response, model, "generate_response", metric_tag)

        return LLMResponse(
            text=response.text,
            sources=None,
            cited_chunks=None,
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

        # Extract sources and cited chunks from grounding metadata
        sources: List[SourceChunk] = []
        cited_chunks: List[SourceChunk] = []
        try:
            if response.candidates:
                candidate = response.candidates[0]
                grounding_meta = getattr(candidate, "grounding_metadata", None)
                if grounding_meta:
                    support_chunks = (
                        getattr(grounding_meta, "support_chunks", None) or []
                    )
                    seen_files: set[str] = set()
                    for chunk in support_chunks:
                        source = getattr(chunk, "chunk_text", "") or ""
                        doc_name = ""
                        doc_uri = ""
                        doc_metadata = getattr(chunk, "source", None)
                        if doc_metadata:
                            doc_name = getattr(doc_metadata, "title", "") or ""
                            doc_uri = getattr(doc_metadata, "uri", "") or ""

                        cited_chunks.append(
                            SourceChunk(
                                file_id=doc_uri,
                                filename=doc_name,
                                text=source,
                                score=1.0,
                            )
                        )

                        if doc_name and doc_name not in seen_files:
                            seen_files.add(doc_name)
                            sources.append(
                                SourceChunk(
                                    file_id=doc_uri,
                                    filename=doc_name,
                                    text="",
                                    score=1.0,
                                )
                            )
            sources = sources[:3]
            cited_chunks = cited_chunks[:5]
        except Exception as e:
            logger.debug(f"Could not extract sources from grounded response: {e}")

        logger.info(
            f"Grounded generation completed. Sources: {len(sources)}, "
            f"Cited chunks: {len(cited_chunks)}"
        )

        return LLMResponse(
            text=response_text,
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
        file_search: bool = False,
        text_format: Optional[Type[BaseModel]] = None,
        metric_tag: str = "",
    ) -> LLMResponse:
        """
        Generate a response using the appropriate service based on model.

        Args:
            input: String or conversation messages
            model: Model to use (defaults to default_model)
            file_search: Enable RAG retrieval (Vertex AI Search grounded generation or file search)
            text_format: Pydantic model for structured output
            metric_tag: Custom tag for metric tracking
        """
        model = model or self.default_model
        service = self._get_service(model)
        return await service.generate_response(
            input, model, file_search, text_format, metric_tag=metric_tag
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
