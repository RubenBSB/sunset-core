import json
import logging
import time
import base64
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, TypedDict, Type, Tuple

from pydantic import BaseModel

from openai import AsyncOpenAI
from google import genai
from google.genai import types

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

    def __init__(self, api_key: str, file_store_id: Optional[str] = None):
        self.api_key = api_key
        super().__init__()
        self.file_store_id = file_store_id
        self._file_store = None  # Lazy-loaded

    async def get_file_store(self):
        """Lazily retrieve and cache the file store."""
        if not self._file_store and self.file_store_id:
            self._file_store = await self.client.vector_stores.retrieve(
                vector_store_id=self.file_store_id
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

    def __init__(self, api_key: str, file_store_id: Optional[str] = None):
        self.api_key = api_key
        super().__init__()
        self.file_store_id = file_store_id
        self._file_store = None  # Lazy-loaded

    async def get_file_store(self):
        """Lazily retrieve and cache the file store."""
        if not self._file_store and self.file_store_id:
            try:
                self._file_store = await self.client.aio.file_search_stores.get(
                    name=self.file_store_id
                )
            except Exception as e:
                logger.error(f"Failed to retrieve Gemini file store: {e}")
                return None
        return self._file_store

    def get_client(self):
        if not hasattr(self, "client"):
            return genai.Client(api_key=self.api_key)
        return self.client

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
        gemini_json_models = ["gemini-3-pro-preview"]
        if any(model.startswith(m) for m in gemini_json_models):
            if text_format:
                config_params["response_mime_type"] = "application/json"
                config_params["response_json_schema"] = text_format.model_json_schema()
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=256)

        config = types.GenerateContentConfig(**config_params)

        response = await self.client.aio.models.generate_content(
            model=model, contents=gemini_messages, config=config
        )

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

        # Log if response is empty (helps debug thinking-only responses)
        if not response.text:
            logger.warning(f"Empty response text. Candidates: {response.candidates}")

        return ToolCallResponse(
            text=response.text or "",
            sources=None,
            cited_chunks=None,
            tool_calls=tool_calls_made if tool_calls_made else None,
        )


class LLMServiceRouter:
    """
    Unified LLM service that routes requests to OpenAI or Gemini based on model name.
    Unified LLM service that routes requests to OpenAI or Gemini based on model name.

    Model routing:
    - Models starting with 'gemini' -> GeminiService
    - All other models -> OpenAIService (default)
    """

    def __init__(
        self,
        openai_api_key: str,
        gemini_api_key: str,
        openai_file_store_id: Optional[str] = None,
        gemini_file_store_id: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
    ):
        self._openai = OpenAIService(
            api_key=openai_api_key,
            file_store_id=openai_file_store_id,
        )
        self._gemini = GeminiService(
            api_key=gemini_api_key,
            file_store_id=gemini_file_store_id,
        )
        self.default_model = default_model

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
    ) -> LLMResponse:
        """Generate a response using the appropriate service based on model."""
        model = model or self.default_model
        service = self._get_service(model)
        return await service.generate_response(input, model, file_search, text_format)

    async def generate_json(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        text_format: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON using the appropriate service based on model."""
        model = model or self.default_model
        service = self._get_service(model)
        return await service.generate_json(messages, model, temperature, text_format)

    async def generate_response_with_tools(
        self,
        input: str | List[Dict[str, Any]],
        model: Optional[str] = None,
        function_tools: Optional[List[Dict[str, Any]]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        temperature: Optional[float] = None,
    ) -> ToolCallResponse:
        """
        Generate a response with function calling support.

        For file search with citations, use generate_response first.
        Routes to the appropriate service based on model.
        """
        model = model or self.default_model
        service = self._get_service(model)
        return await service.generate_response_with_tools(
            input, model, function_tools or [], tool_executor, temperature
        )
