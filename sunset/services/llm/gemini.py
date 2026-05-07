import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

from .base import (
    LLMResponse,
    LLMService,
    SourceChunk,
    ToolCall,
    ToolExecutor,
    _split_tools,
)
from .store import GeminiFileStore

logger = logging.getLogger(__name__)


class GeminiService(LLMService):
    """Google Gemini API implementation with file search store grounding."""

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
        self._store: Optional[GeminiFileStore] = None
        if file_store_id:
            self._store = GeminiFileStore(self.client, file_store_id)

    @property
    def store(self) -> Optional[GeminiFileStore]:
        return self._store

    async def _ensure_store(self) -> Optional[GeminiFileStore]:
        """Lazily create or return the file store."""
        if self._store:
            return self._store
        try:
            self._store = await GeminiFileStore.create(
                self.client, name=self.file_store_name
            )
            return self._store
        except Exception as e:
            logger.error(f"Failed to create Gemini file store: {e}")
            return None

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

    _RESPONSE_NOISE_RE = re.compile(
        r"|".join(
            [
                r"c11n_\d+_\w+\(.*?\)[‡†]",  # citation markers
                r"<ctrl\d+>",  # control tokens
                r"ccall:\w+:\w+\{.*?\}",  # raw tool call leaks
                r"\[cite_start\]",  # cite_start tags
                r"\[cite:\s*\d+\]",  # cite: N tags
                r"\[cite_num\]",  # cite_num tags
            ]
        )
    )

    @classmethod
    def _clean_response_text(cls, text: str) -> str:
        """Strip leaked Gemini internal markers from response text."""
        return cls._RESPONSE_NOISE_RE.sub("", text).strip()

    _MIME_SUFFIXES = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }

    def _parse_data_url(self, data_url: str) -> tuple[bytes, str]:
        """Parse a data URL and return (bytes, mime_type)."""
        match = re.match(r"data:([^;]+);base64,(.+)", data_url)
        if match:
            mime_type = match.group(1)
            b64_data = match.group(2)
            return base64.b64decode(b64_data), mime_type
        raise ValueError("Invalid data URL format")

    async def upload_file(self, file_data: bytes, content_type: str) -> types.File:
        """Upload a file via the Gemini Files API and poll until ACTIVE."""
        loop = asyncio.get_event_loop()
        suffix = self._MIME_SUFFIXES.get(content_type, ".bin")

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            await loop.run_in_executor(
                None, lambda: (os.write(fd, file_data), os.close(fd))
            )

            uploaded = await loop.run_in_executor(
                None,
                lambda: self.client.files.upload(
                    file=tmp_path,
                    config=types.UploadFileConfig(mime_type=content_type),
                ),
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        while uploaded.state != types.FileState.ACTIVE:
            await asyncio.sleep(1)
            uploaded = await loop.run_in_executor(
                None, lambda: self.client.files.get(name=uploaded.name)
            )

        return uploaded

    async def delete_file(self, file_name: str) -> None:
        """Best-effort delete of a previously uploaded file."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.client.files.delete(name=file_name)
            )
        except Exception:
            logger.debug(f"Failed to delete file {file_name}", exc_info=True)

    @asynccontextmanager
    async def managed_file(
        self, file_data: bytes, content_type: str
    ) -> AsyncIterator[types.File]:
        """Upload a file, yield it for use, then delete it."""
        uploaded = await self.upload_file(file_data, content_type)
        try:
            yield uploaded
        finally:
            await self.delete_file(uploaded.name)

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
        file_store = await self._ensure_store() if has_file_search else None

        if has_file_search and file_store:
            tools.append(
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[file_store._store_name]
                    )
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
                        elif part_type == "inline_data":
                            data = part.get("data")
                            mime_type = part.get(
                                "mime_type", "application/octet-stream"
                            )
                            if data:
                                parts.append(
                                    types.Part(
                                        inline_data=types.Blob(
                                            data=data, mime_type=mime_type
                                        )
                                    )
                                )
                        elif part_type == "file":
                            file_uri = part.get("file_uri")
                            mime_type = part.get("mime_type", "application/pdf")
                            if file_uri:
                                parts.append(
                                    types.Part.from_uri(
                                        file_uri=file_uri, mime_type=mime_type
                                    )
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
                    elif part_type == "file":
                        file_uri = part.get("file_uri")
                        mime_type = part.get("mime_type", "application/pdf")
                        if file_uri:
                            parts.append(
                                types.Part.from_uri(
                                    file_uri=file_uri, mime_type=mime_type
                                )
                            )

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
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
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
            thinking_config=types.ThinkingConfig(thinking_budget=256)
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
