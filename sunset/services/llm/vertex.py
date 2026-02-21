import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

from . import LLMResponse, SourceChunk, ToolCall, ToolExecutor, _split_tools
from .base import LLMService
from .store import VertexFileStore

logger = logging.getLogger(__name__)


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
        self._store: Optional[VertexFileStore] = None

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

    @property
    def store(self) -> Optional[VertexFileStore]:
        if not self._store and self.search_data_store_ids:
            self._store = VertexFileStore(
                project=self.project,
                data_store_id=self.search_data_store_ids[0],
            )
        return self._store

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
                    elif part_type == "inline_data":
                        data = part.get("data")
                        mime_type = part.get("mime_type", "application/octet-stream")
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
