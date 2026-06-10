import json
import logging
from typing import Any, Dict, List, Optional, Type

from mistralai import Mistral
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

logger = logging.getLogger(__name__)

# Mistral models that support image input.
_VISION_MODELS = ("pixtral", "mistral-medium", "mistral-large", "mistral-small")


class MistralService(LLMService):
    """Mistral chat completions implementation.

    Function calling, vision (image_url), structured JSON, and pgvector RAG
    via the same search_knowledge tool path used by other providers.
    Mistral has no native vector store — file_search routes to pgvector when
    a retrieval service is available, otherwise it's a no-op.
    """

    def __init__(
        self,
        api_key: str,
        monitoring: Optional[Any] = None,
        retrieval: Optional[Any] = None,
    ):
        self.api_key = api_key
        self.monitoring = monitoring
        self.retrieval = retrieval
        super().__init__()

    def get_client(self):
        if not hasattr(self, "client"):
            return Mistral(api_key=self.api_key)
        return self.client

    @staticmethod
    def _extract_usage(response) -> Usage:
        usage = getattr(response, "usage", None)
        if not usage:
            return {}
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (
            input_tokens + output_tokens
        )
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }

    @property
    def store(self):
        return None

    @staticmethod
    def _supports_vision(model: str) -> bool:
        return any(model.startswith(prefix) for prefix in _VISION_MODELS)

    def _convert_messages(
        self, input: str | List[Dict[str, Any]], model: str
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Mistral's chat format.

        Mistral accepts the same shape (role + content / content parts) but
        we normalize image parts: only data URLs and https URLs are accepted,
        and only on vision-capable models. On a text-only model we drop image
        parts and replace with a placeholder so the model still has context.
        """
        if isinstance(input, str):
            return [{"role": "user", "content": input}]

        supports_vision = self._supports_vision(model)
        out: List[Dict[str, Any]] = []
        for msg in input:
            role = msg.get("role")
            content = msg.get("content")

            # Mistral accepts string content for system/user/assistant.
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                continue

            parts: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("text", "input_text"):
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif ptype in ("image_url", "input_image"):
                    image_url = part.get("image_url") or part.get("url") or ""
                    if not image_url:
                        continue
                    if supports_vision:
                        parts.append({"type": "image_url", "image_url": image_url})
                    else:
                        parts.append({"type": "text", "text": "[image]"})
                elif ptype == "image_description":
                    desc = part.get("description", "no description")
                    parts.append({"type": "text", "text": f"[image: {desc}]"})

            # System role can't carry parts on Mistral — flatten to text.
            if role in ("system", "developer"):
                text = "\n".join(p["text"] for p in parts if p.get("type") == "text")
                if text:
                    out.append({"role": "system", "content": text})
            elif parts:
                out.append({"role": role, "content": parts})

        return out

    def _build_tool_schema(
        self, function_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """OpenAI-style tool dicts pass through to Mistral as-is."""
        return [t for t in function_tools if t.get("type") == "function"]

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

        if has_file_search and self.retrieval is not None:
            return await self._retrieval_generate(
                input, model, regular_tools, tool_executor, metric_tag, temperature
            )
        if has_file_search:
            logger.warning(
                "Mistral file_search requested but no retrieval service configured."
            )

        messages = self._convert_messages(input, model)

        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if regular_tools:
            kwargs["tools"] = self._build_tool_schema(regular_tools)
            kwargs["tool_choice"] = "auto"
        if text_format:
            kwargs["response_format"] = {"type": "json_object"}

        # Tool-calling loop
        if regular_tools and tool_executor:
            tool_calls_made: List[ToolCall] = []
            max_iterations = 10
            response = None

            for _ in range(max_iterations):
                response = await self.client.chat.complete_async(**kwargs)
                msg = response.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []

                if not tool_calls:
                    break

                # Append the assistant message that requested the tool calls.
                kwargs["messages"].append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    tool_name = tc.function.name
                    raw_args = tc.function.arguments
                    tool_args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else (raw_args or {})
                    )
                    tool_calls_made.append(
                        ToolCall(name=tool_name, arguments=tool_args, id=tc.id)
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

                    kwargs["messages"].append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": tc.id,
                            "content": result_str,
                        }
                    )

            text = response.choices[0].message.content if response else ""
            return LLMResponse(
                text=text or "",
                cited_chunks=None,
                tool_calls=tool_calls_made if tool_calls_made else None,
                usage=self._extract_usage(response),
            )

        # Simple generation
        response = await self.client.chat.complete_async(**kwargs)
        text = response.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            cited_chunks=None,
            tool_calls=None,
            usage=self._extract_usage(response),
        )

    async def _retrieval_generate(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        extra_executor: Optional[ToolExecutor] = None,
        metric_tag: str = "",
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """RAG via pgvector. Same shape as the OpenAI / Vertex path."""
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
        try:
            converted = self._convert_messages(messages, model)
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": converted,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            response = await self.client.chat.complete_async(**kwargs)
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:
            logger.error(f"Mistral JSON generation error: {e}")
            return {}
