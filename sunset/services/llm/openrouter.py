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

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _to_openrouter_slug(model: str) -> str:
    """Map an internal model name to OpenRouter's provider/model slug.

    Models already in the "vendor/name" form are returned untouched so callers
    can pass OpenRouter-native slugs directly.
    """
    if "/" in model:
        return model
    if model.startswith("gemini"):
        return f"google/{model}"
    if model.startswith("gpt"):
        return f"openai/{model}"
    if model.startswith("mistral") or model.startswith("pixtral"):
        return f"mistralai/{model}"
    if model.startswith("claude"):
        return f"anthropic/{model}"
    return model


def _response_diagnostics(
    response: Any,
    *,
    iterations: Optional[int] = None,
    hit_max_iterations: Optional[bool] = None,
    final_turn_had_tool_calls: Optional[bool] = None,
) -> Dict[str, Any]:
    """Extract why a completion's text may be empty.

    OpenRouter returns a 200 even when the model produces no text (a tool-only
    turn, a safety/recitation stop, or an answer that landed entirely in the
    reasoning channel). The OpenAI SDK keeps OpenRouter's extra fields
    (``native_finish_reason``, ``provider``) accessible as attributes, and
    Gemini's real stop reason surfaces under ``native_finish_reason``.
    """
    diag: Dict[str, Any] = {}
    if iterations is not None:
        diag["iterations"] = iterations
    if hit_max_iterations is not None:
        diag["hit_max_iterations"] = hit_max_iterations
    if final_turn_had_tool_calls is not None:
        diag["final_turn_had_tool_calls"] = final_turn_had_tool_calls

    choice = None
    try:
        if response is not None and getattr(response, "choices", None):
            choice = response.choices[0]
    except Exception:
        choice = None

    if choice is not None:
        diag["finish_reason"] = getattr(choice, "finish_reason", None)
        native = getattr(choice, "native_finish_reason", None)
        if native:
            diag["native_finish_reason"] = native
        msg = getattr(choice, "message", None)
        if msg is not None:
            reasoning = getattr(msg, "reasoning", None) or getattr(
                msg, "reasoning_details", None
            )
            diag["has_reasoning"] = bool(reasoning)
            refusal = getattr(msg, "refusal", None)
            if refusal:
                diag["refusal"] = str(refusal)[:200]

    provider = getattr(response, "provider", None)
    if provider:
        diag["provider"] = provider
    return diag


class OpenRouterService(LLMService):
    """OpenRouter Chat Completions backend.

    For Gemini models we pin ``provider.only=["google-vertex"]`` so the request
    lands on Vertex (matching the native path) rather than Google AI Studio.
    Function tools pass through as standard OpenAI schema. The ``file_search``
    sentinel routes through pgvector via a search_knowledge function tool —
    same shape as MistralService / VertexAIGeminiService when no native vector
    store is configured.

    Fallback across models is handled server-side by OpenRouter through the
    ``models`` parameter. ``priority=True`` maps to OpenRouter's priority
    routing field — Vertex translates it into the Priority Paygo tier.
    """

    def __init__(
        self,
        api_key: str,
        retrieval: Optional[Any] = None,
        monitoring: Optional[Any] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ):
        self.api_key = api_key
        self.retrieval = retrieval
        self.monitoring = monitoring
        self._site_url = site_url
        self._app_name = app_name
        super().__init__()

    def get_client(self):
        if hasattr(self, "client"):
            return self.client
        default_headers: Dict[str, str] = {}
        if self._site_url:
            default_headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            default_headers["X-Title"] = self._app_name
        return AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
            default_headers=default_headers or None,
        )

    @property
    def store(self):
        return None

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
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cached = getattr(details, "cached_tokens", 0) or 0
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "cached_tokens": int(cached),
        }

    @staticmethod
    def _provider_config(model: str) -> Dict[str, Any]:
        """OpenRouter provider routing for this model.

        Pin Gemini models to google-vertex so we hit the same backend as the
        native path (instead of Google AI Studio, which has different quota
        pools and tool-calling quirks).
        """
        slug = _to_openrouter_slug(model)
        if slug.startswith("google/"):
            return {"only": ["google-vertex"]}
        return {}

    def _build_extra_body(
        self,
        model: str,
        priority: bool,
        models: Optional[List[str]],
    ) -> Optional[Dict[str, Any]]:
        extra_body: Dict[str, Any] = {}
        # Provider pinning: only restrict to google-vertex when EVERY model in
        # the chain is Google. A mixed chain (e.g. gemini + gpt-5.4-mini) drops
        # the restriction so OR can route each model to its native provider.
        all_models = [model] + [m for m in (models or []) if m != model]
        if all(_to_openrouter_slug(m).startswith("google/") for m in all_models):
            extra_body["provider"] = {"only": ["google-vertex"]}
        if models:
            extra_body["models"] = [_to_openrouter_slug(m) for m in models]
        if priority:
            # OpenRouter forwards this to Vertex via X-Vertex-AI-LLM-Request-Type.
            extra_body["priority"] = True
        return extra_body or None

    def _convert_messages(
        self, input: str | List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """OpenRouter accepts the OpenAI Chat Completions shape directly.

        We only need to normalize image parts: data URLs and https URLs are
        accepted, ``image_description`` is flattened to text so non-vision
        models still get context.
        """
        if isinstance(input, str):
            return [{"role": "user", "content": input}]

        out: List[Dict[str, Any]] = []
        for msg in input:
            role = msg.get("role")
            content = msg.get("content")

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
                    parts.append({"type": "image_url", "image_url": {"url": image_url}})
                elif ptype == "image_description":
                    desc = part.get("description", "no description")
                    parts.append({"type": "text", "text": f"[image: {desc}]"})

            if role in ("system", "developer"):
                text = "\n".join(p["text"] for p in parts if p.get("type") == "text")
                if text:
                    out.append({"role": "system", "content": text})
            elif parts:
                out.append({"role": role, "content": parts})

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
        models: Optional[List[str]] = None,
    ) -> LLMResponse:
        has_file_search, regular_tools = _split_tools(function_tools)

        if has_file_search and self.retrieval is not None:
            return await self._retrieval_generate(
                input,
                model,
                regular_tools,
                tool_executor,
                metric_tag,
                temperature,
                priority=priority,
                models=models,
            )
        if has_file_search:
            logger.warning(
                "OpenRouter file_search requested but no retrieval service configured."
            )

        messages = self._convert_messages(input)
        extra_body = self._build_extra_body(model, priority, models)

        kwargs: Dict[str, Any] = {
            "model": _to_openrouter_slug(model),
            "messages": messages,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        if temperature is not None:
            kwargs["temperature"] = temperature
        if regular_tools:
            kwargs["tools"] = [t for t in regular_tools if t.get("type") == "function"]
            kwargs["tool_choice"] = "auto"
        if text_format:
            kwargs["response_format"] = {"type": "json_object"}

        if regular_tools and tool_executor:
            return await self._tool_loop(kwargs, tool_executor)

        response = await self.client.chat.completions.create(**kwargs)
        text = (response.choices[0].message.content or "") if response.choices else ""
        return LLMResponse(
            text=text,
            cited_chunks=None,
            tool_calls=None,
            usage=self._extract_usage(response),
            diagnostics=_response_diagnostics(response),
        )

    async def _tool_loop(
        self,
        kwargs: Dict[str, Any],
        tool_executor: ToolExecutor,
    ) -> LLMResponse:
        tool_calls_made: List[ToolCall] = []
        max_iterations = 10
        response = None
        iterations = 0
        final_turn_had_tool_calls = False

        for _ in range(max_iterations):
            iterations += 1
            response = await self.client.chat.completions.create(**kwargs)
            if not response.choices:
                break
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                final_turn_had_tool_calls = False
                break
            final_turn_had_tool_calls = True

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
                        json.dumps(result) if not isinstance(result, str) else result
                    )
                except Exception as e:
                    logger.error(f"Tool execution error for {tool_name}: {e}")
                    result_str = json.dumps({"error": str(e)})

                kwargs["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )

        if response and response.choices:
            text = response.choices[0].message.content or ""
        else:
            text = ""
        hit_max_iterations = iterations >= max_iterations and final_turn_had_tool_calls
        return LLMResponse(
            text=text,
            cited_chunks=None,
            tool_calls=tool_calls_made if tool_calls_made else None,
            usage=self._extract_usage(response) if response else {},
            diagnostics=_response_diagnostics(
                response,
                iterations=iterations,
                hit_max_iterations=hit_max_iterations,
                final_turn_had_tool_calls=final_turn_had_tool_calls,
            ),
        )

    async def _retrieval_generate(
        self,
        input: str | List[Dict[str, Any]],
        model: str,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        extra_executor: Optional[ToolExecutor] = None,
        metric_tag: str = "",
        temperature: Optional[float] = None,
        priority: bool = False,
        models: Optional[List[str]] = None,
    ) -> LLMResponse:
        """RAG via pgvector. Same shape as MistralService / VertexAIGeminiService."""
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
            priority=priority,
            models=models,
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
            diagnostics=response.get("diagnostics"),
        )

    async def generate_json(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        text_format: Optional[Type[BaseModel]] = None,
        metric_tag: str = "",
    ) -> Dict[str, Any]:
        try:
            kwargs: Dict[str, Any] = {
                "model": _to_openrouter_slug(model),
                "messages": self._convert_messages(messages),
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            extra_body = self._build_extra_body(model, priority=False, models=None)
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = await self.client.chat.completions.create(**kwargs)
            content = (
                response.choices[0].message.content or "{}"
                if response.choices
                else "{}"
            )
            return json.loads(content)
        except Exception as e:
            logger.error(f"OpenRouter JSON generation error: {e}")
            return {}
