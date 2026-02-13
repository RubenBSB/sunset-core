import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from sunset.services.llm import LLMResponse, ToolExecutor
from sunset.services.llm import file_search as file_search

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    history: list[dict[str, Any]]
    last_message: str
    system_prompt: str
    history_limit: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)


class Conversation:
    def __init__(
        self,
        chat_service: "ChatService",
        history: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self._chat = chat_service
        self.history: List[Dict[str, Any]] = list(history or [])
        self.meta: Dict[str, Any] = dict(meta or {})

    async def send(self, message: str) -> LLMResponse:
        self.history.append({"role": "user", "content": message})

        system_prompt = self._chat.system_prompt
        if callable(system_prompt):
            system_prompt = await system_prompt()

        ctx = ConversationContext(
            history=self.history,
            last_message=message,
            system_prompt=system_prompt,
            history_limit=self._chat.history_limit,
            meta=self.meta,
        )

        if self._chat.before_generate:
            await self._chat.before_generate(ctx)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": ctx.system_prompt}
        ]
        limit = self._chat.history_limit
        messages.extend(self.history[-limit:] if limit else self.history)

        response = await self._chat.llm.generate_response(
            input=messages,
            model=self._chat.model,
            function_tools=self._chat.tools or None,
            tool_executor=self._chat.tool_executor,
            temperature=self._chat.temperature,
            metric_tag=self._chat.metric_tag,
        )

        self.history.append({"role": "assistant", "content": response["text"]})

        if self._chat.on_response:
            await self._chat.on_response(ctx, response)

        return response


class ChatService:
    def __init__(
        self,
        llm: Any,
        system_prompt: Union[str, Callable],
        model: str,
        tools: Optional[List[Any]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        history_limit: Optional[int] = None,
        temperature: Optional[float] = None,
        metric_tag: str = "",
        before_generate: Optional[Callable] = None,
        on_response: Optional[Callable] = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools
        self.tool_executor = tool_executor
        self.history_limit = history_limit
        self.temperature = temperature
        self.metric_tag = metric_tag
        self.before_generate = before_generate
        self.on_response = on_response

    def conversation(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Create a new conversation instance."""
        return Conversation(chat_service=self, history=history, meta=meta)
