# ChatService

Multi-turn conversation manager. Wraps an LLM service with system prompts, history management, tool use, and lifecycle hooks.

## Setup

### Infrastructure

No specific `sunset.yaml` entries. Requires an LLM service to be configured (see `llm/README.md`).

## Usage

```python
from sunset.services.chat import ChatService, ConversationContext
from sunset.services.llm import VertexAIGeminiService, file_search

llm = VertexAIGeminiService(project=PROJECT_ID, project_number=PROJECT_NUMBER, location="global")

chat = ChatService(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    model="gemini-2.5-flash",
    tools=[file_search],
    history_limit=20,
    before_generate=before_generate,
    on_response=on_response,
)
```

### Lifecycle hooks

Use `before_generate` and `on_response` to persist conversations to the database:

```python
async def before_generate(ctx: ConversationContext):
    """Load conversation history from DB before generating."""
    conversation = await get_or_create_conversation(ctx.meta["user_id"], session)
    messages = await get_messages(conversation.id, session)
    ctx.history = [{"role": m.role, "content": m.content} for m in messages]
    ctx.meta["conversation_id"] = conversation.id

async def on_response(ctx: ConversationContext, response: LLMResponse):
    """Save the assistant response to DB after generating."""
    await save_message(
        conversation_id=ctx.meta["conversation_id"],
        role="assistant",
        content=response["text"],
        metadata={"cited_chunks": response.get("cited_chunks")},
        session=session,
    )
```

### Handling requests

```python
@router.post("/chat/message")
async def send_message(body: MessageRequest, user: User = Depends(get_current_user)):
    conv = chat.conversation(meta={"user_id": str(user.id)})
    response = await conv.send(body.message)
    return {"text": response["text"], "sources": response.get("cited_chunks")}
```

## API Reference

### `ChatService` Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMService` | required | LLM service instance |
| `system_prompt` | `str \| Callable` | required | System prompt (or async callable returning one) |
| `model` | `str` | required | Model name |
| `tools` | `list` | `None` | Function tools and/or `file_search` sentinel |
| `tool_executor` | `ToolExecutor` | `None` | Async callback for custom tool execution |
| `history_limit` | `int` | `None` | Max messages to include in context |
| `temperature` | `float` | `None` | Sampling temperature |
| `before_generate` | `Callable` | `None` | `async (ConversationContext) -> None` |
| `on_response` | `Callable` | `None` | `async (ConversationContext, LLMResponse) -> None` |

### `ChatService.conversation(history?, meta?) -> Conversation`

Create a new conversation instance.

### `Conversation.send(message) -> LLMResponse` (async)

Send a message and get the assistant's response.
