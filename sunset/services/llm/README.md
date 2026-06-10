# LLMService

Multi-provider LLM service with unified interface for text generation, tool calling, structured output, and file search (RAG). Supports OpenAI, Gemini, Vertex AI, Mistral, and OpenRouter.

## Setup

### Infrastructure

In `sunset.yaml`:

```yaml
infra:
  llm:
    provider: vertexai  # or "openai", "gemini"
```

### Env Vars

Depends on the provider:

**Vertex AI** (recommended for GCP projects): No API key needed — uses Application Default Credentials. The GCP project is already provisioned by sunset.

**OpenAI**: Add to `sunset.env.yaml`:
```yaml
secrets:
  OPENAI_API_KEY: "sk-..."
```

**Gemini**: Add to `sunset.env.yaml`:
```yaml
secrets:
  GEMINI_API_KEY: "..."
```

**Mistral**: Add to `sunset.env.yaml`:
```yaml
secrets:
  MISTRAL_API_KEY: "..."
```

## Usage

### Vertex AI (most common)

```python
from sunset.services.llm import VertexAIGeminiService, file_search, LLMResponse

llm = VertexAIGeminiService(
    project=GCP_PROJECT_ID,
    location="global",
    retrieval=retrieval,  # Optional RetrievalService for file_search
)

response: LLMResponse = await llm.generate_response(
    input=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
    model="gemini-2.5-flash",
    function_tools=[file_search],  # Enable RAG
)

print(response["text"])
print(response["cited_chunks"])  # Source attributions
```

### OpenAI

```python
from sunset.services.llm import OpenAIService

llm = OpenAIService(
    api_key=OPENAI_API_KEY,
    file_store_id="vs_abc123",  # Optional vector store
)

response = await llm.generate_response(
    input="What is the meaning of life?",
    model="gpt-4o",
)
```

### Structured output

```python
from pydantic import BaseModel

class Evaluation(BaseModel):
    score: int
    reasoning: str

result = await llm.generate_json(
    messages=[{"role": "user", "content": "Rate this answer..."}],
    model="gemini-2.5-flash",
    text_format=Evaluation,
)
```

### Document understanding (full file in context)

Pass a GCS file directly to the model without RAG — the entire document is included in the prompt context:

```python
response = await llm.generate_response(
    input=[
        {"role": "user", "content": [
            {"type": "file", "file_uri": "gs://my-bucket/report.pdf", "mime_type": "application/pdf"},
            {"type": "text", "text": "Summarize this document."},
        ]},
    ],
    model="gemini-2.5-flash",
)
```

Supported MIME types include `application/pdf`, `text/plain`, `text/html`, `text/csv`, and others supported by the Gemini API. Only available with Gemini/Vertex AI providers.

### File upload lifecycle (Gemini Files API)

For large files that exceed inline limits, use `managed_file` to upload via the Gemini Files API, use the file in a prompt, and auto-delete it afterward:

```python
from sunset.services.llm import GeminiService

gemini = GeminiService(api_key=GEMINI_API_KEY)

async with gemini.managed_file(pdf_bytes, "application/pdf") as uploaded:
    result = await gemini.generate_json(
        messages=[{"role": "user", "content": [
            {"type": "file", "file_uri": uploaded.uri, "mime_type": "application/pdf"},
            {"type": "text", "text": "Extract the key fields from this document."},
        ]}],
        model="gemini-2.5-flash",
        temperature=0,
    )
```

You can also manage the lifecycle manually:

```python
uploaded = await gemini.upload_file(pdf_bytes, "application/pdf")
try:
    # use uploaded.uri in prompts ...
    pass
finally:
    await gemini.delete_file(uploaded.name)
```

- `upload_file(file_data: bytes, content_type: str) -> types.File` — uploads and polls until ACTIVE
- `delete_file(file_name: str)` — best-effort cleanup
- `managed_file(file_data: bytes, content_type: str)` — async context manager combining both

### Custom tool calling

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

async def execute_tool(name, args):
    if name == "get_weather":
        return {"temp": 22, "condition": "sunny"}

response = await llm.generate_response(
    input=[{"role": "user", "content": "What's the weather in Paris?"}],
    model="gemini-2.5-flash",
    function_tools=tools,
    tool_executor=execute_tool,
)
```

### Multi-provider router

```python
from sunset.services.llm import LLMServiceRouter

llm = LLMServiceRouter(
    openai_api_key=OPENAI_KEY,
    mistral_api_key=MISTRAL_KEY,
    use_vertex_ai=True,
    vertex_project=PROJECT_ID,
    vertex_location="europe-west1",
    vertex_extra_locations=["global"],   # extra location-pinned clients for fallback
)

# Automatically routes to the right provider based on model name
response = await llm.generate_response(input=messages, model="gpt-4o")
response = await llm.generate_response(input=messages, model="gemini-2.5-flash")
response = await llm.generate_response(input=messages, model="mistral-medium-latest")
```

### Cross-provider fallback with circuit breaker

`generate_with_fallback()` runs an explicit chain of `(model, location, priority)` steps and advances on exception, timeout, or empty response. An optional `BreakerProtocol` skips steps whose breaker is open and records 429s / slow successes — the safety-net step (last in chain) is never skipped.

```python
from sunset.services.llm import FallbackStep, LLMFallbackChainExhausted

chain = llm.default_fallback_chain("gemini-3-flash-preview")
# Or build it yourself:
# chain = [
#     FallbackStep("gemini-3-flash-preview", "europe-west1"),
#     FallbackStep("gemini-3-flash-preview", "global"),
#     FallbackStep("gemini-3-flash-preview", "global", priority=True),
#     FallbackStep("gpt-5.4-mini", "openai"),  # safety net
# ]

try:
    response = await llm.generate_with_fallback(
        input=messages,
        chain=chain,
        function_tools=[file_search],
        tool_executor=execute_tool,
        breaker=my_breaker,                    # optional, implements BreakerProtocol
        on_attempt=lambda step, outcome, err, attempt: ...,
        attempt_timeout_s=10.0,
        slow_threshold_s=8.0,
    )
except LLMFallbackChainExhausted:
    ...
```

`priority=True` on a Vertex Gemini step attaches the Priority Paygo header (1.8x billing, separate quota pool). No-op for OpenAI / Mistral.

### Per-provider concurrency gating

The router bounds in-flight calls per provider bucket with local semaphores (`DEFAULT_PROVIDER_CONCURRENCY`, override via the `provider_concurrency` constructor param). On non-final fallback steps, a permit must be acquired within `acquire_timeout_s` (default 2s) or the step is skipped as `"throttled"` and the chain advances — better to fail over than queue behind saturated calls. The final safety-net step always waits. The circuit breaker handles cross-instance coordination; the semaphores are the proactive per-instance layer.

### OpenRouter mode

When constructed with `use_openrouter=True`, every call routes through a single `OpenRouterService` and the native cross-provider chain is bypassed — inter-model fallback is delegated to OpenRouter's server-side `models` array (capped at 3 entries, chain preference order preserved).

```python
llm = LLMServiceRouter(
    use_openrouter=True,
    openrouter_api_key=OPENROUTER_KEY,
    openrouter_site_url="https://example.com",   # optional HTTP-Referer
    openrouter_app_name="myproject",             # optional X-Title
)
```

In OR mode, `default_fallback_chain()` hoists GPT to position 2 (right after the primary) for the fastest cross-provider failover. Gemini models are pinned to the `google-vertex` provider when the whole chain is Google. `OpenRouterService` is also usable standalone (accepts OpenRouter-native `vendor/model` slugs directly).

### Observability (optional)

When the `observability` extra is installed and `init_observability()` has been called (see `sunset/services/observability/README.md`), the router emits OpenTelemetry spans per call/attempt and metrics (`llm.request.total`, `llm.request.duration`, `llm.tokens`, `llm.cost.usd`, `llm.fallback.advanced`, `llm.breaker.*`, `llm.throttled`). Costs come from the pricing table in `pricing.py` (`compute_cost(model, input_tokens, output_tokens)` — USD per 1M tokens; edit there when vendor prices change). Without the extra, all instrumentation no-ops.

## API Reference

### `VertexAIGeminiService` Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str` | required | GCP project ID |
| `project_number` | `str` | `""` | GCP project number (for Discovery Engine) |
| `location` | `str` | `"europe-west1"` | GCP region |
| `monitoring` | `MonitoringService` | `None` | Token usage tracking |
| `retrieval` | `RetrievalService` | `None` | For `file_search` tool |

### `OpenAIService` Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | required | OpenAI API key |
| `file_store_id` | `str` | `None` | Existing vector store ID |
| `monitoring` | `MonitoringService` | `None` | Token usage tracking |
| `retrieval` | `RetrievalService` | `None` | For `file_search` tool |

### Key Methods (all providers)

- `generate_response(input, model, function_tools?, tool_executor?, text_format?, temperature?, metric_tag?, priority?) -> LLMResponse` (async)
- `generate_json(messages, model, temperature?, text_format?) -> dict` (async)

### Router-only methods

- `generate_with_fallback(input, chain, ..., breaker?, on_attempt?, attempt_scope?, attempt_timeout_s?, max_empty_attempts?, slow_threshold_s?) -> LLMResponse` (async)
- `default_fallback_chain(model) -> list[FallbackStep]` — preview models get a multi-step chain (preferred → stable → priority paygo), stable models get local + global, others get a single step. Cross-provider safety net is appended when those providers are configured.

### `FallbackStep`

```python
FallbackStep(model: str, location: str, priority: bool = False)
# location is "openai" / "mistral" or a GCP region for Vertex Gemini.
# breaker_key is "{model}@{location}[+priority]"
```

### `BreakerProtocol`

```python
class BreakerProtocol(Protocol):
    async def is_open(self, key: str) -> bool: ...
    async def record_429(self, key: str) -> bool: ...
    async def record_slow(self, key: str) -> bool: ...
```

### `LLMResponse` TypedDict

```python
{
    "text": str,                          # Generated text
    "cited_chunks": list[SourceChunk],    # Source attributions (if file_search used)
    "tool_calls": list[ToolCall],         # Pending tool calls (if no executor)
    "usage": Usage,                       # Token usage (all providers)
    "diagnostics": dict,                  # Why a completion may be empty (OpenRouter)
}
```

### `Usage` TypedDict

Every provider extracts token usage into `response["usage"]`:

```python
{
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
    "thinking_tokens": int,   # Gemini/Vertex reasoning tokens
    "cached_tokens": int,     # prompt-cache hits
}
```

### `file_search` sentinel

Import and pass as a tool to enable RAG:

```python
from sunset.services.llm import file_search

response = await llm.generate_response(
    input=messages,
    model="gemini-2.5-flash",
    function_tools=[file_search],
)
```
