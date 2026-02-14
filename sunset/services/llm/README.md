# LLMService

Multi-provider LLM service with unified interface for text generation, tool calling, structured output, and file search (RAG). Supports OpenAI, Gemini, and Vertex AI.

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
    use_vertex_ai=True,
    vertex_project=PROJECT_ID,
    vertex_location="global",
)

# Automatically routes to the right provider based on model name
response = await llm.generate_response(input=messages, model="gpt-4o")
response = await llm.generate_response(input=messages, model="gemini-2.5-flash")
```

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

- `generate_response(input, model, function_tools?, tool_executor?, text_format?, temperature?, metric_tag?) -> LLMResponse` (async)
- `generate_json(messages, model, temperature?, text_format?) -> dict` (async)

### `LLMResponse` TypedDict

```python
{
    "text": str,                          # Generated text
    "cited_chunks": list[SourceChunk],    # Source attributions (if file_search used)
    "tool_calls": list[ToolCall],         # Pending tool calls (if no executor)
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
