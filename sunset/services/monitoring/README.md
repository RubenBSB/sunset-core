# MonitoringService

Pushes LLM token usage metrics to Google Cloud Monitoring.

## Setup

### Infrastructure

No specific `sunset.yaml` entries. Uses the GCP project already provisioned. Cloud Monitoring API is enabled by default.

The service auto-creates the custom metric descriptor `custom.googleapis.com/llm/token_usage` on first use.

## Usage

Typically passed to an LLM service for automatic token tracking:

```python
from sunset.services import MonitoringService
from sunset.services.llm import VertexAIGeminiService

monitoring = MonitoringService(project=GCP_PROJECT_ID)

llm = VertexAIGeminiService(
    project=GCP_PROJECT_ID,
    location="global",
    monitoring=monitoring,  # Automatic token tracking
)
```

Or use directly for custom metrics:

```python
monitoring.write_token_metric(
    model="gemini-2.5-flash",
    method="generate_response",
    prompt_tokens=150,
    completion_tokens=200,
    total_tokens=350,
    tag="chat",
)
```

Fire-and-forget: errors are logged, never raised. If `google-cloud-monitoring` is not installed, all operations are no-ops.

## API Reference

### `MonitoringService(project)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `project` | `str` | GCP project ID |

### `write_token_metric(model, method, prompt_tokens, completion_tokens, total_tokens, thinking_tokens=0, cached_tokens=0, tag="")`

Write token usage metrics. Labels: `model`, `method`, `token_type`, `tag`.
