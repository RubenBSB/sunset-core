# Observability Service

OpenTelemetry bootstrap: traces + metrics over OTLP/HTTP to Grafana Cloud (or any OTLP endpoint), with auto-instrumentation for FastAPI, asyncpg, redis, and httpx. No-ops cleanly when no endpoint is configured, so it is always safe to call.

Install the extra: `pip install "sunsetai[observability]"`.

## Setup

Wire it once at process startup, **before** FastAPI/worker objects are constructed:

```python
from sunset.services import init_observability, instrument_fastapi

init_observability(service_name="myproject-api")

app = FastAPI(...)
instrument_fastapi(app)
```

## Configuration

| Source | Key | Description |
|--------|-----|-------------|
| Env | `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP base URL (exporter appends `/v1/traces`, `/v1/metrics`) |
| Env | `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers (e.g. `Authorization=Basic ...` for Grafana Cloud) |
| Secret Manager | `otel-exporter-otlp-endpoint` | Fallback when env is unset (no Cloud Run env wiring needed) |
| Secret Manager | `otel-exporter-otlp-headers` | Fallback when env is unset |
| Env | `GIT_SHA` | Default `service.version` resource attribute |
| Env | `ENV` | Default `deployment.environment` resource attribute |

If no endpoint is found anywhere, observability is disabled and every call is a no-op.

## API

- `init_observability(service_name, service_version=None, environment=None)` — Configure providers + exporters, install auto-instrumentation. Idempotent.
- `instrument_fastapi(app)` — Instrument a FastAPI app. Call after app creation; no-op if disabled.

## Metrics (`observability.metrics`)

Metric names use `llm.*` / `conversations.*` namespaces; attributes follow OTel GenAI semconv (`gen_ai.system`, `gen_ai.request.model`) where applicable. These are emitted by `LLMServiceRouter` via `sunset.services.llm.instrumentation`:

| Metric | Type | Description |
|--------|------|-------------|
| `llm.request.total` | counter | One per fallback attempt, with `status` attribute |
| `llm.request.duration` | histogram (s) | Per-attempt call duration |
| `llm.tokens` | counter | Token usage; `token_type ∈ {input, output, total, thinking, cached}` |
| `llm.cost.usd` | counter (USD) | Cost computed from token usage + pricing table |
| `llm.fallback.advanced` | counter | Fallback chain advanced past a step |
| `llm.breaker.opened` / `llm.breaker.skipped` | counter | Circuit breaker activity |
| `llm.throttled` | counter | Steps skipped on local concurrency semaphore timeout |
| `conversations.active` | gauge | Distinct conversations active in the last ~60s (Redis-backed) |

### Active-conversations gauge

```python
from sunset.services.observability import metrics as obs_metrics

# In FastAPI lifespan (same event loop as the redis client):
obs_metrics.register_redis(redis_service)            # starts the 10s poller
obs_metrics.register_redis(redis_service, start_poller=False)  # writes only (CPU-throttled Cloud Run)

# In request handlers:
await obs_metrics.mark_conversation_active(conversation_id)
```

Only processes running the poller export the gauge, so non-polling writers don't pollute the metric with frozen values.

## Dependencies

`opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`, and the `fastapi` / `asyncpg` / `redis` / `httpx` instrumentation packages (all in the `observability` extra). Each auto-instrumentation fails soft with a warning if its target package is missing.
