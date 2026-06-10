# RedisService

Async Redis client for pub/sub signaling and general use.

## Setup

### Infrastructure

In `sunset.yaml`, enable Redis:

```yaml
infra:
  redis: true
```

`sunset provision` creates a Redis Memorystore instance (Basic tier, 1GB, Redis 7.x) with a VPC connector so Cloud Run services can reach it.

### Local Development

When `redis: true` is configured, `sunset run` starts a `redis:7-alpine` container on port 6379 and sets `REDIS_URL=redis://redis:6379` on all API and worker containers.

### Env Vars

| Variable | Set By | Value |
|----------|--------|-------|
| `REDIS_URL` | Docker Compose (local) | `redis://redis:6379` |
| `REDIS_URL` | Terraform (prod/staging) | `redis://{memorystore_host}:{port}` |

## Usage

```python
from sunset.services import RedisService

redis = RedisService()

# In your FastAPI lifespan:
async def lifespan(app):
    await redis.connect()
    yield
    await redis.close()

# Publish
await redis.publish("events", {"type": "job_complete", "id": "123"})

# Subscribe (async iterator)
async for message in redis.subscribe("events"):
    print(message)  # {"type": "job_complete", "id": "123"}

# Direct client access for other operations
client = redis.client
await client.set("rate:user:123", "1", ex=60)
count = await client.incr("rate:user:123")
```

## Rate Limiting

Built-in decorator for per-IP rate limiting:

```python
from sunset.services.redis.rate_limit import rate_limit

# 10 requests per 60 seconds on this route
@router.post("/expensive")
@rate_limit(limit=10, window=60)
async def expensive():
    ...

# Default: 100 requests per 60 seconds
@router.get("/data")
@rate_limit()
async def get_data():
    ...
```

Keys by client IP + route path. Returns `429 Too Many Requests` with `Retry-After` header when exceeded. Silently passes through if Redis is unavailable.

### Global Rate Limit Middleware

Safety net that applies to all routes **without** a `@rate_limit` decorator:

```python
from sunset.services.redis.rate_limit import RateLimitMiddleware

# In your FastAPI app setup:
app.add_middleware(RateLimitMiddleware, limit=200, window=60)
```

Keys by client IP globally. Routes with `@rate_limit` are skipped (they use their own limits). Silently passes through if Redis is unavailable.

## API Reference

### `RedisService()`

Singleton. Reads `REDIS_URL` from env (default: `redis://localhost:6379`).

### Methods

- `connect() -> Redis` — Connect and return the underlying `redis.asyncio` client. Connections use TCP keepalive, a 10s connect timeout, a 30s health-check interval, and retry-on-timeout — Memorystore silently drops idle connections, which would otherwise surface as `TimeoutError` on the next read.
- `close()` — Close the connection
- `publish(channel, data) -> int` — JSON-serialize and publish; returns subscriber count
- `subscribe(channel) -> AsyncIterator[dict]` — Yields parsed JSON messages from channel

### Properties

- `client -> Redis` — The underlying `redis.asyncio.Redis` instance (raises if not connected)

## Dependencies

```
redis>=5.0.0
```

Install with: `pip install sunset[redis]` or `pip install sunset[all]`
