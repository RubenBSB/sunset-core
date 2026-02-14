# PubSubService

Google Cloud Pub/Sub messaging with subscriber management and SSE notification support.

## Setup

### Infrastructure

In `sunset.yaml`:

```yaml
infra:
  pubsub_topics:
    - extraction
    - notifications
  notification_topics:
    - alerts  # Topics that push to /internal/pubsub endpoint
  workers:
    - name: extraction-worker
      topics: [extraction]
      cpu: "2000m"
      memory: "2Gi"
```

`sunset provision` creates topics, subscriptions, and worker Cloud Run services.

### Env Vars

Automatically set by the Docker compose environment:

- `ENV` — Environment name
- `PUBSUB_EMULATOR_HOST` — Set automatically in local dev (emulator)
- `PUBSUB_PROJECT_ID` — GCP project ID
- `PUBSUB_TOPIC_PREFIX` — Project name prefix for topic paths

## Usage

### Publishing messages

```python
from sunset.services import PubSubService

pubsub = PubSubService()

pubsub.publish_message(
    message={"file_id": "abc123", "action": "process"},
    topic_name="extraction",
)
```

### Subscribing to messages (in workers)

```python
pubsub = PubSubService()

async def handle_extraction(data: dict):
    file_id = data["file_id"]
    # Process the file...

pubsub.register_handler("extraction", handle_extraction)
await pubsub.start_subscriber("extraction")
```

### SSE connections (real-time notifications)

```python
queue = pubsub.add_connection(user_id=str(user.id))

async def event_generator():
    while True:
        message = await queue.get()
        yield f"data: {json.dumps(message)}\n\n"

# On disconnect
pubsub.remove_connection(user_id=str(user.id))
```

## API Reference

### `PubSubService()`

Singleton. No constructor args — reads from environment.

### Key Methods

- `publish_message(message, topic_name) -> None` — Publish a dict as JSON
- `register_handler(subscription_key, handler)` — Register async handler for a topic
- `start_subscriber(topic_name)` / `stop_subscriber(topic_name?)` — Manage subscribers (async)
- `add_connection(user_id) -> asyncio.Queue` / `remove_connection(user_id)` — SSE connections
- `get_topic_path(topic_name) -> str` / `get_subscription_path(topic_name) -> str`
