# AnalyticsService

PostHog event tracking: user identification, login/logout, sessions, and custom events.

## Setup

### Env Vars

Add to `sunset.env.yaml`:

```yaml
secrets:
  POSTHOG_API_KEY: "phc_..."
```

Optionally set a custom host:

```yaml
local:
  POSTHOG_HOST: "https://eu.posthog.com"  # Default: https://app.posthog.com
```

If `POSTHOG_API_KEY` is not set, all tracking calls are silently skipped.

## Usage

```python
from sunset.services import AnalyticsService

analytics = AnalyticsService()

# Track custom event
analytics.track_event(
    user_id=str(user.id),
    event="document_uploaded",
    properties={"file_type": "pdf", "size_mb": 2.5},
)

# Built-in events
analytics.track_user_login(user_id=str(user.id), email=user.email, login_method="google")
analytics.track_user_logout(user_id=str(user.id))
analytics.track_session_start(user_id=str(user.id))
analytics.track_session_end(user_id=str(user.id), session_duration_seconds=300)

# Identify user with properties
analytics.identify_user(user_id=str(user.id), properties={"plan": "pro", "school": "MIT"})
```

## API Reference

### `AnalyticsService()`

No constructor args. Reads `POSTHOG_API_KEY` and `POSTHOG_HOST` from environment.

### Key Methods

- `is_enabled() -> bool`
- `track_event(user_id, event, properties?) -> None`
- `track_user_login(user_id, email, login_method?) -> None`
- `track_user_logout(user_id) -> None`
- `track_session_start(user_id) -> None`
- `track_session_end(user_id, session_duration_seconds?) -> None`
- `identify_user(user_id, properties) -> None`
