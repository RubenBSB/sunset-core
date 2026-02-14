# APNsService

Apple Push Notification service via HTTP/2. Supports standard and silent notifications.

## Setup

### Env Vars

Add to `sunset.env.yaml`:

```yaml
secrets:
  APPLE_NOTIFICATION_KEY_ID: "ABC123"
  APPLE_TEAM_ID: "TEAM456"
  APPLE_BUNDLE_ID: "com.yourapp.bundle"
  APPLE_KEY_FILEPATH: "/path/to/AuthKey.p8"
```

Requires an APNs key (.p8 file) from the Apple Developer portal.

## Usage

```python
from sunset.services.apns import APNsService

apns = APNsService(
    key_id="ABC123",
    team_id="TEAM456",
    bundle_id="com.yourapp.bundle",
    key_file_path="/path/to/AuthKey.p8",
    use_sandbox=True,  # False for production
)

# Send push notification
success = await apns.send_notification(
    device_token="device-token-hex",
    title="New Message",
    body="You have a new message from John",
    badge=3,
    custom_data={"conversation_id": "abc123"},
)

# Silent notification (background refresh)
success = await apns.send_silent_notification(
    device_token="device-token-hex",
    custom_data={"action": "sync"},
)
```

Or use the singleton (reads from env):

```python
apns = APNsService.get_instance()
```

## API Reference

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key_id` | `str` | required | APNs key ID |
| `team_id` | `str` | required | Apple team ID |
| `bundle_id` | `str` | required | App bundle ID |
| `key_file_path` | `str` | required | Path to .p8 private key |
| `use_sandbox` | `bool` | `False` | Use sandbox APNs endpoint |

### Key Methods

- `send_notification(device_token, title, body, badge?, sound?, custom_data?, priority?, collapse_id?) -> bool` (async)
- `send_silent_notification(device_token, custom_data?) -> bool` (async)
