# EmailService

IMAP-based email listener using IDLE for real-time email notifications. Supports Gmail, Outlook, and custom IMAP servers.

## Setup

### Env Vars

Add to `sunset.env.yaml`:

```yaml
secrets:
  EMAIL_ADDRESS: "bot@example.com"
  EMAIL_PASSWORD: "app-specific-password"
  IMAP_HOST: "imap.gmail.com"   # Optional, default
  IMAP_PORT: "993"              # Optional, default
```

For Gmail, use an App Password (not your regular password).

## Usage

```python
from sunset.services.email import EmailService

email_service = EmailService()

# Register callback for new emails
async def handle_email(parsed_email):
    print(f"From: {parsed_email['from']}")
    print(f"Subject: {parsed_email['subject']}")
    print(f"Text: {parsed_email['text']}")
    for attachment in parsed_email['attachments']:
        print(f"Attachment: {attachment['filename']} ({attachment['size']} bytes)")

email_service.on_new_email(handle_email)

# Start listening (runs IMAP IDLE loop with auto-reconnect)
await email_service.start_listener()

# Fetch recent emails manually
emails = await email_service.fetch_recent_emails(count=10)

# Stop on shutdown
await email_service.stop_listener()
```

### Parsed email structure

```python
{
    "message_id": str,
    "from": str,
    "to": str,
    "cc": str | None,
    "subject": str,
    "date": str,
    "text": str,       # Plain text body
    "html": str,       # HTML body
    "attachments": [{"filename": str, "content_type": str, "size": int, "data": bytes}],
    "reply_to": str | None,
    "in_reply_to": str | None,
}
```

## API Reference

### `EmailService()`

Singleton. No constructor args — reads secrets via `SecretsService`.

### Key Methods

- `on_new_email(callback)` — Register async callback for incoming emails
- `start_listener()` / `stop_listener()` — IMAP IDLE loop with auto-reconnect (async)
- `fetch_recent_emails(count=10) -> list[dict]` (async)
- `mark_as_read(message_id)` (async)
