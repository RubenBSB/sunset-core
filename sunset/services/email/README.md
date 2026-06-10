# Email services

This package holds **two** independent services:

- **`EmailSendService`** — outbound transactional send (Resend / SendGrid). See [below](#emailsendservice).
- **`EmailService`** — inbound IMAP listener (IDLE) for receiving emails. See [below](#emailservice-imap).

Install deps via the extra: `pip install "sunsetai[email]"` (pulls resend, sendgrid, aioimaplib).

---

## EmailSendService

Outbound **transactional email** with a pluggable engine (Resend or SendGrid).

### Config

Read from environment variables (or Secret Manager in non-local envs):

| Key                | Purpose                                    | Default                 |
| ------------------ | ------------------------------------------ | ----------------------- |
| `EMAIL_ENGINE`     | `resend` or `sendgrid`                     | `resend`                |
| `EMAIL_FROM`       | Default sender (`"Acme <hello@acme.com>"`) | `onboarding@resend.dev` |
| `RESEND_API_KEY`   | API key when engine = `resend`             | —                       |
| `SENDGRID_API_KEY` | API key when engine = `sendgrid`           | —                       |

Engine SDKs are imported lazily, so only the selected engine's package must be installed.

### Usage

```python
from sunset.services import EmailSendService

ok = await EmailSendService.get_instance().send(
    to="user@example.com",
    subject="Welcome",
    html="<p>Hello!</p>",
    # from_email=...  # optional, overrides EMAIL_FROM
    # text=...        # optional plain-text part
)
```

`send()` runs the blocking SDK call off the event loop and **never raises**: with no API key
(or on a provider error) it logs and returns `False`, so a flow's success never depends on email
delivery. Switch providers by changing `EMAIL_ENGINE` — no code change.

> Resend rejects unverified `from` domains. `onboarding@resend.dev` works out of the box in test
> mode (delivers only to the account owner); set `EMAIL_FROM` to a verified-domain address for prod.

---

## EmailService (IMAP)

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
