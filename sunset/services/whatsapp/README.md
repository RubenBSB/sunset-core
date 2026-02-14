# WhatsAppService

WhatsApp messaging via Meta Graph API or Twilio. Handles message deduplication, typing indicators, media downloads, and webhook processing.

## Setup

### Env Vars

**Meta Graph API** (WhatsAppService):

```yaml
secrets:
  WHATSAPP_PHONE_NUMBER_ID: "123456789"
  WHATSAPP_TOKEN: "EAA..."
```

**Twilio** (TwilioService):

```yaml
secrets:
  TWILIO_ACCOUNT_SID: "AC..."
  TWILIO_AUTH_TOKEN: "..."
  TWILIO_WHATSAPP_FROM: "+14155238886"
```

## Usage

### Meta Graph API

```python
from sunset.services.whatsapp import WhatsappService, extract_webhook_message

wa = WhatsappService.get_instance()

# In your webhook endpoint
@router.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    body = await request.json()
    message_data = extract_webhook_message(body)

    if not message_data or wa.is_duplicate(message_data["id"]):
        return {"status": "ok"}

    async with httpx.AsyncClient() as client:
        await wa.process_webhook_message(
            message_data=message_data,
            http_client=client,
            message_handler=handle_message,
        )
    return {"status": "ok"}

async def handle_message(data: dict) -> str | None:
    """Process message and return reply text."""
    return f"You said: {data['text']}"
```

### Twilio

```python
from sunset.services.whatsapp import TwilioService

twilio = TwilioService.get_instance()
twilio.send_message(to="+33612345678", body="Hello from Twilio!")
```

## API Reference

### `WhatsappService`

Singleton. Reads `WHATSAPP_PHONE_NUMBER_ID` and `WHATSAPP_TOKEN` from secrets.

- `is_duplicate(message_id) -> bool` — Deduplication check
- `download_media(http_client, media_id) -> str` — Download as base64 data URL (async)
- `send_typing_indicator(http_client, message_id)` (async)
- `send_message(http_client, to, text)` (async)
- `process_webhook_message(message_data, http_client, message_handler)` (async)

### `extract_webhook_message(body) -> dict | None`

Extract message data from webhook payload. Returns `{id, name, sender, text, image_media_id}`.

### `TwilioService`

Singleton. Reads Twilio credentials from secrets.

- `send_message(to, body) -> None`
