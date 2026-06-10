# WhatsAppService

WhatsApp messaging via Meta Graph API. Handles message deduplication, typing indicators, media downloads, and webhook processing.

## Setup

### Env Vars

```yaml
secrets:
  WHATSAPP_PHONE_NUMBER_ID: "123456789"
  WHATSAPP_TOKEN: "EAA..."
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

## API Reference

### `WhatsappService`

Singleton. Reads `WHATSAPP_PHONE_NUMBER_ID` and `WHATSAPP_TOKEN` from secrets.

- `is_duplicate(message_id) -> bool` — Deduplication check
- `download_media(http_client, media_id) -> str` — Download as base64 data URL (async)
- `send_typing_indicator(http_client, message_id)` (async)
- `send_message(http_client, to, text)` (async)
- `process_webhook_message(message_data, http_client, message_handler)` (async)

### `extract_webhook_message(body) -> dict | None`

Extract message data from webhook payload. Returns `{id, name, sender, type, text, image_media_id, audio_media_id}` where `type` is `"text" | "image" | "audio"` — dispatch on it instead of re-inspecting the payload.
