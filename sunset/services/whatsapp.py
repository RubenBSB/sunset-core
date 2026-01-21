"""
WhatsApp service for handling Meta Business API interactions.

Project-agnostic service that handles:
- Message deduplication
- Typing indicators
- Media handling
- Message sending

Business logic (reply generation) is injected via callback.

Usage:
    from sunset.services import WhatsAppService, SecretsService

    secrets = SecretsService()
    whatsapp = WhatsAppService(
        phone_number_id=secrets.get_secret("WHATSAPP_PHONE_NUMBER_ID"),
        token=secrets.get_secret("WHATSAPP_TOKEN"),
    )

    await whatsapp.send_message(http_client, "+1234567890", "Hello!")
"""

import asyncio
import base64
import logging
import re
from typing import Any, Awaitable, Callable, Dict, Optional
from collections import OrderedDict
from threading import Lock

import httpx
from fastapi import HTTPException, status


logger = logging.getLogger(__name__)

MessageHandler = Callable[[Dict[str, Any]], Awaitable[Optional[str]]]


class WhatsAppService:
    """Service for WhatsApp bot operations."""

    MAX_CACHE_SIZE = 1000
    TYPING_INDICATOR_DELAY = 1.5
    GRAPH_API_VERSION = "v19.0"

    def __init__(self, phone_number_id: str, token: str):
        self._phone_number_id = phone_number_id
        self._token = token
        self._processed_messages: OrderedDict[str, bool] = OrderedDict()
        self._cache_lock = Lock()
        logger.info("WhatsApp service initialized")

    def is_duplicate(self, message_id: Optional[str]) -> bool:
        """Check if message was already processed."""
        if not message_id:
            return False

        with self._cache_lock:
            if message_id in self._processed_messages:
                return True

            self._processed_messages[message_id] = True
            if len(self._processed_messages) > self.MAX_CACHE_SIZE:
                self._processed_messages.popitem(last=False)

            return False

    def _convert_to_whatsapp_markdown(self, text: str) -> str:
        """Convert standard markdown to WhatsApp format."""
        code_blocks = []

        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        processed = re.sub(r"```[\s\S]*?```", save_code_block, text)

        inline_codes = []

        def save_inline_code(match):
            inline_codes.append(match.group(0))
            return f"__INLINE_CODE_{len(inline_codes) - 1}__"

        processed = re.sub(r"`[^`]+`", save_inline_code, processed)

        processed = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", processed)
        processed = re.sub(r"~~([^~]+)~~", r"~\1~", processed)

        for i, code in enumerate(inline_codes):
            processed = processed.replace(f"__INLINE_CODE_{i}__", code)
        for i, block in enumerate(code_blocks):
            processed = processed.replace(f"__CODE_BLOCK_{i}__", block)

        return processed

    async def download_media(
        self, http_client: httpx.AsyncClient, media_id: str
    ) -> str:
        """Download WhatsApp media and return as base64 data URL."""
        headers = {"Authorization": f"Bearer {self._token}"}
        meta_url = f"https://graph.facebook.com/{self.GRAPH_API_VERSION}/{media_id}"

        try:
            meta_response = await http_client.get(meta_url, headers=headers)
            meta_response.raise_for_status()
            media_url = meta_response.json().get("url")
            if not media_url:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to resolve media URL",
                )

            media_response = await http_client.get(media_url, headers=headers)
            media_response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.exception(f"WhatsApp media request failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Media request failed"
            )

        mime_type = media_response.headers.get("Content-Type", "image/jpeg")
        data = base64.b64encode(media_response.content).decode("ascii")
        return f"data:{mime_type};base64,{data}"

    async def send_typing_indicator(
        self, http_client: httpx.AsyncClient, message_id: str
    ) -> None:
        """Send typing indicator and mark message as read."""
        url = f"https://graph.facebook.com/{self.GRAPH_API_VERSION}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
            "typing_indicator": {"type": "text"},
        }

        try:
            response = await http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f"Failed to send typing indicator: {exc}")

    async def send_message(
        self, http_client: httpx.AsyncClient, to: str, text: str
    ) -> None:
        """Send a WhatsApp text message."""
        whatsapp_text = self._convert_to_whatsapp_markdown(text)

        url = f"https://graph.facebook.com/{self.GRAPH_API_VERSION}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": whatsapp_text[:4096]},
        }

        try:
            response = await http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.exception(f"WhatsApp API request failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="WhatsApp API request failed",
            )

    async def process_webhook_message(
        self,
        message_data: Dict[str, Any],
        http_client: httpx.AsyncClient,
        message_handler: MessageHandler,
    ) -> None:
        """Process incoming WhatsApp webhook message and send reply."""
        typing_task = None
        sender = message_data["sender"]
        message_id = message_data.get("id")

        try:
            if message_id:
                typing_task = asyncio.create_task(
                    self._delayed_typing_indicator(http_client, message_id)
                )

            reply = await message_handler(message_data)

            if typing_task and not typing_task.done():
                typing_task.cancel()

            if reply:
                await self.send_message(http_client, sender, reply)

        except Exception as exc:
            logger.exception(f"Message processing error: {exc}")
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()

    async def _delayed_typing_indicator(
        self, http_client: httpx.AsyncClient, message_id: str
    ) -> None:
        """Send typing indicator after delay."""
        await asyncio.sleep(self.TYPING_INDICATOR_DELAY)
        await self.send_typing_indicator(http_client, message_id)


def extract_webhook_message(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract message data from WhatsApp webhook payload."""
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            contacts = value.get("contacts", [])
            name = contacts[0].get("profile", {}).get("name") if contacts else None

            for message in value.get("messages", []):
                msg_type = message.get("type")
                sender = message.get("from")
                msg_id = message.get("id")
                if not sender:
                    continue

                if msg_type == "text":
                    text_body = message.get("text", {}).get("body")
                    if text_body:
                        return {
                            "id": msg_id,
                            "name": name,
                            "sender": sender,
                            "text": text_body.strip(),
                            "image_media_id": None,
                        }

                if msg_type == "image":
                    image = message.get("image", {}) or {}
                    media_id = image.get("id")
                    if media_id:
                        return {
                            "id": msg_id,
                            "name": name,
                            "sender": sender,
                            "text": (image.get("caption") or "").strip() or None,
                            "image_media_id": media_id,
                        }
    return None
