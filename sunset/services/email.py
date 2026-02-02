import logging
import asyncio
import email
from email.header import decode_header
from typing import Optional, Callable, List, Dict, Any

from aioimaplib import IMAP4_SSL

from sunset.services.secrets import get_secrets

logger = logging.getLogger(__name__)


class EmailService:
    """
    IMAP-based email service that listens for new emails using IDLE.
    Supports any IMAP server (Gmail, Outlook, custom servers).

    Required secrets:
        - IMAP_HOST: IMAP server hostname (default: imap.gmail.com)
        - IMAP_PORT: IMAP server port (default: 993)
        - EMAIL_ADDRESS: Email address to connect with
        - EMAIL_PASSWORD: Email password or app-specific password
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmailService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            secrets = get_secrets()

            self.imap_host = secrets.get_secret("IMAP_HOST", "imap.gmail.com")
            self.imap_port = int(secrets.get_secret("IMAP_PORT", "993"))
            self.email_address = secrets.get_secret("EMAIL_ADDRESS")
            self.email_password = secrets.get_secret("EMAIL_PASSWORD")

            self._client: Optional[IMAP4_SSL] = None
            self._listener_task: Optional[asyncio.Task] = None
            self._shutdown_event = asyncio.Event()
            self._on_email_callbacks: List[Callable] = []

            self._initialized = True
            logger.info(f"EmailService initialized for {self.email_address}")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of EmailService"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def on_new_email(self, callback: Callable):
        """Register a callback for new emails"""
        self._on_email_callbacks.append(callback)

    async def start_listener(self):
        """Start listening for new emails using IMAP IDLE"""
        if self._listener_task is not None:
            logger.warning("Email listener already running")
            return

        self._shutdown_event.clear()
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info("Email listener started")

    async def stop_listener(self):
        """Stop the email listener gracefully"""
        if self._listener_task is None:
            logger.warning("Email listener not running")
            return

        logger.info("Stopping email listener...")
        self._shutdown_event.set()

        if self._client:
            try:
                await self._client.logout()
            except Exception as e:
                logger.warning(f"Error during IMAP logout: {e}")

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        self._listener_task = None
        self._client = None
        logger.info("Email listener stopped")

    async def _connect(self):
        """Connect and authenticate to IMAP server"""
        self._client = IMAP4_SSL(host=self.imap_host, port=self.imap_port)
        await self._client.wait_hello_from_server()

        response = await self._client.login(self.email_address, self.email_password)
        if response.result != "OK":
            raise Exception(f"IMAP login failed: {response}")

        response = await self._client.select("INBOX")
        if response.result != "OK":
            raise Exception(f"Failed to select INBOX: {response}")

        logger.info(f"Connected to IMAP server {self.imap_host}")

    async def _listen_loop(self):
        """Main loop using IMAP IDLE to wait for new emails"""
        retry_delay = 1
        max_retry_delay = 60
        initial_sync_done = False

        while not self._shutdown_event.is_set():
            try:
                if self._client is None:
                    await self._connect()

                # Initial sync: fetch existing unread emails on first connection
                if not initial_sync_done:
                    logger.info("Performing initial sync of unread emails...")
                    await self._process_new_emails()
                    initial_sync_done = True

                # Start IDLE mode (wait up to 5 minutes for server push)
                idle_task = await self._client.idle_start(timeout=300)

                # Wait for new email notification or timeout
                msg = await self._client.wait_server_push()
                logger.info(f"IMAP server push received: {msg}")

                # Stop IDLE to process emails
                self._client.idle_done()
                await asyncio.wait_for(idle_task, timeout=10)

                # Fetch and process new unseen emails
                await self._process_new_emails()

                retry_delay = 1  # Reset on success

            except asyncio.CancelledError:
                logger.info("Email listener cancelled")
                raise
            except Exception as e:
                if self._shutdown_event.is_set():
                    break

                logger.error(f"IMAP error: {e}. Retrying in {retry_delay}s...")
                self._client = None

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=retry_delay
                    )
                    break  # Shutdown requested during retry
                except asyncio.TimeoutError:
                    pass

                retry_delay = min(retry_delay * 2, max_retry_delay)

        logger.info("Email listen loop ended")

    async def _process_new_emails(self):
        """Fetch and process new unseen emails"""
        response = await self._client.search("UNSEEN")
        if response.result != "OK":
            logger.error(f"Failed to search emails: {response}")
            return

        # Parse email IDs from response
        email_ids = response.lines[0].split() if response.lines else []
        logger.info(f"Found {len(email_ids)} unseen emails")

        for email_id in email_ids:
            try:
                email_id_str = (
                    email_id.decode() if isinstance(email_id, bytes) else email_id
                )
                response = await self._client.fetch(email_id_str, "(RFC822)")

                if response.result != "OK":
                    logger.error(f"Failed to fetch email {email_id_str}: {response}")
                    continue

                # Extract raw email from response
                raw_email = self._extract_raw_email(response)
                if raw_email is None:
                    continue

                # Parse the email
                parsed = self._parse_email(raw_email)
                logger.info(f"New email from {parsed['from']}: {parsed['subject']}")

                # Notify all registered callbacks
                for callback in self._on_email_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(parsed)
                        else:
                            callback(parsed)
                    except Exception as e:
                        logger.error(f"Error in email callback: {e}")

            except Exception as e:
                logger.error(f"Error processing email {email_id}: {e}")

    def _extract_raw_email(self, response) -> Optional[bytes]:
        """Extract raw email bytes from IMAP fetch response"""
        # aioimaplib FETCH response structure:
        # lines[0]: b'659 FETCH (RFC822 {5634}' (metadata)
        # lines[1]: bytearray with actual email content
        # lines[2]: b' FLAGS (\\Seen))'
        # lines[3]: b'Success'

        for line in response.lines:
            # Handle both bytes and bytearray
            if isinstance(line, (bytes, bytearray)):
                # Convert bytearray to bytes if needed
                data = bytes(line) if isinstance(line, bytearray) else line

                # Skip IMAP metadata lines
                if b"FETCH" in data and b"RFC822" in data:
                    continue
                if b"FLAGS" in data:
                    continue
                if data in (b")", b"Success"):
                    continue

                # This should be the email content - look for email headers
                if len(data) > 100 and (
                    b"From:" in data or b"Subject:" in data or b"MIME-Version:" in data
                ):
                    return data

        logger.error(
            f"Could not extract email from response: {[type(line) for line in response.lines]}"
        )
        return None

    def _parse_email(self, raw_email: bytes) -> Dict[str, Any]:
        """Parse raw email bytes into structured data"""
        msg = email.message_from_bytes(raw_email)

        # Decode subject
        subject = ""
        if msg["Subject"]:
            decoded_parts = decode_header(msg["Subject"])
            subject_parts = []
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    subject_parts.append(
                        part.decode(encoding or "utf-8", errors="replace")
                    )
                else:
                    subject_parts.append(part)
            subject = "".join(subject_parts)

        # Get body content
        body_text = ""
        body_html = ""
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition") or "")

                if "attachment" in disposition or (
                    part.get_filename()
                    and content_type not in ["text/plain", "text/html"]
                ):
                    # Decode filename if encoded
                    filename = part.get_filename()
                    if filename:
                        decoded_parts = decode_header(filename)
                        filename_parts = []
                        for part_val, encoding in decoded_parts:
                            if isinstance(part_val, bytes):
                                filename_parts.append(
                                    part_val.decode(
                                        encoding or "utf-8", errors="replace"
                                    )
                                )
                            else:
                                filename_parts.append(part_val)
                        filename = "".join(filename_parts)

                    payload = part.get_payload(decode=True)
                    attachments.append(
                        {
                            "filename": filename or "unnamed",
                            "content_type": content_type,
                            "size": len(payload) if payload else 0,
                            "data": payload,
                        }
                    )
                elif content_type == "text/plain" and not body_text:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_text = payload.decode(charset, errors="replace")
                elif content_type == "text/html" and not body_html:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_html = payload.decode(charset, errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body_text = payload.decode(charset, errors="replace")

        return {
            "message_id": msg["Message-ID"],
            "from": msg["From"],
            "to": msg["To"],
            "cc": msg.get("Cc"),
            "subject": subject,
            "date": msg["Date"],
            "text": body_text,
            "html": body_html,
            "attachments": attachments,
            "reply_to": msg.get("Reply-To"),
            "in_reply_to": msg.get("In-Reply-To"),
        }

    async def fetch_recent_emails(self, count: int = 10) -> List[Dict[str, Any]]:
        """Fetch the most recent emails (utility method)"""
        if self._client is None:
            await self._connect()

        response = await self._client.search("ALL")
        if response.result != "OK":
            return []

        email_ids = response.lines[0].split() if response.lines else []
        recent_ids = email_ids[-count:] if len(email_ids) > count else email_ids

        emails = []
        for email_id in reversed(recent_ids):
            try:
                email_id_str = (
                    email_id.decode() if isinstance(email_id, bytes) else email_id
                )
                response = await self._client.fetch(email_id_str, "(RFC822)")
                if response.result == "OK":
                    raw_email = self._extract_raw_email(response)
                    if raw_email:
                        emails.append(self._parse_email(raw_email))
            except Exception as e:
                logger.error(f"Error fetching email {email_id}: {e}")

        return emails

    async def mark_as_read(self, message_id: str):
        """Mark an email as read by message ID"""
        if self._client is None:
            await self._connect()

        # Search for the email by Message-ID
        response = await self._client.search(f'HEADER Message-ID "{message_id}"')
        if response.result != "OK" or not response.lines:
            logger.warning(f"Email not found: {message_id}")
            return

        email_ids = response.lines[0].split()
        for email_id in email_ids:
            email_id_str = (
                email_id.decode() if isinstance(email_id, bytes) else email_id
            )
            await self._client.store(email_id_str, "+FLAGS", "\\Seen")
            logger.info(f"Marked email as read: {message_id}")


def get_email_service() -> EmailService:
    """Get the email service singleton."""
    return EmailService.get_instance()
