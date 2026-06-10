"""Transactional email sending — pluggable engine (Resend or SendGrid).

Outbound counterpart to the inbound IMAP `EmailService` in this same package:
this sends transactional mail (invitations, magic links, notifications).

Config (environment variable or Secret Manager key):
    EMAIL_ENGINE      "resend" (default) or "sendgrid"
    EMAIL_FROM        default sender, e.g. "Acme <hello@acme.com>"
    RESEND_API_KEY    when engine = resend
    SENDGRID_API_KEY  when engine = sendgrid

Engine SDKs are imported lazily, so only the selected engine's package needs to
be installed. Degrades gracefully: with no API key it logs and returns False —
delivery is best-effort and the caller's flow must never fail on email.

Usage:
    from sunset.services import EmailSendService
    await EmailSendService.get_instance().send(
        to="a@b.com", subject="Hi", html="<p>Hello</p>"
    )
"""

import asyncio
import logging
from typing import Optional

from sunset.services.secrets import get_secrets

logger = logging.getLogger(__name__)

_sender: "Optional[EmailSendService]" = None


def get_email_sender() -> "EmailSendService":
    """Get the transactional email sender singleton."""
    global _sender
    if _sender is None:
        _sender = EmailSendService()
    return _sender


class EmailSendService:
    _instance = None

    def __init__(self):
        secrets = get_secrets()
        self.engine = (secrets.get_secret("EMAIL_ENGINE", "resend") or "resend").lower()
        self.from_email = secrets.get_secret("EMAIL_FROM", "onboarding@resend.dev")
        self.resend_api_key = secrets.get_secret("RESEND_API_KEY", "")
        self.sendgrid_api_key = secrets.get_secret("SENDGRID_API_KEY", "")

    @classmethod
    def get_instance(cls) -> "EmailSendService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _api_key(self) -> str:
        return (
            self.sendgrid_api_key if self.engine == "sendgrid" else self.resend_api_key
        )

    def is_enabled(self) -> bool:
        """True when the selected engine has an API key configured."""
        return bool(self._api_key())

    async def send(
        self,
        *,
        to: str,
        subject: str,
        html: str,
        from_email: Optional[str] = None,
        text: Optional[str] = None,
    ) -> bool:
        """Send one HTML email. Returns True on a successful send.

        Runs the blocking SDK call off the event loop. Never raises — no key or
        a provider error logs and returns False.
        """
        sender = from_email or self.from_email
        if not self.is_enabled():
            logger.warning(
                "EmailSendService: no API key for engine '%s' — email to %s "
                "not sent (subject: %s).",
                self.engine,
                to,
                subject,
            )
            return False
        try:
            return await asyncio.to_thread(
                self._send_sync, sender, to, subject, html, text
            )
        except Exception as e:
            logger.error(
                "EmailSendService: failed to send to %s via %s: %s",
                to,
                self.engine,
                e,
            )
            return False

    def _send_sync(
        self,
        sender: str,
        to: str,
        subject: str,
        html: str,
        text: Optional[str],
    ) -> bool:
        if self.engine == "resend":
            import resend

            resend.api_key = self.resend_api_key
            payload = {"from": sender, "to": [to], "subject": subject, "html": html}
            if text:
                payload["text"] = text
            resend.Emails.send(payload)
            return True
        if self.engine == "sendgrid":
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            message = Mail(
                from_email=sender,
                to_emails=to,
                subject=subject,
                html_content=html,
            )
            response = SendGridAPIClient(self.sendgrid_api_key).send(message)
            return response.status_code in (200, 201, 202)
        raise ValueError(
            f"Unknown EMAIL_ENGINE: {self.engine!r} (use 'resend' or 'sendgrid')"
        )
