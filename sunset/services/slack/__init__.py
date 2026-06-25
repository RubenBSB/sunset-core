"""Slack service for OAuth v2 + Web API.

Stateless — callers own token persistence (one TokenSet per workspace).
"""

import hashlib
import hmac
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

SLACK_API = "https://slack.com/api"


def slugify_channel_name(name: str, max_len: int = 80) -> str:
    """Coerce an arbitrary string into a Slack-legal channel name.

    Slack rules: lowercase, 1-80 chars, alphanumerics, hyphens, underscores,
    periods. We normalize to hyphens (the Slack convention) and strip the rest.
    """
    # Fold accents so "Château" → "Chateau" instead of getting stripped.
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    s = folded.strip().lower()
    s = re.sub(r"[\s_·•—–]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "channel"
    return s[:max_len]


class SlackError(Exception):
    """Raised when a Slack API call fails (HTTP or `ok: false`)."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        slack_error: Optional[str] = None,
    ):
        self.status_code = status_code
        self.slack_error = slack_error
        super().__init__(message)


@dataclass
class TokenSet:
    """OAuth result for a single workspace install."""

    access_token: str  # xoxb-… bot token
    team_id: str
    team_name: Optional[str] = None
    bot_user_id: Optional[str] = None
    scope: Optional[str] = None
    app_id: Optional[str] = None
    authed_user_id: Optional[str] = None


@dataclass
class Channel:
    id: str
    name: str
    is_private: bool


@dataclass
class SlackUser:
    id: str
    email: Optional[str]
    name: Optional[str] = None
    real_name: Optional[str] = None


@dataclass
class PostedMessage:
    channel: str
    ts: str


# Default bot scopes for a sunset-style integration. Override in get_install_url
# if a project needs more.
DEFAULT_BOT_SCOPES: tuple[str, ...] = (
    "groups:write",  # create private channels + invite to them
    "channels:manage",  # create public channels too (cheap to include)
    "chat:write",  # post messages
    "users:read",  # lookup users
    "users:read.email",  # lookup by email
)


class SlackService:
    """Async Slack API client. Stateless — callers own token persistence."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret

    def _http(self, timeout: float = 20.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout)

    def _auth_headers(self, access_token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    async def _call(
        self,
        client: httpx.AsyncClient,
        method: str,
        access_token: Optional[str] = None,
        *,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """POST/GET to api.slack.com and surface `ok: false` as SlackError."""
        url = f"{SLACK_API}/{method}"
        headers: dict[str, str] = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        if json is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
            resp = await client.post(url, headers=headers, json=json)
        elif data is not None:
            resp = await client.post(url, headers=headers, data=data)
        else:
            resp = await client.get(url, headers=headers, params=params)

        if resp.status_code >= 400:
            raise SlackError(
                f"Slack HTTP {resp.status_code} on {method}: {resp.text}",
                status_code=resp.status_code,
            )
        body = resp.json()
        if not body.get("ok"):
            err = body.get("error", "unknown_error")
            raise SlackError(
                f"Slack {method} returned ok=false: {err}",
                status_code=resp.status_code,
                slack_error=err,
            )
        return body

    # -------------------------------------------------------------------------
    # OAuth v2
    # -------------------------------------------------------------------------

    def get_install_url(
        self,
        redirect_uri: str,
        scopes: Optional[list[str]] = None,
        state: Optional[str] = None,
        user_scopes: Optional[list[str]] = None,
    ) -> str:
        params: dict[str, str] = {
            "client_id": self._client_id,
            "scope": ",".join(scopes or list(DEFAULT_BOT_SCOPES)),
            "redirect_uri": redirect_uri,
        }
        if user_scopes:
            params["user_scope"] = ",".join(user_scopes)
        if state:
            params["state"] = state
        return f"https://slack.com/oauth/v2/authorize?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> TokenSet:
        async with self._http() as client:
            body = await self._call(
                client,
                "oauth.v2.access",
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
        team = body.get("team") or {}
        authed_user = body.get("authed_user") or {}
        return TokenSet(
            access_token=body["access_token"],
            team_id=team.get("id") or body.get("team_id", ""),
            team_name=team.get("name"),
            bot_user_id=body.get("bot_user_id"),
            scope=body.get("scope"),
            app_id=body.get("app_id"),
            authed_user_id=authed_user.get("id"),
        )

    async def auth_test(self, access_token: str) -> dict[str, Any]:
        """Verify a token + return team/bot identity."""
        async with self._http() as client:
            return await self._call(client, "auth.test", access_token, json={})

    # -------------------------------------------------------------------------
    # Users
    # -------------------------------------------------------------------------

    async def lookup_user_by_email(
        self, access_token: str, email: str
    ) -> Optional[SlackUser]:
        """Resolve email → SlackUser. Returns None if not found (no error)."""
        try:
            async with self._http() as client:
                body = await self._call(
                    client,
                    "users.lookupByEmail",
                    access_token,
                    params={"email": email},
                )
        except SlackError as exc:
            if exc.slack_error == "users_not_found":
                return None
            raise
        u = body.get("user") or {}
        profile = u.get("profile") or {}
        return SlackUser(
            id=u["id"],
            email=profile.get("email") or email,
            name=u.get("name"),
            real_name=profile.get("real_name"),
        )

    async def get_user_info(
        self, access_token: str, user_id: str
    ) -> Optional[SlackUser]:
        """Resolve a Slack user_id → SlackUser (incl. email). None if missing.

        The reverse of `lookup_user_by_email` — used when an inbound event gives
        us a Slack user_id and we need their email to map back to an app user.
        Requires the `users:read.email` scope for the email to be populated.
        """
        try:
            async with self._http() as client:
                body = await self._call(
                    client, "users.info", access_token, params={"user": user_id}
                )
        except SlackError as exc:
            if exc.slack_error == "user_not_found":
                return None
            raise
        u = body.get("user") or {}
        profile = u.get("profile") or {}
        return SlackUser(
            id=u.get("id", user_id),
            email=profile.get("email"),
            name=u.get("name"),
            real_name=profile.get("real_name"),
        )

    # -------------------------------------------------------------------------
    # Conversations
    # -------------------------------------------------------------------------

    async def create_conversation(
        self, access_token: str, name: str, *, is_private: bool = True
    ) -> Channel:
        sanitized = slugify_channel_name(name)
        async with self._http() as client:
            body = await self._call(
                client,
                "conversations.create",
                access_token,
                json={"name": sanitized, "is_private": is_private},
            )
        c = body["channel"]
        return Channel(
            id=c["id"],
            name=c.get("name", sanitized),
            is_private=c.get("is_private", is_private),
        )

    async def invite_to_conversation(
        self, access_token: str, channel_id: str, user_ids: list[str]
    ) -> None:
        """Invite users to a channel. Silently ignores already-in errors."""
        if not user_ids:
            return
        async with self._http() as client:
            try:
                await self._call(
                    client,
                    "conversations.invite",
                    access_token,
                    json={"channel": channel_id, "users": ",".join(user_ids)},
                )
            except SlackError as exc:
                # Treat "already in channel" as a no-op so retries are idempotent.
                if exc.slack_error in {"already_in_channel", "cant_invite_self"}:
                    return
                raise

    async def rename_conversation(
        self, access_token: str, channel_id: str, name: str
    ) -> Channel:
        sanitized = slugify_channel_name(name)
        async with self._http() as client:
            body = await self._call(
                client,
                "conversations.rename",
                access_token,
                json={"channel": channel_id, "name": sanitized},
            )
        c = body["channel"]
        return Channel(
            id=c["id"],
            name=c.get("name", sanitized),
            is_private=c.get("is_private", True),
        )

    async def archive_conversation(self, access_token: str, channel_id: str) -> None:
        async with self._http() as client:
            await self._call(
                client,
                "conversations.archive",
                access_token,
                json={"channel": channel_id},
            )

    # -------------------------------------------------------------------------
    # Messages
    # -------------------------------------------------------------------------

    async def post_message(
        self,
        access_token: str,
        channel: str,
        text: str,
        *,
        blocks: Optional[list[dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
    ) -> PostedMessage:
        payload: dict[str, Any] = {"channel": channel, "text": text}
        if blocks is not None:
            payload["blocks"] = blocks
        if thread_ts is not None:
            payload["thread_ts"] = thread_ts
        async with self._http() as client:
            body = await self._call(
                client, "chat.postMessage", access_token, json=payload
            )
        return PostedMessage(channel=body["channel"], ts=body["ts"])

    async def update_message(
        self,
        access_token: str,
        channel: str,
        ts: str,
        text: str,
        *,
        blocks: Optional[list[dict[str, Any]]] = None,
    ) -> PostedMessage:
        """Edit a message previously posted by the bot (chat.update).

        Used to turn a placeholder ("🤔 …") into the final answer once the
        agent has finished — Slack requires a <3s ack, so we post fast then
        update in place.
        """
        payload: dict[str, Any] = {"channel": channel, "ts": ts, "text": text}
        if blocks is not None:
            payload["blocks"] = blocks
        async with self._http() as client:
            body = await self._call(
                client, "chat.update", access_token, json=payload
            )
        return PostedMessage(channel=body["channel"], ts=body["ts"])

    async def conversations_replies(
        self,
        access_token: str,
        channel: str,
        ts: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch the messages of a thread (oldest → newest).

        Returns raw Slack message dicts ({user, text, ts, bot_id?, …}); the
        caller decides how to turn them into LLM conversation turns.
        """
        async with self._http() as client:
            body = await self._call(
                client,
                "conversations.replies",
                access_token,
                params={"channel": channel, "ts": ts, "limit": limit},
            )
        return body.get("messages", [])

    # -------------------------------------------------------------------------
    # Inbound webhook security
    # -------------------------------------------------------------------------

    @staticmethod
    def verify_signature(
        signing_secret: str,
        timestamp: str,
        body: bytes,
        signature: str,
        *,
        max_skew_seconds: int = 60 * 5,
    ) -> bool:
        """Validate a Slack request signature (events, interactivity, commands).

        Slack signs `v0:<timestamp>:<raw_body>` with HMAC-SHA256 keyed on the
        app's signing secret. We reject stale timestamps (replay window) before
        comparing digests in constant time. `body` must be the *raw* request
        bytes — re-serializing the parsed JSON would change the signature.
        """
        if not signing_secret or not timestamp or not signature:
            return False
        try:
            if abs(time.time() - int(timestamp)) > max_skew_seconds:
                return False
        except (TypeError, ValueError):
            return False
        basestring = b"v0:" + timestamp.encode() + b":" + body
        digest = hmac.new(
            signing_secret.encode(), basestring, hashlib.sha256
        ).hexdigest()
        expected = f"v0={digest}"
        return hmac.compare_digest(expected, signature)
