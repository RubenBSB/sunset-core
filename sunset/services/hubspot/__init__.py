"""HubSpot CRM service for OAuth, deals, properties, files, and webhooks."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

HUBSPOT_API = "https://api.hubapi.com"


class HubspotError(Exception):
    """Raised when a HubSpot API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


@dataclass
class TokenSet:
    access_token: str
    refresh_token: str
    expires_at: datetime
    hub_id: Optional[str] = None


@dataclass
class Property:
    name: str
    label: str
    group_name: str
    type: str
    field_type: Optional[str] = None


@dataclass
class Stage:
    id: str
    label: str
    display_order: int


@dataclass
class Pipeline:
    id: str
    label: str
    stages: list[Stage]


@dataclass
class Deal:
    id: str
    properties: dict
    created_at: str
    updated_at: str


@dataclass
class Contact:
    id: str
    properties: dict
    created_at: str
    updated_at: str


@dataclass
class Subscription:
    id: int
    event_type: str
    property_name: Optional[str]
    active: bool


class HubspotService:
    """Async HubSpot API client. Stateless — callers own token persistence."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        developer_api_key: Optional[str] = None,
        app_id: Optional[str] = None,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._developer_api_key = developer_api_key
        self._app_id = app_id

    def _http(self, timeout: float = 30.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout)

    def _auth_headers(self, access_token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {access_token}"}

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        resp = await client.request(method, url, **kwargs)
        if resp.status_code >= 400:
            raise HubspotError(
                f"HubSpot API error: {resp.status_code} {resp.text}",
                status_code=resp.status_code,
            )
        return resp

    # -------------------------------------------------------------------------
    # OAuth
    # -------------------------------------------------------------------------

    def get_install_url(self, redirect_uri: str, scopes: list[str]) -> str:
        params = urlencode(
            {
                "client_id": self._client_id,
                "redirect_uri": redirect_uri,
                "scope": " ".join(scopes),
            }
        )
        return f"https://app.hubspot.com/oauth/authorize?{params}"

    async def exchange_code(self, code: str, redirect_uri: str) -> TokenSet:
        async with self._http() as client:
            resp = await self._request(
                client,
                "POST",
                f"{HUBSPOT_API}/oauth/v1/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "redirect_uri": redirect_uri,
                    "code": code,
                },
            )
            return self._parse_token_response(resp.json())

    async def refresh_token(self, refresh_token: str) -> TokenSet:
        async with self._http() as client:
            resp = await self._request(
                client,
                "POST",
                f"{HUBSPOT_API}/oauth/v1/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": refresh_token,
                },
            )
            return self._parse_token_response(resp.json())

    async def get_access_token_info(self, access_token: str) -> dict:
        """Fetch hub_id, user, scopes for an access token."""
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/oauth/v1/access-tokens/{access_token}",
            )
            return resp.json()

    def _parse_token_response(self, data: dict) -> TokenSet:
        expires_in = data.get("expires_in", 21600)
        return TokenSet(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            hub_id=str(data["hub_id"]) if data.get("hub_id") else None,
        )

    # -------------------------------------------------------------------------
    # Properties & property groups
    # -------------------------------------------------------------------------

    async def list_properties(
        self, access_token: str, object_type: str = "deals"
    ) -> list[Property]:
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/crm/v3/properties/{object_type}",
                headers=self._auth_headers(access_token),
            )
            return [
                Property(
                    name=p["name"],
                    label=p["label"],
                    group_name=p.get("groupName", ""),
                    type=p.get("type", ""),
                    field_type=p.get("fieldType"),
                )
                for p in resp.json().get("results", [])
            ]

    async def list_property_groups(
        self, access_token: str, object_type: str = "deals"
    ) -> list[dict]:
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/crm/v3/properties/{object_type}/groups",
                headers=self._auth_headers(access_token),
            )
            return [
                {
                    "name": g["name"],
                    "label": g["label"],
                    "display_order": g.get("displayOrder", 0),
                }
                for g in resp.json().get("results", [])
            ]

    async def find_group_by_label(
        self, access_token: str, label: str, object_type: str = "deals"
    ) -> Optional[dict]:
        groups = await self.list_property_groups(access_token, object_type)
        for g in groups:
            if g["label"].lower() == label.lower():
                return g
        return None

    # -------------------------------------------------------------------------
    # Pipelines
    # -------------------------------------------------------------------------

    async def get_pipelines(
        self, access_token: str, object_type: str = "deals"
    ) -> list[Pipeline]:
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/crm/v3/pipelines/{object_type}",
                headers=self._auth_headers(access_token),
            )
            return [
                Pipeline(
                    id=p["id"],
                    label=p["label"],
                    stages=[
                        Stage(
                            id=s["id"],
                            label=s["label"],
                            display_order=s.get("displayOrder", 0),
                        )
                        for s in p.get("stages", [])
                    ],
                )
                for p in resp.json().get("results", [])
            ]

    # -------------------------------------------------------------------------
    # Deals
    # -------------------------------------------------------------------------

    async def get_deal(
        self,
        access_token: str,
        deal_id: str,
        properties: Optional[list[str]] = None,
    ) -> Deal:
        params: dict[str, Any] = {}
        if properties:
            params["properties"] = ",".join(properties)
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/crm/v3/objects/deals/{deal_id}",
                headers=self._auth_headers(access_token),
                params=params,
            )
            return self._parse_deal(resp.json())

    async def list_deals(
        self,
        access_token: str,
        properties: Optional[list[str]] = None,
        stage: Optional[str] = None,
        pipeline: Optional[str] = None,
        limit: int = 100,
    ) -> list[Deal]:
        properties = properties or ["dealname", "amount", "dealstage", "pipeline"]

        if stage or pipeline:
            return await self._search_deals(
                access_token, properties, stage, pipeline, limit
            )

        async with self._http() as client:
            deals: list[Deal] = []
            after: Optional[str] = None
            while True:
                params: dict[str, Any] = {
                    "limit": min(limit, 100),
                    "properties": ",".join(properties),
                    "archived": "false",
                }
                if after:
                    params["after"] = after
                resp = await self._request(
                    client,
                    "GET",
                    f"{HUBSPOT_API}/crm/v3/objects/deals",
                    headers=self._auth_headers(access_token),
                    params=params,
                )
                data = resp.json()
                deals.extend(self._parse_deal(r) for r in data.get("results", []))
                paging = data.get("paging", {}).get("next")
                if not paging:
                    break
                after = paging.get("after")
            return deals

    async def _search_deals(
        self,
        access_token: str,
        properties: list[str],
        stage: Optional[str],
        pipeline: Optional[str],
        limit: int,
    ) -> list[Deal]:
        filters = []
        if stage:
            filters.append(
                {"propertyName": "dealstage", "operator": "EQ", "value": stage}
            )
        if pipeline:
            filters.append(
                {"propertyName": "pipeline", "operator": "EQ", "value": pipeline}
            )

        async with self._http() as client:
            resp = await self._request(
                client,
                "POST",
                f"{HUBSPOT_API}/crm/v3/objects/deals/search",
                headers=self._auth_headers(access_token),
                json={
                    "filterGroups": [{"filters": filters}] if filters else [],
                    "properties": properties,
                    "limit": min(limit, 100),
                },
            )
            return [self._parse_deal(r) for r in resp.json().get("results", [])]

    async def create_deal(
        self,
        access_token: str,
        name: str,
        stage_id: str,
        pipeline_id: str = "default",
        properties: Optional[dict] = None,
    ) -> Deal:
        props = {
            "dealname": name,
            "dealstage": stage_id,
            "pipeline": pipeline_id,
            **(properties or {}),
        }
        async with self._http() as client:
            resp = await self._request(
                client,
                "POST",
                f"{HUBSPOT_API}/crm/v3/objects/deals",
                headers=self._auth_headers(access_token),
                json={"properties": props},
            )
            return self._parse_deal(resp.json())

    async def update_deal(
        self, access_token: str, deal_id: str, properties: dict
    ) -> Deal:
        async with self._http() as client:
            resp = await self._request(
                client,
                "PATCH",
                f"{HUBSPOT_API}/crm/v3/objects/deals/{deal_id}",
                headers=self._auth_headers(access_token),
                json={"properties": properties},
            )
            return self._parse_deal(resp.json())

    async def move_deal(
        self,
        access_token: str,
        deal_id: str,
        stage_id: str,
        pipeline_id: Optional[str] = None,
    ) -> Deal:
        props = {"dealstage": stage_id}
        if pipeline_id:
            props["pipeline"] = pipeline_id
        return await self.update_deal(access_token, deal_id, props)

    async def delete_deal(self, access_token: str, deal_id: str) -> None:
        async with self._http() as client:
            await self._request(
                client,
                "DELETE",
                f"{HUBSPOT_API}/crm/v3/objects/deals/{deal_id}",
                headers=self._auth_headers(access_token),
            )

    # -------------------------------------------------------------------------
    # Contacts
    # -------------------------------------------------------------------------

    async def get_contact(
        self,
        access_token: str,
        contact_id: str,
        properties: Optional[list[str]] = None,
    ) -> Contact:
        params: dict[str, Any] = {}
        if properties:
            params["properties"] = ",".join(properties)
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/crm/v3/objects/contacts/{contact_id}",
                headers=self._auth_headers(access_token),
                params=params,
            )
            return self._parse_contact(resp.json())

    async def list_contacts(
        self,
        access_token: str,
        properties: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[Contact]:
        properties = properties or ["firstname", "lastname", "email", "phone"]
        async with self._http() as client:
            contacts: list[Contact] = []
            after: Optional[str] = None
            while len(contacts) < limit:
                params: dict[str, Any] = {
                    "limit": min(limit - len(contacts), 100),
                    "properties": ",".join(properties),
                    "archived": "false",
                }
                if after:
                    params["after"] = after
                resp = await self._request(
                    client,
                    "GET",
                    f"{HUBSPOT_API}/crm/v3/objects/contacts",
                    headers=self._auth_headers(access_token),
                    params=params,
                )
                data = resp.json()
                contacts.extend(self._parse_contact(r) for r in data.get("results", []))
                paging = data.get("paging", {}).get("next")
                if not paging:
                    break
                after = paging.get("after")
            return contacts

    async def create_contact(self, access_token: str, properties: dict) -> Contact:
        async with self._http() as client:
            resp = await self._request(
                client,
                "POST",
                f"{HUBSPOT_API}/crm/v3/objects/contacts",
                headers=self._auth_headers(access_token),
                json={"properties": properties},
            )
            return self._parse_contact(resp.json())

    async def update_contact(
        self, access_token: str, contact_id: str, properties: dict
    ) -> Contact:
        async with self._http() as client:
            resp = await self._request(
                client,
                "PATCH",
                f"{HUBSPOT_API}/crm/v3/objects/contacts/{contact_id}",
                headers=self._auth_headers(access_token),
                json={"properties": properties},
            )
            return self._parse_contact(resp.json())

    async def delete_contact(self, access_token: str, contact_id: str) -> None:
        async with self._http() as client:
            await self._request(
                client,
                "DELETE",
                f"{HUBSPOT_API}/crm/v3/objects/contacts/{contact_id}",
                headers=self._auth_headers(access_token),
            )

    async def create_or_update_contact_by_email(
        self, access_token: str, email: str, properties: dict
    ) -> Contact:
        """Upsert a contact keyed by email. Creates if missing, patches if present."""
        props = {**properties, "email": email}
        async with self._http() as client:
            try:
                resp = await self._request(
                    client,
                    "POST",
                    f"{HUBSPOT_API}/crm/v3/objects/contacts",
                    headers=self._auth_headers(access_token),
                    json={"properties": props},
                )
                return self._parse_contact(resp.json())
            except HubspotError as e:
                if e.status_code != 409:
                    raise
            resp = await self._request(
                client,
                "PATCH",
                f"{HUBSPOT_API}/crm/v3/objects/contacts/{email}",
                headers=self._auth_headers(access_token),
                params={"idProperty": "email"},
                json={"properties": props},
            )
            return self._parse_contact(resp.json())

    # -------------------------------------------------------------------------
    # Custom objects & associations
    # -------------------------------------------------------------------------

    async def list_custom_object_schemas(self, access_token: str) -> list[dict]:
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/crm/v3/schemas",
                headers=self._auth_headers(access_token),
            )
            return [
                {
                    "name": s["name"],
                    "label": s.get("labels", {}).get("singular", s["name"]),
                    "object_type_id": s["objectTypeId"],
                    "fully_qualified_name": s.get(
                        "fullyQualifiedName", s["objectTypeId"]
                    ),
                }
                for s in resp.json().get("results", [])
            ]

    async def get_associations(
        self,
        access_token: str,
        from_object_type: str,
        from_id: str,
        to_object_type: str,
    ) -> list[str]:
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/crm/v4/objects/{from_object_type}/{from_id}/associations/{to_object_type}",
                headers=self._auth_headers(access_token),
            )
            return [str(a["toObjectId"]) for a in resp.json().get("results", [])]

    async def get_custom_object(
        self,
        access_token: str,
        object_type: str,
        object_id: str,
        properties: Optional[list[str]] = None,
    ) -> dict:
        if properties is None:
            props = await self.list_properties(access_token, object_type)
            properties = [p.name for p in props]
        params = {"properties": ",".join(properties)} if properties else {}
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/crm/v3/objects/{object_type}/{object_id}",
                headers=self._auth_headers(access_token),
                params=params,
            )
            data = resp.json()
            return {
                "id": data["id"],
                "properties": data.get("properties", {}),
                "created_at": data.get("createdAt", ""),
                "updated_at": data.get("updatedAt", ""),
            }

    # -------------------------------------------------------------------------
    # Owners
    # -------------------------------------------------------------------------

    async def get_owner(self, access_token: str, owner_id: str) -> Optional[dict]:
        async with self._http() as client:
            try:
                resp = await self._request(
                    client,
                    "GET",
                    f"{HUBSPOT_API}/crm/v3/owners/{owner_id}",
                    headers=self._auth_headers(access_token),
                )
                return resp.json()
            except HubspotError:
                return None

    async def get_owner_name(self, access_token: str, owner_id: str) -> Optional[str]:
        owner = await self.get_owner(access_token, owner_id)
        if not owner:
            return None
        full = f"{owner.get('firstName', '') or ''} {owner.get('lastName', '') or ''}".strip()
        return full or owner.get("email")

    # -------------------------------------------------------------------------
    # Notes
    # -------------------------------------------------------------------------

    async def create_note_on_deal(
        self,
        access_token: str,
        deal_id: str,
        body: str,
        pin: bool = False,
    ) -> str:
        payload = {
            "properties": {
                "hs_note_body": body,
                "hs_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "associations": [
                {
                    "to": {"id": deal_id},
                    "types": [
                        {
                            "associationCategory": "HUBSPOT_DEFINED",
                            "associationTypeId": 214,  # note -> deal
                        }
                    ],
                }
            ],
        }
        async with self._http() as client:
            resp = await self._request(
                client,
                "POST",
                f"{HUBSPOT_API}/crm/v3/objects/notes",
                headers=self._auth_headers(access_token),
                json=payload,
            )
            note_id = resp.json()["id"]

        if pin:
            try:
                await self.update_deal(
                    access_token, deal_id, {"hs_pinned_engagement_id": note_id}
                )
            except HubspotError as e:
                logger.warning(f"Failed to pin note {note_id}: {e}")

        return note_id

    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------

    async def get_file_info(self, access_token: str, file_id: str) -> dict:
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                f"{HUBSPOT_API}/files/v3/files/{file_id}",
                headers=self._auth_headers(access_token),
            )
            return resp.json()

    async def get_file_signed_url(
        self, access_token: str, file_id: str
    ) -> tuple[str, str, str]:
        """Returns (download_url, filename, content_type)."""
        info = await self.get_file_info(access_token, file_id)
        filename = info.get("name", "unknown")
        content_type = info.get("type", "application/octet-stream")

        async with self._http() as client:
            try:
                resp = await self._request(
                    client,
                    "GET",
                    f"{HUBSPOT_API}/files/v3/files/{file_id}/signed-url",
                    headers=self._auth_headers(access_token),
                )
                url = resp.json().get("url") or info.get("url")
            except HubspotError:
                url = info.get("url")

        if not url:
            raise HubspotError(f"No download URL available for file {file_id}")
        return url, filename, content_type

    async def download_file(
        self, access_token: str, file_id_or_url: str
    ) -> tuple[bytes, str, str]:
        """Returns (file_bytes, content_type, filename)."""
        expected_content_type: Optional[str] = None
        if file_id_or_url.isdigit():
            url, filename, expected_content_type = await self.get_file_signed_url(
                access_token, file_id_or_url
            )
        else:
            url = file_id_or_url
            filename = "unknown"

        async with self._http(timeout=60.0) as client:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code != 200:
                resp = await client.get(
                    url,
                    headers=self._auth_headers(access_token),
                    follow_redirects=True,
                )
            if resp.status_code != 200:
                raise HubspotError(
                    f"Failed to download file: {resp.status_code}",
                    status_code=resp.status_code,
                )

            content_type = resp.headers.get("content-type", "application/octet-stream")
            if "text/html" in content_type and expected_content_type:
                content_type = expected_content_type
            return resp.content, content_type, filename

    # -------------------------------------------------------------------------
    # Webhook subscriptions (Developer API — requires developer_api_key + app_id)
    # -------------------------------------------------------------------------

    def _webhook_url(self, path: str = "") -> str:
        if not self._developer_api_key or not self._app_id:
            raise HubspotError("Webhook API requires developer_api_key and app_id")
        base = f"{HUBSPOT_API}/webhooks/v3/{self._app_id}"
        return f"{base}/{path}" if path else base

    def _webhook_params(self) -> dict:
        return {"hapikey": self._developer_api_key}

    async def list_subscriptions(self) -> list[Subscription]:
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                self._webhook_url("subscriptions"),
                params=self._webhook_params(),
            )
            return [
                Subscription(
                    id=s["id"],
                    event_type=s["eventType"],
                    property_name=s.get("propertyName"),
                    active=s.get("active", False),
                )
                for s in resp.json().get("results", [])
            ]

    async def create_subscription(
        self,
        event_type: str,
        property_name: Optional[str] = None,
        active: bool = True,
    ) -> Subscription:
        payload: dict[str, Any] = {"eventType": event_type, "active": active}
        if property_name:
            payload["propertyName"] = property_name

        async with self._http() as client:
            resp = await self._request(
                client,
                "POST",
                self._webhook_url("subscriptions"),
                params=self._webhook_params(),
                json=payload,
            )
            data = resp.json()
            return Subscription(
                id=data["id"],
                event_type=data["eventType"],
                property_name=data.get("propertyName"),
                active=data.get("active", False),
            )

    async def delete_subscription(self, subscription_id: int) -> None:
        async with self._http() as client:
            await self._request(
                client,
                "DELETE",
                self._webhook_url(f"subscriptions/{subscription_id}"),
                params=self._webhook_params(),
            )

    async def delete_all_subscriptions(self) -> None:
        for sub in await self.list_subscriptions():
            await self.delete_subscription(sub.id)

    async def get_webhook_settings(self) -> dict:
        async with self._http() as client:
            resp = await self._request(
                client,
                "GET",
                self._webhook_url("settings"),
                params=self._webhook_params(),
            )
            return resp.json()

    async def configure_webhook_url(
        self,
        target_url: str,
        max_concurrent_requests: int = 10,
        throttling_period: str = "SECONDLY",
    ) -> None:
        async with self._http() as client:
            await self._request(
                client,
                "PUT",
                self._webhook_url("settings"),
                params=self._webhook_params(),
                json={
                    "targetUrl": target_url,
                    "throttling": {
                        "period": throttling_period,
                        "maxConcurrentRequests": max_concurrent_requests,
                    },
                },
            )

    async def setup_property_subscriptions(
        self,
        properties: list[str],
        event_type: str = "deal.propertyChange",
        webhook_url: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        """Configure webhook URL + replace all subscriptions with the given properties.

        Yields progress events: {"step": str, "current": int, "total": int, "status": str}.
        """
        total = len(properties) + 2
        current = 0

        if webhook_url:
            yield {
                "step": "Configuring webhook URL",
                "current": current,
                "total": total,
                "status": "running",
            }
            if webhook_url.startswith("https://"):
                try:
                    await self.configure_webhook_url(webhook_url)
                    yield {
                        "step": f"Webhook URL: {webhook_url}",
                        "current": current,
                        "total": total,
                        "status": "done",
                    }
                except Exception as e:
                    yield {
                        "step": f"Error configuring URL: {e}",
                        "current": current,
                        "total": total,
                        "status": "error",
                    }
            else:
                yield {
                    "step": "Skipped URL config (HTTPS required)",
                    "current": current,
                    "total": total,
                    "status": "done",
                }
        current += 1

        yield {
            "step": "Cleaning up existing subscriptions",
            "current": current,
            "total": total,
            "status": "running",
        }
        try:
            await self.delete_all_subscriptions()
            yield {
                "step": "Cleaned up subscriptions",
                "current": current,
                "total": total,
                "status": "done",
            }
        except Exception as e:
            yield {
                "step": f"Error cleaning up: {e}",
                "current": current,
                "total": total,
                "status": "error",
            }
        current += 1

        for prop in properties:
            yield {
                "step": f"Subscribing to '{prop}'",
                "current": current,
                "total": total,
                "status": "running",
            }
            try:
                await self.create_subscription(event_type, prop)
                yield {
                    "step": f"Subscribed to '{prop}'",
                    "current": current,
                    "total": total,
                    "status": "done",
                }
            except Exception as e:
                yield {
                    "step": f"Error: {e}",
                    "current": current,
                    "total": total,
                    "status": "error",
                }
            current += 1

        yield {
            "step": "Setup complete",
            "current": total,
            "total": total,
            "status": "complete",
        }

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_deal(data: dict) -> Deal:
        return Deal(
            id=data["id"],
            properties=data.get("properties", {}),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", ""),
        )

    @staticmethod
    def _parse_contact(data: dict) -> Contact:
        return Contact(
            id=data["id"],
            properties=data.get("properties", {}),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", ""),
        )
