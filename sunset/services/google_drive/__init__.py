"""Google Drive API client — OAuth, file operations, and change tracking."""

import asyncio
import logging
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly email"

FILE_FIELDS = "id, name, mimeType, parents, md5Checksum, size, modifiedTime"
FILE_FIELDS_WITH_TRASHED = f"{FILE_FIELDS}, trashed"

_BATCH_SIZE = 100
_CONCURRENCY = 5

GOOGLE_EXPORT_MIME_MAP = {
    "application/vnd.google-apps.document": "application/pdf",
    "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.google-apps.presentation": "application/pdf",
    "application/vnd.google-apps.drawing": "application/pdf",
}


class GoogleDriveError(Exception):
    """Base exception for Google Drive API errors."""


class TokenExpiredError(GoogleDriveError):
    """Access token has expired (401)."""


class InsufficientPermissionsError(GoogleDriveError):
    """Insufficient permissions (403)."""


class NotFoundError(GoogleDriveError):
    """Resource not found (404)."""


class RateLimitError(GoogleDriveError):
    """Rate limit exceeded (429)."""


def _check_response(resp: httpx.Response) -> None:
    if resp.status_code == 401:
        raise TokenExpiredError(f"Token expired: {resp.text}")
    if resp.status_code == 403:
        raise InsufficientPermissionsError(f"Insufficient permissions: {resp.text}")
    if resp.status_code == 404:
        raise NotFoundError(f"Not found: {resp.text}")
    if resp.status_code == 429:
        raise RateLimitError(f"Rate limit exceeded: {resp.text}")
    resp.raise_for_status()


class GoogleDriveService:
    """Async Google Drive API client."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._client = httpx.AsyncClient(timeout=30.0)

    # ── OAuth ──────────────────────────────────────────────────────────

    def get_auth_url(self, redirect_uri: str, state: str | None = None) -> str:
        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": DRIVE_SCOPE,
            "access_type": "offline",
            "prompt": "consent",
        }
        if state:
            params["state"] = state
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict:
        resp = await self._client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        _check_response(resp)
        return resp.json()

    async def refresh_token(self, refresh_token: str) -> dict:
        resp = await self._client.post(
            GOOGLE_TOKEN_URL,
            data={
                "refresh_token": refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "refresh_token",
            },
        )
        _check_response(resp)
        return resp.json()

    async def get_user_info(self, access_token: str) -> dict:
        resp = await self._client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        _check_response(resp)
        return resp.json()

    # ── File & Folder Methods ──────────────────────────────────────────

    async def list_shared_drives(self, access_token: str) -> list[dict]:
        """List all shared drives the user has access to."""
        headers = {"Authorization": f"Bearer {access_token}"}
        drives: list[dict] = []
        page_token: str | None = None

        while True:
            params: dict = {"pageSize": 100}
            if page_token:
                params["pageToken"] = page_token

            resp = await self._client.get(
                f"{DRIVE_API_BASE}/drives",
                headers=headers,
                params=params,
            )
            _check_response(resp)
            data = resp.json()
            drives.extend(data.get("drives", []))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return drives

    async def list_shared_with_me(
        self,
        access_token: str,
        folders_only: bool = False,
    ) -> list[dict]:
        """List files/folders shared with the user (not in My Drive or shared drives)."""
        headers = {"Authorization": f"Bearer {access_token}"}
        query = "sharedWithMe=true"
        if folders_only:
            query += " and mimeType='application/vnd.google-apps.folder'"

        items: list[dict] = []
        page_token: str | None = None

        while True:
            params: dict = {
                "q": query,
                "fields": f"nextPageToken, files({FILE_FIELDS})",
                "pageSize": 1000,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await self._client.get(
                f"{DRIVE_API_BASE}/files",
                headers=headers,
                params=params,
            )
            _check_response(resp)
            data = resp.json()
            items.extend(data.get("files", []))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return items

    async def list_folder_children(
        self,
        access_token: str,
        folder_id: str = "root",
        folders_only: bool = False,
    ) -> list[dict]:
        headers = {"Authorization": f"Bearer {access_token}"}
        query = f"'{folder_id}' in parents"
        if folders_only:
            query += " and mimeType='application/vnd.google-apps.folder'"

        items: list[dict] = []
        page_token: str | None = None

        while True:
            params: dict = {
                "q": query,
                "fields": f"nextPageToken, files({FILE_FIELDS})",
                "pageSize": 1000,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await self._client.get(
                f"{DRIVE_API_BASE}/files",
                headers=headers,
                params=params,
            )
            _check_response(resp)
            data = resp.json()
            items.extend(data.get("files", []))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return items

    async def get_file_metadata(self, access_token: str, file_id: str) -> dict:
        resp = await self._client.get(
            f"{DRIVE_API_BASE}/files/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"fields": FILE_FIELDS_WITH_TRASHED, "supportsAllDrives": "true"},
        )
        _check_response(resp)
        return resp.json()

    async def export_file(
        self, access_token: str, file_id: str, export_mime_type: str
    ) -> bytes:
        resp = await self._client.get(
            f"{DRIVE_API_BASE}/files/{file_id}/export",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"mimeType": export_mime_type, "supportsAllDrives": "true"},
        )
        _check_response(resp)
        return resp.content

    async def download_file(self, access_token: str, file_id: str) -> bytes:
        resp = await self._client.get(
            f"{DRIVE_API_BASE}/files/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"alt": "media", "supportsAllDrives": "true"},
        )
        _check_response(resp)
        return resp.content

    async def build_folder_tree(
        self,
        access_token: str,
        root_folder_id: str,
    ) -> set[str]:
        folder_ids: set[str] = {root_folder_id}
        queue: list[str] = [root_folder_id]
        sem = asyncio.Semaphore(_CONCURRENCY)

        async def _list_subfolders(fid: str) -> list[str]:
            async with sem:
                children = await self.list_folder_children(
                    access_token,
                    fid,
                    folders_only=True,
                )
                return [c["id"] for c in children]

        while queue:
            tasks = [_list_subfolders(fid) for fid in queue]
            results = await asyncio.gather(*tasks)
            queue = []
            for child_ids in results:
                for cid in child_ids:
                    if cid not in folder_ids:
                        folder_ids.add(cid)
                        queue.append(cid)

        return folder_ids

    async def list_all_files(
        self,
        access_token: str,
        folder_ids: set[str],
    ) -> list[dict]:
        headers = {"Authorization": f"Bearer {access_token}"}
        folder_list = list(folder_ids)
        all_files: list[dict] = []

        for i in range(0, len(folder_list), _BATCH_SIZE):
            batch = folder_list[i : i + _BATCH_SIZE]
            parent_clauses = " or ".join(f"'{fid}' in parents" for fid in batch)
            query = f"({parent_clauses}) and trashed=false"

            page_token: str | None = None
            while True:
                params: dict = {
                    "q": query,
                    "fields": f"nextPageToken, files({FILE_FIELDS})",
                    "pageSize": 1000,
                    "supportsAllDrives": "true",
                    "includeItemsFromAllDrives": "true",
                }
                if page_token:
                    params["pageToken"] = page_token

                resp = await self._client.get(
                    f"{DRIVE_API_BASE}/files",
                    headers=headers,
                    params=params,
                )
                _check_response(resp)
                data = resp.json()

                for f in data.get("files", []):
                    if f.get("mimeType") == "application/vnd.google-apps.folder":
                        continue
                    all_files.append(f)

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

        return all_files

    # ── Change Tracking ────────────────────────────────────────────────

    async def get_start_page_token(self, access_token: str) -> str:
        resp = await self._client.get(
            f"{DRIVE_API_BASE}/changes/startPageToken",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"supportsAllDrives": "true"},
        )
        _check_response(resp)
        return resp.json()["startPageToken"]

    async def list_changes(self, access_token: str, page_token: str) -> dict:
        headers = {"Authorization": f"Bearer {access_token}"}
        changes: list[dict] = []
        current_token = page_token

        while True:
            resp = await self._client.get(
                f"{DRIVE_API_BASE}/changes",
                headers=headers,
                params={
                    "pageToken": current_token,
                    "fields": "changes(fileId,file(id,name,mimeType,parents,md5Checksum,size,modifiedTime,trashed),removed),newStartPageToken,nextPageToken",
                    "pageSize": 1000,
                    "supportsAllDrives": "true",
                    "includeItemsFromAllDrives": "true",
                },
            )
            _check_response(resp)
            data = resp.json()

            for c in data.get("changes", []):
                changes.append(
                    {
                        "file_id": c.get("fileId"),
                        "file": c.get("file"),
                        "removed": c.get("removed", False),
                    }
                )

            new_token = data.get("newStartPageToken")
            if new_token:
                return {"changes": changes, "new_page_token": new_token}

            current_token = data["nextPageToken"]

    async def watch_changes(
        self,
        access_token: str,
        page_token: str,
        webhook_url: str,
        channel_id: str,
        channel_token: str,
        expiration_ms: int | None = None,
    ) -> dict:
        body: dict = {
            "id": channel_id,
            "type": "web_hook",
            "address": webhook_url,
            "token": channel_token,
        }
        if expiration_ms is not None:
            body["expiration"] = expiration_ms

        resp = await self._client.post(
            f"{DRIVE_API_BASE}/changes/watch",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            params={"pageToken": page_token, "supportsAllDrives": "true"},
            json=body,
        )
        _check_response(resp)
        data = resp.json()
        return {
            "resourceId": data.get("resourceId"),
            "expiration": data.get("expiration"),
        }

    async def stop_channel(
        self,
        access_token: str,
        channel_id: str,
        resource_id: str,
    ) -> None:
        resp = await self._client.post(
            f"{DRIVE_API_BASE}/channels/stop",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"id": channel_id, "resourceId": resource_id},
        )
        _check_response(resp)

    async def close(self):
        await self._client.aclose()
