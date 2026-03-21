# Google Drive Service

Google Drive API client with OAuth, file operations, and change tracking.

## Setup

```python
from sunset.services import GoogleDriveService

drive = GoogleDriveService(
    client_id="your-google-client-id",
    client_secret="your-google-client-secret",
)
```

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_CLIENT_ID` | Google OAuth2 client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth2 client secret |

## OAuth Flow

```python
# 1. Generate auth URL and redirect user
url = drive.get_auth_url(redirect_uri="https://yourapp.com/callback", state="optional-state")

# 2. Exchange authorization code for tokens
tokens = await drive.exchange_code(code="auth-code", redirect_uri="https://yourapp.com/callback")
# Returns: {"access_token", "refresh_token", "expires_in", "token_type", "scope"}

# 3. Refresh expired tokens
new_tokens = await drive.refresh_token(refresh_token=tokens["refresh_token"])
# Returns: {"access_token", "expires_in"}

# 4. Get user info
user = await drive.get_user_info(access_token=tokens["access_token"])
# Returns: {"email", "name"}
```

## File & Folder Operations

```python
access_token = tokens["access_token"]

# List children of a folder (default: root)
files = await drive.list_folder_children(access_token, folder_id="root")
folders = await drive.list_folder_children(access_token, folder_id="root", folders_only=True)

# Get metadata for a single file
meta = await drive.get_file_metadata(access_token, file_id="file-id")

# Export a Google-native file (Docs, Sheets, etc.) to a standard format
from sunset.services.google_drive import GOOGLE_EXPORT_MIME_MAP
pdf_bytes = await drive.export_file(access_token, file_id="google-doc-id", export_mime_type="application/pdf")

# Download non-Google-native file content
content = await drive.download_file(access_token, file_id="file-id")

# Recursively enumerate all subfolder IDs
folder_ids = await drive.build_folder_tree(access_token, root_folder_id="folder-id")

# List all non-folder files in a set of folders (includes Google-native files)
files = await drive.list_all_files(access_token, folder_ids=folder_ids)
```

## Change Tracking

```python
# Get starting page token
page_token = await drive.get_start_page_token(access_token)

# List changes since a page token
result = await drive.list_changes(access_token, page_token=page_token)
# Returns: {"changes": [{"file_id", "file", "removed"}], "new_page_token"}

# Register a webhook for change notifications
channel = await drive.watch_changes(
    access_token,
    page_token=page_token,
    webhook_url="https://yourapp.com/webhook",
    channel_id="unique-channel-id",
    channel_token="verification-token",
    expiration_ms=3600000,  # optional, 1 hour
)
# Returns: {"resourceId", "expiration"}

# Stop a watch channel
await drive.stop_channel(access_token, channel_id="channel-id", resource_id=channel["resourceId"])
```

## Error Handling

The service raises typed exceptions for common API errors:

| Exception | HTTP Status | Meaning |
|---|---|---|
| `TokenExpiredError` | 401 | Access token expired — refresh it |
| `InsufficientPermissionsError` | 403 | Missing required scopes |
| `NotFoundError` | 404 | File or resource not found |
| `RateLimitError` | 429 | Too many requests — back off |
| `GoogleDriveError` | — | Base exception for all above |

## Cleanup

```python
await drive.close()
```
