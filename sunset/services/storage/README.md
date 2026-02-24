# StorageService

Google Cloud Storage: file uploads, signed URLs, and deletion.

## Setup

### Infrastructure

In `sunset.yaml`, declare the buckets you need:

```yaml
infra:
  storage_buckets:
    - uploads
    - exports
```

`sunset provision` creates GCS buckets prefixed with your project name (e.g. `myapp-uploads-prod`).

### Local Development

When `storage_buckets` is configured in `sunset.yaml`, `sunset run` automatically starts a [fake-gcs-server](https://github.com/fsouza/fake-gcs-server) emulator and sets `STORAGE_EMULATOR_HOST` on all containers. No GCP credentials or bucket pre-creation needed — buckets are created on first access.

The `StorageService` detects the emulator automatically. `generate_signed_url()` returns a direct emulator URL instead of a signed URL.

### Env Vars (Production)

Set in `sunset.env.yaml`:

```yaml
secrets:
  GCS_BUCKET_NAME: "myapp-uploads-prod"
```

For local development **without the emulator** (e.g. testing against real GCS), also set:

```yaml
secrets:
  GCS_SIGNING_SERVICE_ACCOUNT: "your-sa@project.iam.gserviceaccount.com"
```

## Usage

```python
from sunset.services import StorageService

storage = StorageService()

# Upload
gcs_path = storage.upload_file(
    data=file_bytes,
    destination_path="attachments/user-123/photo.jpg",
    content_type="image/jpeg",
)

# Generate signed URL (time-limited access)
url = storage.generate_signed_url(gcs_path, expiration_minutes=15)

# Delete
storage.delete_file(gcs_path)

# Check existence
exists = storage.file_exists(gcs_path)
```

## API Reference

### `StorageService()`

Singleton. No constructor args — reads `GCP_PROJECT_ID` and `GCS_BUCKET_NAME` from secrets.

### Key Methods

- `upload_file(data, destination_path, content_type?) -> str` — Returns `gs://` path
- `generate_signed_url(gcs_path, expiration_minutes=15) -> str`
- `delete_file(gcs_path) -> bool`
- `file_exists(gcs_path) -> bool`
