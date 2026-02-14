# SecretsService

Unified secrets loader. Reads from environment variables locally, GCP Secret Manager in production.

## Setup

### Infrastructure

No `sunset.yaml` entries needed. The service uses the GCP project that's already provisioned.

### Env Vars

Set in `sunset.env.yaml` under `secrets:` for any secret your app needs. The service resolves secrets by:

1. Checking environment variables (uppercased, hyphens → underscores)
2. Querying GCP Secret Manager (lowercased, underscores → hyphens)
3. Falling back to `default` if provided

Secrets added to `sunset.env.yaml` → `secrets:` are automatically created in GCP Secret Manager by `sunset provision`.

## Usage

```python
from sunset.services import SecretsService

secrets = SecretsService()

# Required secret (raises if missing)
jwt_key = secrets.get_secret("JWT_SECRET_KEY")

# Optional secret with fallback
sentry_dsn = secrets.get_secret("sentry-dsn", default="")
```

The service auto-detects the environment via `ENV` variable (`local` vs `production`/`staging`).

## API Reference

### `SecretsService()`

No constructor args. Reads `ENV` and `GCP_PROJECT_ID` from environment.

### `get_secret(secret_name, default=None) -> str`

Retrieve a secret. Results are cached with `lru_cache`.

- `secret_name`: Name of the secret (case-insensitive, hyphens/underscores interchangeable)
- `default`: Fallback value. If `None` and secret not found, raises `ValueError`.
