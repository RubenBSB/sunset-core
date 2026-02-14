# AuthService

JWT access tokens, refresh token rotation, password hashing (Argon2), OAuth helpers, and MFA (TOTP).

## Setup

### Infrastructure

No specific `sunset.yaml` entries. Requires a JWT secret in `sunset.env.yaml`:

```yaml
secrets:
  JWT_SECRET_KEY: "your-secret-key"
```

For OAuth, add provider credentials:

```yaml
secrets:
  GOOGLE_OAUTH_CLIENT_ID: "..."
  GOOGLE_OAUTH_CLIENT_SECRET: "..."
```

### Database

Refresh token rotation requires a `RefreshToken` SQLAlchemy model:

```python
class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, default=False)
    replaced_by = Column(UUID, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

Create a migration: `sunset migrate --create -m "add refresh tokens"`.

## Usage

Canonical pattern from the base template (`server/api/routers/auth/utils.py`):

```python
import os
from sunset.services import AuthService
from api.models import RefreshToken

auth = AuthService(
    jwt_secret=os.environ["JWT_SECRET_KEY"],
    access_token_expire_minutes=15,
    refresh_token_expire_days=7,
    refresh_token_model=RefreshToken,
    is_production=os.environ.get("ENV") == "production",
)

# Create access token
token = auth.create_token(user_id=str(user.id))

# Verify token (FastAPI dependency)
payload = auth.verify_token(token)

# Refresh token flow
raw, db_token = await auth.create_refresh_token(user_id=str(user.id), session=db)
auth.set_refresh_cookie(response, raw)

new_access, new_refresh = await auth.rotate_refresh_token(raw, session=db)
await auth.revoke_refresh_token(raw, session=db)
```

### FastAPI dependency pattern

```python
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = auth.verify_token(credentials.credentials)
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return payload["sub"]
```

## API Reference

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jwt_secret` | `str` | required | JWT signing secret |
| `jwt_algorithm` | `str` | `"HS256"` | JWT algorithm |
| `access_token_expire_minutes` | `int` | `15` | Access token TTL |
| `refresh_token_expire_days` | `int` | `7` | Refresh token TTL |
| `refresh_token_model` | `Any` | `None` | SQLAlchemy model for refresh tokens |
| `is_production` | `bool` | `False` | Controls cookie Secure/SameSite flags |
| `refresh_cookie_name` | `str` | `"refresh_token"` | HttpOnly cookie name |
| `refresh_cookie_path` | `str` | `"/auth"` | Cookie path |

### Key Methods

- `hash_password(password) -> str` — Argon2id hash
- `verify_password(password, hash) -> bool`
- `create_token(user_id, extra_claims?, expires_delta?) -> str`
- `verify_token(token) -> dict | None`
- `create_refresh_token(user_id, session) -> (raw_token, db_record)` (async)
- `rotate_refresh_token(raw_token, session) -> (new_access, new_refresh)` (async)
- `revoke_refresh_token(raw_token, session)` (async)
- `set_refresh_cookie(response, raw_token)` / `clear_refresh_cookie(response)`
- `generate_mfa_secret() -> str` / `verify_mfa_code(secret, code) -> bool`
