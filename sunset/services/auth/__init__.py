"""
Authentication service for password hashing, JWT tokens, refresh token rotation,
cookie management, and MFA (TOTP).

Usage:
    from sunset.services import AuthService

    auth = AuthService(jwt_secret="your-secret")

    # Password hashing
    hash = auth.hash_password("password123")
    is_valid = auth.verify_password("password123", hash)

    # JWT access tokens
    token = auth.create_token(user_id="123", extra_claims={"role": "admin"})
    payload = auth.verify_token(token)

    # Refresh tokens (requires AsyncSession + RefreshToken model)
    raw, db_row = await auth.create_refresh_token(user_id="123", session=db)
    new_access, new_refresh = await auth.rotate_refresh_token(raw, session=db)
    await auth.revoke_refresh_token(raw, session=db)

    # Cookie helpers (for FastAPI/Starlette responses)
    auth.set_refresh_cookie(response, raw)
    auth.clear_refresh_cookie(response)
    raw = auth.get_refresh_token_from_cookie(request)

    # MFA (TOTP)
    secret = auth.generate_mfa_secret()
    qr_uri = auth.get_mfa_provisioning_uri(secret, "user@example.com", "MyApp")
    is_valid = auth.verify_mfa_code(secret, "123456")
"""

import hashlib
import logging
import secrets as secrets_mod
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import argon2
import pyotp
from jose import JWTError, jwt
from sqlalchemy import select, update

logger = logging.getLogger(__name__)

# Argon2 hasher with secure defaults
_hasher = argon2.PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=1,
)


class AuthService:
    """
    Authentication service with password hashing, JWT tokens, refresh token
    rotation, cookie management, and MFA support.

    Args:
        jwt_secret: Secret key for JWT signing (required)
        jwt_algorithm: JWT algorithm (default: HS256)
        access_token_expire_minutes: Short-lived access token TTL (default: 15)
        refresh_token_expire_days: Refresh token TTL (default: 7)
        refresh_token_model: SQLAlchemy model class for refresh tokens (optional,
            required for refresh token methods)
        is_production: Whether the app is running in production (affects cookie
            Secure and SameSite flags)
        refresh_cookie_name: Name of the HttpOnly cookie (default: "refresh_token")
        refresh_cookie_domain: Cookie domain (default: None, uses request origin)
        refresh_cookie_path: Cookie path (default: "/auth")
    """

    def __init__(
        self,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        access_token_expire_minutes: int = 15,
        refresh_token_expire_days: int = 7,
        refresh_token_model: Any = None,
        is_production: bool = False,
        refresh_cookie_name: str = "refresh_token",
        refresh_cookie_path: str = "/auth",
        refresh_cookie_domain: Optional[str] = None,
    ):
        if not jwt_secret:
            raise ValueError("jwt_secret is required")

        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.refresh_token_model = refresh_token_model
        self.is_production = is_production
        self.refresh_cookie_name = refresh_cookie_name
        self.refresh_cookie_path = refresh_cookie_path
        self.refresh_cookie_domain = refresh_cookie_domain

    # =========================================================================
    # Password Hashing (Argon2)
    # =========================================================================

    def hash_password(self, password: str) -> str:
        """Hash a password using Argon2id."""
        return _hasher.hash(password)

    def verify_password(self, password: str, hash: str) -> bool:
        """Verify a password against an Argon2 hash."""
        try:
            _hasher.verify(hash, password)
            return True
        except argon2.exceptions.VerifyMismatchError:
            return False
        except Exception as e:
            logger.warning(f"Password verification error: {e}")
            return False

    def needs_rehash(self, hash: str) -> bool:
        """Check if a password hash needs to be rehashed."""
        return _hasher.check_needs_rehash(hash)

    # =========================================================================
    # JWT Access Tokens
    # =========================================================================

    def create_token(
        self,
        user_id: str,
        extra_claims: Optional[dict] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create a short-lived JWT access token."""
        now = datetime.now(timezone.utc)
        expire = now + (
            expires_delta or timedelta(minutes=self.access_token_expire_minutes)
        )

        payload = {
            "sub": str(user_id),
            "iat": now,
            "exp": expire,
        }

        if extra_claims:
            payload.update(extra_claims)

        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode a JWT token. Returns payload or None."""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
            )
            return payload
        except JWTError as e:
            logger.debug(f"Token verification failed: {e}")
            return None

    def get_user_id_from_token(self, token: str) -> Optional[str]:
        """Extract user_id from a JWT token."""
        payload = self.verify_token(token)
        if payload:
            return payload.get("sub")
        return None

    # =========================================================================
    # Refresh Tokens
    # =========================================================================

    @staticmethod
    def _hash_refresh_token(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode()).hexdigest()

    def _ensure_refresh_model(self):
        if self.refresh_token_model is None:
            raise RuntimeError(
                "refresh_token_model must be set on AuthService to use refresh tokens"
            )

    async def create_refresh_token(self, user_id: str, session: Any) -> tuple:
        """Create a new refresh token stored in the database.

        Args:
            user_id: The user's ID
            session: AsyncSession

        Returns:
            (raw_token, db_record) — raw_token goes into the cookie,
            db_record is the persisted row.
        """
        self._ensure_refresh_model()
        Model = self.refresh_token_model

        raw_token = secrets_mod.token_urlsafe(32)
        token_hash = self._hash_refresh_token(raw_token)
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.refresh_token_expire_days
        )

        db_token = Model(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        session.add(db_token)
        await session.commit()
        return raw_token, db_token

    async def rotate_refresh_token(self, raw_token: str, session: Any) -> tuple:
        """Validate and rotate a refresh token.

        Implements automatic reuse detection: if a revoked token is presented,
        the entire token family (all tokens for that user) is revoked.

        Args:
            raw_token: The raw refresh token from the cookie
            session: AsyncSession

        Returns:
            (new_access_token, new_raw_refresh_token)

        Raises:
            ValueError with a descriptive message on failure.
        """
        self._ensure_refresh_model()
        Model = self.refresh_token_model

        token_hash = self._hash_refresh_token(raw_token)
        result = await session.execute(
            select(Model).where(Model.token_hash == token_hash)
        )
        db_token = result.scalar_one_or_none()

        if not db_token:
            raise ValueError("Invalid refresh token")

        # Reuse detection
        if db_token.revoked:
            await self._revoke_family(db_token.user_id, session)
            raise ValueError("Refresh token reuse detected — all sessions revoked")

        if db_token.expires_at < datetime.now(timezone.utc):
            raise ValueError("Refresh token expired")

        user_id = str(db_token.user_id)

        # Revoke old token
        db_token.revoked = True

        # Issue new pair
        new_raw_refresh, new_db_token = await self.create_refresh_token(
            user_id, session
        )
        db_token.replaced_by = new_db_token.id
        await session.commit()

        new_access_token = self.create_token(user_id=user_id)
        return new_access_token, new_raw_refresh

    async def revoke_refresh_token(self, raw_token: str, session: Any):
        """Revoke a single refresh token (for logout)."""
        self._ensure_refresh_model()
        Model = self.refresh_token_model

        token_hash = self._hash_refresh_token(raw_token)
        result = await session.execute(
            select(Model).where(Model.token_hash == token_hash)
        )
        db_token = result.scalar_one_or_none()
        if db_token:
            db_token.revoked = True
            await session.commit()

    async def revoke_all_user_tokens(self, user_id: str, session: Any):
        """Revoke all refresh tokens for a user."""
        self._ensure_refresh_model()
        await self._revoke_family(user_id, session)

    async def _revoke_family(self, user_id, session: Any):
        """Revoke all active refresh tokens for a user."""
        Model = self.refresh_token_model
        await session.execute(
            update(Model)
            .where(Model.user_id == user_id, not Model.revoked)
            .values(revoked=True)
        )
        await session.commit()

    # =========================================================================
    # Refresh Token Cookie Helpers
    # =========================================================================

    def set_refresh_cookie(self, response: Any, raw_token: str):
        """Set the refresh token as an HttpOnly cookie on a response."""
        kwargs = dict(
            key=self.refresh_cookie_name,
            value=raw_token,
            httponly=True,
            secure=self.is_production,
            samesite="lax"
            if self.refresh_cookie_domain
            else ("none" if self.is_production else "lax"),
            path=self.refresh_cookie_path,
            max_age=self.refresh_token_expire_days * 24 * 3600,
        )
        if self.refresh_cookie_domain:
            kwargs["domain"] = self.refresh_cookie_domain
        response.set_cookie(**kwargs)

    def clear_refresh_cookie(self, response: Any):
        """Clear the refresh token cookie."""
        kwargs = dict(
            key=self.refresh_cookie_name,
            path=self.refresh_cookie_path,
            httponly=True,
            secure=self.is_production,
            samesite="lax"
            if self.refresh_cookie_domain
            else ("none" if self.is_production else "lax"),
        )
        if self.refresh_cookie_domain:
            kwargs["domain"] = self.refresh_cookie_domain
        response.delete_cookie(**kwargs)

    def get_refresh_token_from_cookie(self, request: Any) -> str:
        """Read the refresh token from the request cookies.

        Raises:
            ValueError if no refresh token cookie is present.
        """
        raw_token = request.cookies.get(self.refresh_cookie_name)
        if not raw_token:
            raise ValueError("No refresh token")
        return raw_token

    # =========================================================================
    # MFA (TOTP - Time-based One-Time Password)
    # =========================================================================

    def generate_mfa_secret(self) -> str:
        """Generate a new MFA secret for TOTP."""
        return pyotp.random_base32()

    def get_mfa_provisioning_uri(
        self,
        secret: str,
        email: str,
        issuer: str,
    ) -> str:
        """Get the provisioning URI for QR code generation."""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=email, issuer_name=issuer)

    def verify_mfa_code(self, secret: str, code: str, valid_window: int = 1) -> bool:
        """Verify a TOTP code."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=valid_window)
        except Exception as e:
            logger.warning(f"MFA verification error: {e}")
            return False

    def get_current_mfa_code(self, secret: str) -> str:
        """Get the current valid TOTP code (for testing purposes)."""
        totp = pyotp.TOTP(secret)
        return totp.now()


# Singleton instance
_auth_service: Optional[AuthService] = None


def get_auth_service(jwt_secret: Optional[str] = None, **kwargs) -> AuthService:
    """
    Get or create the AuthService singleton.

    Args:
        jwt_secret: JWT secret (required on first call, optional after)
        **kwargs: Additional arguments passed to AuthService constructor on first call

    Returns:
        AuthService instance
    """
    global _auth_service
    if _auth_service is None:
        if not jwt_secret:
            raise ValueError("jwt_secret is required for first initialization")
        _auth_service = AuthService(jwt_secret=jwt_secret, **kwargs)
    return _auth_service
