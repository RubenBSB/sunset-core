"""
Authentication service for password hashing, JWT tokens, and MFA (TOTP).

Usage:
    from sunset.services import AuthService

    auth = AuthService(jwt_secret="your-secret")

    # Password hashing
    hash = auth.hash_password("password123")
    is_valid = auth.verify_password("password123", hash)

    # JWT tokens
    token = auth.create_token(user_id="123", extra_claims={"role": "admin"})
    payload = auth.verify_token(token)

    # MFA (TOTP)
    secret = auth.generate_mfa_secret()
    qr_uri = auth.get_mfa_provisioning_uri(secret, "user@example.com", "MyApp")
    is_valid = auth.verify_mfa_code(secret, "123456")
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import argon2
import pyotp
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Argon2 hasher with secure defaults
_hasher = argon2.PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=1,
)


class AuthService:
    """
    Authentication service with password hashing, JWT tokens, and MFA support.

    Args:
        jwt_secret: Secret key for JWT signing (required)
        jwt_algorithm: JWT algorithm (default: HS256)
        token_expire_hours: Token expiration in hours (default: 48)
    """

    def __init__(
        self,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        token_expire_hours: int = 48,
    ):
        if not jwt_secret:
            raise ValueError("jwt_secret is required")

        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.token_expire_hours = token_expire_hours

    # =========================================================================
    # Password Hashing (Argon2)
    # =========================================================================

    def hash_password(self, password: str) -> str:
        """
        Hash a password using Argon2id.

        Args:
            password: Plain text password

        Returns:
            Argon2 hash string
        """
        return _hasher.hash(password)

    def verify_password(self, password: str, hash: str) -> bool:
        """
        Verify a password against an Argon2 hash.

        Args:
            password: Plain text password to verify
            hash: Argon2 hash to verify against

        Returns:
            True if password matches, False otherwise
        """
        try:
            _hasher.verify(hash, password)
            return True
        except argon2.exceptions.VerifyMismatchError:
            return False
        except Exception as e:
            logger.warning(f"Password verification error: {e}")
            return False

    def needs_rehash(self, hash: str) -> bool:
        """
        Check if a password hash needs to be rehashed (e.g., after config changes).

        Args:
            hash: Existing Argon2 hash

        Returns:
            True if rehash is needed
        """
        return _hasher.check_needs_rehash(hash)

    # =========================================================================
    # JWT Tokens
    # =========================================================================

    def create_token(
        self,
        user_id: str,
        extra_claims: Optional[dict] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create a JWT token.

        Args:
            user_id: User identifier (stored as 'sub' claim)
            extra_claims: Additional claims to include in the token
            expires_delta: Custom expiration delta (default: token_expire_hours)

        Returns:
            JWT token string
        """
        now = datetime.now(timezone.utc)
        expire = now + (expires_delta or timedelta(hours=self.token_expire_hours))

        payload = {
            "sub": str(user_id),
            "iat": now,
            "exp": expire,
        }

        if extra_claims:
            payload.update(extra_claims)

        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload dict if valid, None if invalid/expired
        """
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
        """
        Extract user_id from a JWT token.

        Args:
            token: JWT token string

        Returns:
            User ID if token is valid, None otherwise
        """
        payload = self.verify_token(token)
        if payload:
            return payload.get("sub")
        return None

    # =========================================================================
    # MFA (TOTP - Time-based One-Time Password)
    # =========================================================================

    def generate_mfa_secret(self) -> str:
        """
        Generate a new MFA secret for TOTP.

        Returns:
            Base32-encoded secret string (store securely in database)
        """
        return pyotp.random_base32()

    def get_mfa_provisioning_uri(
        self,
        secret: str,
        email: str,
        issuer: str,
    ) -> str:
        """
        Get the provisioning URI for QR code generation.

        Use this URI to generate a QR code that users can scan with
        authenticator apps like Google Authenticator or Authy.

        Args:
            secret: MFA secret from generate_mfa_secret()
            email: User's email address
            issuer: Application name (displayed in authenticator app)

        Returns:
            otpauth:// URI string
        """
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=email, issuer_name=issuer)

    def verify_mfa_code(self, secret: str, code: str, valid_window: int = 1) -> bool:
        """
        Verify a TOTP code.

        Args:
            secret: MFA secret stored for the user
            code: 6-digit code entered by the user
            valid_window: Number of 30-second windows to check (default: 1)
                          This allows for slight time drift between client/server.

        Returns:
            True if code is valid, False otherwise
        """
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=valid_window)
        except Exception as e:
            logger.warning(f"MFA verification error: {e}")
            return False

    def get_current_mfa_code(self, secret: str) -> str:
        """
        Get the current valid TOTP code (for testing purposes).

        Args:
            secret: MFA secret

        Returns:
            Current 6-digit code
        """
        totp = pyotp.TOTP(secret)
        return totp.now()


# Singleton instance
_auth_service: Optional[AuthService] = None


def get_auth_service(jwt_secret: Optional[str] = None) -> AuthService:
    """
    Get or create the AuthService singleton.

    Args:
        jwt_secret: JWT secret (required on first call, optional after)

    Returns:
        AuthService instance
    """
    global _auth_service
    if _auth_service is None:
        if not jwt_secret:
            raise ValueError("jwt_secret is required for first initialization")
        _auth_service = AuthService(jwt_secret=jwt_secret)
    return _auth_service
