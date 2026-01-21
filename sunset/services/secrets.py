"""
Secrets service for unified secret management.

Handles:
- Local development: reads from .env.local, falls back to GCP Secret Manager
- Staging/Production: reads from GCP Secret Manager
"""

import os
import logging
from typing import Optional
from functools import lru_cache


logger = logging.getLogger(__name__)


class SecretsService:
    """
    Unified secrets loader.

    Usage:
        from sunset.services import SecretsService

        secrets = SecretsService()
        api_key = secrets.get_secret("OPENAI_API_KEY")
    """

    _instance: Optional["SecretsService"] = None

    def __init__(self):
        self.env = os.getenv("ENV", "local")
        self.project_id = os.getenv("GCP_PROJECT_ID") or os.getenv(
            "GOOGLE_CLOUD_PROJECT"
        )
        self._secret_client = None

        if self.env == "local":
            self._load_local_env()

    @classmethod
    def get_instance(cls) -> "SecretsService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def secret_client(self):
        """Lazy-load the secret manager client."""
        if self._secret_client is None:
            from google.cloud import secretmanager

            self._secret_client = secretmanager.SecretManagerServiceClient()
        return self._secret_client

    def _load_local_env(self):
        """Load .env.local for local development."""
        try:
            from dotenv import load_dotenv

            load_dotenv(".env.local")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. "
                "Install with: pip install python-dotenv"
            )

    def _get_from_remote(self, secret_name: str) -> Optional[str]:
        """Fetch secret from GCP Secret Manager."""
        if not self.project_id:
            return None

        try:
            gcp_secret_name = secret_name.lower().replace("_", "-")
            path = (
                f"projects/{self.project_id}/secrets/{gcp_secret_name}/versions/latest"
            )
            logger.debug(f"Fetching secret from GCP: {gcp_secret_name}")
            response = self.secret_client.access_secret_version(request={"name": path})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.debug(f"Could not fetch secret {secret_name} from GCP: {e}")
            return None

    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str, default: Optional[str] = None) -> str:
        """
        Retrieve a secret.

        Args:
            secret_name: Name of the secret (e.g., "OPENAI_API_KEY" or "openai-api-key")
            default: Default value if secret not found

        Returns:
            The secret value

        Raises:
            ValueError: If secret not found and no default provided
        """
        env_var = secret_name.upper().replace("-", "_")

        if self.env == "local":
            value = os.getenv(env_var)
            if value is not None:
                return value

        value = self._get_from_remote(secret_name)
        if value is not None:
            return value

        if default is not None:
            return default

        raise ValueError(f"Missing secret '{secret_name}' (env var: {env_var})")
