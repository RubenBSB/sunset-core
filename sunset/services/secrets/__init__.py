import os
from functools import lru_cache
from typing import Optional

from google.cloud import secretmanager

_secrets_service: "Optional[SecretsService]" = None


def get_secrets() -> "SecretsService":
    global _secrets_service
    if _secrets_service is None:
        _secrets_service = SecretsService()
    return _secrets_service


class SecretsService:
    """
    Unified secrets loader:
    - Local development: reads from .env.local
    - Staging/Production: reads from GCP Secret Manager
    """

    def __init__(self):
        self.env = os.getenv("ENV", "local")
        self.project_id = os.getenv("GCP_PROJECT_ID") or os.getenv(
            "GOOGLE_CLOUD_PROJECT"
        )

        if self.env == "local":
            pass
        else:
            self.secret_client = secretmanager.SecretManagerServiceClient()

    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str, default: Optional[str] = None) -> str:
        """
        Retrieve a secret:
        1. First check environment variable
        2. Then try GCP Secret Manager (if not local)
        3. Fall back to default if provided
        """
        env_var = secret_name.upper().replace("-", "_")

        # Always check environment first
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value

        # Local: env var is the only source
        if self.env == "local":
            if default is not None:
                return default
            raise ValueError(f"Missing secret {env_var} in environment")

        # Remote: try GCP Secret Manager
        try:
            secret_key = secret_name.lower().replace("_", "-")
            path = f"projects/{self.project_id}/secrets/{secret_key}/versions/latest"
            response = self.secret_client.access_secret_version(request={"name": path})
            return response.payload.data.decode("UTF-8")

        except Exception as e:
            if default is not None:
                return default
            raise ValueError(f"Failed to load secret '{secret_name}': {e}")
