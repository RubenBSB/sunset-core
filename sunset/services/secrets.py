import os
from typing import Optional
from functools import lru_cache

from google.cloud import secretmanager


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
        - Local: from environment (via .env.local)
        - Staging/Prod: from GCP Secret Manager
        """
        if self.env == "local":
            env_var = secret_name.upper().replace("-", "_")
            value = os.getenv(env_var, default)

            if value is None:
                raise ValueError(f"Missing secret {env_var} in .env.local")

            return value

        # Remote (GCP Secret Manager)
        try:
            if secret_name.upper().replace("-", "_") == secret_name:
                secret_name = secret_name.lower().replace("_", "-")
            path = (
                f"projects/{self.project_id}/secrets/" f"{secret_name}/versions/latest"
            )
            response = self.secret_client.access_secret_version(request={"name": path})
            return response.payload.data.decode("UTF-8")

        except Exception as e:
            if default is not None:
                return default
            raise ValueError(f"Failed to load secret '{secret_name}': {e}")


_secrets_service: Optional[SecretsService] = None


def get_secrets() -> SecretsService:
    global _secrets_service
    if _secrets_service is None:
        _secrets_service = SecretsService()
    return _secrets_service
