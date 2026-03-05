import logging
import os
from datetime import timedelta
from typing import Optional
from urllib.parse import quote

import google.auth.credentials
from google.auth import default
from google.auth.transport import requests as google_requests
from google.cloud import storage

from sunset.services.secrets import get_secrets

logger = logging.getLogger(__name__)


class StorageService:
    """
    GCS storage service for secure file uploads.
    Uses signed URLs for time-limited secure access.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StorageService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._emulator_host = os.getenv("STORAGE_EMULATOR_HOST")

            if self._emulator_host:
                self.project_id = "local-test-project"
                self.bucket_name = os.getenv("GCS_BUCKET_NAME", "local-bucket")
                self._credentials = None
                self._client = storage.Client(
                    project=self.project_id,
                    credentials=google.auth.credentials.AnonymousCredentials(),
                )
                self._signing_sa_email = None
                self._emulator_public_host = os.getenv(
                    "STORAGE_EMULATOR_PUBLIC_HOST", "http://localhost:4443"
                )
                self._ensure_emulator_bucket()
                self._initialized = True
                logger.info(
                    f"StorageService initialized with emulator: {self._emulator_host}"
                )
                return

            secrets = get_secrets()
            self.env = os.getenv("ENV", "local")
            self.project_id = secrets.get_secret("GCP_PROJECT_ID")
            self.bucket_name = secrets.get_secret("GCS_BUCKET_NAME")

            self._credentials, _ = default()
            self._client = storage.Client(
                project=self.project_id, credentials=self._credentials
            )

            self._signing_sa_email = self._get_signing_service_account(secrets)

            self._initialized = True
            logger.info(f"StorageService initialized for bucket: {self.bucket_name}")

    def _get_signing_service_account(self, secrets) -> Optional[str]:
        """Get the service account email to use for signing URLs."""

        # For local dev, try to get from secrets/env
        sa_email = secrets.get_secret("GCS_SIGNING_SERVICE_ACCOUNT", default=None)
        if not sa_email:
            logger.warning(
                "No signing service account found. "
                "Set GCS_SIGNING_SERVICE_ACCOUNT for local development with user credentials."
            )
        return sa_email

    def _ensure_emulator_bucket(self):
        """Create the bucket in the emulator if it doesn't exist."""
        bucket = self._client.bucket(self.bucket_name)
        if not bucket.exists():
            self._client.create_bucket(bucket)
            logger.info(f"Created emulator bucket: {self.bucket_name}")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of StorageService"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_bucket(self):
        """Get the storage bucket"""
        return self._client.bucket(self.bucket_name)

    def upload_file(
        self,
        data: bytes,
        destination_path: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a file to GCS.

        Args:
            data: File content as bytes
            destination_path: Path in bucket (e.g., "attachments/email-uuid/file.pdf")
            content_type: MIME type of the file

        Returns:
            The GCS path (gs://bucket/path)
        """
        bucket = self._get_bucket()
        blob = bucket.blob(destination_path)

        blob.upload_from_string(
            data,
            content_type=content_type or "application/octet-stream",
        )

        gcs_path = f"gs://{self.bucket_name}/{destination_path}"
        logger.info(f"Uploaded file to {gcs_path}")
        return gcs_path

    def generate_signed_url(
        self,
        gcs_path: str,
        expiration_minutes: int = 15,
    ) -> str:
        """
        Generate a signed URL for secure, time-limited access.

        Args:
            gcs_path: Full GCS path (gs://bucket/path) or just the blob path
            expiration_minutes: URL validity duration (default: 15 minutes)

        Returns:
            A signed URL for downloading the file
        """
        # Parse gcs_path if it's a full gs:// URL
        if gcs_path.startswith("gs://"):
            # Extract path after bucket name
            parts = gcs_path.replace("gs://", "").split("/", 1)
            blob_path = parts[1] if len(parts) > 1 else ""
        else:
            blob_path = gcs_path

        # Emulator: return direct URL using the public host (reachable from browser)
        if self._emulator_host:
            encoded_path = quote(blob_path, safe="")
            return f"{self._emulator_public_host}/storage/v1/b/{self.bucket_name}/o/{encoded_path}?alt=media"

        bucket = self._get_bucket()
        blob = bucket.blob(blob_path)

        # Check if credentials can sign directly (service account key)
        if hasattr(self._credentials, "sign_bytes"):
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiration_minutes),
                method="GET",
            )
        else:
            # Use IAM signBlob API for user credentials or metadata-based credentials
            if not self._signing_sa_email:
                raise ValueError(
                    "Cannot sign URLs: no signing service account configured. "
                    "Set GCS_SIGNING_SERVICE_ACCOUNT secret for local development."
                )

            # Refresh credentials to ensure we have a valid token
            self._credentials.refresh(google_requests.Request())

            logger.info(f"Signing URL for service account: {self._signing_sa_email}")

            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiration_minutes),
                method="GET",
                service_account_email=self._signing_sa_email,
                access_token=self._credentials.token,
            )

        return url

    def delete_file(self, gcs_path: str) -> bool:
        """
        Delete a file from GCS.

        Args:
            gcs_path: Full GCS path (gs://bucket/path) or just the blob path

        Returns:
            True if deleted, False if not found
        """
        if gcs_path.startswith("gs://"):
            parts = gcs_path.replace("gs://", "").split("/", 1)
            blob_path = parts[1] if len(parts) > 1 else ""
        else:
            blob_path = gcs_path

        bucket = self._get_bucket()
        blob = bucket.blob(blob_path)

        if blob.exists():
            blob.delete()
            logger.info(f"Deleted file: {gcs_path}")
            return True

        logger.warning(f"File not found for deletion: {gcs_path}")
        return False

    def file_exists(self, gcs_path: str) -> bool:
        """Check if a file exists in GCS"""
        if gcs_path.startswith("gs://"):
            parts = gcs_path.replace("gs://", "").split("/", 1)
            blob_path = parts[1] if len(parts) > 1 else ""
        else:
            blob_path = gcs_path

        bucket = self._get_bucket()
        blob = bucket.blob(blob_path)
        return blob.exists()


_storage_service: Optional[StorageService] = None


def get_storage() -> StorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
