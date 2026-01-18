"""
Google Cloud Storage service.

Handles file uploads, downloads, and signed URL generation.

Usage:
    from sunset.services import StorageService
    
    storage = StorageService(bucket_name="my-bucket")
    gcs_path = storage.upload_file(data, "path/to/file.pdf")
    url = storage.generate_signed_url(gcs_path, expiration_minutes=15)
"""

import os
import logging
from typing import Optional
from datetime import timedelta

from google.cloud import storage
from google.auth import default
from google.auth.transport import requests as google_requests


logger = logging.getLogger(__name__)


class StorageService:
    """GCS storage service for secure file operations."""

    _instance: Optional["StorageService"] = None

    def __init__(self, bucket_name: Optional[str] = None, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")

        self._credentials, _ = default()
        self._client = storage.Client(project=self.project_id, credentials=self._credentials)

        self._signing_sa_email = self._get_signing_service_account()
        logger.info(f"StorageService initialized for bucket: {self.bucket_name}")

    def _get_signing_service_account(self) -> Optional[str]:
        if hasattr(self._credentials, "service_account_email"):
            return self._credentials.service_account_email
        
        sa_email = os.getenv("GCS_SIGNING_SERVICE_ACCOUNT")
        if not sa_email:
            logger.warning("No signing service account found. Set GCS_SIGNING_SERVICE_ACCOUNT for local development.")
        return sa_email

    @classmethod
    def get_instance(cls, **kwargs) -> "StorageService":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def _get_bucket(self):
        return self._client.bucket(self.bucket_name)

    def upload_file(self, data: bytes, destination_path: str, content_type: Optional[str] = None) -> str:
        bucket = self._get_bucket()
        blob = bucket.blob(destination_path)
        blob.upload_from_string(data, content_type=content_type or "application/octet-stream")
        gcs_path = f"gs://{self.bucket_name}/{destination_path}"
        logger.info(f"Uploaded file to {gcs_path}")
        return gcs_path

    def generate_signed_url(self, gcs_path: str, expiration_minutes: int = 15) -> str:
        if gcs_path.startswith("gs://"):
            parts = gcs_path.replace("gs://", "").split("/", 1)
            blob_path = parts[1] if len(parts) > 1 else ""
        else:
            blob_path = gcs_path

        bucket = self._get_bucket()
        blob = bucket.blob(blob_path)

        if hasattr(self._credentials, "sign_bytes"):
            url = blob.generate_signed_url(version="v4", expiration=timedelta(minutes=expiration_minutes), method="GET")
        else:
            if not self._signing_sa_email:
                raise ValueError("Cannot sign URLs: no signing service account configured.")
            if not self._credentials.token:
                self._credentials.refresh(google_requests.Request())
            url = blob.generate_signed_url(
                version="v4", expiration=timedelta(minutes=expiration_minutes), method="GET",
                service_account_email=self._signing_sa_email, access_token=self._credentials.token
            )

        return url

    def delete_file(self, gcs_path: str) -> bool:
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

        logger.warning(f"File not found: {gcs_path}")
        return False

    def file_exists(self, gcs_path: str) -> bool:
        if gcs_path.startswith("gs://"):
            parts = gcs_path.replace("gs://", "").split("/", 1)
            blob_path = parts[1] if len(parts) > 1 else ""
        else:
            blob_path = gcs_path

        bucket = self._get_bucket()
        blob = bucket.blob(blob_path)
        return blob.exists()
