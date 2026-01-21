import os
import jwt
import time
import json
import logging
from typing import Optional, Dict, Any

import httpx

from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)


class APNsService:
    _instance = None

    def __init__(
        self,
        key_id: str,
        team_id: str,
        bundle_id: str,
        key_file_path: str,
        use_sandbox: bool = False,
    ):
        self.key_id = key_id
        self.team_id = team_id
        self.bundle_id = bundle_id
        self.use_sandbox = use_sandbox

        # APNs endpoints
        self.apns_url = (
            "https://api.sandbox.push.apple.com"
            if use_sandbox
            else "https://api.push.apple.com"
        )

        # Load the private key
        try:
            with open(key_file_path, "rb") as key_file:
                self.private_key = serialization.load_pem_private_key(
                    key_file.read(), password=None
                )
            logger.info(f"Successfully loaded APNs private key from {key_file_path}")
        except Exception as e:
            logger.error(
                f"Failed to load APNs private key from {key_file_path}: {str(e)}"
            )
            raise

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            key_id = os.getenv("APPLE_NOTIFICATION_KEY_ID")
            team_id = os.getenv("APPLE_TEAM_ID")
            bundle_id = os.getenv("APPLE_BUNDLE_ID")
            key_file_path = os.getenv("APPLE_KEY_FILEPATH")

            logger.info(
                f"Initializing APNs service with key_id={key_id}, team_id={team_id}, bundle_id={bundle_id}, key_file_path={key_file_path}"
            )

            cls._instance = cls(
                key_id=key_id,
                team_id=team_id,
                bundle_id=bundle_id,
                key_file_path=key_file_path,
                use_sandbox=True,
            )
        return cls._instance

    def generate_jwt_token(self) -> str:
        """Generate JWT token for APNs authentication"""
        now = int(time.time())

        headers = {"alg": "ES256", "kid": self.key_id}

        payload = {
            "iss": self.team_id,
            "iat": now,
            "exp": now + 3600,  # Token expires in 1 hour
        }

        return jwt.encode(payload, self.private_key, algorithm="ES256", headers=headers)

    async def send_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        badge: Optional[int] = None,
        sound: str = "default",
        custom_data: Optional[Dict[str, Any]] = None,
        priority: int = 10,
        collapse_id: Optional[str] = None,
    ) -> bool:
        """Send push notification to iOS device"""

        logger.info(
            f"Sending notification to device token: {device_token[:10]}... (truncated)"
        )
        logger.info(f"Using APNs URL: {self.apns_url}")

        # Create the payload
        payload = {"aps": {"alert": {"title": title, "body": body}, "sound": sound}}

        if badge is not None:
            payload["aps"]["badge"] = 0

        if custom_data:
            payload.update(custom_data)

        logger.info(f"Notification payload: {json.dumps(payload)}")

        # Generate JWT token
        try:
            jwt_token = self.generate_jwt_token()
            logger.info("JWT token generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate JWT token: {str(e)}")
            return False

        # Prepare headers
        headers = {
            "authorization": f"bearer {jwt_token}",
            "apns-topic": self.bundle_id,
            "apns-priority": str(priority),
            "content-type": "application/json",
        }

        if collapse_id:
            headers["apns-collapse-id"] = collapse_id

        logger.info(f"Request headers: {headers}")

        # Send the notification
        url = f"{self.apns_url}/3/device/{device_token}"
        logger.info(f"Sending POST request to: {url}")

        try:
            async with httpx.AsyncClient(http2=True) as client:
                response = await client.post(
                    url, json=payload, headers=headers, timeout=30.0
                )

                logger.info(f"APNs response status: {response.status_code}")
                logger.info(f"APNs response headers: {dict(response.headers)}")

                if response.status_code == 200:
                    logger.info("Notification sent successfully")
                    return True
                else:
                    logger.error(
                        f"APNs Error: {response.status_code} - {response.text}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False

    async def send_silent_notification(
        self, device_token: str, custom_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send silent push notification (content-available)"""

        payload = {"aps": {"content-available": 1}}

        if custom_data:
            payload.update(custom_data)

        jwt_token = self.generate_jwt_token()

        headers = {
            "authorization": f"bearer {jwt_token}",
            "apns-topic": self.bundle_id,
            "apns-priority": "5",  # Lower priority for silent notifications
            "content-type": "application/json",
        }

        url = f"{self.apns_url}/3/device/{device_token}"

        try:
            async with httpx.AsyncClient(http2=True) as client:
                response = await client.post(
                    url, json=payload, headers=headers, timeout=30.0
                )

                return response.status_code == 200

        except Exception as e:
            logger.error(f"Error sending silent notification: {e}")
            return False
