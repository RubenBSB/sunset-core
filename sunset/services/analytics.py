import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import posthog

logger = logging.getLogger(__name__)


class AnalyticsService:
    _instance = None

    def __init__(self):
        self.posthog_api_key = os.getenv("POSTHOG_API_KEY")
        self.posthog_host = os.getenv("POSTHOG_HOST", "https://app.posthog.com")

        if self.posthog_api_key:
            posthog.api_key = self.posthog_api_key
            posthog.host = self.posthog_host
            logger.info("PostHog analytics initialized")
        else:
            logger.warning("PostHog API key not found. Analytics disabled.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_enabled(self) -> bool:
        """Check if analytics is enabled (PostHog API key is set)"""
        return bool(self.posthog_api_key)

    def track_event(
        self, user_id: str, event: str, properties: Optional[Dict[str, Any]] = None
    ):
        """Track an event with PostHog"""
        if not self.is_enabled():
            return

        try:
            posthog.capture(
                distinct_id=user_id, event=event, properties=properties or {}
            )
            logger.info(f"Tracked event: {event} for user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to track event {event}: {str(e)}")

    def track_user_login(self, user_id: str, email: str, login_method: str = "apple"):
        """Track user login event"""
        self.track_event(
            user_id=user_id,
            event="user_login",
            properties={
                "email": email,
                "login_method": login_method,
                "timestamp": datetime.utcnow().isoformat(),
                "platform": "ios",
            },
        )

    def track_user_logout(self, user_id: str):
        """Track user logout event"""
        self.track_event(
            user_id=user_id,
            event="user_logout",
            properties={"timestamp": datetime.utcnow().isoformat(), "platform": "ios"},
        )

    def track_session_start(self, user_id: str):
        """Track session start"""
        self.track_event(
            user_id=user_id,
            event="session_start",
            properties={"timestamp": datetime.utcnow().isoformat(), "platform": "ios"},
        )

    def track_session_end(
        self, user_id: str, session_duration_seconds: Optional[int] = None
    ):
        """Track session end with duration"""
        properties = {"timestamp": datetime.utcnow().isoformat(), "platform": "ios"}

        if session_duration_seconds is not None:
            properties["session_duration_seconds"] = session_duration_seconds
            properties["session_duration_minutes"] = round(
                session_duration_seconds / 60, 2
            )

        self.track_event(user_id=user_id, event="session_end", properties=properties)

    def identify_user(self, user_id: str, properties: Dict[str, Any]):
        """Identify user with properties"""
        if not self.is_enabled():
            return

        try:
            posthog.identify(distinct_id=user_id, properties=properties)
            logger.info(f"Identified user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to identify user {user_id}: {str(e)}")


# Singleton instance
analytics = AnalyticsService()
