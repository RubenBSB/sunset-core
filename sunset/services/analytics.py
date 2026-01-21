"""
Analytics service using PostHog.

Usage:
    from sunset.services import AnalyticsService

    analytics = AnalyticsService()
    analytics.track_event(user_id, "button_clicked", {"button": "signup"})
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import posthog


logger = logging.getLogger(__name__)


class AnalyticsService:
    """PostHog analytics service for event tracking."""

    _instance: Optional["AnalyticsService"] = None

    def __init__(self, api_key: Optional[str] = None, host: Optional[str] = None):
        self.api_key = api_key or os.getenv("POSTHOG_API_KEY")
        self.host = host or os.getenv("POSTHOG_HOST", "https://app.posthog.com")

        if self.api_key:
            posthog.api_key = self.api_key
            posthog.host = self.host
            logger.info("PostHog analytics initialized")
        else:
            logger.warning("PostHog API key not found. Analytics disabled.")

    @classmethod
    def get_instance(cls, **kwargs) -> "AnalyticsService":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def track_event(
        self, user_id: str, event: str, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self.is_enabled():
            return

        try:
            posthog.capture(
                distinct_id=user_id, event=event, properties=properties or {}
            )
            logger.debug(f"Tracked event: {event} for user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to track event {event}: {e}")

    def identify_user(self, user_id: str, properties: Dict[str, Any]) -> None:
        if not self.is_enabled():
            return

        try:
            posthog.identify(distinct_id=user_id, properties=properties)
            logger.debug(f"Identified user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to identify user {user_id}: {e}")

    def track_user_login(
        self, user_id: str, email: str, login_method: str = "google"
    ) -> None:
        self.track_event(
            user_id,
            "user_login",
            {
                "email": email,
                "login_method": login_method,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def track_user_signup(
        self, user_id: str, email: str, signup_method: str = "google"
    ) -> None:
        self.track_event(
            user_id,
            "user_signup",
            {
                "email": email,
                "signup_method": signup_method,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
