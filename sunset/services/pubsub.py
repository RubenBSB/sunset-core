"""
Google Cloud Pub/Sub service for async messaging.

Usage:
    from sunset.services import PubSubService

    pubsub = PubSubService()
    pubsub.publish("my-topic", {"user_id": "123", "action": "signup"})
"""

import os
import json
import logging
from typing import Dict, Callable, Any, Optional, Set

from google.cloud import pubsub_v1
from google.api_core.exceptions import AlreadyExists


logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict], Any]


class PubSubService:
    """GCP Pub/Sub service for async messaging."""

    _instance: Optional["PubSubService"] = None

    def __init__(
        self, project_id: Optional[str] = None, topic_prefix: Optional[str] = None
    ):
        env = os.getenv("ENV", "local")
        self.env = "prod" if env == "production" else env
        self.emulator_host = os.getenv("PUBSUB_EMULATOR_HOST")
        self.project_id = (
            project_id
            or os.getenv("PUBSUB_PROJECT_ID")
            or os.getenv("GCP_PROJECT_ID", "local-test-project")
        )
        self.topic_prefix = topic_prefix or os.getenv("PUBSUB_TOPIC_PREFIX", "app")

        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()

        self._message_handlers: Dict[str, MessageHandler] = {}
        self._created_topics: Set[str] = set()

        if self.emulator_host:
            logger.info(f"PubSub using emulator at {self.emulator_host}")
        else:
            logger.info(f"PubSub initialized for env: {self.env}")

    @classmethod
    def get_instance(cls, **kwargs) -> "PubSubService":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def get_topic_path(self, topic_name: str) -> str:
        return self.publisher.topic_path(
            self.project_id, f"{self.topic_prefix}-{topic_name}-{self.env}"
        )

    def get_subscription_path(self, topic_name: str) -> str:
        return self.subscriber.subscription_path(
            self.project_id, f"{self.topic_prefix}-{topic_name}-sub-{self.env}"
        )

    def ensure_topic_exists(self, topic_name: str) -> str:
        topic_path = self.get_topic_path(topic_name)

        if topic_path in self._created_topics:
            return topic_path

        if self.emulator_host:
            try:
                self.publisher.create_topic(request={"name": topic_path})
                logger.info(f"Created topic: {topic_path}")
            except AlreadyExists:
                pass

            sub_path = self.get_subscription_path(topic_name)
            try:
                self.subscriber.create_subscription(
                    request={"name": sub_path, "topic": topic_path}
                )
                logger.info(f"Created subscription: {sub_path}")
            except AlreadyExists:
                pass

        self._created_topics.add(topic_path)
        return topic_path

    def publish(
        self,
        topic_name: str,
        data: Dict[str, Any],
        attributes: Optional[Dict[str, str]] = None,
    ) -> str:
        topic_path = self.ensure_topic_exists(topic_name)
        message_bytes = json.dumps(data).encode("utf-8")

        future = self.publisher.publish(topic_path, message_bytes, **(attributes or {}))
        message_id = future.result()

        logger.debug(f"Published to {topic_name}: {message_id}")
        return message_id

    def handler(self, topic_name: str):
        def decorator(func: MessageHandler):
            self._message_handlers[topic_name] = func
            return func

        return decorator

    def process_message(self, topic_name: str, data: dict) -> Any:
        handler = self._message_handlers.get(topic_name)
        if handler:
            return handler(data)
        logger.warning(f"No handler for topic: {topic_name}")
        return None
