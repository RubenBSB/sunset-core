import asyncio
import datetime
import json
import logging
import os
from typing import Any, Callable, Dict, Optional, Set

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import pubsub_v1

logger = logging.getLogger(__name__)

# Type for message handlers
MessageHandler = Callable[[dict], Any]


class PubSubService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PubSubService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            env = os.getenv("ENV", "local")
            # Map to Terraform naming: production -> prod
            self.env = "prod" if env == "production" else env
            self.emulator_host = os.getenv("PUBSUB_EMULATOR_HOST")
            # Use separate project ID for Pub/Sub (emulator uses fake project)
            self.project_id = os.getenv("PUBSUB_PROJECT_ID") or os.getenv(
                "GCP_PROJECT_ID", "local-test-project"
            )
            self.topic_prefix = os.getenv("PUBSUB_TOPIC_PREFIX", "vedis")

            self.publisher = pubsub_v1.PublisherClient()
            self.subscriber = pubsub_v1.SubscriberClient()

            # Message handlers for different subscriptions
            self._message_handlers: Dict[str, MessageHandler] = {}

            # Active SSE connections
            self.active_connections: Dict[str, asyncio.Queue] = {}

            # Subscriber management (multiple subscribers supported)
            self._subscriber_managers: Dict[str, "PubSubSubscriberManager"] = {}
            self._subscriber_tasks: Dict[str, asyncio.Task] = {}

            # Store reference to main event loop for cross-thread communication
            self._main_loop = None

            # Track created topics to avoid repeated creation attempts
            self._created_topics: Set[str] = set()

            self._initialized = True
            if self.emulator_host:
                logger.info(f"PubSub service using emulator at {self.emulator_host}")
            else:
                logger.info(f"PubSub service initialized for environment: {self.env}")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of PubSubService"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_topic_path(self, topic_name: str) -> str:
        """Get the full topic path for a given topic name."""
        return self.publisher.topic_path(
            self.project_id, f"{self.topic_prefix}-{topic_name}-{self.env}"
        )

    def get_subscription_path(self, topic_name: str) -> str:
        """Get the subscription path for a given topic name."""
        return self.subscriber.subscription_path(
            self.project_id, f"{self.topic_prefix}-{topic_name}-sub-{self.env}"
        )

    def _ensure_topic_exists(self, topic_path: str) -> None:
        """Ensure the topic exists, creating it if necessary (emulator mode only)."""
        if topic_path in self._created_topics:
            return

        if not self.emulator_host:
            # In production, topics should be pre-created via Terraform/IaC
            self._created_topics.add(topic_path)
            return

        try:
            self.publisher.create_topic(name=topic_path)
            logger.info(f"Created topic: {topic_path}")
        except AlreadyExists:
            logger.debug(f"Topic already exists: {topic_path}")
        except Exception as e:
            logger.warning(f"Failed to create topic {topic_path}: {e}")

        self._created_topics.add(topic_path)

    def _ensure_subscription_exists(self, topic_name: str) -> None:
        """Ensure the subscription exists, creating it if necessary (emulator mode only)."""
        if not self.emulator_host:
            return

        topic_path = self.get_topic_path(topic_name)
        subscription_path = self.get_subscription_path(topic_name)

        # Ensure topic exists first
        self._ensure_topic_exists(topic_path)

        try:
            self.subscriber.create_subscription(
                name=subscription_path, topic=topic_path
            )
            logger.info(f"Created subscription: {subscription_path}")
        except AlreadyExists:
            logger.debug(f"Subscription already exists: {subscription_path}")
        except Exception as e:
            logger.warning(f"Failed to create subscription {subscription_path}: {e}")

    def add_connection(self, user_id: str):
        """Add a new SSE connection for a user"""
        if user_id not in self.active_connections:
            self.active_connections[user_id] = asyncio.Queue()
        return self.active_connections[user_id]

    def remove_connection(self, user_id: str):
        """Remove an SSE connection for a user"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"Connection for user {user_id} removed")

    def register_handler(self, subscription_key: str, handler: MessageHandler):
        """Register a handler for a specific subscription."""
        self._message_handlers[subscription_key] = handler
        logger.info(f"Registered handler for subscription: {subscription_key}")

    def get_handler(self, subscription_key: str) -> Optional[MessageHandler]:
        """Get the handler for a subscription."""
        return self._message_handlers.get(subscription_key)

    def publish_message(self, message: dict, topic_name: str = "extraction") -> None:
        """Publish a message to the specified topic"""
        message_id = message.get("file_id") or message.get("content_id") or "unknown"
        encoded = json.dumps(message).encode("utf-8")

        topic_path = self.get_topic_path(topic_name)

        # Ensure topic exists (creates if needed in emulator mode)
        self._ensure_topic_exists(topic_path)

        # Publish message asynchronously without blocking
        future = self.publisher.publish(topic_path, encoded)
        future.add_done_callback(
            lambda f: self._publish_callback(f, message_id, topic_path)
        )
        logger.info(f"Initiated message publishing to {topic_path}")

    def _publish_callback(self, future, message_id, topic_path):
        """Callback for publish operations"""
        try:
            pub_message_id = future.result(timeout=10)
            logger.info(f"Published message {pub_message_id} to {topic_path}")
        except NotFound:
            # Topic was deleted (e.g., emulator restart), clear cache so it gets recreated
            self._created_topics.discard(topic_path)
            logger.error(
                f"Failed to publish to {topic_path}: Topic not found (will recreate on next publish)"
            )
        except Exception as e:
            logger.error(f"Failed to publish to {topic_path}: {e}")

    async def start_subscriber(self, topic_name: str):
        """
        Start a Pub/Sub subscriber for a given topic.

        Args:
            topic_name: Name of the topic (e.g., "extraction")
        """
        if topic_name in self._subscriber_managers:
            logger.warning(f"Pub/Sub subscriber already running for {topic_name}")
            return

        # Store reference to the main event loop
        self._main_loop = asyncio.get_running_loop()

        # Ensure subscription exists (creates topic + subscription if needed in emulator mode)
        self._ensure_subscription_exists(topic_name)

        subscription_path = self.get_subscription_path(topic_name)

        manager = PubSubSubscriberManager(self, subscription_path, topic_name)
        self._subscriber_managers[topic_name] = manager
        self._subscriber_tasks[topic_name] = asyncio.create_task(manager.start())
        logger.info(f"Pub/Sub subscriber started for {subscription_path}")

    async def stop_subscriber(self, topic_name: str = None):
        """Stop Pub/Sub subscriber(s) on application shutdown.

        Args:
            topic_name: Optional topic name to stop. If None, stops all subscribers.
        """
        if not self._subscriber_managers:
            logger.warning("No Pub/Sub subscribers running")
            return

        topics_to_stop = (
            [topic_name] if topic_name else list(self._subscriber_managers.keys())
        )

        for name in topics_to_stop:
            manager = self._subscriber_managers.get(name)
            task = self._subscriber_tasks.get(name)

            if not manager:
                continue

            await manager.stop()

            if task:
                try:
                    await asyncio.wait_for(task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Pub/Sub subscriber for {name} did not stop gracefully, cancelling..."
                    )
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self._subscriber_managers.pop(name, None)
            self._subscriber_tasks.pop(name, None)
            logger.info(f"Pub/Sub subscriber stopped for {name}")

        if not self._subscriber_managers:
            self._main_loop = None


class PubSubSubscriberManager:
    """Manages the Pub/Sub subscriber with retry logic"""

    def __init__(
        self,
        pubsub_service: PubSubService,
        subscription_path: Optional[str] = None,
        subscription_key: Optional[str] = None,
    ):
        self.pubsub_service = pubsub_service
        self.subscriber = pubsub_service.subscriber
        self.subscription_path = subscription_path or pubsub_service.subscription_path
        self.subscription_key = subscription_key
        self.shutdown_event = asyncio.Event()
        self.retry_delay = 1  # Initial retry delay in seconds
        self.max_retry_delay = 60  # Maximum retry delay
        self.retry_multiplier = 2

    async def start(self):
        """Start the subscriber with retry logic"""
        logger.info(f"Starting Pub/Sub subscriber for {self.subscription_path}")
        while not self.shutdown_event.is_set():
            try:
                # Ensure subscription exists before each attempt (handles emulator restarts)
                if self.subscription_key:
                    self.pubsub_service._ensure_subscription_exists(
                        self.subscription_key
                    )

                await self._run_subscriber()
            except Exception as e:
                if self.shutdown_event.is_set():
                    logger.info("Subscriber shutting down")
                    break

                logger.error(
                    f"Subscriber error: {e}. Retrying in {self.retry_delay} seconds..."
                )
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(), timeout=self.retry_delay
                    )
                    break  # Shutdown was requested during retry delay
                except asyncio.TimeoutError:
                    pass  # Continue with retry

                # Exponential backoff with jitter
                self.retry_delay = min(
                    self.retry_delay * self.retry_multiplier, self.max_retry_delay
                )

        logger.info("Pub/Sub subscriber stopped")

    async def _run_subscriber(self):
        """Run the subscriber in a separate thread"""

        def blocking_subscriber():
            flow_control = pubsub_v1.types.FlowControl(max_messages=100)
            logger.info(f"Connecting to Pub/Sub subscription: {self.subscription_path}")

            streaming_pull_future = self.subscriber.subscribe(
                self.subscription_path,
                callback=self._pubsub_callback,
                flow_control=flow_control,
            )

            # Reset retry delay on successful connection
            self.retry_delay = 1

            try:
                # This will block until cancelled or an error occurs
                streaming_pull_future.result()
            except Exception as e:
                streaming_pull_future.cancel()
                raise e

        # Run the blocking subscriber in a thread
        await asyncio.to_thread(blocking_subscriber)

    def _pubsub_callback(self, message):
        """Handle incoming Pub/Sub messages"""
        try:
            data = json.loads(message.data.decode("utf-8"))

            # Check for custom handler first
            if self.subscription_key:
                handler = self.pubsub_service.get_handler(self.subscription_key)
                if handler:
                    logger.info(
                        f"Processing message with handler for {self.subscription_key}"
                    )
                    main_loop = self.pubsub_service._main_loop
                    if main_loop and main_loop.is_running():
                        # Run async handler in main loop
                        future = asyncio.run_coroutine_threadsafe(
                            handler(data), main_loop
                        )
                        # Wait for completion with timeout
                        try:
                            future.result(timeout=300)  # 5 min timeout for extraction
                        except Exception as e:
                            logger.error(f"Handler error: {e}")
                            message.nack()
                            return
                    message.ack()
                    return

            # Default behavior: SSE notification
            content_id = data.get("content_id")
            status = data.get("status")
            user_ids = data.get("user_ids", [])
            content_title = data.get("content_title")

            logger.info(
                f"Received pubsub message for content {content_id} with status {status} for {len(user_ids)} users"
            )

            # Prepare notification message
            notification_message = {
                "content_id": content_id,
                "status": status,
                "content_title": content_title,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            # Notify connected users via SSE
            notified_users = 0
            for user_id in user_ids:
                if user_id in self.pubsub_service.active_connections:
                    try:
                        # Use the stored main event loop reference
                        main_loop = self.pubsub_service._main_loop
                        if main_loop and main_loop.is_running():
                            # Schedule the queue.put() on the main event loop thread-safely
                            asyncio.run_coroutine_threadsafe(
                                self.pubsub_service.active_connections[user_id].put(
                                    notification_message
                                ),
                                main_loop,
                            )
                            notified_users += 1
                            logger.info(
                                f"Notified user {user_id} about content {content_id} status: {status}"
                            )
                        else:
                            logger.warning(
                                f"Main event loop not available for user {user_id}"
                            )
                    except Exception as e:
                        logger.error(f"Failed to notify user {user_id}: {e}")
                        # Remove broken connection
                        self.pubsub_service.remove_connection(user_id)

            logger.info(
                f"Successfully notified {notified_users} out of {len(user_ids)} users"
            )
            message.ack()

        except Exception as e:
            logger.error(f"Error processing pubsub message: {e}")
            message.nack()

    async def stop(self):
        """Stop the subscriber gracefully"""
        logger.info("Stopping Pub/Sub subscriber...")
        self.shutdown_event.set()
