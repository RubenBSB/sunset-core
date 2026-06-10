"""Redis service — thin async wrapper around redis.asyncio for pub/sub and general use."""

import json
import logging
import os
from typing import AsyncIterator

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisService:
    """Async Redis client with pub/sub support. Singleton."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._client: aioredis.Redis | None = None
            self._initialized = True

    async def connect(self) -> aioredis.Redis:
        """Connect to Redis. Returns the underlying client."""
        if self._client is None:
            # keepalive + health checks: Memorystore silently drops idle
            # connections, which surfaces as TimeoutError on the next read.
            self._client = aioredis.from_url(
                self.url,
                decode_responses=True,
                socket_keepalive=True,
                socket_connect_timeout=10,
                health_check_interval=30,
                retry_on_timeout=True,
            )
            logger.info(f"Connected to Redis at {self.url}")
        return self._client

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Redis connection closed")

    @property
    def client(self) -> aioredis.Redis:
        """Get the underlying Redis client. Raises if not connected."""
        if self._client is None:
            raise RuntimeError("RedisService not connected. Call connect() first.")
        return self._client

    async def publish(self, channel: str, data: dict) -> int:
        """Publish a JSON message to a channel. Returns number of subscribers that received it."""
        return await self.client.publish(channel, json.dumps(data))

    async def subscribe(self, channel: str) -> AsyncIterator[dict]:
        """Subscribe to a channel and yield parsed JSON messages."""
        pubsub = self.client.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
