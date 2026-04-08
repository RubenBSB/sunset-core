"""Sunset services - shared library for common functionality."""

__all__ = [
    "SecretsService",
    "AuthService",
    "WhatsAppService",
    "extract_webhook_message",
    "StorageService",
    "PubSubService",
    "AnalyticsService",
    "LLMService",
    "MonitoringService",
    "RetrievalService",
    "ChatService",
    "CrawlService",
    "FirecrawlService",
    "PlaywrightCrawlService",
    "RedisService",
    "InstagramService",
    "YouTubeService",
    "ASRService",
    "ShopifyService",
    "GoogleDriveService",
    "SEOService",
]


def __getattr__(name: str):
    """Lazy import services to avoid loading unused dependencies."""
    if name == "SecretsService":
        from sunset.services.secrets import SecretsService

        return SecretsService
    if name == "AuthService":
        from sunset.services.auth import AuthService

        return AuthService
    if name == "WhatsAppService":
        from sunset.services.whatsapp import WhatsAppService

        return WhatsAppService
    if name == "extract_webhook_message":
        from sunset.services.whatsapp import extract_webhook_message

        return extract_webhook_message
    if name == "StorageService":
        from sunset.services.storage import StorageService

        return StorageService
    if name == "PubSubService":
        from sunset.services.pubsub import PubSubService

        return PubSubService
    if name == "AnalyticsService":
        from sunset.services.analytics import AnalyticsService

        return AnalyticsService
    if name == "LLMService":
        from sunset.services.llm import LLMService

        return LLMService
    if name == "MonitoringService":
        from sunset.services.monitoring import MonitoringService

        return MonitoringService
    if name == "RetrievalService":
        from sunset.services.retrieval import RetrievalService

        return RetrievalService
    if name == "ChatService":
        from sunset.services.chat import ChatService

        return ChatService
    if name == "CrawlService":
        from sunset.services.crawl import CrawlService

        return CrawlService
    if name == "FirecrawlService":
        from sunset.services.crawl import FirecrawlService

        return FirecrawlService
    if name == "PlaywrightCrawlService":
        from sunset.services.crawl import PlaywrightCrawlService

        return PlaywrightCrawlService
    if name == "RedisService":
        from sunset.services.redis import RedisService

        return RedisService
    if name == "InstagramService":
        from sunset.services.instagram import InstagramService

        return InstagramService
    if name == "YouTubeService":
        from sunset.services.youtube import YouTubeService

        return YouTubeService
    if name == "ASRService":
        from sunset.services.asr import ASRService

        return ASRService
    if name == "ShopifyService":
        from sunset.services.shopify import ShopifyService

        return ShopifyService
    if name == "GoogleDriveService":
        from sunset.services.google_drive import GoogleDriveService

        return GoogleDriveService
    if name == "SEOService":
        from sunset.services.seo import SEOService

        return SEOService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
