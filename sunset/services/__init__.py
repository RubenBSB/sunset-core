"""Sunset services - shared library for common functionality."""

__all__ = [
    "SecretsService",
    "AuthService",
    "WhatsappService",
    "extract_webhook_message",
    "EmailSendService",
    "StorageService",
    "PubSubService",
    "AnalyticsService",
    "LLMService",
    "MonitoringService",
    "RetrievalService",
    "MultimodalEmbeddingService",
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
    "HubspotService",
    "SlackService",
    "init_observability",
    "instrument_fastapi",
    "DuffelService",
    "DuffelError",
]


def __getattr__(name: str):
    """Lazy import services to avoid loading unused dependencies."""
    if name == "SecretsService":
        from sunset.services.secrets import SecretsService

        return SecretsService
    if name == "AuthService":
        from sunset.services.auth import AuthService

        return AuthService
    # "WhatsAppService" kept as an alias for the historical (broken) spelling.
    if name in ("WhatsappService", "WhatsAppService"):
        from sunset.services.whatsapp import WhatsappService

        return WhatsappService
    if name == "extract_webhook_message":
        from sunset.services.whatsapp import extract_webhook_message

        return extract_webhook_message
    if name == "EmailSendService":
        from sunset.services.email import EmailSendService

        return EmailSendService
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
    if name == "MultimodalEmbeddingService":
        from sunset.services.retrieval.multimodal import MultimodalEmbeddingService

        return MultimodalEmbeddingService
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
    if name == "HubspotService":
        from sunset.services.hubspot import HubspotService

        return HubspotService
    if name == "SlackService":
        from sunset.services.slack import SlackService

        return SlackService
    if name == "init_observability":
        from sunset.services.observability import init_observability

        return init_observability
    if name == "instrument_fastapi":
        from sunset.services.observability import instrument_fastapi

        return instrument_fastapi
    if name == "DuffelService":
        from sunset.services.duffel import DuffelService

        return DuffelService
    if name == "DuffelError":
        from sunset.services.duffel import DuffelError

        return DuffelError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
