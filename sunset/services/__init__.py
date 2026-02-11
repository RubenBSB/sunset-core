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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
