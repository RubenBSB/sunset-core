"""Sunset services - shared library for common functionality."""

__all__ = [
    "SecretsService",
    "WhatsAppService",
    "extract_webhook_message",
    "StorageService",
    "PubSubService",
    "AnalyticsService",
    "LLMService",
]


def __getattr__(name: str):
    """Lazy import services to avoid loading unused dependencies."""
    if name == "SecretsService":
        from sunset.services.secrets import SecretsService
        return SecretsService
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
