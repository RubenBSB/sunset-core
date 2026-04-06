"""Website crawling service with pluggable backends (Firecrawl, Playwright)."""

from .base import CrawlFile, CrawlPage, CrawlResult, CrawlService, OutputFormat
from .firecrawl import FirecrawlService
from .playwright import PlaywrightCrawlService

__all__ = [
    "OutputFormat",
    "CrawlPage",
    "CrawlFile",
    "CrawlResult",
    "CrawlService",
    "FirecrawlService",
    "PlaywrightCrawlService",
]
