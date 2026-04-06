import logging
import os
from typing import Any

from .base import CrawlPage, CrawlResult, CrawlService, OutputFormat

logger = logging.getLogger(__name__)


class FirecrawlService(CrawlService):
    """Website crawler using the Firecrawl API.

    Args:
        api_key: Firecrawl API key. Falls back to ``FIRECRAWL_API_KEY`` env var.
        only_main_content: Strip headers/navs/footers from output.
        request_delay: Seconds between scrapes (forces concurrency=1 on Firecrawl side).
        poll_interval: Seconds between status checks while waiting for crawl completion.
        timeout: Max seconds to wait for crawl completion.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        only_main_content: bool = True,
        request_delay: float | None = None,
        poll_interval: int = 2,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self.only_main_content = only_main_content
        self.request_delay = request_delay
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from firecrawl import AsyncFirecrawl

            self._client = AsyncFirecrawl(api_key=self.api_key)
        return self._client

    async def crawl(
        self,
        url: str,
        *,
        exclude_paths: list[str] | None = None,
        max_depth: int = 2,
        max_pages: int = 50,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        json_schema: dict[str, Any] | None = None,
        prompt: str | None = None,
    ) -> CrawlResult:
        client = self._get_client()

        scrape_formats = self._build_formats(output_format, json_schema, prompt)
        scrape_options = {
            "formats": scrape_formats,
            "only_main_content": self.only_main_content,
        }

        crawl_params: dict[str, Any] = {
            "limit": max_pages,
            "max_discovery_depth": max_depth,
            "scrape_options": scrape_options,
        }
        if exclude_paths:
            crawl_params["exclude_paths"] = exclude_paths
        if self.request_delay is not None:
            crawl_params["delay"] = self.request_delay
        if prompt:
            crawl_params["prompt"] = prompt

        # Use the blocking crawl() which auto-paginates and aggregates all docs
        result = await client.crawl(
            url=url,
            poll_interval=self.poll_interval,
            timeout=self.timeout,
            **crawl_params,
        )

        data = getattr(result, "data", None) or []
        pages = self._parse_pages(data, output_format)

        if not pages:
            logger.warning(
                "Firecrawl crawl returned no pages (status=%s, total=%s, completed=%s)",
                getattr(result, "status", None),
                getattr(result, "total", None),
                getattr(result, "completed", None),
            )

        output = self._aggregate(pages, output_format)
        return CrawlResult(pages=pages, files=[], output=output)

    def _build_formats(
        self,
        output_format: OutputFormat,
        json_schema: dict[str, Any] | None,
        prompt: str | None,
    ) -> list:
        if output_format == OutputFormat.JSON:
            json_fmt: dict[str, Any] = {"type": "json"}
            if json_schema:
                json_fmt["schema"] = json_schema
            if prompt:
                json_fmt["prompt"] = prompt
            return [json_fmt]
        return ["markdown"]

    def _parse_pages(self, data: list, output_format: OutputFormat) -> list[CrawlPage]:
        pages: list[CrawlPage] = []
        for doc in data:
            metadata = getattr(doc, "metadata", None) or {}
            source_url = (
                getattr(metadata, "source_url", None)
                if not isinstance(metadata, dict)
                else metadata.get("sourceURL", "")
            ) or ""
            title = (
                getattr(metadata, "title", None)
                if not isinstance(metadata, dict)
                else metadata.get("title", "")
            ) or ""
            status_code = (
                getattr(metadata, "status_code", None)
                if not isinstance(metadata, dict)
                else metadata.get("statusCode")
            )

            error = None
            if status_code and status_code >= 400:
                error = f"HTTP {status_code}"

            content = ""
            json_data = None
            if output_format == OutputFormat.JSON:
                json_data = getattr(doc, "json", None) or getattr(
                    doc, "json_data", None
                )
                if json_data:
                    import json

                    content = json.dumps(json_data, indent=2, default=str)
            else:
                content = getattr(doc, "markdown", None) or ""
                if output_format == OutputFormat.TEXT:
                    content = self._strip_markdown(content)

            links = getattr(doc, "links", None) or []

            pages.append(
                CrawlPage(
                    url=source_url,
                    title=title,
                    content=content,
                    depth=0,
                    json_data=json_data,
                    links=links,
                    error=error,
                )
            )
        return pages

    @staticmethod
    def _strip_markdown(text: str) -> str:
        import re

        text = re.sub(r"#{1,6}\s*", "", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"^[-*]\s", "", text, flags=re.MULTILINE)
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        return text.strip()

    @staticmethod
    def _aggregate(pages: list[CrawlPage], output_format: OutputFormat) -> str:
        sections: list[str] = []
        for p in pages:
            if p.failed:
                sections.append(f"{p.url}\nError: {p.error}\n")
            elif output_format == OutputFormat.MARKDOWN:
                header = f"# {p.title}" if p.title else f"# {p.url}"
                sections.append(f"{header}\n\nSource: {p.url}\n\n{p.content}\n")
            elif output_format == OutputFormat.JSON:
                sections.append(f"Source: {p.url}\n{p.content}\n")
            else:
                header = p.title or p.url
                sections.append(f"{header}\n{p.url}\n\n{p.content}\n")

        sep = "\n---\n\n" if output_format == OutputFormat.MARKDOWN else "\n\n"
        return sep.join(sections)

    async def close(self) -> None:
        self._client = None
