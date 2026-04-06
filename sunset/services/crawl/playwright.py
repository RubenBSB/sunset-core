"""Website crawling service using Playwright for JS-rendered pages.

BFS-crawls from seed URLs, converts pages to markdown, and extracts
text from linked files (PDF, DOCX, etc.).
"""

import asyncio
import logging
import os
from collections import deque
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

from .base import CrawlFile, CrawlPage, CrawlResult, CrawlService, OutputFormat

logger = logging.getLogger(__name__)

_FILE_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".csv",
    ".txt",
}


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, "")
    )


def _is_file_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _FILE_EXTENSIONS)


def _extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


class PlaywrightCrawlService(CrawlService):
    """Async website crawler with Playwright and BFS traversal.

    Args:
        allowed_domains: Restrict crawling to these domains.
            ``None`` auto-detects from the seed URL.
        request_delay: Seconds to wait between page loads.
        headless: Run Playwright browser in headless mode.
        timeout: Page navigation timeout in milliseconds.
    """

    def __init__(
        self,
        *,
        allowed_domains: list[str] | None = None,
        request_delay: float = 0.5,
        headless: bool = True,
        timeout: int = 30_000,
    ):
        self.allowed_domains = allowed_domains
        self.request_delay = request_delay
        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser = None

    async def _ensure_browser(self) -> None:
        if self._browser is not None:
            return
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        logger.info("PlaywrightCrawlService: browser launched")

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
            logger.info("PlaywrightCrawlService: browser closed")

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
        from bs4 import BeautifulSoup
        from markdownify import markdownify

        await self._ensure_browser()

        domains = set(self.allowed_domains) if self.allowed_domains else set()
        if not domains:
            domains = {_extract_domain(url)}

        exclude_patterns = self._compile_excludes(exclude_paths)

        visited: set[str] = set()
        pages: list[CrawlPage] = []
        files: list[CrawlFile] = []
        file_urls_seen: set[str] = set()

        queue: deque[tuple[str, int]] = deque()
        normalized = _normalize_url(url)
        queue.append((normalized, 0))
        visited.add(normalized)

        context = await self._browser.new_context()
        page = await context.new_page()

        try:
            while queue and len(pages) < max_pages:
                current_url, depth = queue.popleft()

                crawl_page = await self._visit_page(
                    page, current_url, depth, output_format, markdownify, BeautifulSoup
                )
                pages.append(crawl_page)

                if not crawl_page.failed:
                    for link in crawl_page.links:
                        link_domain = _extract_domain(link)
                        if link_domain not in domains:
                            continue
                        if self._is_excluded(link, exclude_patterns):
                            continue

                        normalized_link = _normalize_url(link)

                        if _is_file_url(normalized_link):
                            if normalized_link not in file_urls_seen:
                                file_urls_seen.add(normalized_link)
                                crawl_file = await self._download_file(
                                    page, normalized_link, current_url
                                )
                                files.append(crawl_file)
                        elif depth < max_depth and normalized_link not in visited:
                            visited.add(normalized_link)
                            queue.append((normalized_link, depth + 1))

                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
        finally:
            await page.close()
            await context.close()

        output = self._aggregate(pages, files, output_format)
        return CrawlResult(pages=pages, files=files, output=output)

    @staticmethod
    def _compile_excludes(exclude_paths: list[str] | None) -> list:
        if not exclude_paths:
            return []
        import re

        return [re.compile(p) for p in exclude_paths]

    @staticmethod
    def _is_excluded(url: str, patterns: list) -> bool:
        path = urlparse(url).path
        return any(p.search(path) for p in patterns)

    async def _visit_page(
        self,
        page,
        url: str,
        depth: int,
        output_format: OutputFormat,
        markdownify,
        BeautifulSoup,
    ) -> CrawlPage:
        try:
            response = await page.goto(
                url, wait_until="domcontentloaded", timeout=self.timeout
            )
            if response and response.status >= 400:
                return CrawlPage(
                    url=url,
                    title="",
                    content="",
                    depth=depth,
                    error=f"HTTP {response.status}",
                )

            title = await page.title() or ""
            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            seen_links: set[str] = set()
            links: list[str] = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                absolute = urljoin(url, href)
                parsed = urlparse(absolute)
                if parsed.scheme in ("http", "https") and absolute not in seen_links:
                    seen_links.add(absolute)
                    links.append(absolute)

            for tag in soup.select("script, style, nav, footer, header, noscript"):
                tag.decompose()

            main = soup.select_one("main, [role='main'], article, .content")
            content_root = main or soup.find("body") or soup
            content_html = str(content_root)

            if output_format == OutputFormat.MARKDOWN:
                content = markdownify(content_html, strip=["img"]).strip()
            else:
                content = content_root.get_text(separator="\n", strip=True)

            logger.info(f"Crawled: {url} (depth={depth}, links={len(links)})")
            return CrawlPage(
                url=url, title=title, content=content, depth=depth, links=links
            )

        except Exception as exc:
            logger.warning(f"Failed to crawl {url}: {exc}")
            return CrawlPage(url=url, title="", content="", depth=depth, error=str(exc))

    async def _download_file(self, page, url: str, source_page: str) -> CrawlFile:
        filename = os.path.basename(urlparse(url).path) or "unknown"
        try:
            response = await page.request.get(url)
            content_type = response.headers.get("content-type", "")
            body = await response.body()
            text = await self._extract_file_text(body, filename, content_type)

            logger.info(f"Downloaded file: {filename} ({len(body)} bytes)")
            return CrawlFile(
                url=url,
                filename=filename,
                content=text,
                mime_type=content_type,
                source_page=source_page,
            )
        except Exception as exc:
            logger.warning(f"Failed to download {url}: {exc}")
            return CrawlFile(
                url=url,
                filename=filename,
                content="",
                mime_type="",
                source_page=source_page,
                error=str(exc),
            )

    async def _extract_file_text(
        self, data: bytes, filename: str, content_type: str
    ) -> str:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf" or "pdf" in content_type:
            return await self._extract_pdf_text(data)
        if ext in (".txt", ".csv", ".md"):
            return data.decode("utf-8", errors="replace")
        return f"[Binary file: {filename}]"

    async def _extract_pdf_text(self, data: bytes) -> str:
        try:
            import pymupdf

            doc = pymupdf.open(stream=data, filetype="pdf")
            texts = []
            for pdf_page in doc:
                texts.append(pdf_page.get_text())
            doc.close()
            return "\n\n".join(texts)
        except ImportError:
            logger.warning("pymupdf not installed — skipping PDF text extraction")
            return "[PDF content — install pymupdf to extract text]"
        except Exception as exc:
            logger.warning(f"Failed to extract PDF text: {exc}")
            return f"[PDF extraction failed: {exc}]"

    @staticmethod
    def _aggregate(
        pages: list[CrawlPage], files: list[CrawlFile], output_format: OutputFormat
    ) -> str:
        sections: list[str] = []
        is_md = output_format == OutputFormat.MARKDOWN

        for p in pages:
            if p.failed:
                if is_md:
                    sections.append(f"# {p.url}\n\n> **Error:** {p.error}\n")
                else:
                    sections.append(f"{p.url}\nError: {p.error}\n")
            else:
                if is_md:
                    header = f"# {p.title}" if p.title else f"# {p.url}"
                    sections.append(f"{header}\n\nSource: {p.url}\n\n{p.content}\n")
                else:
                    header = p.title or p.url
                    sections.append(f"{header}\n{p.url}\n\n{p.content}\n")

        for f in files:
            if f.failed:
                if is_md:
                    sections.append(
                        f"## File: {f.filename}\n\n> **Error:** {f.error}\n"
                    )
                else:
                    sections.append(f"File: {f.filename}\nError: {f.error}\n")
            else:
                if is_md:
                    sections.append(
                        f"## File: {f.filename}\n\nSource: {f.url}\n\n{f.content}\n"
                    )
                else:
                    sections.append(f"File: {f.filename}\n{f.url}\n\n{f.content}\n")

        separator = "\n---\n\n" if is_md else "\n\n"
        return separator.join(sections)
