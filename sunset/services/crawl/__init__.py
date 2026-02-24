"""Website crawling service using Playwright for JS-rendered pages.

BFS-crawls from seed URLs, converts pages to markdown, and extracts
text from linked files (PDF, DOCX, etc.).
"""

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, urlunparse

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
    """Strip fragments and trailing slashes for deduplication."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, "")
    )


def _is_file_url(url: str) -> bool:
    """Check if a URL points to a downloadable file based on extension."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _FILE_EXTENSIONS)


def _extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


@dataclass
class CrawlPage:
    url: str
    title: str
    content: str
    depth: int
    links: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class CrawlFile:
    url: str
    filename: str
    content: str
    mime_type: str
    source_page: str
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class CrawlResult:
    pages: list[CrawlPage]
    files: list[CrawlFile]
    markdown: str


class CrawlService:
    """Async website crawler with Playwright and BFS traversal.

    Args:
        max_depth: Maximum link-follow depth from seed URLs.
        max_pages: Safety cap on total pages to visit.
        output_format: ``"md"`` for markdown, ``"text"`` for plain text.
        allowed_domains: Restrict crawling to these domains.
            ``None`` auto-detects from seed URLs.
        request_delay: Seconds to wait between page loads.
        headless: Run Playwright browser in headless mode.
        timeout: Page navigation timeout in milliseconds.
    """

    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 50,
        output_format: str = "md",
        allowed_domains: list[str] | None = None,
        request_delay: float = 0.5,
        headless: bool = True,
        timeout: int = 30_000,
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.output_format = output_format
        self.allowed_domains = allowed_domains
        self.request_delay = request_delay
        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser = None

    async def _ensure_browser(self) -> None:
        """Lazy-launch the Playwright browser on first use."""
        if self._browser is not None:
            return

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        logger.info("CrawlService: browser launched")

    async def close(self) -> None:
        """Close the Playwright browser and cleanup."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
            logger.info("CrawlService: browser closed")

    async def crawl(self, urls: list[str]) -> CrawlResult:
        """BFS-crawl starting from seed URLs.

        Args:
            urls: Seed URLs to start crawling from.

        Returns:
            CrawlResult with pages, files, and aggregated markdown.
        """
        from bs4 import BeautifulSoup
        from markdownify import markdownify

        await self._ensure_browser()

        # Resolve allowed domains
        domains = set(self.allowed_domains) if self.allowed_domains else set()
        if not domains:
            domains = {_extract_domain(u) for u in urls}

        visited: set[str] = set()
        pages: list[CrawlPage] = []
        files: list[CrawlFile] = []
        file_urls_seen: set[str] = set()

        # BFS queue: (url, depth)
        queue: deque[tuple[str, int]] = deque()
        for url in urls:
            normalized = _normalize_url(url)
            if normalized not in visited:
                queue.append((normalized, 0))
                visited.add(normalized)

        context = await self._browser.new_context()
        page = await context.new_page()

        try:
            while queue and len(pages) < self.max_pages:
                url, depth = queue.popleft()

                crawl_page = await self._visit_page(
                    page, url, depth, markdownify, BeautifulSoup
                )
                pages.append(crawl_page)

                if not crawl_page.failed:
                    for link in crawl_page.links:
                        link_domain = _extract_domain(link)
                        if link_domain not in domains:
                            continue

                        normalized_link = _normalize_url(link)

                        if _is_file_url(normalized_link):
                            if normalized_link not in file_urls_seen:
                                file_urls_seen.add(normalized_link)
                                crawl_file = await self._download_file(
                                    page, normalized_link, url
                                )
                                files.append(crawl_file)
                        elif depth < self.max_depth and normalized_link not in visited:
                            visited.add(normalized_link)
                            queue.append((normalized_link, depth + 1))

                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
        finally:
            await page.close()
            await context.close()

        markdown = self._aggregate(pages, files)
        return CrawlResult(pages=pages, files=files, markdown=markdown)

    async def _visit_page(
        self,
        page,
        url: str,
        depth: int,
        markdownify,
        BeautifulSoup,
    ) -> CrawlPage:
        """Navigate to a URL, extract content and links."""
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

            # Extract links BEFORE removing nav/footer/header
            seen_links: set[str] = set()
            links: list[str] = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                absolute = urljoin(url, href)
                parsed = urlparse(absolute)
                if parsed.scheme in ("http", "https") and absolute not in seen_links:
                    seen_links.add(absolute)
                    links.append(absolute)

            # Remove non-content elements for clean content extraction
            for tag in soup.select("script, style, nav, footer, header, noscript"):
                tag.decompose()

            # Prefer main content area, fall back to body
            main = soup.select_one("main, [role='main'], article, .content")
            content_root = main or soup.find("body") or soup
            content_html = str(content_root)

            if self.output_format == "md":
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
        """Download a file and extract its text content."""
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
        """Extract text from file bytes. Supports PDF via pymupdf."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf" or "pdf" in content_type:
            return await self._extract_pdf_text(data)

        if ext in (".txt", ".csv", ".md"):
            return data.decode("utf-8", errors="replace")

        return f"[Binary file: {filename}]"

    async def _extract_pdf_text(self, data: bytes) -> str:
        """Extract text from PDF bytes using pymupdf."""
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

    def _aggregate(self, pages: list[CrawlPage], files: list[CrawlFile]) -> str:
        """Combine all crawled pages and files into a single output string."""
        sections: list[str] = []

        for p in pages:
            if p.failed:
                if self.output_format == "md":
                    sections.append(f"# {p.url}\n\n> **Error:** {p.error}\n")
                else:
                    sections.append(f"{p.url}\nError: {p.error}\n")
            else:
                if self.output_format == "md":
                    header = f"# {p.title}" if p.title else f"# {p.url}"
                    sections.append(f"{header}\n\nSource: {p.url}\n\n{p.content}\n")
                else:
                    header = p.title or p.url
                    sections.append(f"{header}\n{p.url}\n\n{p.content}\n")

        for f in files:
            if f.failed:
                if self.output_format == "md":
                    sections.append(
                        f"## File: {f.filename}\n\n> **Error:** {f.error}\n"
                    )
                else:
                    sections.append(f"File: {f.filename}\nError: {f.error}\n")
            else:
                if self.output_format == "md":
                    sections.append(
                        f"## File: {f.filename}\n\nSource: {f.url}\n\n{f.content}\n"
                    )
                else:
                    sections.append(f"File: {f.filename}\n{f.url}\n\n{f.content}\n")

        separator = "\n---\n\n" if self.output_format == "md" else "\n\n"
        return separator.join(sections)
