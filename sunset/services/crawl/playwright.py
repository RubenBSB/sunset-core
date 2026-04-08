"""Website crawling service using Playwright for JS-rendered pages.

BFS-crawls from seed URLs with optional sitemap discovery, converts
pages to markdown, and extracts text from linked files (PDF, DOCX, etc.).
"""

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import deque
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

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

_SKIP_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".mp4",
    ".mp3",
    ".zip",
    ".rar",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".css",
    ".js",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_NOISE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^Brochure$",
        r"^Lire la suite$",
        r"^Voir le projet$",
        r"^En savoir plus$",
        r"^En savoir \+$",
        r"^Decouvrir$",
        r"^Retour$",
        r"^Partager$",
        r"^Suivant$",
        r"^Precedent$",
        r"^Fermer$",
        r"^Menu$",
        r"^Rechercher$",
        r"^Accueil$",
        r"^S'inscrire$",
        r"^Candidater$",
        r"^Nous Rencontrer$",
        r"^Telecharger$",
        r"^Je m'inscris$",
        r"^Voir egalement$",
    ]
]


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = "https"
    netloc = parsed.netloc.lower()
    if not netloc.startswith("www."):
        netloc = f"www.{netloc}"
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def _is_file_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _FILE_EXTENSIONS)


def _should_skip(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _SKIP_EXTENSIONS)


def _extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _is_same_domain(url: str, domains: set[str]) -> bool:
    return _extract_domain(url) in domains


def _clean_noise(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if any(rx.match(stripped) for rx in _NOISE_PATTERNS):
            continue
        if stripped.startswith("- "):
            bullet_text = stripped[2:].strip()
            if any(rx.match(bullet_text) for rx in _NOISE_PATTERNS):
                continue
        cleaned.append(line)
    # Collapse runs of 3+ blank lines
    result = []
    blank_count = 0
    for line in cleaned:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return "\n".join(result)


async def _discover_sitemap_urls(
    base_url: str, domains: set[str], timeout: int = 15
) -> set[str]:
    urls: set[str] = set()
    sitemap_locations = [
        f"{base_url}/sitemap.xml",
        f"{base_url}/sitemap_index.xml",
    ]

    async with httpx.AsyncClient(
        headers=_DEFAULT_HEADERS, timeout=timeout, follow_redirects=True
    ) as client:
        # Check robots.txt for sitemap references
        try:
            resp = await client.get(f"{base_url}/robots.txt")
            if resp.is_success:
                for line in resp.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sitemap_locations.append(line.split(":", 1)[1].strip())
        except Exception:
            pass

        visited_sitemaps: set[str] = set()

        async def parse_sitemap(sitemap_url: str) -> None:
            if sitemap_url in visited_sitemaps:
                return
            visited_sitemaps.add(sitemap_url)
            try:
                resp = await client.get(sitemap_url)
                if not resp.is_success:
                    return
                root = ET.fromstring(resp.content)
                ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                for child_sitemap in root.findall(".//sm:sitemap/sm:loc", ns):
                    await parse_sitemap(child_sitemap.text.strip())
                for loc in root.findall(".//sm:url/sm:loc", ns):
                    url = loc.text.strip()
                    if _is_same_domain(url, domains):
                        urls.add(_normalize_url(url))
            except Exception:
                pass

        for loc in sitemap_locations:
            await parse_sitemap(loc)

    logger.info("Sitemap discovery found %d URLs", len(urls))
    return urls


class PlaywrightCrawlService(CrawlService):
    """Async website crawler with Playwright and BFS traversal.

    Args:
        allowed_domains: Restrict crawling to these domains.
            ``None`` auto-detects from the seed URL.
        request_delay: Seconds to wait between page loads.
        headless: Run Playwright browser in headless mode.
        timeout: Page navigation timeout in milliseconds.
        discover_sitemap: Parse robots.txt / sitemap.xml to seed the
            BFS queue before crawling.
        noise_patterns: Extra regex patterns to strip from output.
            Built-in CTA/nav patterns are always applied.
    """

    def __init__(
        self,
        *,
        allowed_domains: list[str] | None = None,
        request_delay: float = 0.5,
        headless: bool = True,
        timeout: int = 30_000,
        discover_sitemap: bool = True,
        noise_patterns: list[str] | None = None,
    ):
        self.allowed_domains = allowed_domains
        self.request_delay = request_delay
        self.headless = headless
        self.timeout = timeout
        self.discover_sitemap = discover_sitemap
        self._extra_noise = (
            [re.compile(p, re.IGNORECASE) for p in noise_patterns]
            if noise_patterns
            else []
        )
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

        # Seed queue from sitemap discovery
        if self.discover_sitemap:
            sitemap_urls = await _discover_sitemap_urls(url, domains)
            for surl in sitemap_urls:
                if surl not in visited and not _should_skip(surl):
                    if not self._is_excluded(surl, exclude_patterns):
                        visited.add(surl)
                        queue.append((surl, 0))

        context = await self._browser.new_context()
        page = await context.new_page()

        try:
            while queue and len(pages) < max_pages:
                current_url, depth = queue.popleft()

                if self._is_excluded(current_url, exclude_patterns):
                    continue

                crawl_page = await self._visit_page(
                    page, current_url, depth, output_format, markdownify, BeautifulSoup
                )
                pages.append(crawl_page)

                if not crawl_page.failed:
                    for link in crawl_page.links:
                        if not _is_same_domain(link, domains):
                            continue
                        if self._is_excluded(link, exclude_patterns):
                            continue

                        normalized_link = _normalize_url(link)

                        if _should_skip(normalized_link):
                            continue

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
        output = _clean_noise(output)
        if self._extra_noise:
            lines = output.split("\n")
            output = "\n".join(
                line
                for line in lines
                if not any(rx.match(line.strip()) for rx in self._extra_noise)
            )
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
            for selector in [
                "[class*='cookie']",
                "[class*='popup']",
                "[class*='modal']",
                "[class*='sidebar']",
                "[class*='widget']",
                "[class*='banner']",
                "[id*='cookie']",
                "[id*='popup']",
                "[id*='modal']",
                "[class*='menu']",
                "[class*='breadcrumb']",
                "[class*='share']",
                "[class*='social']",
            ]:
                for el in soup.select(selector):
                    el.decompose()

            main = soup.select_one("main, [role='main'], article, .content")
            if not main:
                main = soup.find("div", class_=re.compile(r"content|main|page", re.I))
            if not main:
                main = soup.find("div", id=re.compile(r"content|main|page", re.I))
            content_root = main or soup.find("body") or soup
            content_html = str(content_root)

            if output_format == OutputFormat.MARKDOWN:
                content = markdownify(content_html, strip=["img"]).strip()
            else:
                content = content_root.get_text(separator="\n", strip=True)

            # Deduplicate consecutive identical lines
            deduped = []
            for line in content.split("\n"):
                if not deduped or line.strip() != deduped[-1].strip():
                    deduped.append(line)
            content = "\n".join(deduped)

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
