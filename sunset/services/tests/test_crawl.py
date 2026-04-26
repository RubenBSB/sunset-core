"""Tests for PlaywrightCrawlService — URL helpers, noise cleanup,
sitemap discovery, and crawl orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sunset.services.crawl.base import CrawlFile, CrawlPage, OutputFormat
from sunset.services.crawl.playwright import (
    PlaywrightCrawlService,
    _clean_noise,
    _extract_domain,
    _is_file_url,
    _is_same_domain,
    _normalize_url,
    _should_skip,
)

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def test_forces_https(self):
        assert (
            _normalize_url("http://example.com/page") == "https://www.example.com/page"
        )

    def test_adds_www(self):
        assert (
            _normalize_url("https://example.com/page") == "https://www.example.com/page"
        )

    def test_preserves_www(self):
        assert (
            _normalize_url("https://www.example.com/page")
            == "https://www.example.com/page"
        )

    def test_strips_trailing_slash(self):
        assert (
            _normalize_url("https://www.example.com/page/")
            == "https://www.example.com/page"
        )

    def test_root_path_preserved(self):
        assert _normalize_url("https://example.com") == "https://www.example.com/"
        assert _normalize_url("https://example.com/") == "https://www.example.com/"

    def test_strips_fragment(self):
        assert (
            _normalize_url("https://example.com/page#section")
            == "https://www.example.com/page"
        )

    def test_preserves_query(self):
        assert (
            _normalize_url("https://example.com/page?q=1")
            == "https://www.example.com/page?q=1"
        )

    def test_lowercases_netloc(self):
        assert (
            _normalize_url("https://EXAMPLE.COM/Page") == "https://www.example.com/Page"
        )


class TestExtractDomain:
    def test_strips_www(self):
        assert _extract_domain("https://www.example.com/page") == "example.com"

    def test_no_www(self):
        assert _extract_domain("https://example.com/page") == "example.com"


class TestIsSameDomain:
    def test_matching(self):
        domains = {"example.com"}
        assert _is_same_domain("https://www.example.com/page", domains)
        assert _is_same_domain("https://example.com/page", domains)

    def test_not_matching(self):
        domains = {"example.com"}
        assert not _is_same_domain("https://other.com/page", domains)


class TestIsFileUrl:
    def test_pdf(self):
        assert _is_file_url("https://example.com/doc.pdf")

    def test_docx(self):
        assert _is_file_url("https://example.com/report.docx")

    def test_html_page(self):
        assert not _is_file_url("https://example.com/about")


class TestShouldSkip:
    def test_image(self):
        assert _should_skip("https://example.com/logo.png")

    def test_font(self):
        assert _should_skip("https://example.com/font.woff2")

    def test_css(self):
        assert _should_skip("https://example.com/style.css")

    def test_html_page(self):
        assert not _should_skip("https://example.com/about")


# ---------------------------------------------------------------------------
# Noise cleanup
# ---------------------------------------------------------------------------


class TestCleanNoise:
    def test_strips_cta_lines(self):
        text = "Some content\nEn savoir plus\nMore content"
        result = _clean_noise(text)
        assert "En savoir plus" not in result
        assert "Some content" in result
        assert "More content" in result

    def test_strips_bullet_cta(self):
        text = "Items:\n- Candidater\n- Real item"
        result = _clean_noise(text)
        assert "Candidater" not in result
        assert "Real item" in result

    def test_collapses_blank_lines(self):
        text = "A\n\n\n\n\nB"
        result = _clean_noise(text)
        # Allows up to 2 blank lines (text + 2 blanks = 3 newlines max between content)
        assert "\n\n\n\n" not in result
        assert "A" in result
        assert "B" in result

    def test_case_insensitive(self):
        text = "Content\nMENU\nMore"
        result = _clean_noise(text)
        assert "MENU" not in result

    def test_keeps_normal_content(self):
        text = "# Title\n\nParagraph about the school.\n\nAnother paragraph."
        assert _clean_noise(text) == text


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_sitemap_urls():
    """Sitemap discovery parses robots.txt and sitemap XML."""
    from sunset.services.crawl.playwright import _discover_sitemap_urls

    robots_txt = "User-agent: *\nSitemap: https://www.example.com/sitemap.xml\n"
    sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://www.example.com/page1</loc></url>
        <url><loc>https://www.example.com/page2</loc></url>
        <url><loc>https://other.com/page3</loc></url>
    </urlset>"""

    class FakeResponse:
        def __init__(self, text, success=True):
            self._text = text
            self.is_success = success
            self.content = text.encode()

        @property
        def text(self):
            return self._text

    responses = {
        "https://www.example.com/robots.txt": FakeResponse(robots_txt),
        "https://www.example.com/sitemap.xml": FakeResponse(sitemap_xml),
        "https://www.example.com/sitemap_index.xml": FakeResponse("", success=False),
    }

    class FakeClient:
        async def get(self, url):
            return responses.get(url, FakeResponse("", success=False))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    with patch(
        "sunset.services.crawl.playwright.httpx.AsyncClient", return_value=FakeClient()
    ):
        urls = await _discover_sitemap_urls("https://www.example.com", {"example.com"})

    # Should include example.com pages, exclude other.com
    assert len(urls) == 2
    assert all("example.com" in u for u in urls)
    assert not any("other.com" in u for u in urls)


@pytest.mark.asyncio
async def test_discover_sitemap_index():
    """Sitemap index files should be followed recursively."""
    from sunset.services.crawl.playwright import _discover_sitemap_urls

    sitemap_index = """<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap><loc>https://www.example.com/sitemap-pages.xml</loc></sitemap>
    </sitemapindex>"""

    sitemap_pages = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://www.example.com/about</loc></url>
    </urlset>"""

    class FakeResponse:
        def __init__(self, text, success=True):
            self._text = text
            self.is_success = success
            self.content = text.encode()

        @property
        def text(self):
            return self._text

    responses = {
        "https://www.example.com/robots.txt": FakeResponse("", success=False),
        "https://www.example.com/sitemap.xml": FakeResponse(sitemap_index),
        "https://www.example.com/sitemap_index.xml": FakeResponse("", success=False),
        "https://www.example.com/sitemap-pages.xml": FakeResponse(sitemap_pages),
    }

    class FakeClient:
        async def get(self, url):
            return responses.get(url, FakeResponse("", success=False))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    with patch(
        "sunset.services.crawl.playwright.httpx.AsyncClient", return_value=FakeClient()
    ):
        urls = await _discover_sitemap_urls("https://www.example.com", {"example.com"})

    assert any("about" in u for u in urls)


# ---------------------------------------------------------------------------
# PlaywrightCrawlService
# ---------------------------------------------------------------------------


class TestPlaywrightCrawlServiceInit:
    def test_defaults(self):
        svc = PlaywrightCrawlService()
        assert svc.discover_sitemap is True
        assert svc._extra_noise == []
        assert svc.request_delay == 0.5
        assert svc.headless is True

    def test_custom_noise_patterns(self):
        svc = PlaywrightCrawlService(noise_patterns=[r"^Subscribe$", r"^Download$"])
        assert len(svc._extra_noise) == 2
        assert svc._extra_noise[0].match("Subscribe")
        assert not svc._extra_noise[0].match("subscribe to newsletter")

    def test_disable_sitemap(self):
        svc = PlaywrightCrawlService(discover_sitemap=False)
        assert svc.discover_sitemap is False


class TestVisitPage:
    """Test _visit_page content extraction with mocked Playwright."""

    @pytest.mark.asyncio
    async def test_extracts_markdown(self):
        svc = PlaywrightCrawlService()

        html = """<html><head><title>Test Page</title></head>
        <body><main><h1>Hello</h1><p>World</p></main></body></html>"""

        mock_page = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_page.goto.return_value = mock_response
        mock_page.title.return_value = "Test Page"
        mock_page.content.return_value = html

        from bs4 import BeautifulSoup
        from markdownify import markdownify

        result = await svc._visit_page(
            mock_page,
            "https://www.example.com/test",
            0,
            OutputFormat.MARKDOWN,
            markdownify,
            BeautifulSoup,
        )

        assert result.title == "Test Page"
        assert result.url == "https://www.example.com/test"
        assert result.depth == 0
        assert not result.failed
        assert "Hello" in result.content
        assert "World" in result.content

    @pytest.mark.asyncio
    async def test_strips_noise_elements(self):
        svc = PlaywrightCrawlService()

        html = """<html><head><title>Test</title></head><body>
        <nav>Navigation</nav>
        <div class="cookie-banner">Accept cookies</div>
        <main><p>Real content</p></main>
        <footer>Footer</footer></body></html>"""

        mock_page = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_page.goto.return_value = mock_response
        mock_page.title.return_value = "Test"
        mock_page.content.return_value = html

        from bs4 import BeautifulSoup
        from markdownify import markdownify

        result = await svc._visit_page(
            mock_page,
            "https://www.example.com/test",
            0,
            OutputFormat.MARKDOWN,
            markdownify,
            BeautifulSoup,
        )

        assert "Real content" in result.content
        assert "Navigation" not in result.content
        assert "Accept cookies" not in result.content
        assert "Footer" not in result.content

    @pytest.mark.asyncio
    async def test_http_error(self):
        svc = PlaywrightCrawlService()

        mock_page = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 404
        mock_page.goto.return_value = mock_response

        from bs4 import BeautifulSoup
        from markdownify import markdownify

        result = await svc._visit_page(
            mock_page,
            "https://www.example.com/missing",
            1,
            OutputFormat.MARKDOWN,
            markdownify,
            BeautifulSoup,
        )

        assert result.failed
        assert "404" in result.error

    @pytest.mark.asyncio
    async def test_deduplicates_consecutive_lines(self):
        svc = PlaywrightCrawlService()

        html = """<html><head><title>Test</title></head><body>
        <main><p>Line one</p><p>Line one</p><p>Line two</p></main></body></html>"""

        mock_page = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_page.goto.return_value = mock_response
        mock_page.title.return_value = "Test"
        mock_page.content.return_value = html

        from bs4 import BeautifulSoup
        from markdownify import markdownify

        result = await svc._visit_page(
            mock_page,
            "https://www.example.com/test",
            0,
            OutputFormat.TEXT,
            markdownify,
            BeautifulSoup,
        )

        lines = [ln for ln in result.content.split("\n") if ln.strip() == "Line one"]
        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_extracts_links(self):
        svc = PlaywrightCrawlService()

        html = """<html><head><title>Links</title></head><body>
        <main>
            <a href="/page2">Page 2</a>
            <a href="https://example.com/page3">Page 3</a>
            <a href="mailto:test@test.com">Email</a>
        </main></body></html>"""

        mock_page = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_page.goto.return_value = mock_response
        mock_page.title.return_value = "Links"
        mock_page.content.return_value = html

        from bs4 import BeautifulSoup
        from markdownify import markdownify

        result = await svc._visit_page(
            mock_page,
            "https://www.example.com/",
            0,
            OutputFormat.MARKDOWN,
            markdownify,
            BeautifulSoup,
        )

        # Should resolve relative links and include http(s) links
        assert any("page2" in link for link in result.links)
        assert any("page3" in link for link in result.links)
        # mailto should not be included
        assert not any("mailto" in link for link in result.links)


class TestAggregate:
    def test_markdown_output(self):
        pages = [
            CrawlPage(
                url="https://a.com", title="Page A", content="Content A", depth=0
            ),
            CrawlPage(
                url="https://b.com", title="Page B", content="Content B", depth=1
            ),
        ]
        output = PlaywrightCrawlService._aggregate(pages, [], OutputFormat.MARKDOWN)
        assert "# Page A" in output
        assert "# Page B" in output
        assert "---" in output

    def test_text_output(self):
        pages = [
            CrawlPage(
                url="https://a.com", title="Page A", content="Content A", depth=0
            ),
        ]
        output = PlaywrightCrawlService._aggregate(pages, [], OutputFormat.TEXT)
        assert "Page A" in output
        assert "Content A" in output
        assert "---" not in output

    def test_includes_files(self):
        pages = [
            CrawlPage(url="https://a.com", title="Page A", content="Content", depth=0),
        ]
        files = [
            CrawlFile(
                url="https://a.com/doc.pdf",
                filename="doc.pdf",
                content="PDF text",
                mime_type="application/pdf",
                source_page="https://a.com",
            ),
        ]
        output = PlaywrightCrawlService._aggregate(pages, files, OutputFormat.MARKDOWN)
        assert "doc.pdf" in output
        assert "PDF text" in output

    def test_error_pages(self):
        pages = [
            CrawlPage(
                url="https://a.com", title="", content="", depth=0, error="HTTP 500"
            ),
        ]
        output = PlaywrightCrawlService._aggregate(pages, [], OutputFormat.MARKDOWN)
        assert "Error" in output
        assert "500" in output


class TestCrawlExcludeAndSkip:
    def test_compile_excludes(self):
        patterns = PlaywrightCrawlService._compile_excludes([r"/blog/.*", r"/admin"])
        assert len(patterns) == 2

    def test_compile_excludes_none(self):
        assert PlaywrightCrawlService._compile_excludes(None) == []

    def test_is_excluded(self):
        patterns = PlaywrightCrawlService._compile_excludes([r"/blog/.*"])
        assert PlaywrightCrawlService._is_excluded(
            "https://example.com/blog/post-1", patterns
        )
        assert not PlaywrightCrawlService._is_excluded(
            "https://example.com/about", patterns
        )


class TestCrawlOrdering:
    """BFS ordering: seed-page links should be crawled before sitemap URLs."""

    @pytest.mark.asyncio
    async def test_seed_links_crawled_before_sitemap(self):
        # Sitemap floods the queue with many URLs that do NOT include /agenda.
        # /agenda is only discoverable via a link on the homepage.
        # With max_pages smaller than the sitemap size, /agenda must still
        # be reached because seed-linked pages are queued before the sitemap.
        sitemap_urls = {f"https://www.example.com/sitemap-page-{i}" for i in range(50)}
        seed_url = "https://www.example.com/"
        agenda_url = "https://www.example.com/agenda"

        async def fake_visit(
            self, page, url, depth, output_format, markdownify, BeautifulSoup
        ):
            links = [agenda_url] if url == seed_url else []
            return CrawlPage(url=url, title=url, content="x", depth=depth, links=links)

        svc = PlaywrightCrawlService(request_delay=0)
        svc._browser = MagicMock()
        svc._browser.new_context = AsyncMock(
            return_value=MagicMock(
                new_page=AsyncMock(return_value=MagicMock(close=AsyncMock())),
                close=AsyncMock(),
            )
        )

        with (
            patch.object(PlaywrightCrawlService, "_ensure_browser", AsyncMock()),
            patch.object(PlaywrightCrawlService, "_visit_page", fake_visit),
            patch(
                "sunset.services.crawl.playwright._discover_sitemap_urls",
                AsyncMock(return_value=sitemap_urls),
            ),
        ):
            result = await svc.crawl(seed_url, max_pages=10)

        visited_urls = [p.url for p in result.pages]
        assert visited_urls[0] == seed_url, "seed must be processed first"
        assert agenda_url in visited_urls, (
            "seed-linked page must be crawled within max_pages even when the "
            "sitemap floods the queue"
        )
        # /agenda should land before any sitemap URL
        agenda_idx = visited_urls.index(agenda_url)
        first_sitemap_idx = next(
            (i for i, u in enumerate(visited_urls) if "sitemap-page-" in u),
            len(visited_urls),
        )
        assert agenda_idx < first_sitemap_idx


class TestExtraNoiseCleanup:
    @pytest.mark.asyncio
    async def test_extra_noise_applied_to_output(self):
        svc = PlaywrightCrawlService(
            discover_sitemap=False,
            noise_patterns=[r"^Custom CTA$"],
        )

        pages = [
            CrawlPage(
                url="https://example.com",
                title="Test",
                content="Good content\nCustom CTA\nMore content",
                depth=0,
            ),
        ]

        output = svc._aggregate(pages, [], OutputFormat.MARKDOWN)
        output = _clean_noise(output)
        # Apply extra noise
        lines = output.split("\n")
        output = "\n".join(
            ln
            for ln in lines
            if not any(rx.match(ln.strip()) for rx in svc._extra_noise)
        )

        assert "Good content" in output
        assert "Custom CTA" not in output
        assert "More content" in output
