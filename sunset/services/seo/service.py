import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from sunset.services.crawl import CrawlService, OutputFormat
from sunset.services.llm import LLMService

logger = logging.getLogger(__name__)

BLOG_SYSTEM_PROMPT = """\
You are an expert SEO blog writer. Write an article that is:
- Well-structured with clear H2/H3 headings
- Informative, factual, and based on the provided sources
- Written in {language}
- Between 800-1500 words
- Optimized for search engines without keyword stuffing
- Engaging and natural to read

Cite sources inline where relevant using markdown links.

Return ONLY the article body in markdown (no title — it will be added separately)."""

METADATA_SYSTEM_PROMPT = """\
You are an SEO specialist. Given the following blog post content, generate metadata.

Return a JSON object with exactly these fields:
- "title": SEO-optimized title (50-60 chars)
- "description": meta description (150-160 chars)
- "keywords": list of 5-8 relevant keywords
- "slug": URL-friendly slug derived from the title (lowercase, hyphens, no special chars)"""

TRANSLATE_SYSTEM_PROMPT = """\
You are a professional translator. Translate the following content from {source} to {target}.

Rules:
- Preserve all markdown formatting exactly
- Preserve all links and URLs unchanged
- Translate naturally, not literally — adapt idioms and phrasing
- Keep technical terms in their commonly used form in {target}

Return ONLY the translated content, nothing else."""


@dataclass
class SEOMetadata:
    title: str
    description: str
    keywords: list[str]
    slug: str
    json_ld: dict[str, Any] | None = None


@dataclass
class BlogPost:
    content: str
    metadata: SEOMetadata
    language: str
    sources: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SitemapEntry:
    loc: str
    lastmod: str | None = None
    changefreq: str | None = None
    priority: float | None = None


class SEOService:
    """SEO utilities: blog generation, translation, metadata extraction, sitemap building.

    Args:
        llm: LLM service instance for content generation.
        crawl: Crawl service instance for web research.
        model: Model identifier for LLM calls (e.g. "gemini-2.5-flash").
    """

    def __init__(self, llm: LLMService, crawl: CrawlService, model: str):
        self.llm = llm
        self.crawl = crawl
        self.model = model

    async def generate_blog_post(
        self,
        topic: str,
        *,
        language: str = "en",
        max_sources: int = 5,
    ) -> BlogPost:
        """Research a topic on the web and generate an SEO-optimized blog post.

        Args:
            topic: The topic or prompt to research and write about.
            language: Language code for the article (e.g. "en", "fr", "es").
            max_sources: Maximum number of web pages to crawl for research.

        Returns:
            BlogPost with content, metadata, sources, and timestamp.
        """
        # 1. Research: crawl the web for sources
        logger.info("Researching topic: %s", topic)
        research = await self._research(topic, max_sources=max_sources)

        # 2. Write the article
        logger.info("Generating article in '%s'", language)
        content = await self._write_article(topic, research, language)

        # 3. Generate metadata
        metadata = await self.generate_metadata(content, language=language)

        sources = [s["url"] for s in research]
        return BlogPost(
            content=content,
            metadata=metadata,
            language=language,
            sources=sources,
        )

    async def translate(
        self,
        content: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate content preserving markdown structure.

        Args:
            content: Markdown content to translate.
            source_lang: Source language code (e.g. "en").
            target_lang: Target language code (e.g. "fr").

        Returns:
            Translated content in markdown.
        """
        if source_lang == target_lang:
            return content

        system = TRANSLATE_SYSTEM_PROMPT.format(source=source_lang, target=target_lang)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        response = await self.llm.generate_response(input=messages, model=self.model)
        return response["text"]

    async def generate_metadata(
        self,
        content: str,
        *,
        language: str = "en",
        base_url: str | None = None,
    ) -> SEOMetadata:
        """Generate SEO metadata (title, description, keywords, slug) from content.

        Args:
            content: The article content in markdown.
            language: Language code for the content.
            base_url: Optional base URL for JSON-LD generation.

        Returns:
            SEOMetadata with title, description, keywords, slug, and optional JSON-LD.
        """
        messages = [
            {"role": "system", "content": METADATA_SYSTEM_PROMPT},
            {"role": "user", "content": content[:3000]},  # cap to avoid huge inputs
        ]
        from pydantic import BaseModel

        class MetadataResponse(BaseModel):
            title: str
            description: str
            keywords: list[str]
            slug: str

        response = await self.llm.generate_response(
            input=messages,
            model=self.model,
            text_format=MetadataResponse,
        )

        import json

        data = json.loads(response["text"])

        json_ld = None
        if base_url:
            json_ld = {
                "@context": "https://schema.org",
                "@type": "BlogPosting",
                "headline": data["title"],
                "description": data["description"],
                "inLanguage": language,
                "url": f"{base_url.rstrip('/')}/blog/{data['slug']}",
                "keywords": ", ".join(data["keywords"]),
            }

        return SEOMetadata(
            title=data["title"],
            description=data["description"],
            keywords=data["keywords"],
            slug=data["slug"],
            json_ld=json_ld,
        )

    def generate_sitemap(self, entries: list[SitemapEntry]) -> str:
        """Generate a sitemap.xml string from a list of URL entries.

        Args:
            entries: List of SitemapEntry with loc, lastmod, changefreq, priority.

        Returns:
            XML string for sitemap.xml.
        """
        urlset = Element("urlset")
        urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

        for entry in entries:
            url_el = SubElement(urlset, "url")
            SubElement(url_el, "loc").text = entry.loc
            if entry.lastmod:
                SubElement(url_el, "lastmod").text = entry.lastmod
            if entry.changefreq:
                SubElement(url_el, "changefreq").text = entry.changefreq
            if entry.priority is not None:
                SubElement(url_el, "priority").text = str(entry.priority)

        return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(
            urlset, encoding="unicode"
        )

    # -- Private helpers --

    async def _research(
        self, topic: str, *, max_sources: int = 5
    ) -> list[dict[str, str]]:
        """Crawl the web for sources on a topic."""
        result = await self.crawl.crawl(
            f"https://www.google.com/search?q={topic.replace(' ', '+')}",
            max_depth=1,
            max_pages=max_sources,
            output_format=OutputFormat.MARKDOWN,
        )
        sources = []
        for page in result.pages:
            if not page.failed and page.content.strip():
                sources.append(
                    {
                        "url": page.url,
                        "title": page.title,
                        "content": page.content[:2000],
                    }
                )
        return sources

    async def _write_article(
        self, topic: str, sources: list[dict[str, str]], language: str
    ) -> str:
        """Generate the article from researched sources."""
        source_text = "\n\n---\n\n".join(
            f"Source: {s['url']}\nTitle: {s['title']}\n\n{s['content']}"
            for s in sources
        )

        system = BLOG_SYSTEM_PROMPT.format(language=language)
        user_msg = f"Topic: {topic}\n\nResearch sources:\n\n{source_text}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        response = await self.llm.generate_response(input=messages, model=self.model)
        return response["text"]
