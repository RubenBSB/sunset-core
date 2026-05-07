# SEOService

Blog generation, translation, SEO metadata extraction, and sitemap building. Uses `CrawlService` for web research and `LLMService` for content generation.

## Setup

Requires a configured `LLMService` and `CrawlService` — no additional dependencies.

## Usage

### Initialize

```python
from sunset.services.seo import SEOService
from sunset.services.crawl import FirecrawlService
from sunset.services.llm import OpenAIService  # or any LLMService

llm = OpenAIService()
crawl = FirecrawlService()
seo = SEOService(llm=llm, crawl=crawl, model="gpt-4o")
```

### Generate a blog post

Researches the topic on the web, writes an SEO-optimized article, and generates metadata automatically.

```python
post = await seo.generate_blog_post(
    "AI-powered customer support trends in 2025",
    language="en",
    max_sources=5,
)

print(post.metadata.title)        # SEO-optimized title
print(post.metadata.description)  # Meta description
print(post.metadata.keywords)     # ["ai", "customer support", ...]
print(post.metadata.slug)         # "ai-powered-customer-support-trends-2025"
print(post.content)               # Markdown article body
print(post.sources)               # ["https://...", ...]
```

### Translate content

Preserves markdown formatting and links.

```python
french = await seo.translate(post.content, source_lang="en", target_lang="fr")
```

### Generate metadata from existing content

```python
metadata = await seo.generate_metadata(
    my_article_content,
    language="en",
    base_url="https://myapp.com",  # enables JSON-LD generation
)

print(metadata.json_ld)
# {
#   "@context": "https://schema.org",
#   "@type": "BlogPosting",
#   "headline": "...",
#   "url": "https://myapp.com/blog/my-slug",
#   ...
# }
```

### Generate a sitemap

```python
from sunset.services.seo import SitemapEntry

entries = [
    SitemapEntry(loc="https://myapp.com/", changefreq="daily", priority=1.0),
    SitemapEntry(loc="https://myapp.com/blog/my-post", lastmod="2025-01-15"),
]
xml = seo.generate_sitemap(entries)
# Returns valid sitemap.xml string
```

### Cron job example (FastAPI)

```python
from fastapi import APIRouter

router = APIRouter()

@router.post("/api/cron/blog/generate")
async def generate_blog():
    post = await seo.generate_blog_post("latest trends in [your industry]")
    # Save to DB, notify for review, etc.
    await db.execute(
        "INSERT INTO blog_posts (title, slug, content, language, status) VALUES ($1, $2, $3, $4, $5)",
        post.metadata.title, post.metadata.slug, post.content, post.language, "draft",
    )
    return {"slug": post.metadata.slug}
```

## API Reference

### SEOService constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm` | `LLMService` | LLM service for content generation |
| `crawl` | `CrawlService` | Crawl service for web research |
| `model` | `str` | Model identifier (e.g. `"gemini-2.5-flash"`) |

### `generate_blog_post()`

```python
async def generate_blog_post(
    topic: str,
    *,
    language: str = "en",
    max_sources: int = 5,
) -> BlogPost
```

### `translate()`

```python
async def translate(
    content: str,
    source_lang: str,
    target_lang: str,
) -> str
```

### `generate_metadata()`

```python
async def generate_metadata(
    content: str,
    *,
    language: str = "en",
    base_url: str | None = None,
) -> SEOMetadata
```

### `generate_sitemap()`

```python
def generate_sitemap(entries: list[SitemapEntry]) -> str
```

### Data Classes

- **`BlogPost`** — `content`, `metadata: SEOMetadata`, `language`, `sources: list[str]`, `created_at: datetime`
- **`SEOMetadata`** — `title`, `description`, `keywords: list[str]`, `slug`, `json_ld: dict | None`
- **`SitemapEntry`** — `loc`, `lastmod`, `changefreq`, `priority`
