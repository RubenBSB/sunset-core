# CrawlService

Website crawler with pluggable backends. Ships with two implementations:

- **FirecrawlService** — Uses the [Firecrawl](https://firecrawl.dev) API. Handles JS rendering, rate limiting, and structured extraction server-side.
- **PlaywrightCrawlService** — Local BFS crawler using Playwright. Renders JS pages, converts to markdown, and extracts text from linked files (PDF, etc.).

Both inherit from the abstract `CrawlService` base class.

## Setup

### FirecrawlService

```
firecrawl-py
```

| Env Var | Description |
|---------|-------------|
| `FIRECRAWL_API_KEY` | Firecrawl API key (or pass directly to constructor) |

### PlaywrightCrawlService

```
playwright
markdownify
beautifulsoup4
pymupdf
```

After installing, run:

```bash
playwright install chromium
```

## Usage

### FirecrawlService

```python
from sunset.services.crawl import FirecrawlService, OutputFormat

crawl = FirecrawlService()

# Markdown output (default)
result = await crawl.crawl("https://example.com", max_depth=2, max_pages=100)
print(result.output)

# Plain text
result = await crawl.crawl(
    "https://example.com",
    output_format=OutputFormat.TEXT,
)

# Structured JSON extraction with schema
result = await crawl.crawl(
    "https://example.com/products",
    output_format=OutputFormat.JSON,
    json_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "price": {"type": "number"},
        },
    },
    prompt="Extract the product title and price",
)
for page in result.pages:
    print(page.json_data)

# Exclude paths and guide the crawl with a prompt
result = await crawl.crawl(
    "https://docs.example.com",
    exclude_paths=["blog/.*", "changelog/.*"],
    prompt="Focus on API reference pages",
    max_depth=3,
)
```

### PlaywrightCrawlService

```python
from sunset.services.crawl import PlaywrightCrawlService, OutputFormat

crawl = PlaywrightCrawlService(request_delay=0.5)

result = await crawl.crawl("https://example.com", max_depth=2, max_pages=50)
print(result.output)

# Iterate pages and downloaded files
for page in result.pages:
    if not page.failed:
        print(f"{page.title} ({len(page.links)} links)")

for file in result.files:
    print(f"{file.filename}: {len(file.content)} chars")

await crawl.close()
```

## API Reference

### Base `crawl()` signature

```python
async def crawl(
    url: str,
    *,
    exclude_paths: list[str] | None = None,   # Regex patterns to exclude
    max_depth: int = 2,
    max_pages: int = 50,
    output_format: OutputFormat = OutputFormat.MARKDOWN,  # MARKDOWN, TEXT, or JSON
    json_schema: dict | None = None,           # JSON Schema for structured extraction
    prompt: str | None = None,                 # Guide the crawl / extraction
) -> CrawlResult
```

### OutputFormat

| Value | Description |
|-------|-------------|
| `OutputFormat.MARKDOWN` | Markdown output (default) |
| `OutputFormat.TEXT` | Plain text |
| `OutputFormat.JSON` | Structured JSON extraction (requires `json_schema` and/or `prompt`) |

### FirecrawlService constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | API key (falls back to `FIRECRAWL_API_KEY` env var) |
| `only_main_content` | `bool` | `True` | Strip headers/navs/footers |
| `request_delay` | `float \| None` | `None` | Seconds between scrapes |
| `poll_interval` | `int` | `2` | Seconds between status checks |
| `timeout` | `int` | `120` | Max seconds to wait for completion |

### PlaywrightCrawlService constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allowed_domains` | `list[str] \| None` | `None` | Restrict crawling to these domains. `None` auto-detects |
| `request_delay` | `float` | `0.5` | Seconds between page loads |
| `headless` | `bool` | `True` | Run browser in headless mode |
| `timeout` | `int` | `30000` | Page navigation timeout in milliseconds |

### Data Classes

- **`CrawlPage`** — `url`, `title`, `content`, `depth`, `json_data`, `links`, `error`, `failed`
- **`CrawlFile`** — `url`, `filename`, `content`, `mime_type`, `source_page`, `error`, `failed`
- **`CrawlResult`** — `pages: list[CrawlPage]`, `files: list[CrawlFile]`, `output: str`
