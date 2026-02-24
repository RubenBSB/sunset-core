# CrawlService

BFS website crawler using Playwright (async). Renders JS-heavy pages, converts to markdown, and extracts text from linked files (PDF, etc.).

## Setup

### Extra Dependencies

The child project's `requirements.txt` needs:

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

```python
from sunset.services import CrawlService

crawl = CrawlService(
    max_depth=2,
    max_pages=50,
    output_format="md",
    request_delay=0.5,
)

# Crawl from seed URLs
result = await crawl.crawl(["https://example.com"])

# Aggregated markdown of all pages + files
print(result.markdown)

# Iterate individual pages
for page in result.pages:
    if page.failed:
        print(f"FAILED {page.url}: {page.error}")
    else:
        print(f"{page.title} ({len(page.links)} links)")

# Iterate downloaded files
for file in result.files:
    if file.failed:
        print(f"FAILED {file.filename}: {file.error}")
    else:
        print(f"{file.filename}: {len(file.content)} chars")

# Restrict to specific domains
crawl = CrawlService(allowed_domains=["docs.example.com", "api.example.com"])
result = await crawl.crawl(["https://docs.example.com/guide"])

# Plain text output instead of markdown
crawl = CrawlService(output_format="text")
result = await crawl.crawl(["https://example.com"])

# Cleanup
await crawl.close()
```

## API Reference

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | `int` | `2` | Maximum BFS depth from seed URLs |
| `max_pages` | `int` | `50` | Safety cap on total pages to visit |
| `output_format` | `str` | `"md"` | `"md"` for markdown, `"text"` for plain text |
| `allowed_domains` | `list[str] \| None` | `None` | Restrict crawling to these domains. `None` auto-detects from seed URLs |
| `request_delay` | `float` | `0.5` | Seconds between page loads |
| `headless` | `bool` | `True` | Run browser in headless mode |
| `timeout` | `int` | `30000` | Page navigation timeout in milliseconds |

### Key Methods

- `crawl(urls) -> CrawlResult` — BFS-crawl from seed URLs. Returns pages, files, and aggregated markdown (async)
- `close()` — Close the Playwright browser (async)

### Data Classes

- `CrawlPage` — `url`, `title`, `content`, `depth`, `links`, `error`, `failed` (property)
- `CrawlFile` — `url`, `filename`, `content`, `mime_type`, `source_page`, `error`, `failed` (property)
- `CrawlResult` — `pages: list[CrawlPage]`, `files: list[CrawlFile]`, `markdown: str`

Failed pages/files have `error` set (non-None) and `failed == True`. They are included in results for traceability.
