# InstagramService

Instagram scraping via GraphQL API: profiles, posts, and media extraction.

## Setup

### Env Vars

Optional — only needed if using residential proxies to avoid rate limits:

```yaml
secrets:
  OXYLABS_USERNAME: "customer-xxx"
  OXYLABS_PASSWORD: "xxx"
```

### Dependencies

Requires `httpx` (included in sunset base template).

## Usage

```python
from sunset.services import InstagramService

# Without proxy
ig = InstagramService()

# With proxy (Oxylabs residential)
ig = InstagramService(
    proxy_username=secrets.get_secret("oxylabs-username"),
    proxy_password=secrets.get_secret("oxylabs-password"),
)

# Fetch profile
profile = await ig.get_profile("username")
# -> InstagramProfile(id, username, full_name, bio, followers, following, post_count, profile_pic_url)

# Fetch recent posts
posts = await ig.get_posts("username", max_posts=12)
# -> List[InstagramPost]

# Fetch only posts newer than a cutoff (incremental sync)
from datetime import datetime, timezone
since = datetime(2026, 3, 1, tzinfo=timezone.utc)
new_posts = await ig.get_posts("username", max_posts=50, since=since)

# Fetch single post by shortcode
post = await ig.get_post("ABC123")

# Fetch single post by URL
post = await ig.get_post_from_url("https://www.instagram.com/p/ABC123/")

# Cleanup
await ig.close()
```

## API Reference

### `InstagramService(proxy_username?, proxy_password?)`

Not a singleton — create per use or keep alive for session reuse.

### Data Classes

- `InstagramProfile` — `id, username, full_name, bio, followers, following, post_count, profile_pic_url`
- `InstagramPost` — `id, shortcode, url, type, caption, likes, comments, published_at, media, thumbnail_url`
- `MediaItem` — `type ("image"|"video"), url`

### Key Methods

- `get_profile(username) -> InstagramProfile?`
- `get_posts(username, max_posts=12, since=None) -> List[InstagramPost]` — if `since` (datetime) is set, stops paginating once posts older than `since` are hit
- `get_post(shortcode) -> InstagramPost?`
- `get_post_from_url(url) -> InstagramPost?`
- `close()` — Close the underlying HTTP client

### Notes

Instagram rotates GraphQL `doc_id` values every 2-4 weeks. If scraping breaks, check browser DevTools on instagram.com for updated `doc_id` values in `graphql/query` network requests, and update `DOC_ID_POST` / `DOC_ID_PROFILE_POSTS` class attributes.
