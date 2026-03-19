# YouTubeService

YouTube Data API v3: channels, videos, and statistics.

## Setup

### Env Vars

```yaml
secrets:
  YOUTUBE_API_KEY: "AIza..."
```

Get an API key from the [Google Cloud Console](https://console.cloud.google.com/apis/credentials) with the YouTube Data API v3 enabled.

### Dependencies

Requires `httpx` (included in sunset base template).

## Usage

```python
from sunset.services import YouTubeService

yt = YouTubeService(api_key=secrets.get_secret("youtube-api-key"))

# Fetch channel info (accepts @handle, URL, or channel ID)
channel = await yt.get_channel("@mkbhd")
# -> YouTubeChannel(id, title, description, thumbnail_url, subscriber_count, video_count)

# Fetch recent videos
videos = await yt.get_videos("@mkbhd", max_results=20)
# -> List[YouTubeVideo]

# Fetch single video by ID
video = await yt.get_video("dQw4w9WgXcQ")

# Fetch single video by URL
video = await yt.get_video_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
```

## API Reference

### `YouTubeService(api_key)`

Not a singleton — create with your API key.

### Data Classes

- `YouTubeChannel` — `id, title, description, thumbnail_url, subscriber_count, video_count, uploads_playlist_id`
- `YouTubeVideo` — `id, title, description, url, thumbnail_url, published_at, channel_title, view_count, like_count, comment_count`

### Key Methods

- `resolve_channel_id(identifier) -> str?` — Resolve handle/URL/ID to channel ID
- `get_channel(identifier) -> YouTubeChannel?`
- `get_videos(identifier, max_results=20) -> List[YouTubeVideo]`
- `get_video(video_id) -> YouTubeVideo?`
- `get_video_from_url(url) -> YouTubeVideo?`
