# YouTubeService

YouTube Data API v3: channels, videos, and statistics.

## Setup

### Env Vars

```yaml
secrets:
  YOUTUBE_API_KEY: "AIza..."
  # Optional — Oxylabs residential proxy (shared with InstagramService)
  OXYLABS_USERNAME: "customer-..."
  OXYLABS_PASSWORD: "..."
```

Get an API key from the [Google Cloud Console](https://console.cloud.google.com/apis/credentials) with the YouTube Data API v3 enabled.

### Dependencies

Requires `httpx` (included in sunset base template), `youtube-transcript-api`, and `yt-dlp` (`pip install sunset[youtube]`). Audio download also requires `ffmpeg` on the system.

## Usage

```python
from sunset.services import YouTubeService

yt = YouTubeService(
    api_key=secrets.get_secret("youtube-api-key"),
    proxy_username=os.getenv("OXYLABS_USERNAME"),  # optional
    proxy_password=os.getenv("OXYLABS_PASSWORD"),
)

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

# Fetch transcript (accepts video ID or URL)
transcript = await yt.get_transcript("dQw4w9WgXcQ")
transcript = await yt.get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Get full text
print(transcript.text)

# Get timed segments
for seg in transcript.segments:
    print(f"[{seg.start:.1f}s] {seg.text}")

# Fetch transcript in another language
transcript = await yt.get_transcript("dQw4w9WgXcQ", languages=["fr", "en"])

# Download audio as bytes
audio_bytes = await yt.download_audio("dQw4w9WgXcQ")
audio_bytes = await yt.download_audio("dQw4w9WgXcQ", codec="aac", quality="128")
```

## API Reference

### `YouTubeService(api_key, proxy_username=None, proxy_password=None)`

Not a singleton — create with your API key. Pass Oxylabs credentials to proxy all requests (Data API + transcript fetches).

### Data Classes

- `YouTubeChannel` — `id, title, description, thumbnail_url, subscriber_count, video_count, uploads_playlist_id`
- `YouTubeVideo` — `id, title, description, url, thumbnail_url, published_at, channel_title, view_count, like_count, comment_count`
- `Transcript` — `video_id, language, language_code, is_generated, segments, text` (`.text` returns full transcript as string)
- `TranscriptSegment` — `text, start, duration`

### Key Methods

- `resolve_channel_id(identifier) -> str?` — Resolve handle/URL/ID to channel ID
- `get_channel(identifier) -> YouTubeChannel?`
- `get_videos(identifier, max_results=20) -> List[YouTubeVideo]`
- `get_video(video_id) -> YouTubeVideo?`
- `get_video_from_url(url) -> YouTubeVideo?`
- `get_transcript(video_id_or_url, languages=["en"]) -> Transcript?` — Fetches captions; accepts video ID or full URL
- `download_audio(video_id_or_url, codec="mp3", quality="192") -> bytes` — Downloads audio via yt-dlp, returns raw bytes
