"""YouTube Data API v3 service for fetching channel and video metadata."""

import asyncio
import logging
import random
import re
import string
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

logger = logging.getLogger(__name__)

YT_API_BASE = "https://www.googleapis.com/youtube/v3"


@dataclass
class YouTubeVideo:
    """A YouTube video with metadata and statistics."""

    id: str
    title: str
    description: Optional[str]
    url: str
    thumbnail_url: Optional[str]
    published_at: Optional[datetime]
    channel_title: Optional[str]
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0


@dataclass
class YouTubeChannel:
    """Basic YouTube channel info."""

    id: str
    title: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    subscriber_count: int = 0
    video_count: int = 0
    uploads_playlist_id: Optional[str] = None


@dataclass
class TranscriptSegment:
    """A single segment of a video transcript."""

    text: str
    start: float
    duration: float


@dataclass
class Transcript:
    """Full transcript for a YouTube video."""

    video_id: str
    language: str
    language_code: str
    is_generated: bool
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


def _build_proxy_url(username: str, password: str) -> str:
    sess_id = "".join(random.choices(string.digits, k=10))
    base = username if username.startswith("customer-") else f"customer-{username}"
    proxy_user = f"{base}-cc-us-sessid-{sess_id}-sesstime-10"
    encoded_pw = quote(password, safe="")
    return f"http://{proxy_user}:{encoded_pw}@pr.oxylabs.io:7777"


class YouTubeService:
    """Async YouTube Data API v3 client."""

    def __init__(
        self,
        api_key: str,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None,
    ):
        self._api_key = api_key
        self._proxy_username = proxy_username
        self._proxy_password = proxy_password

    def _get_proxy_url(self) -> Optional[str]:
        if self._proxy_username and self._proxy_password:
            return _build_proxy_url(self._proxy_username, self._proxy_password)
        return None

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=30.0, proxy=self._get_proxy_url())

    async def resolve_channel_id(self, identifier: str) -> Optional[str]:
        """Resolve a YouTube URL, handle, or channel ID to a channel ID.

        Accepts:
          - https://www.youtube.com/@Handle
          - https://www.youtube.com/channel/UCxxxxx
          - @Handle
          - UCxxxxx (raw channel ID)
        """
        identifier = identifier.strip()

        channel_match = re.search(r"youtube\.com/channel/(UC[\w-]+)", identifier)
        if channel_match:
            return channel_match.group(1)

        handle_match = re.search(r"youtube\.com/@([\w.-]+)", identifier)
        if handle_match:
            handle = handle_match.group(1)
        elif identifier.startswith("@"):
            handle = identifier[1:]
        elif identifier.startswith("UC") and len(identifier) == 24:
            return identifier
        else:
            handle = identifier

        async with self._http_client() as client:
            resp = await client.get(
                f"{YT_API_BASE}/channels",
                params={"key": self._api_key, "forHandle": handle, "part": "id"},
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if items:
                channel_id = items[0]["id"]
                logger.info(f"Resolved @{handle} -> {channel_id}")
                return channel_id

        logger.warning(f"Could not resolve YouTube identifier: {identifier}")
        return None

    async def get_channel(self, identifier: str) -> Optional[YouTubeChannel]:
        """Fetch channel info by ID, handle, or URL."""
        channel_id = await self.resolve_channel_id(identifier)
        if not channel_id:
            return None

        async with self._http_client() as client:
            resp = await client.get(
                f"{YT_API_BASE}/channels",
                params={
                    "key": self._api_key,
                    "id": channel_id,
                    "part": "snippet,contentDetails,statistics",
                },
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                return None

            item = items[0]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            uploads = (
                item.get("contentDetails", {})
                .get("relatedPlaylists", {})
                .get("uploads")
            )
            thumbnails = snippet.get("thumbnails", {})
            thumb = (
                thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url")
            )

            return YouTubeChannel(
                id=channel_id,
                title=snippet.get("title", ""),
                description=snippet.get("description"),
                thumbnail_url=thumb,
                subscriber_count=int(stats.get("subscriberCount", 0)),
                video_count=int(stats.get("videoCount", 0)),
                uploads_playlist_id=uploads,
            )

    async def get_videos(
        self, identifier: str, max_results: int = 20
    ) -> List[YouTubeVideo]:
        """Fetch recent videos from a channel.

        Args:
            identifier: Channel ID, handle, or URL.
            max_results: Maximum videos to return (max 50).
        """
        channel = await self.get_channel(identifier)
        if not channel or not channel.uploads_playlist_id:
            return []

        async with self._http_client() as client:
            # Fetch playlist items
            resp = await client.get(
                f"{YT_API_BASE}/playlistItems",
                params={
                    "key": self._api_key,
                    "playlistId": channel.uploads_playlist_id,
                    "part": "snippet,contentDetails",
                    "maxResults": min(max_results, 50),
                },
            )
            resp.raise_for_status()

            playlist_items = resp.json().get("items", [])
            if not playlist_items:
                return []

            # Collect video IDs
            video_ids = []
            video_snippets: Dict[str, dict] = {}
            for item in playlist_items:
                vid = item.get("contentDetails", {}).get("videoId")
                if vid:
                    video_ids.append(vid)
                    video_snippets[vid] = item.get("snippet", {})

            # Fetch statistics in batches of 50
            video_stats: Dict[str, dict] = {}
            for i in range(0, len(video_ids), 50):
                batch = video_ids[i : i + 50]
                stats_resp = await client.get(
                    f"{YT_API_BASE}/videos",
                    params={
                        "key": self._api_key,
                        "id": ",".join(batch),
                        "part": "statistics",
                    },
                )
                stats_resp.raise_for_status()
                for stat_item in stats_resp.json().get("items", []):
                    s = stat_item.get("statistics", {})
                    video_stats[stat_item["id"]] = {
                        "view_count": int(s.get("viewCount", 0)),
                        "like_count": int(s.get("likeCount", 0)),
                        "comment_count": int(s.get("commentCount", 0)),
                    }

            # Build results
            videos: List[YouTubeVideo] = []
            for vid in video_ids:
                snippet = video_snippets.get(vid, {})
                stats = video_stats.get(vid, {})

                published_at = None
                if snippet.get("publishedAt"):
                    published_at = datetime.fromisoformat(
                        snippet["publishedAt"].replace("Z", "+00:00")
                    )

                thumbnails = snippet.get("thumbnails", {})
                thumb = (
                    thumbnails.get("maxres", {}).get("url")
                    or thumbnails.get("high", {}).get("url")
                    or thumbnails.get("medium", {}).get("url")
                    or thumbnails.get("default", {}).get("url")
                )

                videos.append(
                    YouTubeVideo(
                        id=vid,
                        title=snippet.get("title", ""),
                        description=snippet.get("description"),
                        url=f"https://www.youtube.com/watch?v={vid}",
                        thumbnail_url=thumb,
                        published_at=published_at,
                        channel_title=snippet.get("channelTitle"),
                        view_count=stats.get("view_count", 0),
                        like_count=stats.get("like_count", 0),
                        comment_count=stats.get("comment_count", 0),
                    )
                )

            return videos

    async def get_video(self, video_id: str) -> Optional[YouTubeVideo]:
        """Fetch a single video by ID."""
        async with self._http_client() as client:
            resp = await client.get(
                f"{YT_API_BASE}/videos",
                params={
                    "key": self._api_key,
                    "id": video_id,
                    "part": "snippet,statistics",
                },
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                return None

            item = items[0]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})

            published_at = None
            if snippet.get("publishedAt"):
                published_at = datetime.fromisoformat(
                    snippet["publishedAt"].replace("Z", "+00:00")
                )

            thumbnails = snippet.get("thumbnails", {})
            thumb = (
                thumbnails.get("maxres", {}).get("url")
                or thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url")
            )

            return YouTubeVideo(
                id=video_id,
                title=snippet.get("title", ""),
                description=snippet.get("description"),
                url=f"https://www.youtube.com/watch?v={video_id}",
                thumbnail_url=thumb,
                published_at=published_at,
                channel_title=snippet.get("channelTitle"),
                view_count=int(stats.get("viewCount", 0)),
                like_count=int(stats.get("likeCount", 0)),
                comment_count=int(stats.get("commentCount", 0)),
            )

    async def get_video_from_url(self, url: str) -> Optional[YouTubeVideo]:
        """Fetch video details from a YouTube URL."""
        match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
        if not match:
            logger.error(f"Invalid YouTube URL: {url}")
            return None
        return await self.get_video(match.group(1))

    @staticmethod
    def _extract_video_id(video_id_or_url: str) -> str:
        match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", video_id_or_url)
        return match.group(1) if match else video_id_or_url

    async def get_transcript(
        self,
        video_id_or_url: str,
        languages: list[str] | None = None,
    ) -> Optional[Transcript]:
        """Fetch the transcript for a video.

        Args:
            video_id_or_url: Video ID or full YouTube URL.
            languages: Language codes in descending priority (default: ["en"]).
        """
        video_id = self._extract_video_id(video_id_or_url)
        langs = languages or ["en"]

        try:
            proxy_url = self._get_proxy_url()
            proxy_config = (
                GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
                if proxy_url
                else None
            )
            ytt = YouTubeTranscriptApi(proxy_config=proxy_config)
            fetched = await asyncio.to_thread(ytt.fetch, video_id, languages=langs)
        except Exception as e:
            logger.error(f"Transcript fetch failed for {video_id}: {e}")
            return None

        segments = [
            TranscriptSegment(text=s.text, start=s.start, duration=s.duration)
            for s in fetched
        ]

        return Transcript(
            video_id=video_id,
            language=fetched.language,
            language_code=fetched.language_code,
            is_generated=fetched.is_generated,
            segments=segments,
        )

    async def download_audio(
        self,
        video_id_or_url: str,
        codec: str = "mp3",
        quality: str = "192",
    ) -> bytes:
        """Download audio from a YouTube video and return raw bytes.

        Args:
            video_id_or_url: Video ID or full YouTube URL.
            codec: Audio codec (mp3, aac, wav, etc.). Default "mp3".
            quality: Audio quality in kbps. Default "192".
        """
        from yt_dlp import YoutubeDL

        video_id = self._extract_video_id(video_id_or_url)
        url = f"https://www.youtube.com/watch?v={video_id}"

        def _download(tmp_dir: str) -> Path:
            opts = {
                "format": "bestaudio/best",
                "outtmpl": f"{tmp_dir}/%(id)s.%(ext)s",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": codec,
                        "preferredquality": quality,
                    }
                ],
                "quiet": True,
                "no_warnings": True,
            }
            proxy_url = self._get_proxy_url()
            if proxy_url:
                opts["proxy"] = proxy_url

            with YoutubeDL(opts) as ydl:
                ydl.download([url])

            files = list(Path(tmp_dir).glob(f"{video_id}.*"))
            if not files:
                raise FileNotFoundError(f"No audio file produced for {video_id}")
            return files[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = await asyncio.to_thread(_download, tmp_dir)
            return audio_path.read_bytes()
