"""YouTube Data API v3 service for fetching channel and video metadata."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import httpx

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


class YouTubeService:
    """Async YouTube Data API v3 client."""

    def __init__(self, api_key: str):
        self._api_key = api_key

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

        async with httpx.AsyncClient(timeout=30.0) as client:
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

        async with httpx.AsyncClient(timeout=30.0) as client:
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

        async with httpx.AsyncClient(timeout=30.0) as client:
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
        async with httpx.AsyncClient(timeout=30.0) as client:
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
