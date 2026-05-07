"""Instagram scraping service using GraphQL API."""

import json
import logging
import random
import re
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


def _build_proxy_url(username: str, password: str) -> str:
    sess_id = "".join(random.choices(string.digits, k=10))
    base = username if username.startswith("customer-") else f"customer-{username}"
    proxy_user = f"{base}-cc-us-sessid-{sess_id}-sesstime-10"
    encoded_pw = quote(password, safe="")
    return f"http://{proxy_user}:{encoded_pw}@pr.oxylabs.io:7777"


@dataclass
class MediaItem:
    """Single media item (image or video) within a post."""

    type: str  # "image" or "video"
    url: str


@dataclass
class InstagramPost:
    """An Instagram post with full details."""

    id: str
    shortcode: str
    url: str
    type: str  # "GraphImage", "GraphVideo", "GraphSidecar"
    caption: Optional[str]
    likes: int
    comments: int
    published_at: datetime
    media: List[MediaItem] = field(default_factory=list)
    thumbnail_url: Optional[str] = None


@dataclass
class InstagramProfile:
    """Instagram profile info."""

    id: str
    username: str
    full_name: Optional[str]
    bio: Optional[str]
    followers: int
    following: int
    post_count: int
    profile_pic_url: Optional[str]


class InstagramService:
    """Async Instagram scraper using GraphQL API.

    NOTE: Instagram rotates GraphQL doc_ids every 2-4 weeks.
    If scraping breaks, check browser DevTools on instagram.com for
    updated doc_id values in graphql/query network requests.
    """

    DOC_ID_POST = "8845758582119845"
    DOC_ID_PROFILE_POSTS = "7950326061742207"
    IG_APP_ID = "936619743392459"

    def __init__(
        self,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None,
    ):
        self._proxy_username = proxy_username
        self._proxy_password = proxy_password
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            proxy_url = None
            if self._proxy_username and self._proxy_password:
                proxy_url = _build_proxy_url(self._proxy_username, self._proxy_password)
                logger.info("Oxylabs residential proxy configured")

            self._client = httpx.AsyncClient(
                timeout=30.0,
                proxy=proxy_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "X-Requested-With": "XMLHttpRequest",
                },
            )
            # Fetch CSRF token
            resp = await self._client.get("https://www.instagram.com/")
            csrf = resp.cookies.get("csrftoken", "")
            self._client.headers.update(
                {
                    "X-CSRFToken": csrf,
                    "X-IG-App-ID": self.IG_APP_ID,
                }
            )
        return self._client

    async def _reset_client(self):
        if self._client:
            await self._client.aclose()
        self._client = None

    async def close(self):
        await self._reset_client()

    async def _get_user_id(self, username: str) -> Optional[str]:
        client = await self._get_client()

        # Method 1: profile page HTML
        try:
            resp = await client.get(f"https://www.instagram.com/{username}/")
            if resp.status_code == 200:
                match = re.search(r'"profilePage_(\d+)"', resp.text)
                if match:
                    return match.group(1)
        except Exception as e:
            logger.warning(f"Profile page error for {username}: {e}")

        # Method 2: search endpoint
        try:
            resp = await client.get(
                "https://www.instagram.com/web/search/topsearch/",
                params={"query": username, "context": "user"},
            )
            if resp.status_code == 200:
                for u in resp.json().get("users", []):
                    node = u.get("user", {})
                    if node.get("username", "").lower() == username.lower():
                        return str(node.get("pk") or node.get("id"))
        except Exception as e:
            logger.warning(f"Search error for {username}: {e}")

        return None

    async def get_profile(self, username: str) -> Optional[InstagramProfile]:
        """Fetch Instagram profile info."""
        client = await self._get_client()
        try:
            resp = await client.get(
                f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}",
                headers={
                    "User-Agent": (
                        "Instagram 275.0.0.27.98 Android "
                        "(33/13; 420dpi; 1080x2400; samsung; SM-G991B; o1s; exynos2100)"
                    ),
                    "X-IG-App-ID": self.IG_APP_ID,
                },
            )
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                user = data.get("user") if isinstance(data, dict) else None
                if user:
                    return InstagramProfile(
                        id=user["id"],
                        username=user["username"],
                        full_name=user.get("full_name"),
                        bio=user.get("biography"),
                        followers=(user.get("edge_followed_by") or {}).get("count", 0),
                        following=(user.get("edge_follow") or {}).get("count", 0),
                        post_count=(user.get("edge_owner_to_timeline_media") or {}).get(
                            "count", 0
                        ),
                        profile_pic_url=user.get("profile_pic_url_hd")
                        or user.get("profile_pic_url"),
                    )
            logger.warning(f"Profile API failed for {username}: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Profile API error for {username}: {e}")

        # Fallback: minimal profile via user ID lookup
        user_id = await self._get_user_id(username)
        if user_id:
            return InstagramProfile(
                id=user_id,
                username=username,
                full_name=None,
                bio=None,
                followers=0,
                following=0,
                post_count=0,
                profile_pic_url=None,
            )
        return None

    async def get_posts(
        self,
        username: str,
        max_posts: int = 12,
        since: datetime | None = None,
    ) -> List[InstagramPost]:
        """Fetch recent posts from an Instagram profile.

        If `since` is provided, stops paginating once a post older than
        `since` is encountered and only returns posts newer than `since`.
        """
        client = await self._get_client()

        profile = await self.get_profile(username)
        if not profile:
            logger.error(f"Could not find profile: {username}")
            return []

        posts: List[InstagramPost] = []
        cursor: Optional[str] = None
        hit_since_cutoff = False

        try:
            while len(posts) < max_posts and not hit_since_cutoff:
                variables: dict = {
                    "id": profile.id,
                    "first": min(12, max_posts - len(posts)),
                }
                if cursor:
                    variables["after"] = cursor

                resp = await client.post(
                    "https://www.instagram.com/graphql/query",
                    data={
                        "variables": json.dumps(variables),
                        "doc_id": self.DOC_ID_PROFILE_POSTS,
                    },
                )

                if resp.status_code != 200:
                    logger.error(f"Posts fetch failed: {resp.status_code}")
                    break

                data = resp.json().get("data") or {}
                timeline = (data.get("user") or {}).get(
                    "edge_owner_to_timeline_media"
                ) or {}
                edges = timeline.get("edges") or []

                if not edges:
                    break

                for edge in edges:
                    if len(posts) >= max_posts:
                        break
                    node = edge.get("node") or {}

                    if since is not None:
                        ts = node.get("taken_at_timestamp", 0)
                        post_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                        if post_time < since:
                            hit_since_cutoff = True
                            break

                    shortcode = node.get("shortcode")
                    if shortcode:
                        post = await self.get_post(shortcode)
                        if post:
                            posts.append(post)

                page_info = timeline.get("page_info") or {}
                if not page_info.get("has_next_page"):
                    break
                cursor = page_info.get("end_cursor")

            posts.sort(key=lambda p: p.published_at, reverse=True)
        except Exception as e:
            logger.exception(f"Error fetching posts for {username}: {e}")
            await self._reset_client()

        return posts

    async def get_post(self, shortcode: str) -> Optional[InstagramPost]:
        """Fetch full details for a single post by shortcode."""
        client = await self._get_client()

        try:
            resp = await client.post(
                "https://www.instagram.com/graphql/query",
                data={
                    "variables": json.dumps({"shortcode": shortcode}),
                    "doc_id": self.DOC_ID_POST,
                },
            )

            if resp.status_code != 200:
                logger.error(f"Post fetch failed for {shortcode}: {resp.status_code}")
                return None

            media = (resp.json().get("data") or {}).get("xdt_shortcode_media")
            if not media:
                return None

            return self._parse_post(media)
        except Exception as e:
            logger.exception(f"Error fetching post {shortcode}: {e}")
            await self._reset_client()
            return None

    async def get_post_from_url(self, url: str) -> Optional[InstagramPost]:
        """Fetch post details from a full Instagram URL."""
        match = re.search(r"/(?:p|reel)/([A-Za-z0-9_-]+)", url)
        if not match:
            logger.error(f"Invalid Instagram URL: {url}")
            return None
        return await self.get_post(match.group(1))

    def _parse_post(self, media: dict) -> InstagramPost:
        shortcode = media["shortcode"]
        typename = media.get("__typename", "").replace("XDT", "")

        caption_edges = media.get("edge_media_to_caption", {}).get("edges", [])
        caption = caption_edges[0]["node"]["text"] if caption_edges else None

        likes = media.get("edge_media_preview_like", {}).get("count", 0)
        comments = media.get("edge_media_to_parent_comment", {}).get(
            "count"
        ) or media.get("edge_media_to_comment", {}).get("count", 0)

        timestamp = media.get("taken_at_timestamp", 0)
        published_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        media_items = self._extract_media(media)
        thumbnail_url = media.get("display_url") or media.get("thumbnail_src")

        return InstagramPost(
            id=media.get("id", shortcode),
            shortcode=shortcode,
            url=f"https://www.instagram.com/p/{shortcode}/",
            type=typename,
            caption=caption,
            likes=likes,
            comments=comments,
            published_at=published_at,
            media=media_items,
            thumbnail_url=thumbnail_url,
        )

    def _extract_media(self, media: dict) -> List[MediaItem]:
        items: List[MediaItem] = []
        typename = media.get("__typename", "")

        if typename == "XDTGraphSidecar":
            for edge in media.get("edge_sidecar_to_children", {}).get("edges", []):
                node = edge["node"]
                if node.get("is_video"):
                    items.append(MediaItem(type="video", url=node.get("video_url", "")))
                else:
                    items.append(
                        MediaItem(type="image", url=node.get("display_url", ""))
                    )
        elif media.get("is_video"):
            items.append(MediaItem(type="video", url=media.get("video_url", "")))
        else:
            items.append(MediaItem(type="image", url=media.get("display_url", "")))

        return items
