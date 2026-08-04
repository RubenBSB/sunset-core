"""Search-engine and frontend-cache notification after content changes.

Two best-effort pings, never raising:
- the Next.js frontend's /api/revalidate endpoint (ISR cache bust)
- IndexNow (the index behind Bing, ChatGPT search and Copilot) for
  created/updated/removed URLs — including OLD urls after a rename or
  unpublish, so engines recrawl and drop them.
"""

import logging

import httpx

logger = logging.getLogger(__name__)


async def notify_site_updated(
    *,
    frontend_url: str,
    revalidate_secret: str = "",
    site_url: str | None = None,
    indexnow_key: str | None = None,
    urls: list[str] | None = None,
) -> None:
    """Ping the frontend revalidation endpoint and, when a key is configured
    and the frontend is served over https (production), IndexNow.

    Args:
        frontend_url: base URL of the Next.js app (e.g. https://example.com).
        revalidate_secret: shared secret for POST {frontend_url}/api/revalidate.
        site_url: canonical site origin for IndexNow (defaults to frontend_url).
        indexnow_key: IndexNow key; the key file must be served at
            {site_url}/{key}.txt. No ping when unset.
        urls: changed URLs (new AND old ones after renames/unpublish).
    """
    web_base = frontend_url.rstrip("/")
    site = (site_url or web_base).rstrip("/")
    is_prod = web_base.startswith("https://")

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(
                f"{web_base}/api/revalidate",
                headers={"x-revalidate-secret": revalidate_secret},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Frontend revalidate ping failed: %s", e)

        if urls and indexnow_key and is_prod:
            try:
                await client.post(
                    "https://api.indexnow.org/indexnow",
                    json={
                        "host": site.removeprefix("https://"),
                        "key": indexnow_key,
                        "keyLocation": f"{site}/{indexnow_key}.txt",
                        "urlList": urls[:100],
                    },
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("IndexNow ping failed: %s", e)
