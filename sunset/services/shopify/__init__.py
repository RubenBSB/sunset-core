"""Shopify Admin API service for OAuth, products, and webhooks."""

import asyncio
import hashlib
import hmac
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

SHOPIFY_API_VERSION = "2024-01"


class ShopifyError(Exception):
    """Raised when a Shopify API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class ShopifyService:
    """Async Shopify Admin API client."""

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=30.0)

    def _admin_url(self, shop_domain: str, path: str) -> str:
        return f"https://{shop_domain}/admin/api/{SHOPIFY_API_VERSION}/{path}"

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with rate-limit retry."""
        for attempt in range(3):
            resp = await client.request(method, url, headers=headers, **kwargs)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "2"))
                logger.warning(f"Shopify rate limited, retrying after {retry_after}s")
                await asyncio.sleep(retry_after)
                continue
            if resp.status_code >= 400:
                raise ShopifyError(
                    f"Shopify API error: {resp.status_code} {resp.text}",
                    status_code=resp.status_code,
                )
            return resp
        raise ShopifyError("Shopify rate limit exceeded after retries", status_code=429)

    # -------------------------------------------------------------------------
    # OAuth
    # -------------------------------------------------------------------------

    def get_install_url(
        self,
        shop_domain: str,
        redirect_uri: str,
        scopes: List[str],
        nonce: str,
    ) -> str:
        params = urlencode(
            {
                "client_id": self._api_key,
                "scope": ",".join(scopes),
                "redirect_uri": redirect_uri,
                "state": nonce,
            }
        )
        return f"https://{shop_domain}/admin/oauth/authorize?{params}"

    async def exchange_token(self, shop_domain: str, code: str) -> Dict[str, Any]:
        async with self._http_client() as client:
            resp = await self._request(
                client,
                "POST",
                f"https://{shop_domain}/admin/oauth/access_token",
                json={
                    "client_id": self._api_key,
                    "client_secret": self._api_secret,
                    "code": code,
                },
            )
            return resp.json()

    # -------------------------------------------------------------------------
    # Products
    # -------------------------------------------------------------------------

    async def fetch_all_products(
        self, shop_domain: str, access_token: str
    ) -> List[Dict[str, Any]]:
        headers = {"X-Shopify-Access-Token": access_token}
        products: List[Dict[str, Any]] = []
        url: Optional[str] = self._admin_url(shop_domain, "products.json")
        params: Optional[Dict[str, Any]] = {"limit": 250}

        async with self._http_client() as client:
            while url:
                resp = await self._request(
                    client, "GET", url, headers=headers, params=params
                )
                data = resp.json()
                products.extend(data.get("products", []))

                # Follow Link header for pagination
                next_url = None
                link_header = resp.headers.get("Link", "")
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        next_url = part.split(";")[0].strip().strip("<>")
                        break

                url = next_url
                params = None  # params are embedded in the Link URL

        return products

    async def fetch_product(
        self, shop_domain: str, access_token: str, product_id: int
    ) -> Dict[str, Any]:
        headers = {"X-Shopify-Access-Token": access_token}
        async with self._http_client() as client:
            resp = await self._request(
                client,
                "GET",
                self._admin_url(shop_domain, f"products/{product_id}.json"),
                headers=headers,
            )
            return resp.json().get("product", {})

    # -------------------------------------------------------------------------
    # Webhooks
    # -------------------------------------------------------------------------

    async def register_webhook(
        self,
        shop_domain: str,
        access_token: str,
        topic: str,
        callback_url: str,
    ) -> Dict[str, Any]:
        headers = {"X-Shopify-Access-Token": access_token}
        async with self._http_client() as client:
            resp = await self._request(
                client,
                "POST",
                self._admin_url(shop_domain, "webhooks.json"),
                headers=headers,
                json={
                    "webhook": {
                        "topic": topic,
                        "address": callback_url,
                        "format": "json",
                    }
                },
            )
            return resp.json().get("webhook", {})

    async def register_product_webhooks(
        self, shop_domain: str, access_token: str, callback_url: str
    ) -> List[Dict[str, Any]]:
        results = []
        for topic in ("products/create", "products/update", "products/delete"):
            webhook = await self.register_webhook(
                shop_domain, access_token, topic, callback_url
            )
            results.append(webhook)
        return results

    @staticmethod
    def verify_webhook(data: bytes, hmac_header: str, secret: str) -> bool:
        digest = hmac.new(secret.encode("utf-8"), data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(digest, hmac_header)
