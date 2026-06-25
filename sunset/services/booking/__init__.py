"""Booking.com hotel client — search + deep-link, for logistics suggestions.

Wraps the Booking.com RapidAPI actor (`mtnrabi/booking-live-api`, Apify Standby
mode): a single `/search` call returns ~25 priced properties for a destination +
dates. Like the flight adapters this service is *provider-only* — it knows the
Booking API and nothing about your domain. Proximity ranking (geocoding hotels
against a venue) is the caller's job, kept out so the service stays a thin,
testable HTTP client.

Two honest limitations baked into the source, surfaced — never hidden:
- `/search` prices ONE room (no_rooms=1) and has no inventory count, so a
  group estimate is `price × rooms`, not a guarantee that X rooms are
  simultaneously bookable.
- search results carry no coordinates; the caller geocodes the name for distance.

`booking_url` rebuilds a Booking deep-link pre-filled with the dates and X
single rooms (`no_rooms=X & group_adults=X`), so a click lands on the live
Booking page showing the real per-group availability/price.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

__all__ = ["BookingService", "BookingError"]

_DEFAULT_HOST = "booking-live-api.p.rapidapi.com"
# Cold-start / gateway statuses of the Standby actor — worth one retry.
_RETRY_STATUSES = (502, 503, 504)
_ID_RE = re.compile(r"/hotel/([a-z]{2}/[^.?/]+)\.html")


def _first(d: Any, *keys: str) -> Any:
    """First non-empty field among several possible names (scraper field names
    aren't guaranteed, so we try variants)."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def _hotels_from(body: Any) -> list[dict[str, Any]]:
    """Pull the property list out of a /search response whatever the envelope."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        for k in ("properties", "results", "hotels", "data", "items"):
            v = body.get(k)
            if isinstance(v, list):
                return v
    return []


def _extract_booking_id(link: Any) -> Optional[str]:
    """hotel_booking_id (e.g. 'ma/la-mamounia') from a Booking link, or None."""
    m = _ID_RE.search(link) if isinstance(link, str) else None
    return m.group(1) if m else None


class BookingError(Exception):
    """Raised when a Booking API call fails (HTTP error or exhausted retries)."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class BookingService:
    """Async Booking.com client (hotel search via the RapidAPI Standby actor)."""

    def __init__(
        self,
        api_key: str,
        *,
        host: str = _DEFAULT_HOST,
        timeout: float = 60.0,
    ):
        self._api_key = api_key
        self._host = host
        self._timeout = timeout

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout)

    def _headers(self) -> dict[str, str]:
        return {
            "X-RapidAPI-Key": self._api_key,
            "X-RapidAPI-Host": self._host,
            "Content-Type": "application/json",
        }

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        """POST to the actor, retrying once on a cold-start gateway status."""
        url = f"https://{self._host}{path}"
        for attempt in range(2):
            async with self._http_client() as client:
                resp = await client.post(url, headers=self._headers(), json=payload)
            if resp.status_code in _RETRY_STATUSES and attempt == 0:
                logger.warning("Booking %s cold-start (%s), retrying", path, resp.status_code)
                await asyncio.sleep(3)
                continue
            if resp.status_code == 429:
                raise BookingError(
                    f"Booking quota exceeded: {resp.text[:200]}", status_code=429
                )
            if resp.status_code >= 400:
                raise BookingError(
                    f"Booking API error: {resp.status_code} {resp.text[:200]}",
                    status_code=resp.status_code,
                )
            return resp.json()
        raise BookingError("Booking request failed after retries")

    async def search_hotels(
        self,
        destination: str,
        checkin: str,
        checkout: str,
        *,
        adults: int = 1,
        currency: str = "EUR",
        max_results: int = 25,
    ) -> list[dict[str, Any]]:
        """Priced properties for a text destination + dates (raw API shape).

        `adults` is the occupancy priced per room (1 = single). `price` in the
        result is the total for the stay at that occupancy, not per night.
        Returns raw property dicts; use `summarize_hotels` for the compact shape.
        """
        body = await self._post(
            "/search",
            {
                "destination": destination,
                "checkin_date": checkin,
                "checkout_date": checkout,
                "adults": adults,
                "children": 0,
                "currency": currency,
            },
        )
        hotels = _hotels_from(body)
        return hotels[:max_results] if max_results is not None else hotels

    @staticmethod
    def summarize_hotels(
        raw_hotels: list[dict[str, Any]], *, currency: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Flatten raw /search properties into the compact shape the UI uses.

        One entry per hotel: name, price (stay total, float), currency,
        review_score/_count, nights, the Booking link, and `booking_id` parsed
        from it. Unnamed entries are dropped.
        """
        out: list[dict[str, Any]] = []
        for h in raw_hotels:
            if not isinstance(h, dict):
                continue
            name = _first(h, "name", "hotel_name", "title")
            if not name:
                continue
            price = _first(h, "price", "price_per_night", "gross_price", "total_price")
            link = _first(h, "link", "url", "booking_link")
            out.append(
                {
                    "name": name,
                    "price": float(price) if isinstance(price, (int, float)) else None,
                    "currency": _first(h, "currency", "currency_code") or currency,
                    "review_score": _first(h, "review_score", "rating", "score"),
                    "review_count": _first(h, "review_count"),
                    "nights": _first(h, "nights"),
                    "link": link,
                    "booking_id": _extract_booking_id(link),
                }
            )
        return out

    @staticmethod
    def booking_url(
        hotel_booking_id: str,
        checkin: str,
        checkout: str,
        *,
        rooms: int = 1,
        adults_per_room: int = 1,
        currency: str = "EUR",
    ) -> str:
        """Booking deep-link pre-filled with the dates + occupancy.

        `rooms` rooms × `adults_per_room` → the page opens on the group's
        availability (`no_rooms=rooms`, `group_adults=rooms × adults_per_room`).
        With the default `adults_per_room=1` that's X single rooms.
        """
        query = urlencode(
            {
                "checkin": checkin,
                "checkout": checkout,
                "group_adults": rooms * adults_per_room,
                "no_rooms": rooms,
                "group_children": 0,
                "selected_currency": currency,
            }
        )
        return f"https://www.booking.com/hotel/{hotel_booking_id}.html?{query}"
