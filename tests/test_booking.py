"""Tests for BookingService — request shape, retry, summarisation, deep-link.

HTTP is faked at the httpx.AsyncClient seam so the real request/parse paths run
without a network or a RapidAPI key.
"""

import asyncio
import os
from unittest.mock import patch

import pytest

from sunset.services.booking import BookingError, BookingService

SEARCH_PATH = "/search"

# Live test runs only when a real key is exported; skipped otherwise.
_LIVE_KEY = bool(os.getenv("BOOKING_RAPIDAPI_KEY"))


class FakeResp:
    def __init__(self, status_code=200, json_body=None, text="", headers=None):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json


class FakeClient:
    """httpx.AsyncClient stand-in routing each request through a handler."""

    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        return self._handler(url, kwargs)


def _patch(handler):
    return patch(
        "sunset.services.booking.httpx.AsyncClient",
        return_value=FakeClient(handler),
    )


def _svc():
    return BookingService(api_key="rapid_test_abc")


def _property(name, price, *, link=None, **extra):
    p = {"name": name, "price": price, "currency": "EUR", "review_score": 9.1}
    if link is not None:
        p["link"] = link
    p.update(extra)
    return p


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_search_builds_body_and_headers():
    seen = {}

    def handler(url, kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers")
        seen["json"] = kwargs.get("json")
        return FakeResp(json_body={"properties": []})

    svc = _svc()
    with _patch(handler):
        hotels = asyncio.run(
            svc.search_hotels("Marrakech", "2026-09-10", "2026-09-12", adults=1)
        )

    assert hotels == []
    assert seen["url"].endswith(SEARCH_PATH)
    assert seen["headers"]["X-RapidAPI-Key"] == "rapid_test_abc"
    assert seen["headers"]["X-RapidAPI-Host"] == "booking-live-api.p.rapidapi.com"
    assert seen["json"]["destination"] == "Marrakech"
    assert seen["json"]["checkin_date"] == "2026-09-10"
    assert seen["json"]["checkout_date"] == "2026-09-12"
    assert seen["json"]["adults"] == 1


def test_search_trims_to_max_results():
    def handler(url, kwargs):
        return FakeResp(
            json_body={"properties": [_property(f"H{i}", 100 + i) for i in range(10)]}
        )

    svc = _svc()
    with _patch(handler):
        hotels = asyncio.run(
            svc.search_hotels("Paris", "2026-09-10", "2026-09-12", max_results=3)
        )
    assert len(hotels) == 3
    assert hotels[0]["name"] == "H0"


def test_search_reads_alternate_envelope():
    # Scraper sometimes wraps the list under `results` instead of `properties`.
    def handler(url, kwargs):
        return FakeResp(json_body={"results": [_property("Riad", 150)]})

    svc = _svc()
    with _patch(handler):
        hotels = asyncio.run(svc.search_hotels("Marrakech", "2026-09-10", "2026-09-12"))
    assert len(hotels) == 1 and hotels[0]["name"] == "Riad"


# ---------------------------------------------------------------------------
# Errors + retry
# ---------------------------------------------------------------------------


def test_429_raises_quota_error():
    def handler(url, kwargs):
        return FakeResp(status_code=429, text="monthly quota reached")

    svc = _svc()
    with _patch(handler):
        with pytest.raises(BookingError) as exc:
            asyncio.run(svc.search_hotels("Paris", "2026-09-10", "2026-09-12"))
    assert exc.value.status_code == 429
    assert "quota" in str(exc.value).lower()


def test_4xx_raises_with_status():
    def handler(url, kwargs):
        return FakeResp(status_code=422, text="bad destination")

    svc = _svc()
    with _patch(handler):
        with pytest.raises(BookingError) as exc:
            asyncio.run(svc.search_hotels("", "2026-09-10", "2026-09-12"))
    assert exc.value.status_code == 422


def test_cold_start_retries_then_succeeds():
    state = {"n": 0}

    def handler(url, kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return FakeResp(status_code=503)
        return FakeResp(json_body={"properties": [_property("H", 100)]})

    svc = _svc()
    with patch("sunset.services.booking.asyncio.sleep", new=_noop_sleep), _patch(handler):
        hotels = asyncio.run(svc.search_hotels("Nice", "2026-09-10", "2026-09-12"))
    assert state["n"] == 2 and len(hotels) == 1


async def _noop_sleep(_seconds):
    return None


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def test_summarize_hotels_normalizes_and_parses_id():
    raw = [
        _property(
            "La Mamounia",
            320,
            link="https://www.booking.com/hotel/ma/la-mamounia.html?aid=1&checkin=x",
            review_count=1200,
            nights=2,
        ),
        {"hotel_name": "Hotel Y", "gross_price": 200, "currency_code": "USD"},
        {"no_name_here": True},  # dropped — no name
    ]
    out = BookingService.summarize_hotels(raw)
    assert len(out) == 2
    assert out[0]["name"] == "La Mamounia"
    assert out[0]["price"] == 320.0
    assert out[0]["booking_id"] == "ma/la-mamounia"
    assert out[0]["review_count"] == 1200
    # tolerant key variants + currency fallback
    assert out[1]["name"] == "Hotel Y" and out[1]["price"] == 200.0
    assert out[1]["currency"] == "USD"
    assert out[1]["booking_id"] is None


def test_summarize_currency_fallback():
    [out] = BookingService.summarize_hotels(
        [{"name": "H", "price": 90}], currency="EUR"
    )
    assert out["currency"] == "EUR"


# ---------------------------------------------------------------------------
# Deep-link
# ---------------------------------------------------------------------------


def test_booking_url_encodes_dates_and_rooms():
    url = BookingService.booking_url(
        "ma/la-mamounia", "2026-09-10", "2026-09-12", rooms=8
    )
    assert url.startswith("https://www.booking.com/hotel/ma/la-mamounia.html?")
    assert "checkin=2026-09-10" in url and "checkout=2026-09-12" in url
    # 8 single rooms → no_rooms=8 and group_adults=8 (1 person per room)
    assert "no_rooms=8" in url and "group_adults=8" in url


# ---------------------------------------------------------------------------
# Live (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _LIVE_KEY, reason="export BOOKING_RAPIDAPI_KEY for the live test")
def test_live_search():
    svc = BookingService(api_key=os.environ["BOOKING_RAPIDAPI_KEY"])
    raw = asyncio.run(svc.search_hotels("Marrakech", "2026-09-10", "2026-09-12"))
    hotels = BookingService.summarize_hotels(raw)
    assert isinstance(hotels, list)
