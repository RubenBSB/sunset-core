"""Tests for DuffelService — request shape, price sort/trim, retry, and the
offer-summarisation helper. HTTP is faked at the httpx.AsyncClient seam so the
real request/parse paths run without a network."""

import asyncio
import os
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from sunset.services.duffel import DuffelError, DuffelService

OFFER_REQUESTS_PATH = "/air/offer_requests"

# Live test runs only when a real token is exported; skipped otherwise.
_LIVE_TOKEN = bool(os.getenv("DUFFEL_API_TOKEN"))


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

    async def request(self, method, url, **kwargs):
        return self._handler(method, url, kwargs)


def _patch(handler):
    return patch(
        "sunset.services.duffel.httpx.AsyncClient",
        return_value=FakeClient(handler),
    )


def _svc():
    return DuffelService(access_token="duffel_test_abc")


def _offer(offer_id, amount, *, slices=None):
    return {
        "id": offer_id,
        "total_amount": amount,
        "total_currency": "EUR",
        "slices": slices
        or [
            {
                "duration": "PT2H5M",
                "segments": [
                    {
                        "origin": {"iata_code": "MAD"},
                        "destination": {"iata_code": "NCE"},
                        "departing_at": "2026-06-13T08:20:00",
                        "arriving_at": "2026-06-13T10:25:00",
                        "marketing_carrier": {"iata_code": "IB"},
                        "marketing_carrier_flight_number": "1234",
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_search_builds_body_and_headers():
    seen = {}

    def handler(method, url, kwargs):
        seen["method"] = method
        seen["params"] = kwargs.get("params")
        seen["headers"] = kwargs.get("headers")
        seen["json"] = kwargs.get("json")
        return FakeResp(json_body={"data": {"offers": [_offer("o1", "100.00")]}})

    svc = _svc()
    with _patch(handler):
        offers = asyncio.run(
            svc.search_flights(
                "MAD", "NCE", "2026-06-13", adults=2, cabin_class="economy"
            )
        )

    assert len(offers) == 1
    assert seen["method"] == "POST"
    assert seen["params"] == {"return_offers": "true"}
    assert seen["headers"]["Authorization"] == "Bearer duffel_test_abc"
    assert seen["headers"]["Duffel-Version"] == "v2"
    data = seen["json"]["data"]
    assert data["slices"] == [
        {"origin": "MAD", "destination": "NCE", "departure_date": "2026-06-13"}
    ]
    assert data["passengers"] == [{"type": "adult"}, {"type": "adult"}]
    assert data["cabin_class"] == "economy"


def test_round_trip_adds_return_slice():
    seen = {}

    def handler(method, url, kwargs):
        seen["json"] = kwargs.get("json")
        return FakeResp(json_body={"data": {"offers": []}})

    svc = _svc()
    with _patch(handler):
        asyncio.run(
            svc.search_flights(
                "CDG",
                "FCO",
                date(2026, 6, 14),
                return_date=datetime(2026, 6, 16, 9, 0),
            )
        )
    slices = seen["json"]["data"]["slices"]
    assert len(slices) == 2
    assert slices[0] == {
        "origin": "CDG",
        "destination": "FCO",
        "departure_date": "2026-06-14",
    }
    assert slices[1] == {
        "origin": "FCO",
        "destination": "CDG",
        "departure_date": "2026-06-16",
    }


def test_offers_sorted_by_price_and_trimmed():
    def handler(method, url, kwargs):
        return FakeResp(
            json_body={
                "data": {
                    "offers": [
                        _offer("expensive", "300.00"),
                        _offer("cheap", "90.00"),
                        _offer("mid", "150.00"),
                    ]
                }
            }
        )

    svc = _svc()
    with _patch(handler):
        offers = asyncio.run(
            svc.search_flights("MAD", "NCE", "2026-06-13", max_offers=2)
        )
    assert [o["id"] for o in offers] == ["cheap", "mid"]


def test_unparseable_price_sinks_last():
    def handler(method, url, kwargs):
        return FakeResp(
            json_body={
                "data": {
                    "offers": [
                        _offer("broken", None),
                        _offer("ok", "120.00"),
                    ]
                }
            }
        )

    svc = _svc()
    with _patch(handler):
        offers = asyncio.run(svc.search_flights("MAD", "NCE", "2026-06-13"))
    assert [o["id"] for o in offers] == ["ok", "broken"]


# ---------------------------------------------------------------------------
# Errors + retry
# ---------------------------------------------------------------------------


def test_429_backs_off_then_succeeds():
    state = {"n": 0}

    def handler(method, url, kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return FakeResp(status_code=429, headers={"Retry-After": "0"})
        return FakeResp(json_body={"data": {"offers": []}})

    svc = _svc()
    with _patch(handler):
        offers = asyncio.run(svc.search_flights("MAD", "NCE", "2026-06-13"))
    assert offers == []
    assert state["n"] == 2


def test_api_error_raises():
    def handler(method, url, kwargs):
        return FakeResp(status_code=422, text="invalid_origin")

    svc = _svc()
    with _patch(handler):
        with pytest.raises(DuffelError) as exc:
            asyncio.run(svc.search_flights("XXX", "NCE", "2026-06-13"))
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# Summarisation (pure)
# ---------------------------------------------------------------------------


def test_summarize_offers():
    raw = [
        _offer(
            "off_1",
            "182.30",
            slices=[
                {
                    "duration": "PT4H10M",
                    "segments": [
                        {
                            "origin": {"iata_code": "MAD"},
                            "destination": {"iata_code": "BCN"},
                            "departing_at": "2026-06-13T08:20:00",
                            "arriving_at": "2026-06-13T09:35:00",
                            "marketing_carrier": {"iata_code": "IB"},
                            "marketing_carrier_flight_number": "1234",
                        },
                        {
                            "origin": {"iata_code": "BCN"},
                            "destination": {"iata_code": "NCE"},
                            "departing_at": "2026-06-13T11:00:00",
                            "arriving_at": "2026-06-13T12:30:00",
                            "marketing_carrier": {"iata_code": "VY"},
                            "marketing_carrier_flight_number": "8000",
                        },
                    ],
                }
            ],
        )
    ]
    out = DuffelService.summarize_offers(raw)
    assert len(out) == 1
    o = out[0]
    assert o["price"] == "182.30"
    assert o["currency"] == "EUR"
    assert o["one_way"] is True
    assert o["carriers"] == ["IB", "VY"]
    # Airlines preserve first-seen order (no owner here → segment order).
    assert [a["iata"] for a in o["airlines"]] == ["IB", "VY"]
    itin = o["itineraries"][0]
    assert itin["stops"] == 1
    assert itin["departure_at"] == "2026-06-13T08:20:00"
    assert itin["arrival_at"] == "2026-06-13T12:30:00"
    assert itin["segments"][0]["from"] == "MAD"
    assert itin["segments"][1]["to"] == "NCE"


def test_summarize_includes_airline_logos():
    """Owner airline leads the `airlines` list and carries Duffel's logo URL;
    a different operating carrier on a later segment is appended once."""
    raw = [
        {
            "id": "off_1",
            "total_amount": "100.00",
            "total_currency": "EUR",
            "total_emissions_kg": "120",
            "conditions": {
                "refund_before_departure": {"allowed": False},
                "change_before_departure": {"allowed": True},
            },
            "owner": {
                "iata_code": "BA",
                "name": "British Airways",
                "logo_symbol_url": "https://assets.duffel.com/img/airlines/.../BA.svg",
                "conditions_of_carriage_url": "https://www.britishairways.com/legal",
            },
            "slices": [
                {
                    "duration": "PT3H",
                    "segments": [
                        {
                            "origin": {"iata_code": "LHR"},
                            "origin_terminal": "5",
                            "destination": {"iata_code": "MAD"},
                            "destination_terminal": "4S",
                            "departing_at": "2026-07-01T08:00:00",
                            "arriving_at": "2026-07-01T11:00:00",
                            "marketing_carrier": {
                                "iata_code": "IB",
                                "name": "Iberia",
                                "logo_symbol_url": "https://assets.duffel.com/img/airlines/.../IB.svg",
                            },
                            "operating_carrier": {"iata_code": "I2", "name": "Iberia Express"},
                            "marketing_carrier_flight_number": "1",
                            "aircraft": {"name": "Airbus A320"},
                            "passengers": [
                                {
                                    "cabin_class": "economy",
                                    "cabin_class_marketing_name": "Economy Basic",
                                    "fare_basis_code": "ABATO",
                                    "baggages": [
                                        {"type": "checked", "quantity": 0},
                                        {"type": "carry_on", "quantity": 1},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    [o] = DuffelService.summarize_offers(raw)
    assert [a["iata"] for a in o["airlines"]] == ["BA", "IB"]  # owner first
    ba = o["airlines"][0]
    assert ba["name"] == "British Airways"
    assert ba["logo"].endswith("/BA.svg")
    assert ba["site_url"] == "https://www.britishairways.com/legal"
    assert o["emissions_kg"] == "120"
    assert o["cabin"] == "Economy Basic"
    assert o["fare_basis"] == "ABATO"
    assert o["baggage"] == {"checked": 0, "carry_on": 1}
    assert o["refundable"] is False and o["changeable"] is True
    seg = o["itineraries"][0]["segments"][0]
    assert seg["aircraft"] == "Airbus A320"
    assert seg["from_terminal"] == "5" and seg["to_terminal"] == "4S"
    assert seg["operated_by"] == "Iberia Express"


def test_summarize_handles_empty():
    assert DuffelService.summarize_offers([]) == []


# ---------------------------------------------------------------------------
# Place resolution
# ---------------------------------------------------------------------------


def _places_handler(data):
    def handler(method, url, kwargs):
        assert "/places/suggestions" in url
        return FakeResp(json_body={"data": data})

    return handler


def test_resolve_place_prefers_city():
    data = [
        {"type": "airport", "iata_code": "CDG", "name": "Charles de Gaulle"},
        {"type": "city", "iata_code": "PAR", "name": "Paris"},
        {"type": "airport", "iata_code": "ORY", "name": "Orly"},
    ]
    svc = _svc()
    with _patch(_places_handler(data)):
        assert asyncio.run(svc.resolve_place("Paris")) == "PAR"


def test_resolve_place_falls_back_to_airport():
    data = [{"type": "airport", "iata_code": "ATH", "name": "Athens"}]
    svc = _svc()
    with _patch(_places_handler(data)):
        assert asyncio.run(svc.resolve_place("Athens")) == "ATH"


def test_resolve_place_none_when_empty():
    svc = _svc()
    with _patch(_places_handler([])):
        assert asyncio.run(svc.resolve_place("Arcueil")) is None
    assert asyncio.run(svc.resolve_place("")) is None  # no request made


# ---------------------------------------------------------------------------
# Live smoke test (skipped without a token)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    not _LIVE_TOKEN,
    reason="export DUFFEL_API_TOKEN to run the live test",
)
def test_live_search():
    """Real round-trip against Duffel (test token → Duffel Airways offers)."""
    svc = DuffelService(access_token=os.environ["DUFFEL_API_TOKEN"])
    departure = (date.today() + timedelta(days=30)).isoformat()
    offers = asyncio.run(svc.search_flights("LHR", "JFK", departure, max_offers=3))
    assert isinstance(offers, list)
    summary = DuffelService.summarize_offers(offers)
    for o in summary:
        assert "price" in o and "itineraries" in o
