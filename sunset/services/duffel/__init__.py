"""Duffel API client — flight search / suggestion.

Duffel is the modern REST flight API we use to *suggest* flights matching an
arrival deadline (Amadeus Self-Service was decommissioned in July 2026). Search
is a single call: create an Offer Request with `return_offers=true` and read the
offers inline. Auth is a static bearer token — test vs live is decided by the
token prefix (`duffel_test_…` / `duffel_live_…`), same base URL either way.

Ticketing (placing the order) is intentionally out of scope: the agency books in
its GDS and pastes the PNR back. `summarize_offers` returns the same compact
shape as the other flight adapters, so callers stay provider-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any, Optional, Union

import httpx

logger = logging.getLogger(__name__)

__all__ = ["DuffelService", "DuffelError"]

_BASE_URL = "https://api.duffel.com"
_DEFAULT_VERSION = "v2"


def _as_date_str(value: Union[str, date, datetime]) -> str:
    """Normalise a date input to the YYYY-MM-DD Duffel expects."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _offer_price(offer: dict[str, Any]) -> float:
    """Numeric total for sorting; unparseable offers sink to the bottom."""
    try:
        return float(offer.get("total_amount"))
    except (TypeError, ValueError):
        return float("inf")


def _baggage_counts(segment: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """(checked, carry_on) allowance for a segment, or (None, None) when Duffel
    didn't return baggage info. 0 is a real answer (basic fare, no bag)."""
    paxs = segment.get("passengers") or []
    bags = paxs[0].get("baggages") if paxs else None
    if bags is None:
        return None, None
    checked = sum(b.get("quantity", 0) for b in bags if b.get("type") == "checked")
    carry = sum(b.get("quantity", 0) for b in bags if b.get("type") == "carry_on")
    return checked, carry


def _cond_allowed(cond: Any) -> Optional[bool]:
    """Whether a Duffel condition (refund/change before departure) is allowed.
    None when Duffel didn't price the condition (common — not 'not allowed')."""
    return cond.get("allowed") if isinstance(cond, dict) else None


def _min_opt(current: Optional[int], value: Optional[int]) -> Optional[int]:
    """Running minimum that treats None as 'unknown' (ignored unless both None)."""
    if value is None:
        return current
    return value if current is None else min(current, value)


class DuffelError(Exception):
    """Raised when a Duffel API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class DuffelService:
    """Async Duffel client (flight shopping)."""

    def __init__(
        self,
        access_token: str,
        *,
        version: str = _DEFAULT_VERSION,
        timeout: float = 30.0,
    ):
        self._token = access_token
        self._version = version
        self._timeout = timeout

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Duffel-Version": self._version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make a Duffel request, backing off once on a rate limit (429)."""
        url = f"{_BASE_URL}{path}"
        for _ in range(3):
            async with self._http_client() as client:
                resp = await client.request(
                    method, url, headers=self._headers(), params=params, json=json
                )
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "1"))
                logger.warning("Duffel rate limited, retrying after %ss", retry_after)
                await asyncio.sleep(retry_after)
                continue
            if resp.status_code >= 400:
                raise DuffelError(
                    f"Duffel API error: {resp.status_code} {resp.text}",
                    status_code=resp.status_code,
                )
            return resp.json()
        raise DuffelError("Duffel request failed after retries")

    async def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: Union[str, date, datetime],
        *,
        return_date: Optional[Union[str, date, datetime]] = None,
        adults: int = 1,
        max_offers: int = 10,
        cabin_class: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search flight offers via a one-shot Offer Request.

        `origin`/`destination` are IATA codes. Omitting `return_date` searches
        one-way; supplying it adds the return slice. Offers come back sorted by
        price ascending and trimmed to `max_offers`. Returns raw Duffel offers;
        use `summarize_offers` for the compact, UI-friendly shape.
        """
        slices = [
            {
                "origin": origin,
                "destination": destination,
                "departure_date": _as_date_str(departure_date),
            }
        ]
        if return_date is not None:
            slices.append(
                {
                    "origin": destination,
                    "destination": origin,
                    "departure_date": _as_date_str(return_date),
                }
            )
        data: dict[str, Any] = {
            "slices": slices,
            "passengers": [{"type": "adult"} for _ in range(max(1, adults))],
        }
        if cabin_class:
            data["cabin_class"] = cabin_class

        body = await self._request(
            "POST",
            "/air/offer_requests",
            params={"return_offers": "true"},
            json={"data": data},
        )
        offers = (body.get("data") or {}).get("offers") or []
        offers.sort(key=_offer_price)
        if max_offers is not None:
            offers = offers[:max_offers]
        return offers

    async def resolve_place(self, query: str) -> Optional[str]:
        """Resolve a free-text city/airport name to an IATA code via Duffel's
        place suggestions (`GET /places/suggestions`).

        Prefers a metropolitan *city* code (e.g. "Paris" → PAR, which searches
        all its airports) over a single airport; falls back to the first airport
        match. None when nothing matches (unknown town, a suburb, …).
        """
        q = (query or "").strip()
        if not q:
            return None
        body = await self._request(
            "GET", "/places/suggestions", params={"query": q}
        )
        places = body.get("data") or []
        for place in places:
            if place.get("type") == "city" and place.get("iata_code"):
                return place["iata_code"]
        for place in places:
            if place.get("iata_code"):
                return place["iata_code"]
        return None

    @staticmethod
    def summarize_offers(offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten raw Duffel offers into the compact shape the UI/booker uses.

        One entry per offer: price, currency, carriers, airlines (with name +
        logo), and per-slice legs with departure/arrival times and stop count —
        everything needed to pick a flight against an arrival deadline. Same
        shape across flight adapters.
        """
        out: list[dict[str, Any]] = []
        for offer in offers:
            carriers: set[str] = set()
            # Order-preserving, deduped airline list — owner (the marketing
            # airline) first, then any other operating carrier across segments.
            # Each carries Duffel's hosted logo so the UI can show it.
            airlines: list[dict[str, Any]] = []
            seen_airlines: set[str] = set()

            def _note_airline(carrier: dict[str, Any]) -> None:
                code = carrier.get("iata_code")
                if not code or code in seen_airlines:
                    return
                seen_airlines.add(code)
                airlines.append(
                    {
                        "iata": code,
                        "name": carrier.get("name"),
                        "logo": carrier.get("logo_symbol_url"),
                        # The airline's own site (its conditions-of-carriage
                        # page) — Duffel exposes no deep link to the flight
                        # itself, this is the only carrier URL in an offer.
                        "site_url": carrier.get("conditions_of_carriage_url"),
                    }
                )

            _note_airline(offer.get("owner") or {})
            # Fare attributes — cabin + fare basis come from the first segment;
            # baggage is the worst leg (min) so we never over-promise an allowance.
            cabin: Optional[str] = None
            fare_basis: Optional[str] = None
            bags_checked: Optional[int] = None
            bags_carry_on: Optional[int] = None
            itineraries: list[dict[str, Any]] = []
            for sl in offer.get("slices", []):
                segments = sl.get("segments", [])
                seg_out: list[dict[str, Any]] = []
                for seg in segments:
                    origin = seg.get("origin") or {}
                    dest = seg.get("destination") or {}
                    mc = seg.get("marketing_carrier") or {}
                    oc = seg.get("operating_carrier") or {}
                    carrier = mc.get("iata_code")
                    if carrier:
                        carriers.add(carrier)
                        _note_airline(mc)
                    pax0 = (seg.get("passengers") or [{}])[0]
                    if cabin is None:
                        cabin = pax0.get("cabin_class_marketing_name") or pax0.get(
                            "cabin_class"
                        )
                    if fare_basis is None:
                        fare_basis = pax0.get("fare_basis_code")
                    checked, carry = _baggage_counts(seg)
                    bags_checked = _min_opt(bags_checked, checked)
                    bags_carry_on = _min_opt(bags_carry_on, carry)
                    seg_out.append(
                        {
                            "from": origin.get("iata_code"),
                            "from_terminal": seg.get("origin_terminal"),
                            "departure_at": seg.get("departing_at"),
                            "to": dest.get("iata_code"),
                            "to_terminal": seg.get("destination_terminal"),
                            "arrival_at": seg.get("arriving_at"),
                            "carrier": carrier,
                            "flight_number": seg.get(
                                "marketing_carrier_flight_number"
                            ),
                            "aircraft": (seg.get("aircraft") or {}).get("name"),
                            # Codeshare: the airline actually flying it, when it
                            # differs from the marketing carrier on the ticket.
                            "operated_by": (
                                oc.get("name")
                                if oc.get("iata_code")
                                and oc.get("iata_code") != carrier
                                else None
                            ),
                        }
                    )
                itineraries.append(
                    {
                        "duration": sl.get("duration"),
                        "stops": max(0, len(segments) - 1),
                        "departure_at": seg_out[0]["departure_at"] if seg_out else None,
                        "arrival_at": seg_out[-1]["arrival_at"] if seg_out else None,
                        "segments": seg_out,
                    }
                )
            conditions = offer.get("conditions") or {}
            out.append(
                {
                    "id": offer.get("id"),
                    "price": offer.get("total_amount"),
                    "currency": offer.get("total_currency"),
                    "one_way": len(offer.get("slices", [])) <= 1,
                    "carriers": sorted(carriers),
                    "airlines": airlines,
                    "cabin": cabin,
                    "fare_basis": fare_basis,
                    "baggage": {"checked": bags_checked, "carry_on": bags_carry_on},
                    "refundable": _cond_allowed(conditions.get("refund_before_departure")),
                    "changeable": _cond_allowed(conditions.get("change_before_departure")),
                    "emissions_kg": (
                        str(offer["total_emissions_kg"])
                        if offer.get("total_emissions_kg") is not None
                        else None
                    ),
                    "itineraries": itineraries,
                }
            )
        return out
