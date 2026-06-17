# Duffel Service

Async **Duffel** client for flight search / suggestion.

Duffel is the modern REST flight API we use to **suggest** flights matching an
arrival deadline. (It replaces Amadeus Self-Service, which was decommissioned in
July 2026.) Issuing the order/ticket is intentionally **not** part of this
service — the agency books in its GDS and pastes the PNR back.

## Setup

```python
from sunset.services import DuffelService

duffel = DuffelService(access_token="duffel_test_...")  # or duffel_live_...
```

Test vs live is decided by the token prefix; the base URL is the same.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DUFFEL_API_TOKEN` | Duffel access token (`duffel_test_…` or `duffel_live_…`) |

Create a token at <https://app.duffel.com> → Developers → Access tokens. Test
tokens are free and return synthetic "Duffel Airways" offers in test mode.

## Usage

```python
# One-way search (IATA codes). Offers come back sorted by price, trimmed.
offers = await duffel.search_flights(
    origin="MAD",
    destination="NCE",
    departure_date="2026-06-13",   # str, date, or datetime
    adults=2,
    max_offers=5,
    cabin_class="economy",         # economy | premium_economy | business | first
)

# Round-trip: add return_date (appends the return slice).
offers = await duffel.search_flights("CDG", "FCO", "2026-06-14", return_date="2026-06-16")

# Compact, UI-friendly shape (price, carriers, per-leg times + stops) — the same
# shape across flight adapters, so callers stay provider-agnostic.
flights = DuffelService.summarize_offers(offers)
# [{"id": "off_...", "price": "182.30", "currency": "EUR", "one_way": True,
#   "carriers": ["IB"],
#   "airlines": [{"iata": "IB", "name": "Iberia",
#                 "logo": "https://assets.duffel.com/img/airlines/.../IB.svg"}],
#   "itineraries": [{"duration": "PT2H5M", "stops": 0,
#     "departure_at": "2026-06-13T08:20:00", "arrival_at": "2026-06-13T10:25:00",
#     "segments": [...]}]}]
# `airlines` is owner-first, deduped, and carries Duffel's hosted logo URLs.
```

Pick a flight whose itinerary `arrival_at` is on or before the required-arrival
deadline (back-calculated from the event's balance time).

```python
# Resolve a free-text city/airport name → IATA (prefers a metro city code that
# covers all the city's airports). None when nothing matches.
iata = await duffel.resolve_place("Paris")   # "PAR"
iata = await duffel.resolve_place("Athens")  # "ATH"
iata = await duffel.resolve_place("Arcueil") # None (a suburb, no airport)
```

## Error Handling

All API calls raise `DuffelError` on failure. Rate limiting (HTTP 429) is retried
automatically using the `Retry-After` header.

```python
from sunset.services.duffel import DuffelError

try:
    offers = await duffel.search_flights("MAD", "NCE", "2026-06-13")
except DuffelError as e:
    print(f"API error: {e} (status: {e.status_code})")
```
