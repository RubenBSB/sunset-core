# BookingService

Hotel search + Booking.com deep-links, for logistics suggestions (find hotels
near an event, priced for the dates). Wraps the Booking.com RapidAPI actor
(`mtnrabi/booking-live-api`, Apify Standby mode).

Provider-only by design: it knows the Booking API, nothing about your domain.
Proximity ranking (geocoding hotels against a venue) stays with the caller, so
the service is a thin, testable HTTP client — same split as the flight adapters.

## Setup

### Env Vars

```yaml
secrets:
  RAPIDAPI_KEY: "your-rapidapi-key"   # from rapidapi.com/mtnrabi/api/booking-live-api
```

No extra dependency — uses the core `httpx`.

## Usage

```python
from sunset.services.booking import BookingService

booking = BookingService(api_key="...")  # host defaults to the actor's host

# 1. Priced properties for a destination + dates (raw API shape)
raw = await booking.search_hotels(
    "Marrakech", "2026-09-10", "2026-09-12", adults=1, currency="EUR"
)

# 2. Compact, UI-friendly shape (name, price, score, booking_id, link…)
hotels = BookingService.summarize_hotels(raw, currency="EUR")

# 3. Deep-link pre-filled with the dates + X single rooms
for h in hotels:
    if h["booking_id"]:
        url = BookingService.booking_url(
            h["booking_id"], "2026-09-10", "2026-09-12", rooms=8
        )  # no_rooms=8 & group_adults=8 → 8 single rooms on the live page
```

## API Reference

### `BookingService(api_key, *, host="booking-live-api.p.rapidapi.com", timeout=60.0)`

- `async search_hotels(destination, checkin, checkout, *, adults=1, currency="EUR", max_results=25) -> list[dict]`
  Raw property dicts. Retries once on a cold-start gateway status (502/503/504).
- `summarize_hotels(raw_hotels, *, currency=None) -> list[dict]` (static)
  `{name, price, currency, review_score, review_count, nights, link, booking_id}`
  per hotel; unnamed entries dropped.
- `booking_url(hotel_booking_id, checkin, checkout, *, rooms=1, adults_per_room=1, currency="EUR") -> str` (static)
  Deep-link with `no_rooms=rooms` and `group_adults=rooms × adults_per_room`.

### Errors

Raises `BookingError` (carries `.status_code`) on HTTP error or exhausted
retries; `status_code == 429` means the RapidAPI quota is spent.

## Limitations (intrinsic to this source)

- `/search` prices **one room** and exposes no inventory count → a group
  estimate is `price × rooms`, **not** a guarantee that X rooms are
  simultaneously bookable.
- Results carry **no coordinates** — geocode the hotel name for distance.
- ~25 "popular" properties per destination, not exhaustive.

For a real allotment guarantee + GPS-radius search, a contracted API (RateHawk
`allotment`, or Booking's official Demand API) would be the upgrade path.
