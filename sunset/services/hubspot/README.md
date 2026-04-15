# HubSpot Service

Async HubSpot CRM API client for OAuth, deals, properties, files, custom objects, and webhook subscriptions. Stateless — callers own token persistence.

## Setup

```python
from sunset.services import HubspotService

hubspot = HubspotService(
    client_id="your-client-id",
    client_secret="your-client-secret",
    # Optional — only needed for webhook subscription management:
    developer_api_key="dev-api-key",
    app_id="123456",
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HUBSPOT_CLIENT_ID` | Public app client ID |
| `HUBSPOT_CLIENT_SECRET` | Public app client secret |
| `HUBSPOT_DEVELOPER_API_KEY` | Developer API key (webhook management only) |
| `HUBSPOT_APP_ID` | Public app ID (webhook management only) |

## OAuth

The service is stateless — it returns tokens, you persist them.

```python
# 1. Redirect user to install URL
url = hubspot.get_install_url(
    redirect_uri="https://example.com/hubspot/oauth/callback",
    scopes=["crm.objects.deals.read", "crm.objects.deals.write", "files"],
)

# 2. Exchange code for tokens in your callback
tokens = await hubspot.exchange_code(code, redirect_uri)
# tokens.access_token, tokens.refresh_token, tokens.expires_at, tokens.hub_id
# → persist these somewhere (DB, encrypted)

# 3. When expired, refresh
new_tokens = await hubspot.refresh_token(tokens.refresh_token)

# 4. Inspect a token
info = await hubspot.get_access_token_info(tokens.access_token)
```

## Deals

```python
deal = await hubspot.get_deal(access_token, deal_id, properties=["dealname", "amount"])

deals = await hubspot.list_deals(access_token, stage="appointmentscheduled", limit=50)

deal = await hubspot.create_deal(access_token, name="New deal", stage_id="qualified")
deal = await hubspot.update_deal(access_token, deal_id, {"amount": "5000"})
deal = await hubspot.move_deal(access_token, deal_id, stage_id="closedwon")
await hubspot.delete_deal(access_token, deal_id)
```

## Properties & Pipelines

```python
properties = await hubspot.list_properties(access_token)  # object_type="deals" by default
groups = await hubspot.list_property_groups(access_token)
group = await hubspot.find_group_by_label(access_token, "Dossier ADV")

pipelines = await hubspot.get_pipelines(access_token)
```

## Files

```python
# Get a signed download URL + metadata
url, filename, content_type = await hubspot.get_file_signed_url(access_token, file_id)

# Download bytes directly (accepts file ID or URL)
data, content_type, filename = await hubspot.download_file(access_token, file_id)
```

## Custom Objects & Associations

```python
schemas = await hubspot.list_custom_object_schemas(access_token)

ids = await hubspot.get_associations(
    access_token, from_object_type="deals", from_id=deal_id,
    to_object_type="p12345_dossier_adv",
)

obj = await hubspot.get_custom_object(access_token, object_type, obj_id)
```

## Owners & Notes

```python
name = await hubspot.get_owner_name(access_token, owner_id)

note_id = await hubspot.create_note_on_deal(
    access_token, deal_id, body="Follow-up scheduled", pin=True,
)
```

## Webhook Subscriptions

Requires `developer_api_key` and `app_id` in the constructor.

```python
subs = await hubspot.list_subscriptions()
await hubspot.create_subscription("deal.propertyChange", property_name="dealstage")
await hubspot.delete_subscription(sub_id)
await hubspot.configure_webhook_url("https://example.com/hubspot/webhook")

# Or full setup with progress events (async generator)
async for event in hubspot.setup_property_subscriptions(
    properties=["dealstage", "kbis"],
    webhook_url="https://example.com/hubspot/webhook",
):
    print(event)  # {"step": ..., "current": N, "total": M, "status": ...}
```

## Error Handling

All methods raise `HubspotError` on HTTP failure.

```python
from sunset.services.hubspot import HubspotError

try:
    await hubspot.get_deal(access_token, deal_id)
except HubspotError as e:
    print(f"{e} (status: {e.status_code})")
```
