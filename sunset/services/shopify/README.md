# Shopify Service

Async Shopify Admin API client for OAuth, product management, and webhook handling.

## Setup

```python
from sunset.services import ShopifyService

shopify = ShopifyService(api_key="your-api-key", api_secret="your-api-secret")
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SHOPIFY_API_KEY` | Shopify app API key (client ID) |
| `SHOPIFY_API_SECRET` | Shopify app API secret (client secret) |

## Usage

### OAuth

```python
# Generate install URL for a shop
url = shopify.get_install_url(
    shop_domain="myshop.myshopify.com",
    redirect_uri="https://example.com/callback",
    scopes=["read_products", "write_products"],
    nonce="random-state-string",
)

# Exchange authorization code for access token
result = await shopify.exchange_token("myshop.myshopify.com", code="auth-code")
access_token = result["access_token"]
```

### Products

```python
# Fetch all products (handles pagination automatically)
products = await shopify.fetch_all_products("myshop.myshopify.com", access_token)

# Fetch a single product
product = await shopify.fetch_product("myshop.myshopify.com", access_token, product_id=123)
```

### Webhooks

```python
# Register a single webhook
webhook = await shopify.register_webhook(
    "myshop.myshopify.com", access_token,
    topic="products/create",
    callback_url="https://example.com/webhooks/shopify",
)

# Register all product webhooks (create, update, delete)
webhooks = await shopify.register_product_webhooks(
    "myshop.myshopify.com", access_token,
    callback_url="https://example.com/webhooks/shopify",
)

# Verify incoming webhook signature
from sunset.services import ShopifyService

is_valid = ShopifyService.verify_webhook(
    data=request_body_bytes,
    hmac_header=request.headers["X-Shopify-Hmac-SHA256"],
    secret="your-api-secret",
)
```

## Error Handling

All API methods raise `ShopifyError` on failure. Rate limiting (HTTP 429) is handled automatically with up to 3 retries using the `Retry-After` header.

```python
from sunset.services.shopify import ShopifyError

try:
    products = await shopify.fetch_all_products(shop_domain, access_token)
except ShopifyError as e:
    print(f"API error: {e} (status: {e.status_code})")
```
