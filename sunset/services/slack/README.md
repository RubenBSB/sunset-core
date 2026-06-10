# Slack Service

Async Slack API client for OAuth v2 + Web API (private channels, invites, messages, user lookup). Stateless — callers own token persistence.

## Setup

```python
from sunset.services import SlackService

slack = SlackService(
    client_id="your-client-id",
    client_secret="your-client-secret",
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SLACK_CLIENT_ID` | Slack app client ID |
| `SLACK_CLIENT_SECRET` | Slack app client secret |

The service is stateless. Provide IDs at construction; persist tokens in your DB.

## OAuth v2 (multi-workspace)

```python
# 1. Build the install URL and redirect the user
url = slack.get_install_url(
    redirect_uri="https://example.com/slack/oauth/callback",
    state="csrf-or-org-id",
)

# 2. Exchange the code in your callback
tokens = await slack.exchange_code(code, redirect_uri="https://example.com/slack/oauth/callback")
# tokens.access_token (xoxb-…), tokens.team_id, tokens.team_name,
# tokens.bot_user_id, tokens.scope, tokens.app_id, tokens.authed_user_id
# → persist these keyed on whatever tenant concept your app has.

# 3. Verify a token anytime
info = await slack.auth_test(tokens.access_token)
```

Default bot scopes: `groups:write`, `channels:manage`, `chat:write`, `users:read`, `users:read.email`. Override with `scopes=[...]` if you need more.

## Users

```python
user = await slack.lookup_user_by_email(token, "alice@inspiration.fr")
# Returns SlackUser(id, email, name, real_name) — or None if the email isn't in
# the workspace (rather than raising). Other errors still raise SlackError.
```

## Conversations

```python
ch = await slack.create_conversation(token, "Mariage Durand X3F4", is_private=True)
# ch.id, ch.name, ch.is_private — channel name is auto-slugified to Slack rules
# (lowercase, hyphens, ≤80 chars).

await slack.invite_to_conversation(token, ch.id, ["U02ABC", "U02DEF"])
# Silently no-ops on "already_in_channel" / "cant_invite_self" so retries
# are idempotent.

ch = await slack.rename_conversation(token, ch.id, "new-name")
# Name is auto-slugified like create_conversation. Requires the bot to be a
# member of the channel.

await slack.archive_conversation(token, ch.id)
```

## Messages

```python
msg = await slack.post_message(token, ch.id, "Hello team 🌅")
# msg.channel, msg.ts

# Block kit also supported:
await slack.post_message(token, ch.id, "Fallback text", blocks=[
    {"type": "section", "text": {"type": "mrkdwn", "text": "*Hello* team"}},
])
```

## Errors

All methods raise `SlackError` on HTTP failure or `ok: false`. The exception carries the original Slack error code:

```python
from sunset.services.slack import SlackError

try:
    await slack.create_conversation(token, "Mariage")
except SlackError as exc:
    if exc.slack_error == "name_taken":
        ...
    else:
        raise
```

Common Slack error codes you'll handle:

- `name_taken` — channel name already exists
- `users_not_found` — `lookup_user_by_email` returns `None` instead of raising
- `not_in_channel` / `channel_not_found` — the bot was removed or the channel was archived
- `missing_scope` — your install needs to be re-authorized with a broader scope set
