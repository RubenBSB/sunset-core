"""Rate limiting via Redis — FastAPI decorator and middleware."""

import functools
import inspect
import logging
from typing import Callable

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

RATE_LIMITED_ATTR = "_sunset_rate_limited"

# Number of proxies between the app and the internet that we trust to set
# X-Forwarded-For correctly. 1 is right for Cloud Run without an external
# load balancer (the Google Front End appends the real client IP as the last
# XFF entry). Bump to 2 if you put an external HTTPS LB / Cloud Armor in front.
DEFAULT_TRUSTED_PROXIES = 1


def get_client_ip(
    request: Request, trusted_proxies: int = DEFAULT_TRUSTED_PROXIES
) -> str:
    """Extract the real client IP, resistant to X-Forwarded-For spoofing.

    XFF is a comma-separated list. Anything the client sent is preserved on
    the left; our infra appends the real peer on the right. Only the last
    `trusted_proxies` entries are trustworthy — everything before that is
    attacker-controlled.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            idx = max(0, len(parts) - trusted_proxies)
            return parts[idx]
    if request.client:
        return request.client.host
    return "unknown"


def get_client_identity(
    request: Request, trusted_proxies: int = DEFAULT_TRUSTED_PROXIES
) -> str:
    """Identity key for rate limiting. Prefers a widget/session header over IP,
    since IP collapses all users behind a NAT or proxy into one bucket.
    """
    session_id = request.headers.get("x-session-id")
    if session_id:
        return f"s:{session_id[:64]}"
    return f"ip:{get_client_ip(request, trusted_proxies)}"


def rate_limit(limit: int = 100, window: int = 60) -> Callable:
    """
    Decorator that enforces per-IP rate limiting via Redis.

    Args:
        limit: Max requests allowed within the window.
        window: Window size in seconds.

    Usage:
        @router.post("/expensive")
        @rate_limit(limit=10, window=60)
        async def expensive():
            ...
    """

    def decorator(func: Callable) -> Callable:
        # Detect if the endpoint already declares a Request parameter
        sig = inspect.signature(func)
        existing_request_param = None
        for name, param in sig.parameters.items():
            if param.annotation is Request:
                existing_request_param = name
                break

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get Request from existing param or from the injected one
            request: Request | None = kwargs.get(
                existing_request_param or "_rl_request"
            )
            if request is not None:
                try:
                    redis_svc = getattr(request.app.state, "redis", None)
                    if redis_svc is None or redis_svc._client is None:
                        kwargs.pop("_rl_request", None)
                        return await func(*args, **kwargs)
                    client = redis_svc.client
                    identity = get_client_identity(request)
                    key = f"rl:{identity}:{request.url.path}"
                    pipe = client.pipeline()
                    pipe.incr(key)
                    pipe.expire(key, window, nx=True)
                    count, _ = await pipe.execute()
                    if count > limit:
                        ttl = await client.ttl(key)
                        raise HTTPException(
                            status_code=429,
                            detail="Too many requests",
                            headers={"Retry-After": str(max(ttl, 1))},
                        )
                except HTTPException:
                    raise
                except Exception:
                    logger.warning(
                        "Rate limit check failed, allowing request", exc_info=True
                    )

            # Remove injected param before calling the original function
            kwargs.pop("_rl_request", None)
            return await func(*args, **kwargs)

        # Only add a hidden Request param if the endpoint doesn't already have one
        if existing_request_param is None:
            params = list(sig.parameters.values())
            params.append(
                inspect.Parameter(
                    "_rl_request",
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=Request,
                )
            )
            wrapper.__signature__ = sig.replace(parameters=params)
        setattr(wrapper, RATE_LIMITED_ATTR, True)

        return wrapper

    return decorator


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Global rate limit safety net. Applies to all routes that don't
    already have the @rate_limit decorator.

    Usage:
        app.add_middleware(RateLimitMiddleware, limit=200, window=60)
    """

    def __init__(self, app, limit: int = 200, window: int = 60):
        super().__init__(app)
        self.limit = limit
        self.window = window

    async def dispatch(self, request: Request, call_next):
        # Skip if the matched route already has a per-route rate limit
        route = request.scope.get("route")
        if route and getattr(
            getattr(route, "endpoint", None), RATE_LIMITED_ATTR, False
        ):
            return await call_next(request)

        try:
            redis_svc = getattr(request.app.state, "redis", None)
            if redis_svc is None or redis_svc._client is None:
                return await call_next(request)
            client = redis_svc.client
            identity = get_client_identity(request)
            key = f"rl:global:{identity}"
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window, nx=True)
            count, _ = await pipe.execute()
            if count > self.limit:
                ttl = await client.ttl(key)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests"},
                    headers={"Retry-After": str(max(ttl, 1))},
                )
        except Exception:
            logger.warning(
                "Global rate limit check failed, allowing request", exc_info=True
            )

        return await call_next(request)
