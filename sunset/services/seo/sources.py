"""SSRF-hardened validation of model-supplied source URLs.

LLM-generated articles cite URLs the model chose — untrusted input. Probing
them from application infrastructure (typically inside a VPC, next to Redis,
databases and cloud metadata endpoints) is a classic SSRF vector, so:

- https on port 443 only, no credentials in the URL
- literal IPs and cloud-metadata hosts refused outright
- DNS resolution required; if ANY returned address is private, loopback,
  link-local, multicast, reserved or unspecified (IPv4, IPv6 or IPv4-mapped),
  the host is rejected — mixed answers included (DNS-rebinding style)
- redirects followed manually (max 3), every hop re-validated
- response bodies are never read (HEAD, or a streamed GET closed immediately)
"""

import asyncio
import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

# Cloud metadata endpoints — never probe, whatever DNS says.
BLOCKED_HOSTS = {"metadata.google.internal", "metadata.goog", "169.254.169.254"}
MAX_REDIRECTS = 3
_MD_LINK = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")


def ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def host_resolves_public(host: str) -> bool:
    """Resolve and require EVERY returned address to be public — a single
    private A/AAAA record (DNS rebinding style) rejects the host."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    addrs = {info[4][0] for info in infos}
    if not addrs:
        return False
    return all(ip_is_public(ipaddress.ip_address(a)) for a in addrs)


def url_allowed(url: str) -> bool:
    """Static + DNS checks for a model-supplied URL before any request."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https" or parsed.username or parsed.password:
        return False
    if parsed.port not in (None, 443):
        return False
    host = (parsed.hostname or "").strip("[]").lower()
    if not host or host in BLOCKED_HOSTS:
        return False
    try:
        return ip_is_public(ipaddress.ip_address(host))  # literal IP
    except ValueError:
        pass  # hostname — resolve it
    return host_resolves_public(host)


async def probe_source(client: httpx.AsyncClient, url: str) -> str | None:
    """Liveness probe with manual redirects, re-validating every hop.
    Bodies are never read. 404/410 and blocked/looping URLs drop the source;
    anything else — including 403 bot-walls — keeps it: the citation exists,
    and important claims still go through human review before publication."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not await asyncio.to_thread(url_allowed, current):
            return None
        try:
            resp = await client.head(current, follow_redirects=False)
            if resp.status_code in (405, 501):
                async with client.stream("GET", current, follow_redirects=False) as r:
                    resp = r
        except Exception:  # noqa: BLE001 — network hiccup ≠ dead source
            return url
        if resp.is_redirect:
            location = resp.headers.get("location")
            if not location:
                return None
            current = urljoin(current, location)
            continue
        return None if resp.status_code in (404, 410) else url
    return None  # redirect loop / too many hops


async def ensure_sources(
    sources: list[str] | None,
    content: str,
    *,
    exclude_prefix: str | None = None,
    limit: int = 15,
    user_agent: str = "SunsetBot/1.0",
) -> list[str]:
    """Return a validated source list for an AI-written article.

    Models with built-in web search sometimes cite links in prose without
    filling the structured sources list — recover them from the markdown,
    then apply the SSRF guard and drop dead links (hard 404/410 only).

    Args:
        sources: the structured source list from the model, possibly empty.
        content: article markdown, scanned for links when sources is empty.
        exclude_prefix: drop self-links (e.g. "https://example.com").
        limit: maximum number of sources kept.
        user_agent: UA header for the probe.
    """
    candidates = list(sources) if sources else []
    if not candidates:
        candidates = _MD_LINK.findall(content or "")
    cleaned: list[str] = []
    for url in candidates:
        url = url.rstrip(".,;")
        if exclude_prefix and url.startswith(exclude_prefix):
            continue
        if url_allowed(url) and url not in cleaned:
            cleaned.append(url)
    cleaned = cleaned[:limit]

    async with httpx.AsyncClient(
        timeout=5, headers={"User-Agent": user_agent}
    ) as client:
        checked = await asyncio.gather(*(probe_source(client, u) for u in cleaned))
    return [u for u in checked if u]
