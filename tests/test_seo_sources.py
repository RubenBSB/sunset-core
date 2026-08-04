"""SSRF guard tests for the article source validator.

The URLs come from the LLM (untrusted) and are probed from inside the VPC —
every private/loopback/link-local/metadata target must be rejected, including
after redirects and with DNS answers pointing at private space (rebinding).
"""

import ipaddress
import socket
from unittest.mock import patch

import httpx
import pytest

from sunset.services.seo.sources import (
    ip_is_public as _ip_is_public,
    probe_source as _probe_source,
    url_allowed as _url_allowed,
)


def _fake_resolver(mapping):
    def getaddrinfo(host, port, **kwargs):
        ips = mapping.get(host)
        if ips is None:
            raise OSError(f"no such host: {host}")
        family = socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]

    return getaddrinfo


# ── static URL rules ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/report",  # not https
        "https://user:pass@example.com/",  # credentials
        "https://example.com:8443/",  # non-standard port
        "https://localhost/admin",
        "https://127.0.0.1/",
        "https://[::1]/",
        "https://169.254.169.254/computeMetadata/v1/",
        "https://metadata.google.internal/computeMetadata/v1/",
        "https://10.0.0.8/internal",
        "https://192.168.1.20/",
        "https://172.16.0.1/",
        "https://[fd00::1]/",  # unique-local IPv6
        "https://[::ffff:10.0.0.8]/",  # IPv4-mapped private
    ],
)
def test_url_rejected_statically(url):
    assert _url_allowed(url) is False


def test_public_hostname_allowed():
    with patch("socket.getaddrinfo", _fake_resolver({"example.com": ["93.184.216.34"]})):
        assert _url_allowed("https://example.com/article") is True


def test_hostname_resolving_private_rejected():
    with patch("socket.getaddrinfo", _fake_resolver({"evil.example": ["10.0.0.5"]})):
        assert _url_allowed("https://evil.example/") is False


def test_dns_rebinding_mixed_answers_rejected():
    # one public + one private A record — must reject the whole host
    with patch(
        "socket.getaddrinfo",
        _fake_resolver({"rebind.example": ["93.184.216.34", "169.254.169.254"]}),
    ):
        assert _url_allowed("https://rebind.example/") is False


def test_unresolvable_hostname_rejected():
    with patch("socket.getaddrinfo", _fake_resolver({})):
        assert _url_allowed("https://nxdomain.example/") is False


@pytest.mark.parametrize(
    "ip,public",
    [
        ("8.8.8.8", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("169.254.169.254", False),
        ("224.0.0.1", False),
        ("::1", False),
        ("fe80::1", False),
        ("2606:4700::1111", True),
    ],
)
def test_ip_classification(ip, public):
    assert _ip_is_public(ipaddress.ip_address(ip)) is public


# ── redirect handling ───────────────────────────────────────────────────────


def _mock_transport(routes):
    def handler(request: httpx.Request) -> httpx.Response:
        return routes[str(request.url)]

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_redirect_to_private_ip_dropped():
    routes = {
        "https://ok.example/paper": httpx.Response(
            302, headers={"location": "https://169.254.169.254/latest/meta-data/"}
        ),
    }
    with patch("socket.getaddrinfo", _fake_resolver({"ok.example": ["93.184.216.34"]})):
        async with httpx.AsyncClient(transport=_mock_transport(routes)) as client:
            assert await _probe_source(client, "https://ok.example/paper") is None


@pytest.mark.asyncio
async def test_redirect_loop_dropped():
    routes = {
        "https://loop.example/": httpx.Response(
            301, headers={"location": "https://loop.example/"}
        ),
    }
    with patch("socket.getaddrinfo", _fake_resolver({"loop.example": ["93.184.216.34"]})):
        async with httpx.AsyncClient(transport=_mock_transport(routes)) as client:
            assert await _probe_source(client, "https://loop.example/") is None


@pytest.mark.asyncio
async def test_dead_source_dropped_live_source_kept():
    routes = {
        "https://dead.example/gone": httpx.Response(404),
        "https://alive.example/study": httpx.Response(200),
        "https://paywall.example/report": httpx.Response(403),
    }
    resolver = _fake_resolver(
        {h: ["93.184.216.34"] for h in ("dead.example", "alive.example", "paywall.example")}
    )
    with patch("socket.getaddrinfo", resolver):
        async with httpx.AsyncClient(transport=_mock_transport(routes)) as client:
            assert await _probe_source(client, "https://dead.example/gone") is None
            assert (
                await _probe_source(client, "https://alive.example/study")
                == "https://alive.example/study"
            )
            # 403 bot-walls keep the citation
            assert (
                await _probe_source(client, "https://paywall.example/report")
                == "https://paywall.example/report"
            )
