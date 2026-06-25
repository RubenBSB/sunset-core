"""Tests for EmailSendService — engine dispatch + graceful degradation.

Attributes are set directly on a constructed instance so the tests exercise the
send/dispatch logic without coupling to secret resolution (get_secret is cached).
"""

import asyncio

from sunset.services.email.send import EmailSendService


def _svc(**overrides) -> EmailSendService:
    svc = EmailSendService()
    svc.engine = overrides.get("engine", "resend")
    svc.from_email = overrides.get("from_email", "Acme <hi@acme.co>")
    svc.resend_api_key = overrides.get("resend_api_key", "")
    svc.sendgrid_api_key = overrides.get("sendgrid_api_key", "")
    return svc


def test_disabled_without_key():
    svc = _svc(engine="resend", resend_api_key="")
    assert svc.is_enabled() is False
    # No key → no send attempt, returns False (never raises).
    assert asyncio.run(svc.send(to="a@b.co", subject="s", html="<p>x</p>")) is False


def test_resend_dispatch(monkeypatch):
    import resend

    sent = {}
    monkeypatch.setattr(
        resend.Emails, "send", lambda payload: sent.update(payload) or {"id": "e1"}
    )
    svc = _svc(
        engine="resend", resend_api_key="re_test", from_email="Acme <hi@acme.co>"
    )
    ok = asyncio.run(svc.send(to="x@y.co", subject="Hi", html="<p>hello</p>"))
    assert ok is True
    assert sent["from"] == "Acme <hi@acme.co>"
    assert sent["to"] == ["x@y.co"]
    assert sent["subject"] == "Hi"
    assert sent["html"] == "<p>hello</p>"


def test_from_override(monkeypatch):
    import resend

    sent = {}
    monkeypatch.setattr(resend.Emails, "send", lambda payload: sent.update(payload))
    svc = _svc(engine="resend", resend_api_key="re_test")
    asyncio.run(
        svc.send(
            to="x@y.co", subject="Hi", html="<p>h</p>", from_email="Other <o@acme.co>"
        )
    )
    assert sent["from"] == "Other <o@acme.co>"


def test_unknown_engine_returns_false():
    # Unknown engine still reports enabled (resend key path), but the send raises
    # internally and is swallowed → False.
    svc = _svc(engine="mailgun", resend_api_key="re_test")
    assert asyncio.run(svc.send(to="x@y.co", subject="s", html="<p>x</p>")) is False
