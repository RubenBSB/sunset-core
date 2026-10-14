"""SecretsService: no Secret Manager client unless a remote lookup needs one."""

import pytest

from sunset.services.secrets import SecretsService


@pytest.mark.parametrize("env", ["local", "test"])
def test_env_only_modes_never_build_a_client(monkeypatch, env):
    monkeypatch.setenv("ENV", env)
    monkeypatch.setenv("SOME_KEY", "from-env")
    monkeypatch.delenv("MISSING_KEY", raising=False)
    svc = SecretsService()

    assert svc._secret_client is None
    assert svc.get_secret("some-key") == "from-env"
    assert svc.get_secret("missing-key", default="") == ""
    with pytest.raises(ValueError, match="MISSING_KEY"):
        svc.get_secret("missing-key")
    assert svc._secret_client is None


def test_remote_lookup_builds_the_client_lazily(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("GCP_PROJECT_ID", "demo")
    monkeypatch.delenv("REMOTE_KEY", raising=False)
    svc = SecretsService()
    assert svc._secret_client is None

    class FakeClient:
        def access_secret_version(self, request):
            assert request["name"] == "projects/demo/secrets/remote-key/versions/latest"

            class Payload:
                data = b"from-gcp"

            class Response:
                payload = Payload()

            return Response()

    built = []
    monkeypatch.setattr(
        "sunset.services.secrets.secretmanager.SecretManagerServiceClient",
        lambda: built.append(1) or FakeClient(),
    )
    assert svc.get_secret("remote-key") == "from-gcp"
    assert built == [1]
