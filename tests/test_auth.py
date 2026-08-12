from __future__ import annotations

import json

from vctx.app.auth import OpenRouterAuth
from vctx.net import NetRequest, NetResponse


class MemoryKeyring:
    priority = 1.0

    def __init__(self) -> None:
        self.value: str | None = None

    def get_password(self, service: str, username: str) -> str | None:
        assert (service, username) == ("vctx", "openrouter")
        return self.value

    def set_password(self, service: str, username: str, password: str) -> None:
        assert (service, username) == ("vctx", "openrouter")
        self.value = password

    def delete_password(self, service: str, username: str) -> None:
        assert (service, username) == ("vctx", "openrouter")
        self.value = None


class AuthNet:
    def __init__(self) -> None:
        self.request_value: NetRequest | None = None

    def request(self, request: NetRequest) -> NetResponse:
        self.request_value = request
        return NetResponse(
            url=request.url,
            status_code=200,
            body=json.dumps({"key": "stored-secret"}).encode(),
        )


def test_openrouter_pkce_exchange_status_and_logout() -> None:
    keyring, net = MemoryKeyring(), AuthNet()
    auth = OpenRouterAuth(keyring=keyring, net=net)
    session = auth.begin("http://127.0.0.1:43123/callback", verifier="v" * 64)

    assert "code_challenge_method=S256" in session.authorization_url
    assert "callback_url=http%3A%2F%2F127.0.0.1%3A43123%2Fcallback" in session.authorization_url
    auth.finish(session, code="returned-code")
    assert auth.status().authenticated is True
    assert net.request_value is not None
    assert net.request_value.retry.max_attempts == 1
    assert b'"code_verifier": "' + b"v" * 64 in (net.request_value.body or b"")
    assert "stored-secret" not in auth.status().model_dump_json()
    auth.logout()
    assert auth.status().authenticated is False
