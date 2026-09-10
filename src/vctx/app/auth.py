from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from vctx.ai import CredentialRef, Keyring
from vctx.net import NetRequest, NetRuntime, RetryPolicy

_SERVICE = "vctx"
_ACCOUNT = "openrouter"
_AUTHORIZE_URL = "https://openrouter.ai/auth"
_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"


class AuthError(RuntimeError):
    pass


class AuthSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    authorization_url: str
    callback_url: str
    verifier: str


class AuthStatus(BaseModel):
    authenticated: bool
    provider: str = "openrouter"
    storage: str = "system-keyring"


class CredentialPresence(BaseModel):
    kind: str
    status: str


def probe_credential_presence(
    reference: CredentialRef,
    *,
    env_files: list[Path] | None = None,
    environ: Mapping[str, str] | None = None,
    keyring: Keyring | None = None,
) -> CredentialPresence:
    if reference.kind == "env":
        env = environ if environ is not None else os.environ
        present = bool(env.get(reference.name)) or _dotenv_has(reference.name, env_files or [])
        return CredentialPresence(kind="env", status="present" if present else "missing")
    if keyring is None or keyring.priority <= 0:
        return CredentialPresence(kind="keyring", status="inaccessible")
    try:
        present = bool(keyring.get_password("vctx", reference.name))
    except Exception:
        return CredentialPresence(kind="keyring", status="inaccessible")
    return CredentialPresence(kind="keyring", status="present" if present else "missing")


def _dotenv_has(name: str, paths: list[Path]) -> bool:
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            key, separator, value = line.strip().partition("=")
            if separator and key.strip() == name and value.strip().strip("'\""):
                return True
    return False


class _Exchange(BaseModel):
    key: str


@dataclass
class OpenRouterAuth:
    keyring: Keyring
    net: NetRuntime | None

    def __init__(self, *, keyring: Keyring, net: NetRuntime | None = None) -> None:
        if keyring.priority <= 0:
            raise AuthError("no secure system credential backend is available")
        self.keyring = keyring
        self.net = net

    def begin(self, callback_url: str, *, verifier: str | None = None) -> AuthSession:
        verifier = verifier or secrets.token_urlsafe(64)
        if not 43 <= len(verifier) <= 128:
            raise AuthError("PKCE verifier must contain 43 to 128 characters")
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        query = urlencode(
            {
                "callback_url": callback_url,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        return AuthSession(
            authorization_url=f"{_AUTHORIZE_URL}?{query}",
            callback_url=callback_url,
            verifier=verifier,
        )

    def finish(self, session: AuthSession, *, code: str) -> None:
        body = json.dumps(
            {
                "code": code,
                "code_verifier": session.verifier,
                "code_challenge_method": "S256",
            }
        ).encode()
        if self.net is None:
            raise AuthError("OpenRouter login requires an admitted HTTP runtime")
        try:
            response = self.net.request(
                NetRequest(
                    method="POST",
                    url=_EXCHANGE_URL,
                    headers={"Content-Type": "application/json"},
                    body=body,
                    timeout_s=30,
                    purpose="openrouter_auth",
                    provider_id="openrouter",
                    retry=RetryPolicy(),
                )
            )
        except Exception as exc:
            raise AuthError(
                f"OpenRouter authorization exchange failed: {type(exc).__name__}"
            ) from exc
        if not 200 <= response.status_code < 300:
            raise AuthError(
                f"OpenRouter authorization exchange failed: HTTP {response.status_code}"
            )
        try:
            key = _Exchange.model_validate_json(response.body).key
        except ValidationError as exc:
            raise AuthError(
                "OpenRouter authorization exchange returned an invalid response"
            ) from exc
        self.keyring.set_password(_SERVICE, _ACCOUNT, key)

    def status(self) -> AuthStatus:
        return AuthStatus(authenticated=self.keyring.get_password(_SERVICE, _ACCOUNT) is not None)

    def logout(self) -> None:
        if self.keyring.get_password(_SERVICE, _ACCOUNT) is not None:
            self.keyring.delete_password(_SERVICE, _ACCOUNT)


def system_keyring() -> Keyring:
    try:
        import keyring
    except ModuleNotFoundError as exc:
        raise AuthError("install the core keyring dependency") from exc
    backend = keyring.get_keyring()
    if backend.priority <= 0:
        raise AuthError("no secure system credential backend is available")
    return backend


def desktop_login(
    auth: OpenRouterAuth,
    *,
    open_browser: Callable[[str], object] = webbrowser.open,
    timeout_s: int = 180,
) -> None:
    code: list[str] = []

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            value = parse_qs(urlsplit(self.path).query).get("code", [])
            if value:
                code.append(value[0])
            body = b"Authorization received. You can close this window."
            self.send_response(200 if value else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = HTTPServer(("127.0.0.1", 0), Callback)
    server.timeout = timeout_s
    session = auth.begin(f"http://127.0.0.1:{server.server_port}")
    open_browser(session.authorization_url)
    server.handle_request()
    server.server_close()
    if not code:
        raise AuthError("OpenRouter authorization timed out or returned no code")
    auth.finish(session, code=code[0])
