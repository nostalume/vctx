from __future__ import annotations

import httpx
import pytest

from vctx.net import HttpxNetRuntime, NetRequest, NetResponse, NetRetryPolicy


class _FakeAsyncResponse:
    def __init__(self, *, url: str, status_code: int, body: bytes) -> None:
        self.url = url
        self.status_code = status_code
        self.headers = {"content-type": "text/plain"}
        self.content = body


class _FakeAsyncClient:
    seen_requests: list[tuple[str, str, bytes | None]] = []
    status_by_url: dict[str, list[int]] = {}

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        content: bytes | None,
        timeout: float,
    ) -> _FakeAsyncResponse:
        del headers, timeout
        self.seen_requests.append((method, url, content))
        status_code = _pop_status(self.status_by_url, url)
        return _FakeAsyncResponse(
            url=url,
            status_code=status_code,
            body=f"response:{status_code}:{url}".encode(),
        )


class _FakeSyncResponse:
    def __init__(self, *, url: str, status_code: int, body: bytes) -> None:
        self.url = url
        self.status_code = status_code
        self.headers = {"content-type": "text/plain"}
        self.content = body


class _FakeSyncClient:
    seen_requests: list[tuple[str, str, bytes | None]] = []
    status_by_url: dict[str, list[int]] = {}

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __enter__(self) -> _FakeSyncClient:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        content: bytes | None,
        timeout: float,
    ) -> _FakeSyncResponse:
        del headers, timeout
        self.seen_requests.append((method, url, content))
        status_code = _pop_status(self.status_by_url, url)
        body = b"single" if status_code == 200 else f"status:{status_code}".encode()
        return _FakeSyncResponse(url=url, status_code=status_code, body=body)


def test_httpx_net_runtime_request_many_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_httpx(monkeypatch)
    _reset_fakes()
    runtime = HttpxNetRuntime(max_connections=4, per_provider_concurrency=2)

    responses = runtime.request_many(
        [
            _post_request("https://example.invalid/first", b"first"),
            _post_request("https://example.invalid/second", b"second"),
        ]
    )

    assert [response.body for response in responses] == [
        b"response:200:https://example.invalid/first",
        b"response:200:https://example.invalid/second",
    ]
    assert _FakeAsyncClient.seen_requests == [
        ("POST", "https://example.invalid/first", b"first"),
        ("POST", "https://example.invalid/second", b"second"),
    ]


def test_httpx_net_runtime_request_uses_sync_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_httpx(monkeypatch)
    _reset_fakes()
    runtime = HttpxNetRuntime(max_connections=4, per_provider_concurrency=2)

    response = runtime.request(_post_request("https://example.invalid/single", b"payload"))

    assert response == NetResponse(
        url="https://example.invalid/single",
        status_code=200,
        headers={"content-type": "text/plain"},
        body=b"single",
    )


def test_httpx_net_runtime_retries_get_status_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_httpx(monkeypatch)
    _reset_fakes()
    _FakeSyncClient.status_by_url = {"https://example.invalid/retry": [503, 200]}
    runtime = HttpxNetRuntime(retry=_fast_retry_policy())

    response = runtime.request(_get_request("https://example.invalid/retry"))

    assert response.status_code == 200
    assert _FakeSyncClient.seen_requests == [
        ("GET", "https://example.invalid/retry", None),
        ("GET", "https://example.invalid/retry", None),
    ]


def test_httpx_net_runtime_does_not_retry_post_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_httpx(monkeypatch)
    _reset_fakes()
    _FakeSyncClient.status_by_url = {"https://example.invalid/post": [503, 200]}
    runtime = HttpxNetRuntime(retry=_fast_retry_policy())

    response = runtime.request(_post_request("https://example.invalid/post", b"payload"))

    assert response.status_code == 503
    assert _FakeSyncClient.seen_requests == [("POST", "https://example.invalid/post", b"payload")]


def test_httpx_net_runtime_request_many_retries_get_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_httpx(monkeypatch)
    _reset_fakes()
    _FakeAsyncClient.status_by_url = {
        "https://example.invalid/first": [503, 200],
        "https://example.invalid/second": [200],
    }
    runtime = HttpxNetRuntime(per_provider_concurrency=2, retry=_fast_retry_policy())

    responses = runtime.request_many(
        [
            _get_request("https://example.invalid/first"),
            _get_request("https://example.invalid/second"),
        ]
    )

    assert [response.status_code for response in responses] == [200, 200]
    first_attempts = _FakeAsyncClient.seen_requests.count(
        ("GET", "https://example.invalid/first", None)
    )
    second_attempts = _FakeAsyncClient.seen_requests.count(
        ("GET", "https://example.invalid/second", None)
    )
    assert first_attempts == 2
    assert second_attempts == 1


def _patch_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(httpx, "Client", _FakeSyncClient)


def _reset_fakes() -> None:
    _FakeAsyncClient.seen_requests = []
    _FakeAsyncClient.status_by_url = {}
    _FakeSyncClient.seen_requests = []
    _FakeSyncClient.status_by_url = {}


def _pop_status(status_by_url: dict[str, list[int]], url: str) -> int:
    statuses = status_by_url.get(url)
    if not statuses:
        return 200
    return statuses.pop(0)


def _fast_retry_policy() -> NetRetryPolicy:
    return NetRetryPolicy(attempts=3, initial_backoff_s=0, max_backoff_s=0)


def _get_request(url: str) -> NetRequest:
    return NetRequest(
        method="GET",
        url=url,
        timeout_s=5,
        purpose="subtitle_fetch",
        provider_id="test-source",
    )


def _post_request(url: str, body: bytes) -> NetRequest:
    return NetRequest(
        method="POST",
        url=url,
        headers={"x-test": "1"},
        body=body,
        timeout_s=5,
        purpose="vision_description",
        provider_id="test-vlm",
    )
