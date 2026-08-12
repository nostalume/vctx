from __future__ import annotations

from typing import Literal

import httpx
import pytest

from vctx.net import HttpxNetRuntime, NetError, NetRequest, RetryPolicy


def _request(method: Literal["GET", "POST"], retry: RetryPolicy) -> NetRequest:
    return NetRequest(
        method=method,
        url="https://example.test/resource",
        timeout_s=5,
        purpose="subtitle_fetch" if method == "GET" else "evidence_plan",
        retry=retry,
    )


def test_runtime_executes_declared_status_retry_and_reports_attempts() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 200, request=request, content=b"ok")

    client = httpx.Client(transport=httpx.MockTransport(handle))
    with HttpxNetRuntime(client=client) as net:
        response = net.request(
            _request("GET", RetryPolicy(max_attempts=2, statuses=(503,), backoff_s=0))
        )

    assert (response.status_code, response.attempts, calls) == (200, 2, 2)
    assert client.is_closed


def test_ambiguous_post_timeout_is_not_retried() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("response status unknown", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    with (
        HttpxNetRuntime(client=client) as net,
        pytest.raises(NetError) as raised,
    ):
        net.request(
            _request(
                "POST",
                RetryPolicy(max_attempts=2, retry_connect=True, backoff_s=0),
            )
        )

    assert raised.value.attempts == 1
    assert isinstance(raised.value.cause, httpx.ReadTimeout)
    assert calls == 1
