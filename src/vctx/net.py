from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, Protocol, runtime_checkable
from urllib.request import Request, urlopen

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

NetPurpose = Literal[
    "model_registry",
    "subtitle_fetch",
    "vision_description",
    "asr_transcription",
    "knowledge_flow_extraction",
    "essential_case_extraction",
]
NetMethod = Literal["GET", "POST"]


class NetRequest(BaseModel):
    method: NetMethod
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes | None = None
    timeout_s: float
    purpose: NetPurpose
    provider_id: str | None = None


class NetResponse(BaseModel):
    url: str
    status_code: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes


class NetRetryPolicy(BaseModel):
    attempts: int = 3
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 4.0
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


class NetRuntime(Protocol):
    def request(self, request: NetRequest) -> NetResponse: ...


@runtime_checkable
class BatchNetRuntime(NetRuntime, Protocol):
    def request_many(self, requests: Sequence[NetRequest]) -> list[NetResponse]: ...


class UrllibNetRuntime:
    def __init__(self, *, retry: NetRetryPolicy | None = None) -> None:
        self.retry = retry or NetRetryPolicy()

    def request(self, request: NetRequest) -> NetResponse:
        return _request_with_get_retry(request, self.retry, lambda: self._request_once(request))

    def _request_once(self, request: NetRequest) -> NetResponse:
        raw_request = Request(
            request.url,
            data=request.body,
            headers=request.headers,
            method=request.method,
        )
        with urlopen(raw_request, timeout=request.timeout_s) as response:  # noqa: S310
            return NetResponse(
                url=response.url,
                status_code=response.status,
                headers={key: value for key, value in response.headers.items()},
                body=response.read(),
            )


class HttpxNetRuntime:
    def __init__(
        self,
        *,
        max_connections: int = 8,
        per_provider_concurrency: int = 2,
        retry: NetRetryPolicy | None = None,
    ) -> None:
        self.max_connections = max_connections
        self.per_provider_concurrency = per_provider_concurrency
        self.retry = retry or NetRetryPolicy()

    def request(self, request: NetRequest) -> NetResponse:
        return _request_with_get_retry(request, self.retry, lambda: self._request_once(request))

    def _request_once(self, request: NetRequest) -> NetResponse:
        with httpx.Client(limits=self._limits()) as client:
            response = client.request(
                request.method,
                request.url,
                headers=request.headers,
                content=request.body,
                timeout=request.timeout_s,
            )
        return _httpx_response(response)

    def request_many(self, requests: Sequence[NetRequest]) -> list[NetResponse]:
        return asyncio.run(self._request_many(requests))

    async def _request_many(self, requests: Sequence[NetRequest]) -> list[NetResponse]:
        semaphore = asyncio.Semaphore(self.per_provider_concurrency)
        async with httpx.AsyncClient(limits=self._limits()) as client:
            return await asyncio.gather(
                *[
                    self._bounded_request_with_retry(client, semaphore, request)
                    for request in requests
                ]
            )

    async def _bounded_request_with_retry(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        request: NetRequest,
    ) -> NetResponse:
        return await _async_request_with_get_retry(
            request,
            self.retry,
            lambda: self._bounded_request_once(client, semaphore, request),
        )

    async def _bounded_request_once(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        request: NetRequest,
    ) -> NetResponse:
        async with semaphore:
            response = await client.request(
                request.method,
                request.url,
                headers=request.headers,
                content=request.body,
                timeout=request.timeout_s,
            )
        return _httpx_response(response)

    def _limits(self) -> httpx.Limits:
        return httpx.Limits(max_connections=self.max_connections)


class _RetryableStatusError(RuntimeError):
    def __init__(self, response: NetResponse) -> None:
        super().__init__(f"retryable HTTP status: {response.status_code}")
        self.response = response


def _request_with_get_retry(
    request: NetRequest,
    retry: NetRetryPolicy,
    request_once: Callable[[], NetResponse],
) -> NetResponse:
    if request.method != "GET" or retry.attempts <= 1:
        return request_once()

    retrying = Retrying(
        stop=stop_after_attempt(retry.attempts),
        wait=wait_exponential(
            multiplier=retry.initial_backoff_s,
            max=retry.max_backoff_s,
        ),
        retry=retry_if_exception(_is_retryable_get_exception),
        reraise=True,
    )
    try:
        for attempt in retrying:
            with attempt:
                return _raise_retryable_status(request_once(), retry)
    except _RetryableStatusError as exc:
        return exc.response
    raise RuntimeError("retry ended without response")


async def _async_request_with_get_retry(
    request: NetRequest,
    retry: NetRetryPolicy,
    request_once: Callable[[], Awaitable[NetResponse]],
) -> NetResponse:
    if request.method != "GET" or retry.attempts <= 1:
        return _ensure_net_response(await request_once())

    retrying = AsyncRetrying(
        stop=stop_after_attempt(retry.attempts),
        wait=wait_exponential(
            multiplier=retry.initial_backoff_s,
            max=retry.max_backoff_s,
        ),
        retry=retry_if_exception(_is_retryable_get_exception),
        reraise=True,
    )
    try:
        async for attempt in retrying:
            with attempt:
                response = _ensure_net_response(await request_once())
                if response.status_code in retry.retry_status_codes:
                    raise _RetryableStatusError(response)
                return response
    except _RetryableStatusError as exc:
        return exc.response
    raise RuntimeError("retry ended without response")


def _raise_retryable_status(response: NetResponse, retry: NetRetryPolicy) -> NetResponse:
    if response.status_code in retry.retry_status_codes:
        raise _RetryableStatusError(response)
    return response


def _is_retryable_get_exception(exc: BaseException) -> bool:
    return isinstance(exc, _RetryableStatusError | httpx.TransportError | OSError)


def _ensure_net_response(value: object) -> NetResponse:
    if not isinstance(value, NetResponse):
        raise TypeError(f"expected NetResponse, got {type(value).__name__}")
    return value


def _httpx_response(response: httpx.Response) -> NetResponse:
    return NetResponse(
        url=str(response.url),
        status_code=response.status_code,
        headers={key: value for key, value in response.headers.items()},
        body=response.content,
    )
