from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import httpx

NetPurpose = Literal[
    "model_registry",
    "subtitle_fetch",
    "vision_description",
    "asr_transcription",
    "evidence_plan",
    "summary",
    "openrouter_auth",
    "source_observe",
    "source_media",
]
NetMethod = Literal["GET", "POST"]


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=1, ge=1, le=5)
    statuses: tuple[int, ...] = ()
    retry_connect: bool = False
    retry_timeouts: bool = False
    backoff_s: float = Field(default=0.5, ge=0, le=60)
    max_backoff_s: float = Field(default=4, ge=0, le=60)
    honor_retry_after: bool = True


class NetRequest(BaseModel):
    method: NetMethod
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes | None = None
    timeout_s: float
    connect_timeout_s: float | None = None
    purpose: NetPurpose
    provider_id: str | None = None
    max_response_bytes: int = Field(default=8 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class NetResponse(BaseModel):
    url: str
    status_code: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes
    attempts: int = Field(default=1, ge=1)


class NetError(RuntimeError):
    def __init__(self, *, attempts: int, cause: Exception) -> None:
        super().__init__(f"network request failed after {attempts} attempt(s): {cause}")
        self.attempts = attempts
        self.cause = cause


class NetRuntime(Protocol):
    def request(self, request: NetRequest) -> NetResponse: ...


@runtime_checkable
class BatchNetRuntime(NetRuntime, Protocol):
    def request_many(self, requests: Sequence[NetRequest]) -> list[NetResponse]: ...


@runtime_checkable
class StreamingNetRuntime(NetRuntime, Protocol):
    def iter_request(self, request: NetRequest, *, block_size: int) -> Iterator[NetResponse]: ...


class HttpxNetRuntime:
    def __init__(
        self,
        *,
        max_connections: int = 8,
        per_provider_concurrency: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.per_provider_concurrency = per_provider_concurrency
        httpx_module = _httpx()
        self._httpx = httpx_module
        self._client = client or httpx_module.Client(
            limits=httpx_module.Limits(max_connections=max_connections),
            mounts={
                "http://127.0.0.1": httpx_module.HTTPTransport(),
                "http://localhost": httpx_module.HTTPTransport(),
                "http://[::1]": httpx_module.HTTPTransport(),
            },
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def request(self, request: NetRequest) -> NetResponse:
        return _request_with_retry(request, lambda: self._request_once(request))

    def _request_once(self, request: NetRequest) -> NetResponse:
        with self._client.stream(
            request.method,
            request.url,
            headers=request.headers,
            content=request.body,
            timeout=self._httpx.Timeout(
                request.timeout_s,
                connect=request.connect_timeout_s or request.timeout_s,
            ),
        ) as response:
            body = bytearray()
            for block in response.iter_bytes(64 * 1024):
                if len(body) + len(block) > request.max_response_bytes:
                    raise ValueError("response exceeds declared byte limit")
                body.extend(block)
            return NetResponse(
                url=str(response.url),
                status_code=response.status_code,
                headers={key: value for key, value in response.headers.items()},
                body=bytes(body),
            )

    def request_many(self, requests: Sequence[NetRequest]) -> list[NetResponse]:
        if not requests:
            return []
        workers = min(self.per_provider_concurrency, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self.request, requests))

    def iter_request(self, request: NetRequest, *, block_size: int) -> Iterator[NetResponse]:
        attempts = request.retry.max_attempts
        for attempt in range(1, attempts + 1):
            emitted = False
            received = 0
            try:
                with self._client.stream(
                    request.method,
                    request.url,
                    headers=request.headers,
                    content=request.body,
                    timeout=self._httpx.Timeout(
                        request.timeout_s,
                        connect=request.connect_timeout_s or request.timeout_s,
                    ),
                ) as response:
                    if response.status_code in request.retry.statuses and attempt < attempts:
                        _wait(request.retry, attempt, _headers(response))
                        continue
                    headers = {key: value for key, value in response.headers.items()}
                    for block in response.iter_bytes(block_size):
                        if received + len(block) > request.max_response_bytes:
                            raise ValueError("response exceeds declared byte limit")
                        received += len(block)
                        emitted = True
                        yield NetResponse(
                            url=str(response.url),
                            status_code=response.status_code,
                            headers=headers,
                            body=block,
                            attempts=attempt,
                        )
                    if not emitted:
                        yield NetResponse(
                            url=str(response.url),
                            status_code=response.status_code,
                            headers=headers,
                            body=b"",
                            attempts=attempt,
                        )
                    return
            except Exception as exc:
                if emitted or attempt == attempts or not _retryable_exception(exc, request.retry):
                    raise NetError(attempts=attempt, cause=exc) from exc
                _wait(request.retry, attempt)


def _request_with_retry(
    request: NetRequest,
    request_once: Callable[[], NetResponse],
) -> NetResponse:
    for attempt in range(1, request.retry.max_attempts + 1):
        try:
            response = request_once()
        except Exception as exc:
            if attempt == request.retry.max_attempts or not _retryable_exception(
                exc, request.retry
            ):
                raise NetError(attempts=attempt, cause=exc) from exc
            _wait(request.retry, attempt)
            continue
        if (
            response.status_code not in request.retry.statuses
            or attempt == request.retry.max_attempts
        ):
            return response.model_copy(update={"attempts": attempt})
        _wait(request.retry, attempt, response.headers)
    raise AssertionError("unreachable retry state")


def _retryable_exception(exc: Exception, policy: RetryPolicy) -> bool:
    httpx = _httpx()
    if isinstance(exc, httpx.ConnectTimeout):
        return policy.retry_connect
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return policy.retry_timeouts
    return policy.retry_connect and isinstance(exc, httpx.ConnectError | ConnectionError | OSError)


def _httpx() -> Any:
    return importlib.import_module("httpx")


def _wait(policy: RetryPolicy, attempt: int, headers: dict[str, str] | None = None) -> None:
    value = headers.get("Retry-After") if policy.honor_retry_after and headers else None
    try:
        delay = float(value) if value is not None else -1
    except ValueError:
        delay = -1
    if not 0 <= delay <= 60:
        delay = min(policy.backoff_s * 2 ** (attempt - 1), policy.max_backoff_s)
    time.sleep(delay)


def _headers(response: Any) -> dict[str, str]:
    return {key: value for key, value in response.headers.items()}
