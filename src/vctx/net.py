from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Literal, Protocol, Self, runtime_checkable

import httpx
from pydantic import BaseModel, Field
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt

NetPurpose = Literal[
    "model_registry",
    "subtitle_fetch",
    "vision_description",
    "asr_transcription",
    "evidence_plan",
    "openrouter_auth",
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


class HttpxNetRuntime:
    def __init__(
        self,
        *,
        max_connections: int = 8,
        per_provider_concurrency: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.per_provider_concurrency = per_provider_concurrency
        self._client = client or httpx.Client(limits=httpx.Limits(max_connections=max_connections))

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
        response = self._client.request(
            request.method,
            request.url,
            headers=request.headers,
            content=request.body,
            timeout=httpx.Timeout(
                request.timeout_s,
                connect=request.connect_timeout_s or request.timeout_s,
            ),
        )
        return NetResponse(
            url=str(response.url),
            status_code=response.status_code,
            headers={key: value for key, value in response.headers.items()},
            body=response.content,
        )

    def request_many(self, requests: Sequence[NetRequest]) -> list[NetResponse]:
        if not requests:
            return []
        workers = min(self.per_provider_concurrency, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self.request, requests))


class _RetrySignal(RuntimeError):
    def __init__(
        self,
        *,
        cause: Exception | None = None,
        response: NetResponse | None = None,
    ) -> None:
        super().__init__(str(cause) if cause is not None else "retryable HTTP response")
        self.cause = cause
        self.response = response


def _request_with_retry(
    request: NetRequest,
    request_once: Callable[[], NetResponse],
) -> NetResponse:
    attempts = 0

    def attempt() -> NetResponse:
        nonlocal attempts
        attempts += 1
        try:
            response = request_once()
        except Exception as exc:
            if _retryable_exception(exc, request.retry):
                raise _RetrySignal(cause=exc) from exc
            raise NetError(attempts=attempts, cause=exc) from exc
        if response.status_code in request.retry.statuses:
            raise _RetrySignal(response=response)
        return response.model_copy(update={"attempts": attempts})

    retrying = Retrying(
        stop=stop_after_attempt(request.retry.max_attempts),
        wait=lambda state: _retry_delay(state, request.retry),
        retry=retry_if_exception_type(_RetrySignal),
        reraise=True,
    )
    try:
        return retrying(attempt)
    except _RetrySignal as signal:
        if signal.response is not None:
            return signal.response.model_copy(update={"attempts": attempts})
        assert signal.cause is not None
        raise NetError(attempts=attempts, cause=signal.cause) from signal.cause


def _retryable_exception(exc: Exception, policy: RetryPolicy) -> bool:
    if isinstance(exc, httpx.ConnectTimeout):
        return policy.retry_connect
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return policy.retry_timeouts
    return policy.retry_connect and isinstance(
        exc, httpx.ConnectError | ConnectionError | OSError
    )


def _retry_delay(state: RetryCallState, policy: RetryPolicy) -> float:
    if policy.honor_retry_after and state.outcome is not None:
        signal = state.outcome.exception()
        if isinstance(signal, _RetrySignal) and signal.response is not None:
            value = signal.response.headers.get("Retry-After")
            if value is not None:
                try:
                    parsed = float(value)
                except ValueError:
                    pass
                else:
                    if 0 <= parsed <= 60:
                        return parsed
    delay = policy.backoff_s * (2 ** max(0, state.attempt_number - 1))
    return min(delay, policy.max_backoff_s)
