from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from itertools import chain
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from vctx.errors import ProviderError
from vctx.net import NetRequest, NetResponse, NetRuntime, RetryPolicy, StreamingNetRuntime

_CHUNK_BYTES = 4 * 1024 * 1024
_BLOCK_BYTES = 256 * 1024
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)\Z")


class LocatorExpired(ProviderError):
    pass


class TransferState(BaseModel):
    total: int = Field(gt=0)
    validator: str | None = None
    completed: set[int] = Field(default_factory=set)
    chunk_bytes: int = _CHUNK_BYTES


def download_ranged(
    net: NetRuntime,
    url: str,
    destination: Path,
    *,
    headers: dict[str, str] | None = None,
    refresh: bool = False,
) -> Path:
    """Download a URL through bounded byte ranges without persisting its locator."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f"{destination.name}.part")
    state_path = destination.with_name(f"{destination.name}.ranges.json")
    lock_path = destination.with_name(f"{destination.name}.lock")
    with _exclusive_lock(lock_path):
        if destination.is_file() and not refresh:
            return destination
        if refresh:
            destination.unlink(missing_ok=True)
        first_stream = _request_range(net, url, 0, _CHUNK_BYTES - 1, headers or {})
        first = next(first_stream)
        if first.status_code == 200:
            _write_whole(part, chain((first,), first_stream))
            os.replace(part, destination)
            state_path.unlink(missing_ok=True)
            return destination
        total, start, end = _range_facts(first)
        validator = _header(first, "etag") or _header(first, "last-modified")
        state = _load_state(state_path)
        if (
            state is None
            or state.total != total
            or state.validator != validator
            or state.chunk_bytes != _CHUNK_BYTES
        ):
            state = TransferState(total=total, validator=validator)
            with part.open("wb") as stream:
                stream.truncate(total)
        _write_range(part, chain((first,), first_stream), start, end, total)
        _sync(part)
        state.completed.add(start // _CHUNK_BYTES)
        _save_state(state_path, state)

        missing = [
            index
            for index in range((total + _CHUNK_BYTES - 1) // _CHUNK_BYTES)
            if index not in state.completed
        ]
        workers = _workers(total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_download_range, net, url, part, total, index, headers or {}): index
                for index in missing
            }
            committed: list[int] = []
            for future in as_completed(futures):
                committed.append(future.result())
                if len(committed) < workers:
                    continue
                _sync(part)
                state.completed.update(committed)
                _save_state(state_path, state)
                committed.clear()
            if committed:
                _sync(part)
                state.completed.update(committed)
                _save_state(state_path, state)
        if len(state.completed) != (total + _CHUNK_BYTES - 1) // _CHUNK_BYTES:
            raise ProviderError("source range download is incomplete")
        if part.stat().st_size != total:
            raise ProviderError("source range download has an invalid size")
        os.replace(part, destination)
        state_path.unlink(missing_ok=True)
        return destination


def _request_range(
    net: NetRuntime, url: str, start: int, end: int, headers: dict[str, str]
) -> Iterator[NetResponse]:
    request = _range_request(url, start, end, headers)
    if isinstance(net, StreamingNetRuntime):
        return net.iter_request(request, block_size=_BLOCK_BYTES)
    return iter((net.request(request),))


def _download_range(
    net: NetRuntime,
    url: str,
    part: Path,
    total: int,
    index: int,
    headers: dict[str, str],
) -> int:
    expected_start = index * _CHUNK_BYTES
    responses = _request_range(
        net, url, expected_start, min(total - 1, expected_start + _CHUNK_BYTES - 1), headers
    )
    first = next(responses)
    actual_total, start, end = _range_facts(first)
    if actual_total != total or start != expected_start:
        raise ProviderError("source range response changed identity")
    _write_range(part, chain((first,), responses), start, end, total)
    return index


def _range_request(url: str, start: int, end: int, headers: dict[str, str]) -> NetRequest:
    return NetRequest(
        method="GET",
        url=url,
        headers={**headers, "Range": f"bytes={start}-{end}"},
        timeout_s=60,
        connect_timeout_s=10,
        purpose="source_media",
        provider_id="source-transfer",
        retry=RetryPolicy(
            max_attempts=3,
            statuses=(408, 429, 500, 502, 503, 504),
            retry_connect=True,
            retry_timeouts=True,
        ),
    )


def _range_facts(response: NetResponse) -> tuple[int, int, int]:
    if response.status_code in {401, 403, 404, 410, 412}:
        raise LocatorExpired("source media locator expired")
    if response.status_code != 206:
        raise ProviderError(f"source media returned HTTP {response.status_code}")
    value = _header(response, "content-range")
    match = _CONTENT_RANGE.fullmatch(value or "")
    if match is None:
        raise ProviderError("source range response lacks a valid Content-Range")
    start, end, total = (int(item) for item in match.groups())
    if end < start or end >= total:
        raise ProviderError("source range response has inconsistent bounds")
    return total, start, end


def _write_range(
    part: Path, responses: Iterator[NetResponse], start: int, end: int, total: int
) -> None:
    if end >= total:
        raise ProviderError("source range exceeds expected size")
    with part.open("r+b") as stream:
        stream.seek(start)
        written = 0
        for response in responses:
            size = len(response.body)
            if written + size > end - start + 1:
                raise ProviderError("source range response has inconsistent bounds")
            stream.write(response.body)
            written += size
        if written != end - start + 1:
            raise ProviderError("source range response has inconsistent bounds")


def _write_whole(part: Path, responses: Iterator[NetResponse]) -> None:
    written = 0
    with part.open("wb") as stream:
        for response in responses:
            stream.write(response.body)
            written += len(response.body)
        if not written:
            raise ProviderError("source media response is empty")
        stream.flush()
        os.fsync(stream.fileno())


def _sync(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _load_state(path: Path) -> TransferState | None:
    try:
        return TransferState.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError, ValidationError, ValueError:
        return None


def _save_state(path: Path, state: TransferState) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(state.model_dump_json(), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _header(response: NetResponse, name: str) -> str | None:
    wanted = name.casefold()
    return next(
        (value for key, value in response.headers.items() if key.casefold() == wanted), None
    )


def _workers(total: int) -> int:
    if total < 32 * 1024 * 1024:
        return 4
    if total < 128 * 1024 * 1024:
        return 8
    return 16


@contextmanager
def _exclusive_lock(path: Path, *, timeout_s: float = 30) -> Iterator[None]:
    deadline = time.monotonic() + timeout_s
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ProviderError("timed out waiting for source media lock") from None
            time.sleep(0.05)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)
