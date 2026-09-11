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


class RangeTransfer:
    def __init__(
        self,
        net: NetRuntime,
        url: str,
        destination: Path,
        *,
        headers: dict[str, str] | None = None,
        refresh: bool = False,
    ) -> None:
        self.net = net
        self.url = url
        self.destination = destination
        self.headers = headers or {}
        self.refresh = refresh
        self.part = destination.with_name(f"{destination.name}.part")
        self.state_path = destination.with_name(f"{destination.name}.ranges.json")
        self.lock_path = destination.with_name(f"{destination.name}.lock")

    def download(self) -> Path:
        """Download through bounded byte ranges without persisting the locator."""

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(self.lock_path):
            if self.destination.is_file() and not self.refresh:
                return self.destination
            if self.refresh:
                self.destination.unlink(missing_ok=True)
            first_stream = self._request(0, _CHUNK_BYTES - 1)
            first = next(first_stream)
            if first.status_code == 200:
                self._write_whole(chain((first,), first_stream))
                os.replace(self.part, self.destination)
                self.state_path.unlink(missing_ok=True)
                return self.destination
            total, start, end = _range_facts(first)
            validator = _header(first, "etag") or _header(first, "last-modified")
            state = self._load_state()
            if (
                state is None
                or state.total != total
                or state.validator != validator
                or state.chunk_bytes != _CHUNK_BYTES
            ):
                state = TransferState(total=total, validator=validator)
                with self.part.open("wb") as stream:
                    stream.truncate(total)
            self._write_range(chain((first,), first_stream), start, end, total)
            self._sync()
            state.completed.add(start // _CHUNK_BYTES)
            self._save_state(state)

            missing = [
                index
                for index in range((total + _CHUNK_BYTES - 1) // _CHUNK_BYTES)
                if index not in state.completed
            ]
            workers = _workers(total)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._download_range, total, index): index for index in missing
                }
                committed: list[int] = []
                for future in as_completed(futures):
                    committed.append(future.result())
                    if len(committed) < workers:
                        continue
                    self._sync()
                    state.completed.update(committed)
                    self._save_state(state)
                    committed.clear()
                if committed:
                    self._sync()
                    state.completed.update(committed)
                    self._save_state(state)
            if len(state.completed) != (total + _CHUNK_BYTES - 1) // _CHUNK_BYTES:
                raise ProviderError("source range download is incomplete")
            if self.part.stat().st_size != total:
                raise ProviderError("source range download has an invalid size")
            os.replace(self.part, self.destination)
            self.state_path.unlink(missing_ok=True)
            return self.destination

    def _request(self, start: int, end: int) -> Iterator[NetResponse]:
        request = _range_request(self.url, start, end, self.headers)
        if isinstance(self.net, StreamingNetRuntime):
            return self.net.iter_request(request, block_size=_BLOCK_BYTES)
        return iter((self.net.request(request),))

    def _download_range(self, total: int, index: int) -> int:
        expected_start = index * _CHUNK_BYTES
        responses = self._request(expected_start, min(total - 1, expected_start + _CHUNK_BYTES - 1))
        first = next(responses)
        actual_total, start, end = _range_facts(first)
        if actual_total != total or start != expected_start:
            raise ProviderError("source range response changed identity")
        self._write_range(chain((first,), responses), start, end, total)
        return index

    def _write_range(
        self, responses: Iterator[NetResponse], start: int, end: int, total: int
    ) -> None:
        if end >= total:
            raise ProviderError("source range exceeds expected size")
        with self.part.open("r+b") as stream:
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

    def _write_whole(self, responses: Iterator[NetResponse]) -> None:
        written = 0
        with self.part.open("wb") as stream:
            for response in responses:
                stream.write(response.body)
                written += len(response.body)
            if not written:
                raise ProviderError("source media response is empty")
            stream.flush()
            os.fsync(stream.fileno())

    def _sync(self) -> None:
        with self.part.open("r+b") as stream:
            os.fsync(stream.fileno())

    def _load_state(self) -> TransferState | None:
        try:
            return TransferState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
        except OSError, ValidationError, ValueError:
            return None

    def _save_state(self, state: TransferState) -> None:
        temporary = self.state_path.with_name(f".{self.state_path.name}.tmp")
        try:
            temporary.write_text(state.model_dump_json(), encoding="utf-8")
            os.replace(temporary, self.state_path)
        finally:
            temporary.unlink(missing_ok=True)


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
