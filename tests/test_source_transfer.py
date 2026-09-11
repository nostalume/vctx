from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from vctx.errors import ProviderError
from vctx.net import NetRequest, NetResponse
from vctx.source.transfer import RangeTransfer


class StreamingRangeNet:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.largest = 0

    def request(self, request: NetRequest) -> NetResponse:
        del request
        raise AssertionError("media ranges must not use whole-body request")

    def iter_request(self, request: NetRequest, *, block_size: int) -> Iterator[NetResponse]:
        first, last = request.headers["Range"].removeprefix("bytes=").split("-")
        start, end = int(first), min(int(last), len(self.body) - 1)
        headers = {"Content-Range": f"bytes {start}-{end}/{len(self.body)}", "ETag": "stable"}
        for offset in range(start, end + 1, block_size):
            chunk = self.body[offset : min(end + 1, offset + block_size)]
            self.largest = max(self.largest, len(chunk))
            yield NetResponse(url=request.url, status_code=206, headers=headers, body=chunk)


def test_range_download_streams_bounded_blocks_without_whole_responses(tmp_path: Path) -> None:
    body = bytes(range(256)) * 40_000
    net = StreamingRangeNet(body)

    result = RangeTransfer(net, "https://media.example/video", tmp_path / "video.m4s").download()

    assert result.read_bytes() == body
    assert net.largest <= 256 * 1024


def test_range_rejects_surplus_before_writing_it(tmp_path: Path) -> None:
    transfer = RangeTransfer(StreamingRangeNet(b""), "https://example", tmp_path / "target")
    transfer.part.write_bytes(b"old")
    response = NetResponse(url="https://example", status_code=206, body=b"xx")

    with pytest.raises(ProviderError, match="bounds"):
        transfer._write_range(iter((response,)), 0, 0, 1)

    assert transfer.part.read_bytes() == b"old"
