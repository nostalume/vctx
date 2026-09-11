from __future__ import annotations

import json
from pathlib import Path

import pytest

from vctx.config import YtDlpSourceOptions
from vctx.net import NetRequest, NetResponse
from vctx.source.bilibili import BilibiliSourceAdapter
from vctx.source.session import (
    AsrAudioRequest,
    MediaPermit,
    ObservePermit,
    SubtitlePermit,
    VisualVideoRequest,
)


class BilibiliNet:
    def __init__(self) -> None:
        self.requests: list[NetRequest] = []
        self.media = {
            "https://cdn.example/audio-low.m4s": b"audio-low",
            "https://cdn.example/audio-high.m4s": b"audio-high",
            "https://cdn.example/video-16.m4s": b"video-16",
            "https://cdn.example/video-32.m4s": b"video-32",
        }

    def request(self, request: NetRequest) -> NetResponse:
        self.requests.append(request)
        if "web-interface/view" in request.url:
            return _json_response(
                request.url,
                {
                    "code": 0,
                    "data": {
                        "bvid": "BV1Tpbj6eEDZ",
                        "cid": 42,
                        "title": "Public video",
                        "duration": 12,
                        "owner": {"name": "Uploader", "mid": 42},
                    },
                },
            )
        if "player/playurl" in request.url:
            return _json_response(
                request.url,
                {
                    "code": 0,
                    "data": {
                        "quality": 64,
                        "dash": {
                            "audio": [
                                {
                                    "id": 30216,
                                    "bandwidth": 32,
                                    "baseUrl": "https://cdn.example/audio-low.m4s",
                                },
                                {
                                    "id": 30280,
                                    "bandwidth": 64,
                                    "baseUrl": "https://cdn.example/audio-high.m4s",
                                },
                            ],
                            "video": [
                                {
                                    "id": 16,
                                    "height": 360,
                                    "bandwidth": 64,
                                    "baseUrl": "https://cdn.example/video-16.m4s",
                                },
                                {
                                    "id": 32,
                                    "height": 480,
                                    "bandwidth": 96,
                                    "baseUrl": "https://cdn.example/video-32.m4s",
                                },
                            ],
                        },
                    },
                },
            )
        if "player/v2" in request.url:
            return _json_response(request.url, {"code": 0, "data": {"subtitle": {"subtitles": []}}})
        body = self.media[request.url]
        value = request.headers.get("Range")
        assert value is not None and value.startswith("bytes=")
        first, last = value.removeprefix("bytes=").split("-")
        start = int(first)
        end = min(int(last), len(body) - 1)
        return NetResponse(
            url=request.url,
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{len(body)}",
                "ETag": '"stable"',
            },
            body=body[start : end + 1],
        )


def test_bilibili_uses_typed_anonymous_api_and_actual_available_streams(
    tmp_path: Path,
) -> None:
    net = BilibiliNet()
    url = "https://www.bilibili.com/video/BV1Tpbj6eEDZ"
    session = BilibiliSourceAdapter(net=net).observe(
        url,
        permit=ObservePermit(operation="prepare", network="allowed"),
        options=YtDlpSourceOptions(),
    )

    audio = session.media(
        request=AsrAudioRequest(temp_dir=tmp_path),
        permit=MediaPermit(network="allowed"),
    )
    video = session.media(
        request=VisualVideoRequest(profile="high", temp_dir=tmp_path),
        permit=MediaPermit(network="allowed"),
    )

    assert session.record.metadata.title == "Public video"
    assert session.record.source_capabilities == {"audio", "video"}
    assert audio.local_path.read_bytes() == b"audio-high"
    assert audio.capabilities == {"audio"} and audio.format_id == "30280"
    assert video.local_path.read_bytes() == b"video-32"
    assert video.capabilities == {"video"} and video.format_id == "32"
    assert all("Cookie" not in request.headers for request in net.requests)
    assert all(
        "cdn.example" not in receipt.detail for receipt in session.receipts if receipt.detail
    )


def test_bilibili_rejects_provider_error_without_raw_payload() -> None:
    class FailedNet:
        def request(self, request: NetRequest) -> NetResponse:
            return _json_response(request.url, {"code": -404, "message": "not found"})

    with pytest.raises(Exception, match="Bilibili metadata unavailable"):
        BilibiliSourceAdapter(net=FailedNet()).observe(
            "https://www.bilibili.com/video/BV1Tpbj6eEDZ",
            permit=ObservePermit(operation="prepare", network="allowed"),
            options=YtDlpSourceOptions(),
        )


def test_bilibili_prefers_typed_anonymous_official_subtitle() -> None:
    class SubtitleNet(BilibiliNet):
        def request(self, request: NetRequest) -> NetResponse:
            if "player/v2" in request.url:
                return _json_response(
                    request.url,
                    {
                        "code": 0,
                        "data": {
                            "subtitle": {
                                "subtitles": [
                                    {
                                        "id": 7,
                                        "lan": "zh-CN",
                                        "subtitle_url": "//aisubtitle.hdslb.com/example.json",
                                        "ai_type": 0,
                                    }
                                ]
                            }
                        },
                    },
                )
            if "aisubtitle.hdslb.com" in request.url:
                return _json_response(
                    request.url,
                    {"body": [{"from": 1.25, "to": 2.5, "content": "字幕文本"}]},
                )
            return super().request(request)

    session = BilibiliSourceAdapter(net=SubtitleNet()).observe(
        "https://www.bilibili.com/video/BV1Tpbj6eEDZ",
        permit=ObservePermit(operation="prepare", network="allowed"),
        options=YtDlpSourceOptions(),
    )
    payload = session.transcript(permit=SubtitlePermit(network="allowed"))

    assert session.record.source_capabilities == {"audio", "video", "subtitle"}
    assert payload.provenance.method == "official_subtitles"
    assert "00:00:01,250 --> 00:00:02,500" in payload.text


def test_bilibili_rejects_malformed_subtitle_index() -> None:
    class MalformedNet(BilibiliNet):
        def request(self, request: NetRequest) -> NetResponse:
            if "player/v2" in request.url:
                return _json_response(
                    request.url,
                    {"code": 0, "data": {"subtitle": {"subtitles": [{"id": "bad"}]}}},
                )
            return super().request(request)

    with pytest.raises(Exception, match="subtitle index returned an invalid response"):
        BilibiliSourceAdapter(net=MalformedNet()).observe(
            "https://www.bilibili.com/video/BV1Tpbj6eEDZ",
            permit=ObservePermit(operation="prepare", network="allowed"),
            options=YtDlpSourceOptions(),
        )


def test_bilibili_refreshes_one_expired_media_locator(tmp_path: Path) -> None:
    class RefreshingNet(BilibiliNet):
        plays = 0

        def request(self, request: NetRequest) -> NetResponse:
            if "player/playurl" in request.url:
                self.plays += 1
                locator = f"https://cdn.example/audio-{self.plays}.m4s"
                return _json_response(
                    request.url,
                    {
                        "code": 0,
                        "data": {
                            "dash": {
                                "audio": [{"id": 30280, "bandwidth": 64, "baseUrl": locator}],
                                "video": [],
                            }
                        },
                    },
                )
            if request.url.endswith("audio-1.m4s"):
                return NetResponse(url=request.url, status_code=403, body=b"")
            if request.url.endswith("audio-2.m4s"):
                return NetResponse(
                    url=request.url,
                    status_code=206,
                    headers={"Content-Range": "bytes 0-4/5", "ETag": "fresh"},
                    body=b"audio",
                )
            return super().request(request)

    net = RefreshingNet()
    session = BilibiliSourceAdapter(net=net).observe(
        "https://www.bilibili.com/video/BV1Tpbj6eEDZ",
        permit=ObservePermit(operation="prepare", network="allowed"),
        options=YtDlpSourceOptions(),
    )

    asset = session.media(
        request=AsrAudioRequest(temp_dir=tmp_path),
        permit=MediaPermit(network="allowed"),
    )

    assert asset.local_path.read_bytes() == b"audio"
    assert net.plays == 2


def _json_response(url: str, payload: object) -> NetResponse:
    return NetResponse(url=url, status_code=200, body=json.dumps(payload).encode())
