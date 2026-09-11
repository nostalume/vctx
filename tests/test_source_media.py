from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from vctx.config import YtDlpSourceOptions
from vctx.errors import OperationCancelledError, ProviderError
from vctx.net import NetRuntime
from vctx.source.session import (
    AsrAudioRequest,
    MediaAsset,
    MediaPermit,
    MediaRegistry,
    Revision,
    SourceRecord,
    SourceRef,
    VideoMetadata,
    VisualVideoRequest,
)
from vctx.source.ytdlp import YtDlpInfo, YtDlpParams, YtDlpSession


def test_visual_media_auto_falls_back_without_audio_merge_and_explicit_quality_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.source.ytdlp as ytdlp_module

    formats: list[dict[str, object]] = [
        {"format_id": "video-720", "vcodec": "vp9", "height": 720, "filesize": 900_000_000},
        {"format_id": "video-480", "vcodec": "vp9", "height": 480, "filesize": 10_000_000},
    ]
    info: YtDlpInfo = {
        "id": "lecture",
        "extractor": "example",
        "formats": formats,  # type: ignore[dict-item]
    }
    source = SourceRef(kind="url", value="https://video.example/lecture")
    record = SourceRecord(
        source_id="example__lecture",
        revision=Revision(kind="observed", value="revision"),
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata=VideoMetadata(id="example__lecture", source=source),
    )
    captured: list[YtDlpParams] = []

    class FakeYoutubeDL:
        def __init__(self, params: YtDlpParams) -> None:
            captured.append(params)

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def process_ie_result(self, observed: YtDlpInfo, download: bool) -> YtDlpInfo:
            assert observed is not info and download is True
            path = tmp_path / "cache" / "video.webm"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"video-only")
            return {
                **observed,
                "format_id": "video-480",
                "requested_downloads": [{"filepath": str(path)}],
            }

    monkeypatch.setattr(ytdlp_module._yt_dlp(), "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(
        ytdlp_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=20_000_000_000, free=500_000_000),
    )
    net = cast(NetRuntime, None)
    session = YtDlpSession(info, YtDlpSourceOptions(), record, net)
    permit = MediaPermit(network="allowed")

    asset = session.media(
        request=VisualVideoRequest(profile="auto", temp_dir=tmp_path / "cache"), permit=permit
    )
    assert asset.profile == "fast"
    assert session.receipts[-1].detail == (
        "media-quality auto selected fast because cache space is constrained"
    )
    assert captured[0]["format"] == "bestvideo[height<=480]/bestvideo"
    assert "+" not in str(captured[0]["format"])

    with pytest.raises(ProviderError, match="insufficient cache space"):
        session.media(
            request=VisualVideoRequest(profile="high", temp_dir=tmp_path / "cache"),
            permit=permit,
        )
    assert len(captured) == 1

    def interrupt(_self: object, _observed: YtDlpInfo, download: bool) -> YtDlpInfo:
        assert download is True
        (tmp_path / "cache" / "interrupted.part").write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeYoutubeDL, "process_ie_result", interrupt)
    with pytest.raises(OperationCancelledError) as cancelled:
        session.media(
            request=VisualVideoRequest(profile="fast", temp_dir=tmp_path / "cache"),
            permit=permit,
        )
    assert cancelled.value.exit_code == 130
    partials = list((tmp_path / "cache").glob("*.part"))
    assert len(partials) == 1 and partials[0].read_bytes() == b"partial"


def test_media_registry_keeps_audio_and_video_assets_by_capability(tmp_path: Path) -> None:
    source = SourceRef(kind="url", value="https://video.example/lecture")
    audio = MediaAsset(
        id="audio",
        source=source,
        local_path=tmp_path / "a.m4s",
        purpose="asr",
        format_id="a",
        provider="fixture",
        capabilities={"audio"},
    )
    video = MediaAsset(
        id="video",
        source=source,
        local_path=tmp_path / "v.m4s",
        purpose="visual",
        format_id="v",
        provider="fixture",
        capabilities={"video"},
    )
    registry = MediaRegistry()

    registry.adopt(audio)
    registry.adopt(video)

    assert registry.find(AsrAudioRequest()) is audio
    assert registry.find(VisualVideoRequest(profile="high")) is video
