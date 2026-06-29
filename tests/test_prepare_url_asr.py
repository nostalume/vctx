from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import TypeAlias, cast

import pytest
from typer.testing import CliRunner

from vctx.cli import app
from vctx.models.media import MediaAsset
from vctx.sources.ytdlp_source import YtDlpInfo, YtDlpParams
from vctx.transcript import TranscriptPayload, TranscriptProvenance

runner = CliRunner()

JsonScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]  # noqa: UP040
JsonObject: TypeAlias = dict[str, JsonValue]  # noqa: UP040


class FakeYoutubeDLMedia:
    info: YtDlpInfo = {}
    downloaded_path: Path
    calls: list[tuple[bool, YtDlpParams]] = []

    def __init__(self, params: YtDlpParams) -> None:
        self.params = params

    def __enter__(self) -> FakeYoutubeDLMedia:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def extract_info(self, value: str, download: bool = False) -> YtDlpInfo:
        assert value == "https://video.example/watch?v=no-captions"
        self.calls.append((download, self.params))
        if download:
            paths = self.params["paths"]
            assert isinstance(paths, dict)
            type(self).downloaded_path = Path(paths["home"]) / "example__no-captions.m4a"
            type(self).downloaded_path.parent.mkdir(parents=True, exist_ok=True)
            type(self).downloaded_path.write_bytes(b"fake downloaded audio")
            return {
                **self.info,
                "requested_downloads": [{"filepath": str(self.downloaded_path)}],
            }
        return self.info


def test_prepare_url_without_subtitles_downloads_media_and_runs_asr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as ytdlp_module
    import vctx.transforms.asr as asr_module

    FakeYoutubeDLMedia.calls = []
    FakeYoutubeDLMedia.downloaded_path = tmp_path / "downloaded" / "lecture.m4a"
    FakeYoutubeDLMedia.info = {
        "id": "no-captions",
        "title": "No Captions",
        "duration": 2,
        "webpage_url": "https://video.example/watch?v=no-captions",
        "language": "en",
        "extractor": "example",
        "subtitles": {},
        "automatic_captions": {},
    }
    monkeypatch.setattr(ytdlp_module.yt_dlp, "YoutubeDL", FakeYoutubeDLMedia)

    class FakeAsrAdapter:
        def __init__(self, **kwargs: JsonValue) -> None:
            del kwargs

        def transcribe(self, media_asset: MediaAsset) -> TranscriptPayload:
            assert media_asset.local_path == FakeYoutubeDLMedia.downloaded_path
            return TranscriptPayload(
                text="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nURL ASR text.\n",
                format="vtt",
                provenance=TranscriptProvenance(
                    method="asr", language="en", format="vtt", provider="faster-whisper"
                ),
            )

    monkeypatch.setattr(asr_module, "FasterWhisperAsrAdapter", FakeAsrAdapter)
    config_path = tmp_path / "vctx.toml"
    config_path.write_text(
        """
[transforms.asr]
instance = "local-default"

[instances.asr.local-default]
type = "local-faster-whisper"
model = "tiny"
""".strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "prepare",
            "https://video.example/watch?v=no-captions",
            "--out",
            str(out_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = cast(JsonObject, json.loads((out_dir / "manifest.json").read_text(encoding="utf-8")))
    assert manifest["status"] == "ok"
    assert _step_status(manifest, "source.media") == "ok"
    assert _step_status(manifest, "transform.asr") == "ok"
    assert manifest["transform_evidence"] == [
        {
            "capability": "asr",
            "selected_route": "local",
            "provider_id": "faster-whisper",
            "model_id": "tiny",
            "requires_user_config": False,
            "uploaded": False,
            "cost_may_apply": False,
            "deterministic": False,
            "source_artifacts": [],
            "output_artifacts": [],
            "reason": "default local ASR route available",
            "warnings": [],
        }
    ]
    clean = cast(
        JsonObject,
        json.loads((out_dir / "transcript.clean.json").read_text(encoding="utf-8")),
    )
    segments = cast(list[JsonObject], clean["segments"])
    assert segments[0]["text"] == "URL ASR text."
    assert any(download for download, _params in FakeYoutubeDLMedia.calls)
    download_params = [params for download, params in FakeYoutubeDLMedia.calls if download][0]
    assert download_params["skip_download"] is False
    assert download_params["paths"] == {
        "home": str(out_dir / "media"),
        "temp": str(tmp_path / "cache" / "tmp" / "yt-dlp"),
    }
    assert FakeYoutubeDLMedia.downloaded_path.parent == out_dir / "media"


def _step_status(manifest: JsonObject, name: str) -> str:
    step = _step(manifest, name)
    status = step["status"]
    assert isinstance(status, str)
    return status


def _step(manifest: JsonObject, name: str) -> JsonObject:
    steps = cast(list[JsonObject], manifest["steps"])
    for raw_step in steps:
        assert isinstance(raw_step, dict)
        if raw_step["name"] == name:
            return raw_step
    raise AssertionError(f"missing manifest step: {name}")
