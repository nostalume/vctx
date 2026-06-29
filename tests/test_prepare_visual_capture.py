from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from vctx.cli import app
from vctx.transcript import TranscriptPayload, TranscriptProvenance

runner = CliRunner()


def test_prepare_visual_workflow_writes_capture_records_and_frame_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.transforms.asr as asr_module
    import vctx.transforms.visual_frames as visual_frames_module
    import vctx.transforms.visual_ocr as visual_ocr_module
    import vctx.transforms.visual_routes as visual_routes_module
    from vctx.models.visual import Evidence, FrameAsset

    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"fake mp4 bytes")
    config = tmp_path / "vctx.toml"
    config.write_text(
        """
[transforms.asr]
instance = "local-default"

[instances.asr.local-default]
type = "local-faster-whisper"
model = "tiny"
cache = "persistent"
""".strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    def fake_transcribe(self: object, media_asset: object) -> TranscriptPayload:
        del self, media_asset
        return TranscriptPayload(
            text="WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nThis chart compares child mortality.\n",
            format="vtt",
            provenance=TranscriptProvenance(
                method="asr",
                language="en",
                format="vtt",
                provider="faster-whisper",
            ),
        )

    def fake_extract_frames(
        media_asset: object,
        sample_action: object,
        frames_dir: Path,
    ) -> list[FrameAsset]:
        del media_asset, sample_action
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / "frame-0001.png"
        frame_path.write_bytes(b"fake png bytes")
        return [
            FrameAsset(
                id="frame-0001",
                timestamp_seconds=1.0,
                path=frame_path,
                source="cover",
                evidence=[Evidence(kind="probe", name="test-frame", weight=1.0)],
            )
        ]

    def fake_rapidocr_available() -> bool:
        return True

    def fake_extract_text(self: object, frame: object) -> str:
        del self, frame
        raise visual_ocr_module.OcrExecutionError("fixture OCR failure")

    monkeypatch.setattr(asr_module.FasterWhisperAsrAdapter, "transcribe", fake_transcribe)
    monkeypatch.setattr(visual_frames_module, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(visual_routes_module, "rapidocr_available", fake_rapidocr_available)
    monkeypatch.setattr(visual_ocr_module.RapidOcrAdapter, "extract_text", fake_extract_text)

    result = runner.invoke(
        app,
        [
            "prepare",
            str(media),
            "--out",
            str(out_dir),
            "--config",
            str(config),
            "--workflow",
            "visual",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "visual_records.json").exists()
    assert (out_dir / "visual" / "frames" / "frame-0001.png").exists()

    visual_records = json.loads(
        (out_dir / "visual_records.json").read_text(encoding="utf-8")
    )
    assert visual_records["records"][0]["kind"] == "capture"
    assert visual_records["records"][0]["artifact_path"] == "visual/frames/frame-0001.png"

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert _step_status(manifest, "transform.visual_plan") == "ok"
    assert _step_status(manifest, "transform.visual_capture") == "ok"
    assert {artifact["path"] for artifact in manifest["artifacts"]} >= {
        "visual_records.json",
        "visual/frames/frame-0001.png",
    }

    context = (out_dir / "context.md").read_text(encoding="utf-8")
    assert "## Visual references" in context
    assert (
        '<visual_ref id="frame-0001" timestamp="00:00:01" '
        'path="visual/frames/frame-0001.png">'
    ) in context

    readable = (out_dir / "readable.md").read_text(encoding="utf-8")
    assert "## Visual references" in readable
    assert "### 00:00:01 — frame-0001" in readable
    assert "![Frame frame-0001 at 00:00:01](visual/frames/frame-0001.png)" in readable


def _step_status(manifest: dict[str, Any], name: str) -> str:
    steps = manifest["steps"]
    assert isinstance(steps, list)
    for raw_step in steps:
        assert isinstance(raw_step, dict)
        step = cast(dict[str, Any], raw_step)
        if step["name"] == name:
            status = step["status"]
            assert isinstance(status, str)
            return status
    raise AssertionError(f"missing manifest step: {name}")


def test_prepare_visual_no_motives_skips_visual_media_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as ytdlp_module

    class FakeYoutubeDL:
        calls: list[bool] = []

        def __init__(self, params: dict[str, object]) -> None:
            self.params = params

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
            assert value == "https://video.example/watch?v=plain"
            self.calls.append(download)
            if download:
                raise AssertionError("visual media download should be gated by motives")
            return {
                "id": "plain",
                "title": "Plain Talk",
                "duration": 3,
                "webpage_url": value,
                "language": "en",
                "extractor": "example",
                "subtitles": {"en": [{"ext": "vtt", "url": "https://cdn.example/plain.vtt"}]},
                "automatic_captions": {},
            }

    class FakeSubtitleRuntime:
        def request(self, request: object) -> object:
            from vctx.net import NetResponse

            return NetResponse(
                url="https://cdn.example/plain.vtt",
                status_code=200,
                headers={"content-type": "text/vtt"},
                body=(
                    b"WEBVTT\n\n"
                    b"00:00:00.000 --> 00:00:03.000\n"
                    b"This is spoken explanation only.\n"
                ),
            )

    monkeypatch.setattr(ytdlp_module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(ytdlp_module, "UrllibNetRuntime", FakeSubtitleRuntime)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "prepare",
            "https://video.example/watch?v=plain",
            "--out",
            str(out_dir),
            "--workflow",
            "visual",
        ],
    )

    assert result.exit_code == 0, result.output
    assert True not in FakeYoutubeDL.calls
    assert not (out_dir / "visual_records.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert _step_status(manifest, "transform.visual_plan") == "skipped"
