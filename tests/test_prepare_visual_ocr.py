from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from vctx.cli import app
from vctx.transcript import TranscriptPayload, TranscriptProvenance

runner = CliRunner()


def test_prepare_visual_workflow_runs_available_local_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.transforms.asr as asr_module
    import vctx.transforms.visual_frames as visual_frames_module
    import vctx.transforms.visual_ocr as visual_ocr_module
    import vctx.transforms.visual_routes as visual_routes_module
    from vctx.models.visual import Evidence, FrameAsset

    media = tmp_path / "slides.mp4"
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
            text="WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nLook at the slide text.\n",
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

    def fake_extract_text(self: object, frame: FrameAsset) -> str:
        del self, frame
        return "CAP theorem slide"

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
    visual_records = json.loads(
        (out_dir / "visual_records.json").read_text(encoding="utf-8")
    )
    visual_scores = json.loads((out_dir / "visual_scores.json").read_text(encoding="utf-8"))
    assert [record["kind"] for record in visual_records["records"]] == ["ocr", "capture"]
    assert visual_records["records"][0]["text"] == "CAP theorem slide"
    assert "satisfaction" not in visual_records
    assert [check["status"] for check in visual_scores["satisfaction"]] == [
        "satisfied",
        "satisfied",
    ]

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert _step_status(manifest, "transform.visual_plan") == "ok"
    assert _step_detail(manifest, "transform.visual_plan") == "local OCR: rapidocr"
    assert _step_status(manifest, "transform.visual_capture") == "ok"
    assert _step_status(manifest, "transform.visual_satisfaction") == "ok"
    assert {artifact["path"] for artifact in manifest["artifacts"]} >= {
        "visual_records.json",
        "visual_scores.json",
    }

    context = (out_dir / "context.md").read_text(encoding="utf-8")
    assert (
        '<visual_ref id="frame-0001" timestamp="00:00:01" '
        'path="visual/frames/frame-0001.png">'
    ) in context
    assert "<ocr" in context
    assert "CAP theorem slide</ocr>" in context

    readable = (out_dir / "readable.md").read_text(encoding="utf-8")
    assert "![Frame frame-0001 at 00:00:01](visual/frames/frame-0001.png)" in readable
    assert "- OCR: CAP theorem slide" in readable


def test_prepare_visual_writes_satisfaction_warning_for_missed_formula_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.transforms.asr as asr_module
    import vctx.transforms.visual_frames as visual_frames_module
    import vctx.transforms.visual_ocr as visual_ocr_module
    import vctx.transforms.visual_routes as visual_routes_module
    from vctx.models.visual import Evidence, FrameAsset

    media = tmp_path / "formula.mp4"
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
            text="WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nThis formula is shown on the board.\n",
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

    def fake_extract_text(self: object, frame: FrameAsset) -> str:
        del self, frame
        return "CAP theorem slide"

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
    visual_scores = json.loads((out_dir / "visual_scores.json").read_text(encoding="utf-8"))
    assert any(
        check["status"] == "missed"
        and check["operation"] == "ocr"
        and check["reason"] == "formula_or_equation"
        for check in visual_scores["satisfaction"]
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert _step_status(manifest, "transform.visual_satisfaction") == "warning"
    assert any("visual satisfaction missed" in warning for warning in manifest["warnings"])


def _step_status(manifest: dict[str, Any], name: str) -> str:
    return _step_value(manifest, name, "status")


def _step_detail(manifest: dict[str, Any], name: str) -> str:
    return _step_value(manifest, name, "detail")


def _step_value(manifest: dict[str, Any], name: str, key: str) -> str:
    steps = manifest["steps"]
    assert isinstance(steps, list)
    for raw_step in steps:
        assert isinstance(raw_step, dict)
        step = cast(dict[str, Any], raw_step)
        if step["name"] == name:
            value = step[key]
            assert isinstance(value, str)
            return value
    raise AssertionError(f"missing manifest step: {name}")
