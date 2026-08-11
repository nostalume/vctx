from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from tests.support import asr_ready_segments
from vctx.cli import app
from vctx.source.session import MediaAsset

runner = CliRunner()


def test_prepare_visual_workflow_runs_configured_vlm_description(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.asr as asr_module
    import vctx.transforms.visual_frames as visual_frames_module
    import vctx.transforms.visual_vlm as visual_vlm_module
    from vctx.models.visual import Evidence, FrameAsset

    media = tmp_path / "architecture diagram.mp4"
    media.write_bytes(b"fake mp4 bytes")
    env_file = tmp_path / ".env"
    env_file.write_text("VISION_KEY=test-secret\n", encoding="utf-8")
    config = tmp_path / "vctx.toml"
    config.write_text(
        f"""
[runtime]
env_files = ["{env_file.name}"]

[transforms.asr]
use = "instance:local-default"

[transforms.visual_context]
use = "instance:test-vlm"

[instances.asr.local-default]
type = "local-faster-whisper"
model = "tiny"
cache = "persistent"

[instances.vision.test-vlm]
type = "openai-compatible-vision"
base_url = "https://example.invalid/v1/chat/completions"
api_key_env = "VISION_KEY"
model = "vision-test"
""".strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    def fake_transcribe(self: object, media_asset: MediaAsset) -> object:
        del self
        return asr_ready_segments(
            media_asset.id, [(0.0, 2.0, "This architecture diagram shows the flow.")]
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

    def fake_describe(self: object, frame: FrameAsset) -> str:
        del self, frame
        return "A left-to-right service diagram from ingestion to context pack."

    monkeypatch.setattr(asr_module.FasterWhisperAsrAdapter, "transcribe", fake_transcribe)
    monkeypatch.setattr(visual_frames_module, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        visual_vlm_module.OpenAiCompatibleVisionAdapter,
        "describe",
        fake_describe,
    )

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
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    lane = out_dir / manifest["sources"][0]["path"]
    visual_records = json.loads(
        (lane / "visual_records.json").read_text(encoding="utf-8")
    )
    assert [record["kind"] for record in visual_records["records"]] == [
        "description",
        "capture",
    ]
    assert (
        visual_records["records"][0]["text"]
        == "A left-to-right service diagram from ingestion to context pack."
    )

    assert _step_status(manifest, "transform.visual_plan") == "ok"
    assert _step_detail(manifest, "transform.visual_plan") == "configured VLM: test-vlm"
    assert _step_status(manifest, "transform.visual_capture") == "ok"

    context = (lane / "context.md").read_text(encoding="utf-8")
    assert (
        '<visual_ref id="frame-0001" timestamp="00:00:01" '
        'path="frame-0001.png">'
    ) in context
    assert "<description" in context
    assert (
        "A left-to-right service diagram from ingestion to context pack.</description>"
        in context
    )

    readable = (lane / "read.md").read_text(encoding="utf-8")
    assert "![Frame frame-0001 at 00:00:01](frame-0001.png)" in readable
    assert "- DESCRIPTION: A left-to-right service diagram" in readable


def test_prepare_visual_workflow_keeps_capture_when_vlm_description_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.asr as asr_module
    import vctx.transforms.visual_frames as visual_frames_module
    import vctx.transforms.visual_vlm as visual_vlm_module
    from vctx.models.visual import Evidence, FrameAsset

    media = tmp_path / "architecture diagram.mp4"
    media.write_bytes(b"fake mp4 bytes")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    config = tmp_path / "vctx.toml"
    config.write_text(
        """
[transforms.asr]
use = "instance:local-default"

[transforms.visual_context]
use = "openrouter:qwen/qwen3-vl-8b-instruct"

[instances.asr.local-default]
type = "local-faster-whisper"
model = "tiny"
cache = "persistent"
""".strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    def fake_transcribe(self: object, media_asset: MediaAsset) -> object:
        del self
        return asr_ready_segments(
            media_asset.id, [(10.0, 14.0, "This diagram explains the visual workflow.")]
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
                timestamp_seconds=12.0,
                path=frame_path,
                source="transcript_anchor",
                evidence=[Evidence(kind="transcript", name="diagram", weight=0.9)],
            )
        ]

    def fake_describe(self: object, frame: FrameAsset) -> str:
        del self, frame
        raise visual_vlm_module.VisionExecutionError("fixture VLM failure")

    monkeypatch.setattr(asr_module.FasterWhisperAsrAdapter, "transcribe", fake_transcribe)
    monkeypatch.setattr(visual_frames_module, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        visual_vlm_module.OpenAiCompatibleVisionAdapter,
        "describe",
        fake_describe,
    )

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
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    lane = out_dir / manifest["sources"][0]["path"]
    visual_records = json.loads(
        (lane / "visual_records.json").read_text(encoding="utf-8")
    )
    assert [record["kind"] for record in visual_records["records"]] == ["capture"]

    assert _step_status(manifest, "transform.visual_plan") == "ok"
    assert _step_status(manifest, "transform.visual_capture") == "ok"
    assert "test-secret" not in json.dumps(manifest)


def _step_status(manifest: dict[str, Any], name: str) -> str:
    return _step_value(manifest, name, "status")


def _step_detail(manifest: dict[str, Any], name: str) -> str:
    return _step_value(manifest, name, "detail")


def _step_value(manifest: dict[str, Any], name: str, key: str) -> str:
    steps = manifest["sources"][0]["steps"]
    assert isinstance(steps, list)
    for raw_step in steps:
        assert isinstance(raw_step, dict)
        step = cast(dict[str, Any], raw_step)
        if step["name"] == name:
            value = step[key]
            assert isinstance(value, str)
            return value
    raise AssertionError(f"missing manifest step: {name}")
