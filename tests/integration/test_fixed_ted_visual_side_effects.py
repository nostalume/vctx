from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.cli import app
from vctx.config import (
    CapabilityEnabled,
    PrepareRequest,
    WorkflowProfile,
    resolve_config,
)
from vctx.models import SourceRef
from vctx.models.manifest import ArtifactRef, Manifest, ManifestBuilder
from vctx.models.metadata import VideoMetadata
from vctx.models.visual import VisualRecordSet
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment

_FIXED_TED_URL = "https://www.ted.com/talks/terry_moore_how_to_tie_your_shoes"
_TEMPLATE_PATH = Path("docs/examples/vctx.visual-auto-side-effect.toml")
_RUN_TED_VISUAL = os.environ.get("VCTX_RUN_TED_VISUAL_INTEGRATION") == "1"
_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
_SECRET_SENTINEL = "vctx-secret-sentinel-do-not-leak"
_VISUAL_REF = (
    '<visual_ref id="frame-0001" timestamp="00:00:01" '
    'path="visual/frames/frame-0001.png">\n'
)
_FRAME_MARKDOWN = "![Frame frame-0001 at 00:00:01](visual/frames/frame-0001.png)"

runner = CliRunner()


def test_visual_auto_template_uses_gitignored_env_file_and_no_ocr_language_config(
    tmp_path: Path,
) -> None:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    gitignore = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    assert 'env_files = [".env"]' in template
    assert "OPENROUTER_API_KEY" in template
    assert ".env" in gitignore
    assert ".env.*" in gitignore
    assert "[instances.ocr" not in template
    assert "ocr_language" not in template
    assert "rec_lang" not in template
    assert "det_lang" not in template

    resolved = resolve_config(
        PrepareRequest(
            input=_FIXED_TED_URL,
            out_dir=tmp_path / "out",
            config_path=_TEMPLATE_PATH,
        )
    )

    assert resolved.runtime.workflow == WorkflowProfile.VISUAL
    assert resolved.runtime.env_files == [_TEMPLATE_PATH.parent / ".env"]
    assert resolved.source.yt_dlp.media_profile == "balanced"
    assert resolved.source.yt_dlp.subtitle_languages == ["en", "ja", "zh-Hans", "zh-Hant"]
    assert resolved.transforms.visual_context.enabled == CapabilityEnabled.TRUE
    assert resolved.transforms.visual_context.model == "auto"
    assert resolved.transforms.visual_context.instance is None
    assert resolved.instances.asr == {}


def test_visual_side_effect_invariant_checker_accepts_complete_fixture(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixture"
    _write_visual_fixture(
        out_dir,
        records=[
            {
                "id": "ocr-0001",
                "timestamp_seconds": 1.0,
                "frame_id": "frame-0001",
                "kind": "ocr",
                "text": "鞋带",
            },
            {
                "id": "capture-0001",
                "timestamp_seconds": 1.0,
                "frame_id": "frame-0001",
                "kind": "capture",
                "artifact_path": "visual/frames/frame-0001.png",
            },
        ],
        context=_VISUAL_REF + '<ocr frame_id="frame-0001">鞋带</ocr>\n',
        readable=f"{_FRAME_MARKDOWN}\n- OCR: 鞋带",
    )

    _assert_visual_side_effect_invariants(
        out_dir,
        source_url=_FIXED_TED_URL,
        openrouter_secret=_OPENROUTER_KEY,
    )


def test_visual_side_effect_invariant_checker_accepts_capture_only_fixture(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "capture-only"
    _write_visual_fixture(
        out_dir,
        records=[
            {
                "id": "capture-0001",
                "timestamp_seconds": 1.0,
                "frame_id": "frame-0001",
                "kind": "capture",
                "artifact_path": "visual/frames/frame-0001.png",
            }
        ],
        context=_VISUAL_REF,
        readable=_FRAME_MARKDOWN,
    )

    _assert_visual_side_effect_invariants(
        out_dir,
        source_url=_FIXED_TED_URL,
        openrouter_secret=None,
    )


def test_visual_side_effect_invariant_checker_preserves_native_multilingual_ocr(
    tmp_path: Path,
) -> None:
    native_text = "鞋带 / 靴ひも / résumé"
    out_dir = tmp_path / "native-ocr"
    _write_visual_fixture(
        out_dir,
        records=[
            {
                "id": "ocr-0001",
                "timestamp_seconds": 1.0,
                "frame_id": "frame-0001",
                "kind": "ocr",
                "text": native_text,
            },
            {
                "id": "capture-0001",
                "timestamp_seconds": 1.0,
                "frame_id": "frame-0001",
                "kind": "capture",
                "artifact_path": "visual/frames/frame-0001.png",
            },
        ],
        context=_VISUAL_REF + f'<ocr frame_id="frame-0001">{native_text}</ocr>\n',
        readable=f"{_FRAME_MARKDOWN}\n- OCR: {native_text}",
    )

    _assert_visual_side_effect_invariants(
        out_dir,
        source_url=_FIXED_TED_URL,
        openrouter_secret=None,
    )

    visual_records = VisualRecordSet.model_validate_json(
        (out_dir / "visual_records.json").read_text(encoding="utf-8")
    )
    assert any(record.text == native_text for record in visual_records.records)
    assert native_text in (out_dir / "context.md").read_text(encoding="utf-8")
    assert native_text in (out_dir / "readable.md").read_text(encoding="utf-8")


def test_visual_side_effect_invariant_checker_rejects_absolute_artifact_path(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "absolute-artifact"
    outside = tmp_path / "outside-frame.png"
    outside.write_bytes(b"fake png")
    _write_visual_fixture(
        out_dir,
        records=[
            {
                "id": "capture-0001",
                "timestamp_seconds": 1.0,
                "frame_id": "frame-0001",
                "kind": "capture",
                "artifact_path": str(outside),
            }
        ],
        context=_VISUAL_REF,
        readable=_FRAME_MARKDOWN,
    )

    with pytest.raises(AssertionError, match="artifact_path must be relative"):
        _assert_visual_side_effect_invariants(
            out_dir,
            source_url=_FIXED_TED_URL,
            openrouter_secret=None,
        )


def test_visual_side_effect_invariant_checker_rejects_secret_leak(tmp_path: Path) -> None:
    out_dir = tmp_path / "secret-leak"
    _write_visual_fixture(
        out_dir,
        records=[
            {
                "id": "capture-0001",
                "timestamp_seconds": 1.0,
                "frame_id": "frame-0001",
                "kind": "capture",
                "artifact_path": "visual/frames/frame-0001.png",
            }
        ],
        context=_VISUAL_REF + _SECRET_SENTINEL,
        readable=_FRAME_MARKDOWN,
    )

    with pytest.raises(AssertionError):
        _assert_visual_side_effect_invariants(
            out_dir,
            source_url=_FIXED_TED_URL,
            openrouter_secret=_SECRET_SENTINEL,
        )


@pytest.mark.skipif(
    not _RUN_TED_VISUAL,
    reason="set VCTX_RUN_TED_VISUAL_INTEGRATION=1 to run fixed TED visual side-effect integration",
)
def test_fixed_ted_visual_workflow_preserves_pipeline_invariants(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixed-ted-visual"

    result = runner.invoke(
        app,
        [
            "prepare",
            _FIXED_TED_URL,
            "--out",
            str(out_dir),
            "--config",
            str(_TEMPLATE_PATH),
            "--workflow",
            "visual",
        ],
    )

    assert result.exit_code == 0, result.output
    _assert_visual_side_effect_invariants(
        out_dir,
        source_url=_FIXED_TED_URL,
        openrouter_secret=_OPENROUTER_KEY,
    )


def _write_visual_fixture(
    out_dir: Path,
    *,
    records: list[dict[str, object]],
    context: str,
    readable: str,
) -> None:
    frames_dir = out_dir / "visual" / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame-0001.png").write_bytes(b"fake png")

    manifest_builder = ManifestBuilder.start("fixture", "test")
    manifest_builder.add_step("source.detect", "ok", "local")
    manifest_builder.add_step("metadata.extract", "ok", "fixture")
    manifest_builder.add_step("source.media", "ok", "source media: video: fixture.mp4")
    manifest_builder.add_step("transcript.extract", "ok", "fixture")
    manifest_builder.add_step("transcript.parse", "ok", "vtt")
    manifest_builder.add_step("transform.visual_plan", "ok", "sample -> ocr -> capture")
    manifest_builder.add_step("transform.visual_capture", "ok", f"{len(records)} visual records")
    manifest = manifest_builder.finish(
        "ok",
        [
            ArtifactRef(kind="manifest", path="manifest.json", media_type="application/json"),
            ArtifactRef(kind="metadata", path="metadata.json", media_type="application/json"),
            ArtifactRef(
                kind="visual_records",
                path="visual_records.json",
                media_type="application/json",
            ),
        ],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    metadata = VideoMetadata(
        id="fixture",
        title="Fixture",
        source_type="url",
        source=SourceRef(kind="url", value=_FIXED_TED_URL),
        raw_provider="yt-dlp",
        extractor="TedTalk",
        webpage_url=_FIXED_TED_URL,
    )
    (out_dir / "metadata.json").write_text(metadata.model_dump_json(), encoding="utf-8")

    transcript = Transcript(
        video_id="fixture",
        provenance=TranscriptProvenance(
            method="official_subtitles",
            language="en",
            provider="yt-dlp",
            format="vtt",
        ),
        segments=[TranscriptSegment(id="s1", start=0.0, end=2.0, text="Tie your shoes.")],
    )
    (out_dir / "transcript.raw.json").write_text(transcript.model_dump_json(), encoding="utf-8")
    (out_dir / "transcript.clean.json").write_text(transcript.model_dump_json(), encoding="utf-8")
    (out_dir / "chunks.json").write_text('{"chunks": []}', encoding="utf-8")
    (out_dir / "transcript.md").write_text("Tie your shoes.", encoding="utf-8")
    (out_dir / "context.md").write_text(context, encoding="utf-8")
    (out_dir / "readable.md").write_text(readable, encoding="utf-8")
    (out_dir / "visual_records.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False),
        encoding="utf-8",
    )


def _assert_visual_side_effect_invariants(
    out_dir: Path, *, source_url: str, openrouter_secret: str | None
) -> None:
    _assert_required_artifacts(out_dir)

    manifest = Manifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.status in {"ok", "partial"}
    assert _step_status(manifest, "source.detect") == "ok"
    assert _step_status(manifest, "metadata.extract") == "ok"
    assert _step_status(manifest, "source.media") == "ok"
    assert _step_status(manifest, "transcript.extract") == "ok"
    assert _step_status(manifest, "transcript.parse") == "ok"
    assert _step_status(manifest, "transform.visual_plan") == "ok"
    assert _step_status(manifest, "transform.visual_capture") == "ok"

    metadata = VideoMetadata.model_validate_json(
        (out_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata.source_type == "url"
    assert metadata.raw_provider == "yt-dlp"
    assert metadata.webpage_url is not None
    assert source_url.removeprefix("https://www.") in metadata.webpage_url.removeprefix(
        "https://www."
    )

    raw = Transcript.model_validate_json(
        (out_dir / "transcript.raw.json").read_text(encoding="utf-8")
    )
    clean = Transcript.model_validate_json(
        (out_dir / "transcript.clean.json").read_text(encoding="utf-8")
    )
    assert raw.provenance.provider == "yt-dlp"
    assert raw.provenance.method in {"official_subtitles", "automatic_subtitles"}
    assert len(raw.segments) >= 1
    assert len(clean.segments) >= 1
    assert all("#EXTM3U" not in segment.text for segment in raw.segments)

    visual_records = VisualRecordSet.model_validate_json(
        (out_dir / "visual_records.json").read_text(encoding="utf-8")
    )
    assert visual_records.records, "visual workflow should keep at least capture provenance"
    capture_records = [record for record in visual_records.records if record.kind == "capture"]
    assert capture_records, "visual workflow should keep captured frame artifacts"
    frame_ids = {record.frame_id for record in capture_records}
    assert frame_ids

    for record in visual_records.records:
        assert record.frame_id
        if record.kind in {"ocr", "description"}:
            assert record.text is not None
            assert record.text.strip()
            assert record.frame_id in frame_ids
        if record.artifact_path is not None:
            artifact_path = Path(record.artifact_path)
            assert not artifact_path.is_absolute(), "artifact_path must be relative"
            resolved_artifact = out_dir / artifact_path
            assert resolved_artifact.is_file(), record.artifact_path

    context = (out_dir / "context.md").read_text(encoding="utf-8")
    readable = (out_dir / "readable.md").read_text(encoding="utf-8")
    assert "<visual_ref" in context
    assert "![Frame" in readable

    if openrouter_secret:
        _assert_secret_not_leaked(out_dir, openrouter_secret)


def _assert_required_artifacts(out_dir: Path) -> None:
    required = {
        "manifest.json",
        "metadata.json",
        "transcript.raw.json",
        "transcript.clean.json",
        "chunks.json",
        "context.md",
        "readable.md",
        "transcript.md",
        "visual_records.json",
    }
    missing = [name for name in sorted(required) if not (out_dir / name).is_file()]
    assert missing == []


def _step_status(manifest: Manifest, name: str) -> str:
    for step in manifest.steps:
        if step.name == name:
            return step.status
    raise AssertionError(f"missing manifest step: {name}")


def _assert_secret_not_leaked(out_dir: Path, secret: str) -> None:
    if not secret:
        return
    checked = [
        out_dir / "manifest.json",
        out_dir / "metadata.json",
        out_dir / "context.md",
        out_dir / "readable.md",
        out_dir / "visual_records.json",
    ]
    for path in checked:
        assert secret not in path.read_text(encoding="utf-8")
