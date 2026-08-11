from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.artifact.manifest import Manifest
from vctx.cli import app
from vctx.models.knowledge_flow import KnowledgeFlow
from vctx.source.session import VideoMetadata
from vctx.transcript import Transcript

_FIXED_TED_URL = "https://www.ted.com/talks/terry_moore_how_to_tie_your_shoes"
_RUN_NETWORK = os.environ.get("VCTX_RUN_NETWORK_INTEGRATION") == "1"

runner = CliRunner()

pytestmark = pytest.mark.skipif(
    not _RUN_NETWORK,
    reason="set VCTX_RUN_NETWORK_INTEGRATION=1 to run fixed-source network integration",
)


def test_fixed_ted_source_writes_transcript_context_pack(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixed-ted-source"

    result = runner.invoke(
        app,
        [
            "prepare",
            _FIXED_TED_URL,
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    manifest = Manifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    lane = out_dir / manifest.sources[0].path
    _assert_required_artifacts(lane)
    assert manifest.status == "ok"
    assert _step_status(manifest, "source.detect") == "ok"
    assert _step_status(manifest, "metadata.extract") == "ok"
    assert _step_status(manifest, "transcript.extract") == "ok"
    assert _step_status(manifest, "transcript.parse") == "ok"

    metadata = VideoMetadata.model_validate_json(
        (lane / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata.source_type == "url"
    assert metadata.raw_provider == "yt-dlp"
    assert metadata.extractor == "TedTalk"
    assert metadata.webpage_url is not None
    assert "ted.com/talks/terry_moore_how_to_tie_your_shoes" in metadata.webpage_url

    transcript = Transcript.model_validate_json(
        (lane / "transcript.json").read_text(encoding="utf-8")
    )
    assert transcript.provenance.provider == "yt-dlp"
    assert transcript.provenance.method in {"official_subtitles", "automatic_subtitles"}
    assert transcript.provenance.language_evidence.kind == "detected"
    assert transcript.provenance.language_evidence.code == "en"
    assert transcript.provenance.format == "vtt"
    assert len(transcript.segments) >= 5
    assert all("#EXTM3U" not in segment.text for segment in transcript.segments)
    transcript_chars = sum(len(segment.text) for segment in transcript.segments)
    assert transcript_chars >= 1000

    context = (lane / "context.md").read_text(encoding="utf-8")
    readable = (lane / "read.md").read_text(encoding="utf-8")
    assert "# Agent Context Pack" in context
    assert "Source:" in readable
    assert "tie" in readable.lower()

    knowledge_flow_path = lane / "knowledge_flow.json"
    if knowledge_flow_path.exists():
        KnowledgeFlow.model_validate_json(knowledge_flow_path.read_text(encoding="utf-8"))


def _assert_required_artifacts(out_dir: Path) -> None:
    required = {
        "metadata.json",
        "transcript.json",
        "chunks.json",
        "context.md",
        "read.md",
    }
    missing = [name for name in sorted(required) if not (out_dir / name).is_file()]
    assert missing == []


def _step_status(manifest: Manifest, name: str) -> str:
    for step in manifest.sources[0].steps:
        if step.name == name:
            return step.status
    raise AssertionError(f"missing manifest step: {name}")
