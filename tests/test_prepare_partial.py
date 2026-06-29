from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


class FakeYoutubeDL:
    info: dict[str, object] = {}

    def __init__(self, params: dict[str, object]) -> None:
        self.params = params

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
        assert value == "https://video.example/watch?v=abc"
        assert download is False
        return self.info


def test_prepare_metadata_workflow_writes_metadata_only_partial_pack(tmp_path: Path) -> None:
    source = tmp_path / "lecture.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:01,000
hello
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["prepare", str(source), "--out", str(out_dir), "--workflow", "metadata"],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote partial context pack" in result.output
    assert "Workflow: metadata" in result.output
    assert "Status: partial" in result.output
    assert "Warnings:" in result.output
    assert "Metadata:" in result.output
    assert "Context:" not in result.output
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert not (out_dir / "transcript.clean.json").exists()
    assert not (out_dir / "chunks.json").exists()
    assert not (out_dir / "context.md").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert {artifact["path"] for artifact in manifest["artifacts"]} == {"metadata.json"}
    assert manifest["warnings"] == ["metadata workflow selected; transcript pipeline skipped"]
    assert _step_status(manifest, "transcript.extract") == "skipped"


def test_prepare_url_without_subtitles_writes_metadata_partial_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as module

    FakeYoutubeDL.info = {
        "id": "abc",
        "title": "Lecture",
        "webpage_url": "https://video.example/watch?v=abc",
        "extractor": "example",
        "subtitles": {},
        "automatic_captions": {},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["prepare", "https://video.example/watch?v=abc", "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote partial context pack" in result.output
    assert "Workflow: default" in result.output
    assert "Status: partial" in result.output
    assert "Routes:" in result.output
    assert "Metadata:" in result.output
    assert "Context:" not in result.output
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert not (out_dir / "transcript.clean.json").exists()
    assert not (out_dir / "chunks.json").exists()

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "example__abc"

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert _step_status(manifest, "transcript.extract") == "warning"
    assert _step_status(manifest, "transform.asr") == "warning"
    assert "No subtitles found" in "\n".join(manifest["warnings"])
    assert "Provide a transcript file" in "\n".join(manifest["warnings"])


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
