from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from tests.support import asr_ready
from vctx.cli import app
from vctx.source.session import MediaAsset

runner = CliRunner()


def test_prepare_local_media_runs_asr_and_writes_full_context_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.asr as asr_module

    media = tmp_path / "lecture.wav"
    media.write_bytes(b"fake wav bytes")
    config = tmp_path / "vctx.toml"
    config.write_text(
        """
[transforms.asr]
use = "instance:local-default"

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
        return asr_ready(media_asset.id, "Hello from fake ASR.", model="tiny")

    monkeypatch.setattr(asr_module.FasterWhisperAsrAdapter, "transcribe", fake_transcribe)

    result = runner.invoke(
        app,
        ["prepare", str(media), "--out", str(out_dir), "--config", str(config)],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote context pack" in result.output
    assert "Wrote partial context pack" not in result.output
    assert (out_dir / "manifest.json").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    source_entry = manifest["sources"][0]
    lane = out_dir / source_entry["path"]
    assert {path.name for path in lane.iterdir()} >= {
        "metadata.json", "transcript.json", "chunks.json", "context.md"
    }
    assert manifest["status"] == "ok"
    assert _step_status(manifest, "transcript.extract") == "warning"
    assert _step_status(manifest, "source.media") == "ok"
    assert _step_status(manifest, "transform.asr") == "ok"
    assert _step_detail(manifest, "transform.asr") == "faster-whisper:tiny"

    transcript = json.loads((lane / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["provenance"]["method"] == "asr"
    assert transcript["provenance"]["provider"] == "faster-whisper"
    assert transcript["segments"][0]["text"] == "Hello from fake ASR."

    context = (lane / "context.md").read_text(encoding="utf-8")
    assert "Hello from fake ASR." in context


def _step_status(manifest: dict[str, Any], name: str) -> str:
    step = _step(manifest, name)
    status = step["status"]
    assert isinstance(status, str)
    return status


def _step_detail(manifest: dict[str, Any], name: str) -> str:
    step = _step(manifest, name)
    detail = step["detail"]
    assert isinstance(detail, str)
    return detail


def _step(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    steps = manifest["sources"][0]["steps"]
    assert isinstance(steps, list)
    for raw_step in steps:
        assert isinstance(raw_step, dict)
        step = cast(dict[str, Any], raw_step)
        if step["name"] == name:
            return step
    raise AssertionError(f"missing manifest step: {name}")
