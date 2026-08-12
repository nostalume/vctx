from __future__ import annotations

import json
from pathlib import Path

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
        "metadata.json",
        "transcript.json",
        "chunks.json",
        "context.md",
    }
    assert manifest["status"] == "ok"
    asr_route = next(
        item
        for item in source_entry["effects"]
        if item["operation"] == "asr" and item["route"] == "local"
    )
    assert (asr_route["provider"], asr_route["model"]) == ("faster-whisper", "tiny")

    transcript = json.loads((lane / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["provenance"]["method"] == "asr"
    assert transcript["provenance"]["provider"] == "faster-whisper"
    assert transcript["segments"][0]["text"] == "Hello from fake ASR."
    context = (lane / "context.md").read_text(encoding="utf-8")
    assert "Hello from fake ASR." in context
