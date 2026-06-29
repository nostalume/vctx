from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


def test_chunk_command_writes_chunkset_from_transcript_json(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.clean.json"
    transcript.write_text(
        json.dumps(
            {
                "video_id": "lecture",
                "provenance": {"method": "local_file", "language": "en", "format": "srt"},
                "segments": [
                    {"id": "seg_000001", "start": 0.0, "end": 1.0, "text": "hello"},
                    {"id": "seg_000002", "start": 1.0, "end": 2.0, "text": "world"},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "chunks.json"

    result = runner.invoke(
        app,
        ["chunk", str(transcript), "--out", str(out), "--chunk-max-chars", "8"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["video_id"] == "lecture"
    assert [chunk["text"] for chunk in payload["chunks"]] == ["hello", "world"]
    assert "Wrote chunks:" in result.output


def test_chunk_help_uses_output_file_and_no_negation_flags() -> None:
    result = runner.invoke(app, ["chunk", "--help"])

    assert result.exit_code == 0
    assert "--out" in result.output
    assert "--chunk-max-chars" in result.output
    assert "--no-" not in result.output
