from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


def _write_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "lecture",
                "source_type": "local-file",
                "source": {"kind": "file", "value": "lecture.srt"},
                "title": "Lecture",
            }
        ),
        encoding="utf-8",
    )


def _write_transcript(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "video_id": "lecture",
                "provenance": {"method": "local_file", "language": "en", "format": "srt"},
                "segments": [
                    {"id": "seg_000001", "start": 0.0, "end": 1.0, "text": "hello world"}
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_chunks(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "video_id": "lecture",
                "strategy": "chars-v1",
                "chunks": [
                    {
                        "id": "chunk_0001",
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello world",
                        "segment_ids": ["seg_000001"],
                        "char_count": 11,
                        "approx_token_count": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_render_command_writes_context_markdown(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    transcript = tmp_path / "transcript.clean.json"
    chunks = tmp_path / "chunks.json"
    out = tmp_path / "context.md"
    _write_metadata(metadata)
    _write_transcript(transcript)
    _write_chunks(chunks)

    result = runner.invoke(
        app,
        [
            "render",
            "--metadata",
            str(metadata),
            "--transcript",
            str(transcript),
            "--chunks",
            str(chunks),
            "--out",
            str(out),
            "--format",
            "context",
        ],
    )

    assert result.exit_code == 0, result.output
    content = out.read_text(encoding="utf-8")
    assert "# Agent Context Pack" in content
    assert '<chunk id="chunk_0001"' in content
    assert "Wrote render: " in result.output


def test_render_command_writes_transcript_markdown_without_chunks(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    transcript = tmp_path / "transcript.clean.json"
    out = tmp_path / "transcript.md"
    _write_metadata(metadata)
    _write_transcript(transcript)

    result = runner.invoke(
        app,
        [
            "render",
            "--metadata",
            str(metadata),
            "--transcript",
            str(transcript),
            "--out",
            str(out),
            "--format",
            "transcript",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "# Transcript — Lecture" in out.read_text(encoding="utf-8")


def test_render_help_uses_decisive_format_flag_without_negation_pairs() -> None:
    result = runner.invoke(app, ["render", "--help"])

    assert result.exit_code == 0
    assert "--format" in result.output
    assert "--no-" not in result.output
