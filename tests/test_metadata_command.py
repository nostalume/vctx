from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


def test_metadata_command_prints_human_readable_local_metadata(tmp_path: Path) -> None:
    source = tmp_path / "lecture.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    result = runner.invoke(app, ["metadata", str(source)])

    assert result.exit_code == 0, result.output
    assert "title: lecture" in result.output
    assert "source_type: local-file" in result.output
    assert "raw_provider: local-file" in result.output


def test_metadata_command_prints_json_local_metadata(tmp_path: Path) -> None:
    source = tmp_path / "lecture.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    result = runner.invoke(app, ["metadata", str(source), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"] == "local__lecture"
    assert payload["source_type"] == "local-file"
    assert payload["source"] == {"kind": "file", "value": str(source)}
    assert payload["raw_provider"] == "local-file"


def test_metadata_help_uses_decisive_json_flag_without_negation_pair() -> None:
    result = runner.invoke(app, ["metadata", "--help"])

    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--no-json" not in result.output
