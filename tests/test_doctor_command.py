from __future__ import annotations

from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


def test_doctor_reports_environment_checks() -> None:
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "python:" in result.output
    assert "yt-dlp:" in result.output
    assert "cache:" in result.output
    assert "ffmpeg:" in result.output


def test_doctor_help_has_no_negation_flags() -> None:
    result = runner.invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    assert "--no-" not in result.output
