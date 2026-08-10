from __future__ import annotations

import json

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


def test_doctor_reports_selected_product_policy_as_json(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "doctor",
            "--workflow",
            "visual",
            "--offline",
            "--no-retain-media",
            "--cache-dir",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["profile"] in {"core", "asr", "visual", "full"}
    assert report["workflow"] == "visual"
    assert report["offline"] is True
    assert report["retention"] == "omit"
    assert report["capabilities"]["asr"]["selector"] == "auto"
    assert report["capabilities"]["ocr"]["selector"] == "auto"
    assert report["capabilities"]["vision"]["selector"] == "auto"
