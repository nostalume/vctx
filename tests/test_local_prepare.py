from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


def test_prepare_local_srt_writes_context_pack(tmp_path: Path) -> None:
    source = tmp_path / "lecture.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:02,000
Hello <b>world</b>.

2
00:00:02,000 --> 00:00:05,000
This is a second caption.
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert "INFO vctx.app.prepare" not in result.output
    assert "Workflow: default" in result.output
    assert "Status: ok" in result.output
    assert "Config: built-in defaults + CLI" in result.output
    assert "Artifacts:" in result.output
    assert "  - metadata.json" in result.output
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "transcript.raw.json").exists()
    assert (out_dir / "transcript.clean.json").exists()
    assert (out_dir / "chunks.json").exists()
    assert (out_dir / "context.md").exists()
    assert (out_dir / "readable.md").exists()
    assert (out_dir / "transcript.md").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["input"] == str(source)
    assert {artifact["path"] for artifact in manifest["artifacts"]} >= {
        "metadata.json",
        "context.md",
        "readable.md",
    }

    clean = json.loads((out_dir / "transcript.clean.json").read_text(encoding="utf-8"))
    assert clean["segments"][0]["text"] == "Hello world."

    context = (out_dir / "context.md").read_text(encoding="utf-8")
    assert "# Agent Context Pack" in context
    assert '<chunk id="chunk_0001" start="00:00:00" end="00:00:05">' in context
    assert "Hello world." in context


def test_prepare_local_srt_writes_knowledge_flow_json_for_arrow_chain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "flow.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:02,000
URL acquisition -> transcript extraction -> knowledge flow
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out_dir)])

    assert result.exit_code == 0, result.output
    flow = json.loads((out_dir / "knowledge_flow.json").read_text(encoding="utf-8"))
    assert {node["label"] for node in flow["nodes"]} == {
        "URL acquisition",
        "transcript extraction",
        "knowledge flow",
    }
    assert flow["edges"][0]["evidence"] == ["seg_000001"]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert {artifact["path"] for artifact in manifest["artifacts"]} >= {
        "knowledge_flow.json"
    }
    context = (out_dir / "context.md").read_text(encoding="utf-8")
    assert "## Knowledge flow" in context
    assert "URL acquisition -> transcript extraction -> knowledge flow" in context


def test_prepare_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "lecture.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:01,000
hello
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("keep", encoding="utf-8")

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out_dir)])

    assert result.exit_code == 5
    assert "output directory already exists" in result.output
    assert (out_dir / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_prepare_verbose_emits_phase_logs(tmp_path: Path) -> None:
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
        ["prepare", str(source), "--out", str(out_dir), "--verbose"],
    )

    assert result.exit_code == 0, result.output
    assert "INFO vctx.app.prepare prepare.start" in result.output
    assert "INFO vctx.app.prepare source.detect" in result.output
    assert "INFO vctx.app.prepare prepare.finish status=ok" in result.output
    assert "duration_ms=" in result.output
    assert "Workflow: default" in result.output


def test_prepare_verbose_writes_logs_to_stderr_not_stdout(tmp_path: Path) -> None:
    source = tmp_path / "lecture.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:01,000
hello
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["uv", "run", "vctx", "prepare", str(source), "--out", str(out_dir), "--verbose"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Wrote context pack:" in result.stdout
    assert "Workflow: default" in result.stdout
    assert "INFO vctx.app.prepare" not in result.stdout
    assert "INFO vctx.app.prepare prepare.start" in result.stderr
    assert "INFO vctx.app.prepare prepare.finish status=ok" in result.stderr



def test_prepare_debug_emits_debug_details(tmp_path: Path) -> None:
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
        ["prepare", str(source), "--out", str(out_dir), "--debug"],
    )

    assert result.exit_code == 0, result.output
    assert "DEBUG vctx.app.prepare prepare.output formats=" in result.output


def test_prepare_log_file_writes_logs_without_secret_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-token")
    source = tmp_path / "lecture.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:01,000
hello
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    log_file = tmp_path / "run.log"

    result = runner.invoke(
        app,
        ["prepare", str(source), "--out", str(out_dir), "--log-file", str(log_file)],
    )

    assert result.exit_code == 0, result.output
    assert "INFO vctx.app.prepare" not in result.output
    log_text = log_file.read_text(encoding="utf-8")
    assert "INFO vctx.app.prepare prepare.start" in log_text
    assert "prepare.finish status=ok" in log_text
    assert "super-secret-token" not in log_text
    assert "super-secret-token" not in result.output
