from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.cli import app
from vctx.supervise import run_supervised

runner = CliRunner()


def test_deadline_terminates_worker_tree_and_bounds_diagnostics(tmp_path: Path) -> None:
    marker = tmp_path / "escaped"
    descendant = (
        f"import time,pathlib; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('leaked')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}]); time.sleep(20)"
    )
    started = time.monotonic()

    result = run_supervised([sys.executable, "-c", parent], b"private-input", timeout_s=0.1)
    time.sleep(2)

    assert result.returncode == 124
    assert time.monotonic() - started < 4
    assert not marker.exists()
    assert b"private-input" not in result.stdout + result.stderr


def test_prepare_runs_through_private_worker_with_unchanged_stdout(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nBounded.\n", encoding="utf-8")
    out = tmp_path / "pack"

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out), "--max-runtime", "10"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("Wrote context pack:")
    assert (out / "manifest.json").is_file()


def test_supervisor_relays_stderr_before_child_exit(tmp_path: Path) -> None:
    acknowledged = tmp_path / "acknowledged"
    child = (
        "import pathlib,sys,time; "
        "sys.stderr.write('phase-ready\\n'); sys.stderr.flush(); "
        f"marker=pathlib.Path({str(acknowledged)!r}); "
        "deadline=time.monotonic()+2; "
        'exec("while not marker.exists() and time.monotonic() < deadline:\\n time.sleep(0.01)"); '
        "sys.exit(0 if marker.exists() else 9)"
    )
    chunks: list[bytes] = []

    def relay(block: bytes) -> None:
        chunks.append(block)
        acknowledged.write_text("seen", encoding="utf-8")

    result = run_supervised(
        [sys.executable, "-c", child],
        b"",
        timeout_s=5,
        stderr_sink=relay,
    )

    assert result.returncode == 0
    assert b"phase-ready" in b"".join(chunks)
    assert b"phase-ready" in result.stderr


def test_supervised_prepare_writes_bounded_json_phase_events(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nProfiled.\n", encoding="utf-8")
    out = tmp_path / "pack"
    profile = tmp_path / "profile.jsonl"

    result = runner.invoke(
        app,
        [
            "prepare",
            str(source),
            "--out",
            str(out),
            "--max-runtime",
            "10",
            "--profile-json",
            str(profile),
        ],
    )

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in profile.read_text(encoding="utf-8").splitlines()]
    assert events[0] == {"schema": 1, "event": "start", "phase": "prepare.total"}
    assert events[-1]["event"] == "finish"
    assert events[-1]["phase"] == "prepare.total"
    assert isinstance(events[-1]["duration_ms"], int)
    assert all(len(json.dumps(event).encode()) < 1024 for event in events)


def test_profile_events_are_visible_before_supervised_worker_exit(tmp_path: Path) -> None:
    profile = tmp_path / "profile.jsonl"
    acknowledged = tmp_path / "acknowledged"
    child = (
        "import logging,pathlib,sys,time; "
        "from vctx.app.progress import configure_logging,phase; "
        f"profile=pathlib.Path({str(profile)!r}); marker=pathlib.Path({str(acknowledged)!r}); "
        "configure_logging(verbose=False,debug=False,log_file=None,profile_json=profile); "
        "scope=phase(logging.getLogger('vctx'),'supervised.live'); scope.__enter__(); "
        "sys.stderr.write('profile-ready\\n'); sys.stderr.flush(); "
        "deadline=time.monotonic()+2; "
        'exec("while not marker.exists() and time.monotonic() < deadline:\\n time.sleep(0.01)"); '
        "scope.__exit__(None,None,None); sys.exit(0 if marker.exists() else 9)"
    )

    def observe(block: bytes) -> None:
        if b"profile-ready" not in block or not profile.is_file():
            return
        events = [json.loads(line) for line in profile.read_text(encoding="utf-8").splitlines()]
        if events == [{"schema": 1, "event": "start", "phase": "supervised.live"}]:
            acknowledged.write_text("seen", encoding="utf-8")

    result = run_supervised([sys.executable, "-c", child], b"", timeout_s=5, stderr_sink=observe)

    assert result.returncode == 0
    assert acknowledged.is_file()


@pytest.mark.parametrize("seconds", [0, 86401])
def test_prepare_rejects_invalid_runtime_bound(tmp_path: Path, seconds: int) -> None:
    result = runner.invoke(
        app, ["prepare", "missing", "--out", str(tmp_path / "out"), "--max-runtime", str(seconds)]
    )
    assert result.exit_code == 2
