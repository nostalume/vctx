from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.artifact.manifest import Manifest
from vctx.artifact.publish import PackPublisher
from vctx.cli import app
from vctx.errors import OperationCancelledError
from vctx.source.local import LocalFileSession
from vctx.source.session import SubtitlePermit
from vctx.transcript import TranscriptPayload

runner = CliRunner()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


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
    assert "Status: ok" in result.output
    assert (out_dir / "manifest.json").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    source_entry = manifest["sources"][0]
    lane = out_dir / source_entry["path"]
    lane_names = {path.name for path in lane.iterdir()}
    assert lane_names >= {
        "metadata.json",
        "subtitle.und.srt",
        "transcript.json",
        "chunks.json",
        "context.md",
        "read.md",
    }
    assert [name for name in lane_names if name.startswith("transcript")] == [
        "transcript.json"
    ]
    assert lane_names.isdisjoint({"visual_records.json", "visual_scores.json"})
    assert manifest["schema_version"] == "2"
    assert manifest["status"] == "ok"
    assert "input" not in manifest
    assert source_entry["id"].startswith("local__")
    assert source_entry["assets"][0]["path"] == "subtitle.und.srt"
    assert {artifact["path"] for artifact in source_entry["artifacts"]} >= {
        "metadata.json",
        "context.md",
        "read.md",
    }

    transcript = json.loads((lane / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["segments"][0]["text"] == "Hello world."
    assert (lane / "subtitle.und.srt").read_bytes() == source.read_bytes()

    context = (lane / "context.md").read_text(encoding="utf-8")
    assert "# Agent Context Pack" in context
    assert '<chunk id="chunk_0001" start="00:00:00" end="00:00:05">' in context
    assert "Hello world." in context


def test_prepare_multiple_inputs_writes_independent_source_lanes(tmp_path: Path) -> None:
    first = tmp_path / "a" / "video.srt"
    second = tmp_path / "b" / "video.srt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("1\n00:00:00,000 --> 00:00:01,000\nFirst source.\n", encoding="utf-8")
    second.write_text("1\n00:00:00,000 --> 00:00:01,000\nSecond source.\n", encoding="utf-8")
    out = tmp_path / "pack"

    result = runner.invoke(app, ["prepare", str(first), str(second), "--out", str(out)])

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "2"
    assert len(manifest["sources"]) == 2
    assert {path.name for path in out.iterdir()} == {
        "manifest.json",
        *(source["key"] for source in manifest["sources"]),
    }
    for source in manifest["sources"]:
        lane = out / source["path"]
        assert lane.is_dir()
        assert (lane / "metadata.json").is_file()
        assert (lane / "context.md").is_file()
        assert not (lane / "manifest.json").exists()
        for artifact in source["artifacts"]:
            body = (lane / artifact["path"]).read_bytes()
            assert artifact["bytes"] == len(body)
        assert artifact["sha256"] == hashlib.sha256(body).hexdigest()


def test_prepare_adds_to_valid_pack_without_rewriting_existing_lane(tmp_path: Path) -> None:
    first = tmp_path / "first.srt"
    second = tmp_path / "second.srt"
    first.write_text("1\n00:00:00,000 --> 00:00:01,000\nFirst.\n", encoding="utf-8")
    second.write_text("1\n00:00:00,000 --> 00:00:01,000\nSecond.\n", encoding="utf-8")
    out = tmp_path / "pack"
    assert runner.invoke(app, ["prepare", str(first), "--out", str(out)]).exit_code == 0
    before_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    first_lane = out / before_manifest["sources"][0]["path"]
    before = {path.name: path.read_bytes() for path in first_lane.iterdir()}

    result = runner.invoke(app, ["prepare", str(second), "--out", str(out)])

    assert result.exit_code == 0, result.output
    after_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert after_manifest["pack_id"] == before_manifest["pack_id"]
    assert len(after_manifest["sources"]) == 2
    assert {path.name: path.read_bytes() for path in first_lane.iterdir()} == before


def test_prepare_replaces_changed_source_and_preserves_its_sibling(tmp_path: Path) -> None:
    first = tmp_path / "first.srt"
    second = tmp_path / "second.srt"
    first.write_text("1\n00:00:00,000 --> 00:00:01,000\nOld.\n", encoding="utf-8")
    second.write_text("1\n00:00:00,000 --> 00:00:01,000\nStable.\n", encoding="utf-8")
    out = tmp_path / "pack"
    initial = runner.invoke(app, ["prepare", str(first), str(second), "--out", str(out)])
    assert initial.exit_code == 0, initial.output
    before = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    stable = next(source for source in before["sources"] if source["title"] == "second")
    stable_bytes = _tree_bytes(out / stable["path"])
    first.write_text("1\n00:00:00,000 --> 00:00:01,000\nNew.\n", encoding="utf-8")

    result = runner.invoke(app, ["prepare", str(first), "--out", str(out)])

    assert result.exit_code == 0, result.output
    after = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    changed = next(source for source in after["sources"] if source["title"] == "first")
    assert "New." in (out / changed["path"] / "context.md").read_text(encoding="utf-8")
    assert _tree_bytes(out / stable["path"]) == stable_bytes


def test_prepare_reuses_matching_revision_unless_overwrite_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nStable.\n", encoding="utf-8")
    out = tmp_path / "pack"
    assert runner.invoke(app, ["prepare", str(source), "--out", str(out)]).exit_code == 0
    calls = 0
    original = LocalFileSession.transcript

    def tracked(self: LocalFileSession, *, permit: SubtitlePermit) -> TranscriptPayload:
        nonlocal calls
        calls += 1
        return original(self, permit=permit)

    monkeypatch.setattr(LocalFileSession, "transcript", tracked)
    assert runner.invoke(app, ["prepare", str(source), "--out", str(out)]).exit_code == 0
    assert calls == 0
    assert runner.invoke(
        app, ["prepare", str(source), "--out", str(out), "--overwrite"]
    ).exit_code == 0
    assert calls == 1


def test_prepare_swap_failure_preserves_verified_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.srt"
    second = tmp_path / "second.srt"
    first.write_text("1\n00:00:00,000 --> 00:00:01,000\nFirst.\n", encoding="utf-8")
    second.write_text("1\n00:00:00,000 --> 00:00:01,000\nSecond.\n", encoding="utf-8")
    out = tmp_path / "pack"
    assert runner.invoke(app, ["prepare", str(first), "--out", str(out)]).exit_code == 0
    before = _tree_bytes(out)
    from vctx.artifact import publish

    replace = publish.os.replace

    def fail_stage_swap(source: Path | str, target: Path | str) -> None:
        if Path(source).name.endswith(".stage") and Path(target) == out.resolve():
            raise OSError("injected publication failure")
        replace(source, target)

    monkeypatch.setattr(publish.os, "replace", fail_stage_swap)
    result = runner.invoke(app, ["prepare", str(second), "--out", str(out)])

    assert result.exit_code == 1
    assert _tree_bytes(out) == before
    assert not any(".vctx-" in path.name for path in tmp_path.iterdir())


def test_prepare_refuses_corrupt_pack_even_with_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nOriginal.\n", encoding="utf-8")
    out = tmp_path / "pack"
    assert runner.invoke(app, ["prepare", str(source), "--out", str(out)]).exit_code == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    context = out / manifest["sources"][0]["path"] / "context.md"
    context.write_text("corrupt", encoding="utf-8")

    result = runner.invoke(
        app, ["prepare", str(source), "--out", str(out), "--overwrite"]
    )

    assert result.exit_code == 5
    assert "not a verified vctx schema-2 pack" in result.output
    assert context.read_text(encoding="utf-8") == "corrupt"


def test_prepare_recovers_old_pack_after_interrupted_backup_rename(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nStable.\n", encoding="utf-8")
    out = tmp_path / "pack"
    assert runner.invoke(app, ["prepare", str(source), "--out", str(out)]).exit_code == 0
    old = Manifest.model_validate_json((out / "manifest.json").read_text(encoding="utf-8"))
    publisher = PackPublisher(out)
    marker = {
        "target": str(publisher.target),
        "state": "ready",
        "pack_id": str(old.pack_id),
        "old_run_id": str(old.updated_run_id),
        "new_run_id": "interrupted",
    }
    publisher.marker.write_text(
        json.dumps(marker), encoding="utf-8"
    )
    out.replace(publisher.backup)
    publisher.stage.mkdir()
    (publisher.stage / "unfinished").write_text("discard", encoding="utf-8")
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not publisher.stage.exists() and not publisher.backup.exists()


def test_prepare_cancelled_refresh_preserves_prior_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nStable.\n", encoding="utf-8")
    out = tmp_path / "pack"
    assert runner.invoke(app, ["prepare", str(source), "--out", str(out)]).exit_code == 0
    before = _tree_bytes(out)

    def cancel(*_args: object, **_kwargs: object) -> None:
        raise OperationCancelledError("cancelled")

    monkeypatch.setattr(LocalFileSession, "transcript", cancel)
    result = runner.invoke(
        app, ["prepare", str(source), "--out", str(out), "--overwrite"]
    )

    assert result.exit_code == 130
    assert _tree_bytes(out) == before


def test_prepare_deduplicates_input_and_isolates_rejected_sibling(tmp_path: Path) -> None:
    source = tmp_path / "video.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nUsable.\n", encoding="utf-8")
    out = tmp_path / "pack"

    result = runner.invoke(
        app, ["prepare", str(source), str(source), str(tmp_path / "missing"), "--out", str(out)]
    )

    assert result.exit_code == 3
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert len(manifest["sources"]) == 1
    assert (out / manifest["sources"][0]["path"] / "context.md").is_file()


def test_existing_pack_publishes_valid_addition_beside_rejected_input(tmp_path: Path) -> None:
    first = tmp_path / "first.srt"
    second = tmp_path / "second.srt"
    first.write_text("1\n00:00:00,000 --> 00:00:01,000\nFirst.\n", encoding="utf-8")
    second.write_text("1\n00:00:00,000 --> 00:00:01,000\nSecond.\n", encoding="utf-8")
    out = tmp_path / "pack"
    assert runner.invoke(app, ["prepare", str(first), "--out", str(out)]).exit_code == 0

    result = runner.invoke(
        app, ["prepare", str(second), str(tmp_path / "missing"), "--out", str(out)]
    )

    assert result.exit_code == 3
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert {source["title"] for source in manifest["sources"]} == {"first", "second"}


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
    assert "not a verified vctx schema-2 pack" in result.output
    assert (out_dir / "existing.txt").read_text(encoding="utf-8") == "keep"


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
