from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import vctx.model_store as model_store


def test_model_lock_records_process_start_identity(tmp_path: Path) -> None:
    lock = tmp_path / "model.lock"

    with model_store.model_lock(lock):
        record = json.loads(lock.read_text(encoding="utf-8"))

    assert record["pid"] == os.getpid()
    assert record["process_start"]


def test_prune_recovers_lease_after_pid_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = "b" * 24
    workspace = tmp_path / ".incomplete" / "asr" / identity
    workspace.mkdir(parents=True)
    (workspace.parent / f"{identity}.lock").write_text(
        json.dumps({"pid": os.getpid(), "process_start": "former-process"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_store, "_process_state", lambda _pid: (True, "current-process"))

    report = model_store.prune_model_cache(
        tmp_path, incomplete=True, unreferenced=False, dry_run=False
    )

    assert report.incomplete == [f".incomplete/asr/{identity}"]
    assert not workspace.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows console-signal regression")
def test_prune_does_not_signal_live_windows_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = "a" * 24
    workspace = tmp_path / ".incomplete" / "asr" / identity
    workspace.mkdir(parents=True)
    (workspace.parent / f"{identity}.lock").write_text(
        json.dumps({"pid": os.getpid(), "created_ns": time.time_ns()}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        model_store.os,
        "kill",
        lambda *_args: pytest.fail("Windows PID probe sent a console control event"),
    )

    report = model_store.prune_model_cache(
        tmp_path, incomplete=True, unreferenced=False, dry_run=False
    )

    assert report.incomplete == []
    assert workspace.is_dir()


def test_dead_model_lease_is_reclaimed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stage = model_store.incomplete_dir("asr", "small", tmp_path)
    stage.parent.mkdir(parents=True)
    lock = stage.parent / f"{stage.name}.lock"
    lock.write_text(json.dumps({"pid": 2147483647, "created_ns": 0}), encoding="utf-8")

    def download(_capability: str, _model: str, target: Path, **_kwargs: object) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"model")
        (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(model_store, "download_model", download)
    assert model_store.pull_models(["asr"], cache_dir=tmp_path)[0].state == "ready"
    assert not lock.exists()
