from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

import vctx.model_download as model_download
import vctx.model_store as models_module
from vctx.cli import app
from vctx.model_store import require_prepared_model
from vctx.supervise import SupervisedResult

runner = CliRunner()


def test_models_status_is_network_free_and_machine_readable(tmp_path: Path) -> None:
    result = runner.invoke(app, ["models", "status", "--cache-dir", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    records = json.loads(result.output)
    assert [(item["capability"], item["state"]) for item in records] == [
        ("asr", "missing"),
        ("ocr", "missing"),
    ]
    assert records[0]["cache_path"] == "asr/small"
    assert all(not Path(item["cache_path"]).is_absolute() for item in records)


def test_models_pull_uses_adapter_boundary_and_verify_detects_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    def download(capability: str, _model_id: str, target: Path, **_kwargs: object) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "weights.bin").write_bytes(b"prepared")
        if capability == "asr":
            (target / "model.bin").write_bytes(b"prepared")
            (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(models_module, "download_model", download)
    pull = runner.invoke(
        app,
        ["models", "pull", "asr", "ocr", "--cache-dir", str(tmp_path), "--asr", "local:tiny"],
    )
    assert pull.exit_code == 0, pull.output
    assert "asr: ready" in pull.output
    assert "ocr: ready" in pull.output
    status = json.loads(
        runner.invoke(
            app,
            [
                "models",
                "status",
                "asr",
                "--cache-dir",
                str(tmp_path),
                "--asr",
                "local:tiny",
                "--json",
            ],
        ).output
    )[0]
    assert status["cache_path"].startswith("generations/asr/")

    verify = runner.invoke(
        app,
        [
            "models",
            "verify",
            "asr",
            "ocr",
            "--cache-dir",
            str(tmp_path),
            "--asr",
            "local:tiny",
            "--json",
        ],
    )
    assert [item["state"] for item in json.loads(verify.output)] == ["ready", "ready"]

    model_root = tmp_path / "models" / status["cache_path"]
    (model_root / "weights.bin").write_bytes(b"corrupt")
    corrupt = runner.invoke(
        app,
        [
            "models",
            "verify",
            "asr",
            "--cache-dir",
            str(tmp_path),
            "--asr",
            "local:tiny",
            "--json",
        ],
    )
    assert json.loads(corrupt.output)[0]["state"] == "corrupt"


def test_asr_model_pull_preserves_previous_model_on_download_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    fail = False

    def download(
        _capability: str,
        _model_id: str,
        target: Path,
        *,
        conservative: bool,
        max_runtime: int,
        **_kwargs: object,
    ) -> None:
        assert conservative is False
        assert max_runtime == 3600
        (target / "model.bin").write_bytes(b"replacement" if fail else b"original")
        (target / "config.json").write_text("{}", encoding="utf-8")
        if fail:
            raise RuntimeError("interrupted download")

    monkeypatch.setattr(models_module, "download_model", download)
    args = ["models", "pull", "asr", "--cache-dir", str(tmp_path)]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    receipt_path = next((tmp_path / "models" / "receipts" / "asr").glob("*.json"))
    receipt = json.loads(receipt_path.read_text())
    target = tmp_path / "models" / receipt["cache_path"]
    before = {path.name: path.read_bytes() for path in target.iterdir()}

    fail = True
    second = runner.invoke(app, [*args, "--refresh"])

    assert second.exit_code == 1
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    assert any(path.name == ".incomplete" for path in (tmp_path / "models").iterdir())


def test_asr_pull_reuses_recoverable_huggingface_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    destinations: list[Path] = []
    modes: list[bool] = []
    fail = True

    def download(
        _capability: str,
        _model_id: str,
        destination: Path,
        *,
        conservative: bool,
        max_runtime: int,
        **_kwargs: object,
    ) -> None:
        nonlocal fail
        assert max_runtime == 3600
        destinations.append(destination)
        modes.append(conservative)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "partial.bin").write_bytes(b"reusable")
        if fail:
            raise RuntimeError("interrupted")
        (destination / "model.bin").write_bytes(b"model")
        (destination / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(models_module, "download_model", download)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    args = ["models", "pull", "asr", "--cache-dir", str(tmp_path)]

    first = runner.invoke(app, args)
    assert first.exit_code == 1
    assert destinations[0].is_dir() and (destinations[0] / "partial.bin").is_file()
    assert os.environ.get("HF_XET_HIGH_PERFORMANCE") is None

    fail = False
    second = runner.invoke(app, [*args, "--refresh"])

    assert second.exit_code == 0, second.output
    assert destinations == [destinations[0], destinations[0]]
    assert modes == [False, False]


def test_models_status_and_prune_own_incomplete_workspaces(tmp_path: Path) -> None:
    identity = hashlib.sha256(b"small").hexdigest()[:24]
    incomplete = tmp_path / "models" / ".incomplete" / "asr" / identity
    incomplete.mkdir(parents=True)
    (incomplete / "partial.bin").write_bytes(b"partial")

    status = runner.invoke(app, ["models", "status", "asr", "--cache-dir", str(tmp_path), "--json"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)[0]["state"] == "incomplete"

    dry = runner.invoke(
        app,
        [
            "models",
            "prune",
            "--incomplete",
            "--dry-run",
            "--json",
            "--cache-dir",
            str(tmp_path),
        ],
    )
    assert dry.exit_code == 0, dry.output
    assert json.loads(dry.output)["incomplete"] and incomplete.is_dir()

    prune = runner.invoke(app, ["models", "prune", "--incomplete", "--cache-dir", str(tmp_path)])
    assert prune.exit_code == 0, prune.output
    assert "pruned" in prune.output and not incomplete.exists()


def test_models_prune_removes_only_unreferenced_generations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    def download(
        _capability: str,
        _model_id: str,
        target: Path,
        **_kwargs: object,
    ) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"referenced")
        (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(models_module, "download_model", download)
    receipt = models_module.pull_models(
        ["asr"], cache_dir=tmp_path / "models", asr_model_id="small"
    )[0]
    referenced = tmp_path / "models" / receipt.cache_path
    orphan = tmp_path / "models" / "generations" / "asr" / ("f" * 16) / ("e" * 24)
    orphan.mkdir(parents=True)
    (orphan / "weights.bin").write_bytes(b"orphan")

    args = [
        "models",
        "prune",
        "--unreferenced",
        "--cache-dir",
        str(tmp_path),
        "--json",
    ]
    dry = runner.invoke(app, [*args, "--dry-run"])

    assert dry.exit_code == 0, dry.output
    report = json.loads(dry.output)
    assert report["generations"] == [orphan.relative_to(tmp_path / "models").as_posix()]
    assert referenced.is_dir() and orphan.is_dir()

    prune = runner.invoke(app, args)

    assert prune.exit_code == 0, prune.output
    assert referenced.is_dir() and not orphan.exists()


def test_models_prune_preserves_live_and_ambiguous_state(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    identity = hashlib.sha256(b"small").hexdigest()[:24]
    active = model_root / ".incomplete" / "asr" / identity
    active.mkdir(parents=True)
    (active / "partial.bin").write_bytes(b"partial")
    (active.parent / f"{identity}.lock").write_text(
        json.dumps({"pid": os.getpid(), "created_ns": time.time_ns()}),
        encoding="utf-8",
    )
    orphan = model_root / "generations" / "asr" / ("f" * 16) / ("e" * 24)
    orphan.mkdir(parents=True)
    (orphan / "weights.bin").write_bytes(b"orphan")
    receipt = model_root / "receipts" / "asr" / "invalid.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("not-json", encoding="utf-8")

    incomplete = runner.invoke(
        app, ["models", "prune", "--incomplete", "--cache-dir", str(tmp_path), "--json"]
    )
    unreferenced = runner.invoke(
        app, ["models", "prune", "--unreferenced", "--cache-dir", str(tmp_path)]
    )

    assert incomplete.exit_code == 0
    assert json.loads(incomplete.output)["incomplete"] == []
    assert active.is_dir()
    assert unreferenced.exit_code == 1
    assert "invalid receipt" in unreferenced.output
    assert orphan.is_dir()


def test_warm_model_requirement_uses_receipt_file_facts_not_content_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    def download(
        _capability: str,
        _model_id: str,
        target: Path,
        **_kwargs: object,
    ) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"model-bytes")
        (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(models_module, "download_model", download)
    models_module.pull_models(["asr"], cache_dir=tmp_path)
    monkeypatch.setattr(
        models_module,
        "tree_integrity",
        lambda _root: pytest.fail("warm readiness rehashed model content"),
    )

    assert require_prepared_model("asr", tmp_path).state == "ready"

    monkeypatch.setattr(
        models_module,
        "download_model",
        lambda *_args, **_kwargs: pytest.fail("ready pull entered download path"),
    )
    assert models_module.pull_models(["asr"], cache_dir=tmp_path)[0].state == "ready"


def test_verify_refreshes_changed_facts_and_keeps_named_receipts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    def download(
        _capability: str,
        model_id: str,
        target: Path,
        **_kwargs: object,
    ) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(model_id.encode())
        (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(models_module, "download_model", download)
    small = models_module.pull_models(["asr"], cache_dir=tmp_path, asr_model_id="small")[0]
    tiny = models_module.pull_models(["asr"], cache_dir=tmp_path, asr_model_id="tiny")[0]
    assert small.cache_path != tiny.cache_path
    assert len(list((tmp_path / "receipts" / "asr").glob("*.json"))) == 2

    weights = tmp_path / small.cache_path / "model.bin"
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert (
        models_module.model_status(["asr"], cache_dir=tmp_path, asr_model_id="small")[0].state
        == "changed"
    )
    assert (
        models_module.verify_models(["asr"], cache_dir=tmp_path, asr_model_id="small")[0].state
        == "ready"
    )
    assert require_prepared_model("asr", tmp_path, asr_model_id="small").state == "ready"


def test_dead_model_lease_is_reclaimed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:

    stage = models_module.incomplete_dir("asr", "small", tmp_path)
    stage.parent.mkdir(parents=True)
    lock = stage.parent / f"{stage.name}.lock"
    lock.write_text(json.dumps({"pid": 2147483647, "created_ns": 0}), encoding="utf-8")

    def download(_capability: str, _model: str, target: Path, **_kwargs: object) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"model")
        (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(models_module, "download_model", download)
    assert models_module.pull_models(["asr"], cache_dir=tmp_path)[0].state == "ready"
    assert not lock.exists()


def test_hub_download_fast_mode_is_child_scoped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    observed: dict[str, object] = {}

    def run(command: list[str], payload: bytes, **kwargs: object) -> SupervisedResult:
        observed.update(command=command, payload=payload, **kwargs)
        return SupervisedResult(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(model_download, "run_supervised", run)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)

    model_download.download_model(
        "asr",
        "small",
        tmp_path,
        cache_root=tmp_path,
        conservative=False,
        max_runtime=3600,
    )

    assert os.environ.get("HF_XET_HIGH_PERFORMANCE") is None
    env = cast(dict[str, str], observed["environment"])
    command = cast(list[str], observed["command"])
    assert env["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert command[:3] == [sys.executable, "-m", "vctx.model_hub"]
