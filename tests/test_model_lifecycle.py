from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

import vctx.model.download as model_download
import vctx.model.store as models_module


def test_models_pull_uses_adapter_boundary_and_verify_detects_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    def download(capability: str, _model_id: str, target: Path, **_kwargs: object) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "weights.bin").write_bytes(b"prepared")
        if capability == "asr":
            (target / "model.bin").write_bytes(b"prepared")
            (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(model_download, "download_model", download)
    cache_root = tmp_path / "models"
    store = models_module.ModelStore(cache_root)
    pulled = store.pull(["asr", "ocr"], asr_model_id="tiny")
    assert [receipt.state for receipt in pulled] == ["ready", "ready"]
    status = store.status(["asr"], asr_model_id="tiny")[0]
    assert status.cache_path.startswith("generations/asr/")
    assert [receipt.state for receipt in store.verify(["asr", "ocr"], asr_model_id="tiny")] == [
        "ready",
        "ready",
    ]

    model_root = cache_root / status.cache_path
    (model_root / "weights.bin").write_bytes(b"corrupt")
    assert store.verify(["asr"], asr_model_id="tiny")[0].state == "corrupt"


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

    monkeypatch.setattr(model_download, "download_model", download)
    cache_root = tmp_path / "models"
    store = models_module.ModelStore(cache_root)
    receipt = store.pull(["asr"])[0]
    target = cache_root / receipt.cache_path
    before = {path.name: path.read_bytes() for path in target.iterdir()}

    fail = True
    with pytest.raises(models_module.ModelLifecycleError, match="interrupted download"):
        store.pull(["asr"], refresh=True)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    assert any(path.name == ".incomplete" for path in cache_root.iterdir())


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

    monkeypatch.setattr(model_download, "download_model", download)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    cache_root = tmp_path / "models"
    store = models_module.ModelStore(cache_root)

    with pytest.raises(models_module.ModelLifecycleError, match="interrupted"):
        store.pull(["asr"])
    assert destinations[0].is_dir() and (destinations[0] / "partial.bin").is_file()
    assert os.environ.get("HF_XET_HIGH_PERFORMANCE") is None

    fail = False
    assert store.pull(["asr"], refresh=True)[0].state == "ready"
    assert destinations == [destinations[0], destinations[0]]
    assert modes == [False, False]


def test_models_status_and_prune_own_incomplete_workspaces(tmp_path: Path) -> None:
    identity = hashlib.sha256(b"small").hexdigest()[:24]
    incomplete = tmp_path / "models" / ".incomplete" / "asr" / identity
    incomplete.mkdir(parents=True)
    (incomplete / "partial.bin").write_bytes(b"partial")

    model_root = tmp_path / "models"
    store = models_module.ModelStore(model_root)
    assert store.status(["asr"])[0].state == "incomplete"

    dry = store.prune(incomplete=True, unreferenced=False, dry_run=True)
    assert dry.incomplete and incomplete.is_dir()

    prune = store.prune(incomplete=True, unreferenced=False, dry_run=False)
    assert prune.incomplete and not incomplete.exists()


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

    monkeypatch.setattr(model_download, "download_model", download)
    store = models_module.ModelStore(tmp_path / "models")
    receipt = store.pull(["asr"], asr_model_id="small")[0]
    referenced = tmp_path / "models" / receipt.cache_path
    orphan = tmp_path / "models" / "generations" / "asr" / ("f" * 16) / ("e" * 24)
    orphan.mkdir(parents=True)
    (orphan / "weights.bin").write_bytes(b"orphan")

    dry = store.prune(incomplete=False, unreferenced=True, dry_run=True)
    assert dry.generations == [orphan.relative_to(tmp_path / "models").as_posix()]
    assert referenced.is_dir() and orphan.is_dir()

    store.prune(incomplete=False, unreferenced=True, dry_run=False)
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

    store = models_module.ModelStore(model_root)
    incomplete = store.prune(incomplete=True, unreferenced=False, dry_run=False)
    assert incomplete.incomplete == []
    assert active.is_dir()
    with pytest.raises(models_module.ModelLifecycleError, match="invalid receipt"):
        store.prune(incomplete=False, unreferenced=True, dry_run=False)
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

    monkeypatch.setattr(model_download, "download_model", download)
    store = models_module.ModelStore(tmp_path)
    store.pull(["asr"])
    monkeypatch.setattr(
        models_module,
        "tree_integrity",
        lambda _root: pytest.fail("warm readiness rehashed model content"),
    )

    assert store.require("asr").state == "ready"

    monkeypatch.setattr(
        model_download,
        "download_model",
        lambda *_args, **_kwargs: pytest.fail("ready pull entered download path"),
    )
    assert store.pull(["asr"])[0].state == "ready"


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

    monkeypatch.setattr(model_download, "download_model", download)
    store = models_module.ModelStore(tmp_path)
    small = store.pull(["asr"], asr_model_id="small")[0]
    tiny = store.pull(["asr"], asr_model_id="tiny")[0]
    assert small.cache_path != tiny.cache_path
    assert len(list((tmp_path / "receipts" / "asr").glob("*.json"))) == 2

    weights = tmp_path / small.cache_path / "model.bin"
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert store.status(["asr"], asr_model_id="small")[0].state == "changed"
    assert store.verify(["asr"], asr_model_id="small")[0].state == "ready"
    assert store.require("asr", asr_model_id="small").state == "ready"
