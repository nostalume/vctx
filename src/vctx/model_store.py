from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

from vctx.artifact.bundle import encode_json
from vctx.errors import VctxError
from vctx.model_download import download_model

_IDENTITY = re.compile(r"[0-9a-f]{16}")
_WORKSPACE_IDENTITY = re.compile(r"[0-9a-f]{24}")
_GENERATION = re.compile(r"[0-9a-f]{24}")


class ModelLifecycleError(VctxError):
    pass


ModelCapability = Literal["asr", "ocr"]
ModelState = Literal["ready", "missing", "incomplete", "changed", "corrupt"]


class ModelFileFact(BaseModel):
    path: str
    size: int
    mtime_ns: int


class ModelReceipt(BaseModel):
    capability: ModelCapability
    provider: str
    model_id: str
    state: ModelState
    cache_path: str
    bytes: int
    package_version: str
    integrity: str | None = None
    message: str | None = None
    files: list[ModelFileFact] = Field(default_factory=list)


class ModelPruneReport(BaseModel):
    dry_run: bool
    incomplete: list[str]
    generations: list[str]
    bytes: int

    @property
    def count(self) -> int:
        return len(self.incomplete) + len(self.generations)


def incomplete_dir(capability: str, model_id: str, cache_root: Path) -> Path:
    import hashlib

    identity = hashlib.sha256(model_id.encode()).hexdigest()[:24]
    return cache_root / ".incomplete" / capability / identity


def receipt_path(capability: str, cache_root: Path, *, model_id: str) -> Path:
    import hashlib

    identity = hashlib.sha256(model_id.encode()).hexdigest()[:24]
    return cache_root / "receipts" / capability / f"{identity}.json"


def model_identity(
    capability: ModelCapability, asr_model_id: str = "small"
) -> tuple[str, str, str]:
    if capability == "asr":
        return ("faster-whisper", asr_model_id, "faster-whisper")
    return ("rapidocr", "rapidocr", "rapidocr")


def rapidocr_config_path(cache_root: Path) -> Path:
    return cache_root / "ocr" / "rapidocr" / "config.yaml"


def pull_models(
    capabilities: list[str] | None,
    *,
    cache_dir: Path,
    asr_model_id: str = "small",
    conservative: bool = False,
    refresh: bool = False,
    max_runtime: int = 3600,
) -> list[ModelReceipt]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return [
        _pull_model(
            capability,
            cache_dir,
            asr_model_id=asr_model_id,
            conservative=conservative,
            refresh=refresh,
            max_runtime=max_runtime,
        )
        for capability in _capabilities(capabilities)
    ]


def model_status(
    capabilities: list[str] | None, *, cache_dir: Path, asr_model_id: str = "small"
) -> list[ModelReceipt]:
    return [
        inspect_model(capability, cache_dir, verify=False, asr_model_id=asr_model_id)
        for capability in _capabilities(capabilities)
    ]


def verify_models(
    capabilities: list[str] | None, *, cache_dir: Path, asr_model_id: str = "small"
) -> list[ModelReceipt]:
    return [
        inspect_model(capability, cache_dir, verify=True, asr_model_id=asr_model_id)
        for capability in _capabilities(capabilities)
    ]


def require_prepared_model(
    capability: ModelCapability, cache_root: Path, *, asr_model_id: str = "small"
) -> ModelReceipt:
    receipt = inspect_model(capability, cache_root, verify=False, asr_model_id=asr_model_id)
    if receipt.state != "ready":
        raise ModelLifecycleError(
            f"local {capability.upper()} model is {receipt.state}; "
            f"run: vctx models pull {capability}"
        )
    return receipt


def inspect_model(
    capability: ModelCapability, cache_root: Path, *, verify: bool, asr_model_id: str
) -> ModelReceipt:
    provider, model_id, package = model_identity(capability, asr_model_id)
    expected_dir = cache_root / capability / model_id
    path = receipt_path(capability, cache_root, model_id=model_id)
    legacy = cache_root / "receipts" / f"{capability}.json"
    if not path.is_file() and legacy.is_file():
        path = legacy
    base = ModelReceipt(
        capability=capability,
        provider=provider,
        model_id=model_id,
        state="missing",
        cache_path=expected_dir.relative_to(cache_root).as_posix(),
        bytes=0,
        package_version=package_version(package),
    )
    try:
        prepared = path.is_file()
    except OSError:
        prepared = False
    if not prepared:
        incomplete = incomplete_dir(capability, model_id, cache_root)
        if incomplete.is_dir():
            return base.model_copy(
                update={
                    "state": "incomplete",
                    "bytes": _tree_size(incomplete),
                    "message": "resumable download is incomplete",
                }
            )
        return base.model_copy(update={"message": "model is not prepared"})
    try:
        recorded = ModelReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return base.model_copy(update={"state": "corrupt", "message": "invalid receipt"})
    if (recorded.capability, recorded.model_id) != (capability, model_id):
        return base.model_copy(update={"state": "corrupt", "message": "receipt identity mismatch"})
    model_dir = (cache_root / recorded.cache_path).resolve()
    if not model_dir.is_relative_to(cache_root.resolve()) or not model_dir.is_dir():
        return base.model_copy(update={"state": "corrupt", "message": "model generation missing"})
    if not verify:
        if recorded.files and recorded.files != tree_facts(model_dir):
            return recorded.model_copy(
                update={"state": "changed", "message": "model files changed; run models verify"}
            )
        return recorded
    digest, size = tree_integrity(model_dir)
    if size == 0 or digest != recorded.integrity or size != recorded.bytes:
        return recorded.model_copy(
            update={
                "state": "corrupt",
                "bytes": size,
                "integrity": digest,
                "message": "integrity mismatch",
            }
        )
    refreshed = recorded.model_copy(
        update={"state": "ready", "message": None, "files": tree_facts(model_dir)}
    )
    write_atomic(path, encode_json(refreshed))
    return refreshed


def tree_integrity(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(data)
                size += len(data)
    return digest.hexdigest(), size


def tree_facts(root: Path) -> list[ModelFileFact]:
    facts = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        facts.append(
            ModelFileFact(
                path=path.relative_to(root).as_posix(),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return facts


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _capabilities(values: list[str] | None) -> list[ModelCapability]:
    values = values or ["asr", "ocr"]
    invalid = [value for value in values if value not in {"asr", "ocr"}]
    if invalid:
        raise ModelLifecycleError(f"unsupported model capability: {', '.join(invalid)}")
    return cast(list[ModelCapability], list(dict.fromkeys(values)))


def _pull_model(
    capability: ModelCapability,
    cache_root: Path,
    *,
    asr_model_id: str,
    conservative: bool,
    refresh: bool,
    max_runtime: int,
) -> ModelReceipt:
    provider, model_id, package = model_identity(capability, asr_model_id)

    if not refresh:
        current = inspect_model(
            capability,
            cache_root,
            verify=False,
            asr_model_id=asr_model_id,
        )
        if current.state == "ready":
            path = receipt_path(capability, cache_root, model_id=model_id)
            if not path.is_file():
                write_atomic(path, encode_json(current))
            return current

    workspace = _acquire_model(
        capability,
        model_id,
        cache_root,
        conservative=conservative,
        max_runtime=max_runtime,
    )
    digest, size = tree_integrity(workspace)
    if size == 0:
        raise ModelLifecycleError(f"{provider} model pull produced no files")
    model_dir = (
        _publish_generation(cache_root, capability, model_id, workspace, digest)
        if capability == "asr"
        else workspace
    )
    receipt = ModelReceipt(
        capability=capability,
        provider=provider,
        model_id=model_id,
        state="ready",
        cache_path=model_dir.relative_to(cache_root).as_posix(),
        bytes=size,
        package_version=package_version(package),
        integrity=digest,
        files=tree_facts(model_dir),
    )
    write_atomic(receipt_path(capability, cache_root, model_id=model_id), encode_json(receipt))
    return receipt


def _acquire_model(
    capability: ModelCapability,
    model_id: str,
    cache_root: Path,
    *,
    conservative: bool,
    max_runtime: int,
) -> Path:
    target = cache_root / capability / model_id
    workspace = incomplete_dir(capability, model_id, cache_root) if capability == "asr" else target
    lock = workspace.parent / f"{workspace.name}.lock"
    try:
        if capability == "asr":
            with model_lock(lock):
                workspace.mkdir(parents=True, exist_ok=True)
                download_model(
                    capability,
                    model_id,
                    workspace,
                    cache_root=cache_root,
                    conservative=conservative,
                    max_runtime=max_runtime,
                )
                _validate_ctranslate2(workspace)
        else:
            download_model(
                capability,
                model_id,
                workspace,
                cache_root=cache_root,
                conservative=conservative,
                max_runtime=max_runtime,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise ModelLifecycleError(f"failed to pull {capability} model: {exc}") from exc
    return workspace


def _publish_generation(
    cache_root: Path,
    capability: ModelCapability,
    model_id: str,
    workspace: Path,
    digest: str,
) -> Path:
    identity = hashlib.sha256(model_id.encode()).hexdigest()[:16]
    generation = cache_root / "generations" / capability / identity / digest[:24]
    generation.parent.mkdir(parents=True, exist_ok=True)
    if generation.exists():
        if workspace != generation:
            shutil.rmtree(workspace)
    else:
        os.replace(workspace, generation)
    return generation


def _validate_ctranslate2(root: Path) -> None:
    missing = [name for name in ("model.bin", "config.json") if not (root / name).is_file()]
    if missing:
        raise ValueError(f"download is not a CTranslate2 model: missing {', '.join(missing)}")


@contextmanager
def model_lock(path: Path, *, timeout_s: float = 30.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _dead_lock(path):
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise ModelLifecycleError(
                    f"timed out waiting for model lock: {path.name}"
                ) from None
            time.sleep(0.05)
    try:
        pid = os.getpid()
        os.write(
            descriptor,
            json.dumps(
                {
                    "pid": pid,
                    "process_start": _process_start_identity(pid),
                    "created_ns": time.time_ns(),
                }
            ).encode(),
        )
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _dead_lock(path: Path) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        pid = int(record["pid"])
        running, observed = _process_state(pid)
        if not running:
            return True
        expected = record.get("process_start")
        if isinstance(expected, str) and observed is not None:
            return expected != observed
    except OSError, ValueError, KeyError, TypeError, json.JSONDecodeError:
        return True
    return False


def _process_state(pid: int) -> tuple[bool, str | None]:
    if pid <= 0:
        return False, None
    if os.name == "nt":
        return _windows_process_state(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, None
    except PermissionError:
        return True, None
    return True, _posix_process_start(pid)


def _windows_process_state(pid: int) -> tuple[bool, str | None]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    wait_for_single_object.restype = wintypes.DWORD
    get_process_times = kernel32.GetProcessTimes
    get_process_times.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    get_process_times.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    handle = open_process(0x101000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5, None
    try:
        if wait_for_single_object(handle, 0) == 0:
            return False, None
        creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not get_process_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return True, None
        return True, str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
    finally:
        close_handle(handle)


def _process_start_identity(pid: int) -> str | None:
    return _process_state(pid)[1]


def _posix_process_start(pid: int) -> str | None:
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
            return fields.split()[19]
        except OSError, IndexError:
            return None
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    value = result.stdout.decode(errors="replace").strip()
    return value or None


def prune_model_cache(
    root: Path,
    *,
    incomplete: bool,
    unreferenced: bool,
    dry_run: bool,
) -> ModelPruneReport:
    workspaces = _incomplete_candidates(root) if incomplete else []
    generations = _generation_candidates(root) if unreferenced else []
    targets = [*workspaces, *generations]
    proposed = ModelPruneReport(
        dry_run=dry_run,
        incomplete=[_relative(root, path) for path in workspaces],
        generations=[_relative(root, path) for path in generations],
        bytes=sum(_tree_size(path) for path in targets),
    )
    if dry_run:
        return proposed
    removed_workspaces: list[str] = []
    removed_generations: list[str] = []
    removed_bytes = 0
    for path in workspaces:
        lock = path.parent / f"{path.name}.lock"
        try:
            with model_lock(lock, timeout_s=0):
                size = _tree_size(path)
                shutil.rmtree(path)
        except ModelLifecycleError:
            continue
        removed_workspaces.append(_relative(root, path))
        removed_bytes += size
        _remove_empty_parents(path.parent, root / ".incomplete")
    for path in generations:
        size = _tree_size(path)
        shutil.rmtree(path)
        removed_generations.append(_relative(root, path))
        removed_bytes += size
        _remove_empty_parents(path.parent, root / "generations")
    return ModelPruneReport(
        dry_run=False,
        incomplete=removed_workspaces,
        generations=removed_generations,
        bytes=removed_bytes,
    )


def _incomplete_candidates(root: Path) -> list[Path]:
    base = root / ".incomplete"
    candidates = []
    for path in base.glob("*/*"):
        lock = path.parent / f"{path.name}.lock"
        if (
            path.is_dir()
            and not path.is_symlink()
            and path.parent.name in {"asr", "ocr"}
            and _WORKSPACE_IDENTITY.fullmatch(path.name)
            and (not lock.exists() or _dead_lock(lock))
        ):
            candidates.append(path)
    return sorted(candidates)


def _generation_candidates(root: Path) -> list[Path]:
    base = root / "generations"
    referenced = _receipt_references(root, base)
    return sorted(
        path
        for path in base.glob("*/*/*")
        if path.is_dir()
        and not path.is_symlink()
        and path.parent.parent.name in {"asr", "ocr"}
        and _IDENTITY.fullmatch(path.parent.name)
        and _GENERATION.fullmatch(path.name)
        and path.resolve() not in referenced
    )


def _receipt_references(root: Path, generations: Path) -> set[Path]:
    references: set[Path] = set()
    for path in (root / "receipts").rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            relative = value["cache_path"]
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ModelLifecycleError(
                f"cannot prune generations with invalid receipt: {path.name}"
            ) from exc
        if target.is_relative_to(generations.resolve()):
            references.add(target)
    return references


def _tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _remove_empty_parents(path: Path, boundary: Path) -> None:
    while path != boundary.parent:
        try:
            path.rmdir()
        except OSError:
            return
        if path == boundary:
            return
        path = path.parent
