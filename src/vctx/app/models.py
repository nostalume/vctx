from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from vctx.artifact.bundle import encode_json
from vctx.config import ResolvedConfig
from vctx.errors import VctxError

ModelCapability = Literal["asr", "ocr"]
ModelState = Literal["ready", "missing", "corrupt"]


class ModelLifecycleError(VctxError):
    pass


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


@dataclass(frozen=True)
class Models:
    cache_dir: Path
    asr_model_id: str

    @classmethod
    def open(cls, resolved: ResolvedConfig) -> Models:
        return cls(resolved.cache.model_dir, select_asr_model_id(resolved))

    def pull(self, capabilities: list[str] | None) -> list[ModelReceipt]:
        return pull_models(
            capabilities, cache_dir=self.cache_dir, asr_model_id=self.asr_model_id
        )

    def status(self, capabilities: list[str] | None) -> list[ModelReceipt]:
        return model_status(
            capabilities, cache_dir=self.cache_dir, asr_model_id=self.asr_model_id
        )

    def verify(self, capabilities: list[str] | None) -> list[ModelReceipt]:
        return verify_models(
            capabilities, cache_dir=self.cache_dir, asr_model_id=self.asr_model_id
        )


def select_asr_model_id(resolved: ResolvedConfig) -> str:
    policy = resolved.asr
    model_ref = policy.model_ref()
    if model_ref is not None:
        if model_ref.startswith("local:"):
            return model_ref.removeprefix("local:")
        raise ModelLifecycleError("models pull manages named local ASR models, not path/HF refs")
    instance_name = policy.instance_name()
    if instance_name is None:
        return "small"
    instance = resolved.instances.asr[instance_name]
    if instance.type != "local-faster-whisper":
        raise ModelLifecycleError("selected ASR instance is online and has no local model to pull")
    model_id = instance.model or "small"
    if Path(model_id).is_absolute():
        raise ModelLifecycleError("selected ASR model is an explicit local path and needs no pull")
    return model_id


def pull_models(
    capabilities: list[str] | None,
    *,
    cache_dir: Path,
    asr_model_id: str = "small",
) -> list[ModelReceipt]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return [
        _pull(capability, cache_dir, asr_model_id=asr_model_id)
        for capability in _capabilities(capabilities)
    ]


def model_status(
    capabilities: list[str] | None, *, cache_dir: Path, asr_model_id: str = "small"
) -> list[ModelReceipt]:
    return [
        _inspect(capability, cache_dir, verify=False, asr_model_id=asr_model_id)
        for capability in _capabilities(capabilities)
    ]


def verify_models(
    capabilities: list[str] | None, *, cache_dir: Path, asr_model_id: str = "small"
) -> list[ModelReceipt]:
    return [
        _inspect(capability, cache_dir, verify=True, asr_model_id=asr_model_id)
        for capability in _capabilities(capabilities)
    ]


def render_model_receipts(receipts: list[ModelReceipt], *, json_output: bool) -> str:
    if json_output:
        return json.dumps([item.model_dump(mode="json") for item in receipts], indent=2) + "\n"
    return "".join(
        f"{item.capability}: {item.state} ({item.provider}/{item.model_id})\n" for item in receipts
    )


def require_prepared_model(
    capability: ModelCapability, cache_root: Path, *, asr_model_id: str = "small"
) -> ModelReceipt:
    receipt = _inspect(capability, cache_root, verify=True, asr_model_id=asr_model_id)
    if receipt.state != "ready":
        raise ModelLifecycleError(
            f"local {capability.upper()} model is {receipt.state}; "
            f"run: vctx models pull {capability}"
        )
    return receipt


def rapidocr_config_path(cache_root: Path) -> Path:
    return cache_root / "ocr" / "rapidocr" / "config.yaml"


def _capabilities(values: list[str] | None) -> list[ModelCapability]:
    values = values or ["asr", "ocr"]
    invalid = [value for value in values if value not in {"asr", "ocr"}]
    if invalid:
        raise ModelLifecycleError(f"unsupported model capability: {', '.join(invalid)}")
    selected: list[ModelCapability] = []
    for value in values:
        capability: ModelCapability = "asr" if value == "asr" else "ocr"
        if capability not in selected:
            selected.append(capability)
    return selected


def _identity(capability: ModelCapability, asr_model_id: str = "small") -> tuple[str, str, str]:
    if capability == "asr":
        return ("faster-whisper", asr_model_id, "faster-whisper")
    return ("rapidocr", "rapidocr", "rapidocr")


def _model_dir(capability: ModelCapability, cache_root: Path, *, asr_model_id: str) -> Path:
    _provider, model_id, _package = _identity(capability, asr_model_id)
    return cache_root / capability / model_id


def _receipt_path(capability: ModelCapability, cache_root: Path) -> Path:
    return cache_root / "receipts" / f"{capability}.json"


def _pull(capability: ModelCapability, cache_root: Path, *, asr_model_id: str) -> ModelReceipt:
    provider, model_id, package = _identity(capability, asr_model_id)
    model_dir = _pull_model(capability, model_id, cache_root)
    digest, size = _tree_integrity(model_dir)
    if size == 0:
        raise ModelLifecycleError(f"{provider} model pull produced no files")
    receipt = ModelReceipt(
        capability=capability,
        provider=provider,
        model_id=model_id,
        state="ready",
        cache_path=model_dir.relative_to(cache_root).as_posix(),
        bytes=size,
        package_version=_package_version(package),
        integrity=digest,
    )
    path = _receipt_path(capability, cache_root)
    _write_atomic(path, encode_json(receipt))
    return receipt


def _inspect(
    capability: ModelCapability, cache_root: Path, *, verify: bool, asr_model_id: str
) -> ModelReceipt:
    provider, model_id, package = _identity(capability, asr_model_id)
    model_dir = _model_dir(capability, cache_root, asr_model_id=asr_model_id)
    path = _receipt_path(capability, cache_root)
    base = ModelReceipt(
        capability=capability,
        provider=provider,
        model_id=model_id,
        state="missing",
        cache_path=model_dir.relative_to(cache_root).as_posix(),
        bytes=0,
        package_version=_package_version(package),
    )
    try:
        prepared = path.is_file() and model_dir.is_dir()
    except OSError:
        prepared = False
    if not prepared:
        return base.model_copy(update={"message": "model is not prepared"})
    try:
        recorded = ModelReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return base.model_copy(update={"state": "corrupt", "message": "invalid receipt"})
    if not verify:
        return recorded
    digest, size = _tree_integrity(model_dir)
    if size == 0 or digest != recorded.integrity or size != recorded.bytes:
        return recorded.model_copy(
            update={
                "state": "corrupt",
                "bytes": size,
                "integrity": digest,
                "message": "integrity mismatch",
            }
        )
    return recorded


def _pull_model(capability: str, model_id: str, cache_root: Path) -> Path:
    target = cache_root / capability / model_id
    if capability == "asr":
        return _pull_asr_model(model_id, target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        module = importlib.import_module("rapidocr")
        module_file = module.__file__
        if module_file is None:
            raise ImportError("rapidocr has no filesystem package location")
        template = Path(module_file).with_name("config.yaml")
        config = template.read_text(encoding="utf-8").replace(
            "model_root_dir: null", f'model_root_dir: "{target.as_posix()}"'
        )
        config_path = rapidocr_config_path(cache_root)
        config_path.write_text(config, encoding="utf-8")
        module.download_models(config_path)
    except (ImportError, OSError, RuntimeError) as exc:
        raise ModelLifecycleError(f"failed to pull {capability} model: {exc}") from exc
    return target


def _pull_asr_model(model_id: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stage = target.parent / f".{target.name}.{token}.stage"
    backup = target.parent / f".{target.name}.{token}.backup"
    try:
        stage.mkdir()
        module = importlib.import_module("faster_whisper")
        module.download_model(model_id, output_dir=str(stage))
        _validate_ctranslate2(stage)
        if target.exists():
            os.replace(target, backup)
        os.replace(stage, target)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise ModelLifecycleError(f"failed to pull asr model: {exc}") from exc
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if target.exists():
            shutil.rmtree(backup, ignore_errors=True)
    return target


def _validate_ctranslate2(root: Path) -> None:
    missing = [name for name in ("model.bin", "config.json") if not (root / name).is_file()]
    if missing:
        raise ValueError(f"download is not a CTranslate2 model: missing {', '.join(missing)}")


def _write_atomic(path: Path, content: str) -> None:
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


def _tree_integrity(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(data)
        size += len(data)
    return digest.hexdigest(), size


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"
