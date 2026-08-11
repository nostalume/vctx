from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from vctx.config import PrepareRequest, WorkflowProfile, resolve_config
from vctx.errors import VctxError
from vctx.io import model_to_json

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


def resolve_asr_model_id(
    *, config_path: Path | None, cache_dir: Path | None, selector: str | None
) -> str:
    resolved = resolve_config(
        PrepareRequest(
            inputs=["model-lifecycle"],
            out_dir=Path("."),
            workflow=WorkflowProfile.TRANSCRIPT,
            config_path=config_path,
            cache_dir=cache_dir,
            asr_use=selector,
        )
    )
    policy = resolved.transforms.asr
    model_ref = policy.model_ref()
    if model_ref is not None:
        if model_ref.startswith("local:"):
            return model_ref.removeprefix("local:")
        raise ModelLifecycleError("models pull manages named local ASR models, not path/HF refs")
    instance_name = policy.instance_name()
    if instance_name is None:
        return "base"
    instance = resolved.instances.asr[instance_name]
    if instance.type != "local-faster-whisper":
        raise ModelLifecycleError("selected ASR instance is online and has no local model to pull")
    model_id = instance.model or instance.model_policy
    if Path(model_id).is_absolute():
        raise ModelLifecycleError("selected ASR model is an explicit local path and needs no pull")
    return "base" if model_id == "auto" else model_id


def resolve_model_dir(*, config_path: Path | None, cache_dir: Path | None) -> Path:
    return resolve_config(
        PrepareRequest(inputs=["model-lifecycle"], out_dir=Path("."), config_path=config_path,
                       cache_dir=cache_dir)
    ).cache.model_dir


def manage_models(
    action: Literal["pull", "status", "verify"],
    capabilities: list[str] | None,
    *,
    cache_dir: Path,
    asr_model_id: str = "base",
) -> list[ModelReceipt]:
    cache_root = cache_dir
    if action == "pull":
        cache_root.mkdir(parents=True, exist_ok=True)
    selected = _capabilities(capabilities)
    if action == "pull":
        return [_pull(capability, cache_root, asr_model_id=asr_model_id) for capability in selected]
    return [
        _inspect(capability, cache_root, verify=action == "verify", asr_model_id=asr_model_id)
        for capability in selected
    ]


def render_model_receipts(receipts: list[ModelReceipt], *, json_output: bool) -> str:
    if json_output:
        return json.dumps([item.model_dump(mode="json") for item in receipts], indent=2) + "\n"
    return "".join(
        f"{item.capability}: {item.state} ({item.provider}/{item.model_id})\n"
        for item in receipts
    )


def require_prepared_model(
    capability: ModelCapability, cache_root: Path, *, asr_model_id: str = "base"
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


def _identity(capability: ModelCapability, asr_model_id: str = "base") -> tuple[str, str, str]:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model_to_json(receipt), encoding="utf-8")
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
    if not path.is_file() or not model_dir.is_dir():
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
    target.mkdir(parents=True, exist_ok=True)
    try:
        if capability == "asr":
            module = importlib.import_module("faster_whisper")
            module.download_model(model_id, output_dir=str(target))
        else:
            module = importlib.import_module("rapidocr")
            template = Path(module.__file__).with_name("config.yaml")
            config = template.read_text(encoding="utf-8").replace(
                "model_root_dir: null", f'model_root_dir: "{target.as_posix()}"'
            )
            config_path = rapidocr_config_path(cache_root)
            config_path.write_text(config, encoding="utf-8")
            module.download_models(config_path)
    except (ImportError, OSError, RuntimeError) as exc:
        raise ModelLifecycleError(f"failed to pull {capability} model: {exc}") from exc
    return target


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
