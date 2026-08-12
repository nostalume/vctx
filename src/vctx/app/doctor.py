from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from vctx.ai import AiTask, Keyring, admit_ai_binding
from vctx.app.auth import AuthError, system_keyring
from vctx.app.models import model_status, select_asr_model_id
from vctx.config import (
    CapabilityPolicy,
    PrepareRequest,
    PrepareTarget,
    ResolvedConfig,
    load_resolved_config,
)


def doctor_report(
    *,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
    target: PrepareTarget = PrepareTarget.TRANSCRIPT,
    asr: str | None = None,
    ocr: str | None = None,
    vision: str | None = None,
    offline: bool | None = None,
    retain_media: bool | None = None,
    json_output: bool = False,
) -> str:
    resolved = load_resolved_config(
        PrepareRequest(
            inputs=["doctor"],
            out_dir=Path("."),
            config_path=config_path,
            cache_dir=cache_dir,
            target=target,
            asr_use=asr,
            ocr_use=ocr,
            vision_use=vision,
            offline=offline,
            retain_media=retain_media,
        )
    )
    asr_model_id = select_asr_model_id(resolved)
    models = {
        item.capability: item
        for item in model_status(
            ["asr", "ocr"],
            cache_dir=resolved.cache.model_dir,
            asr_model_id=asr_model_id,
        )
    }
    try:
        keyring: Keyring | None = system_keyring()
    except AuthError:
        keyring = None
    report: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "vctx": _package_version("vctx"),
        "yt-dlp": _package_version("yt-dlp"),
        "profile": _installed_profile(),
        "target": resolved.target.value,
        "offline": resolved.runtime.offline,
        "retention": "retain" if resolved.output.retain_media else "omit",
        "config": {
            "origin": resolved.config_file.origin,
            "path": str(resolved.config_file.path) if resolved.config_file.path else None,
        },
        "cache": {
            "source_dir": str(resolved.cache.source_dir),
            "model_dir": str(resolved.cache.model_dir),
            "source_state": _cache_state(resolved.cache.source_dir),
        },
        "capabilities": {
            "asr": _capability(
                resolved.asr,
                models["asr"].state if resolved.asr.enabled else "disabled",
            ),
            "ocr": _capability(
                resolved.evidence.ocr,
                models["ocr"].state if resolved.evidence.ocr.enabled else "disabled",
            ),
            "vision": _capability(
                resolved.evidence.vision,
                _read_ai_readiness(
                    resolved, "vision_description", resolved.evidence.vision, keyring
                ),
            ),
            "planner": _capability(
                resolved.evidence.planner,
                _read_ai_readiness(resolved, "evidence_plan", resolved.evidence.planner, keyring),
            ),
        },
    }
    if json_output:
        return json.dumps(report, indent=2) + "\n"
    lines = [
        *(f"{key}: {report[key]}" for key in ("python", "vctx", "yt-dlp", "profile", "target")),
        f"offline: {str(report['offline']).lower()}",
        f"retention: {report['retention']}",
        _config_line(report["config"]),
        f"cache.source: {report['cache']['source_state']} ({report['cache']['source_dir']})",
        f"cache.models: {report['cache']['model_dir']}",
        *(
            f"capability.{name}: {value['selector']} ({value['readiness']})"
            for name, value in report["capabilities"].items()
        ),
    ]
    return "\n".join(lines) + "\n"


def _capability(policy: CapabilityPolicy, readiness: str) -> dict[str, str]:
    return {"selector": _selector(policy), "readiness": readiness}


def _selector(policy: CapabilityPolicy) -> str:
    if policy.disabled():
        return "none"
    if policy.auto():
        return "auto"
    instance = policy.instance_name()
    if instance is not None:
        return f"instance:{instance}"
    return policy.model_ref() or "unknown"


def _read_ai_readiness(
    resolved: ResolvedConfig,
    task: AiTask,
    policy: CapabilityPolicy,
    keyring: Keyring | None,
) -> str:
    if policy.disabled():
        return "disabled"
    if resolved.runtime.offline:
        return "unavailable-offline"
    binding = admit_ai_binding(
        task=task,
        instance_name=policy.instance_name(),
        auto=policy.auto(),
        instances=resolved.instances.ai,
        offline=resolved.runtime.offline,
        env_files=resolved.runtime.env_files,
        keyring=keyring,
    )
    if binding is None:
        return "unavailable-auth"
    source = (
        binding.route.credential.partition(":")[0] if binding.route.credential else "none"
    )
    return f"configured-{source}"


def _installed_profile() -> str:
    has_asr = _package_version("faster-whisper") != "missing"
    has_visual = all(
        _package_version(package) != "missing" for package in ("av", "onnxruntime", "rapidocr")
    )
    if has_asr and has_visual:
        return "full"
    if has_asr:
        return "asr"
    if has_visual:
        return "visual"
    return "core"


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _config_line(config: dict[str, str | None]) -> str:
    path = f" ({config['path']})" if config["path"] else ""
    return f"config: {config['origin']}{path}"


def _cache_state(cache_dir: Path) -> str:
    if not cache_dir.exists():
        return "missing"
    if not cache_dir.is_dir():
        return "error-not-directory"
    return "present"
