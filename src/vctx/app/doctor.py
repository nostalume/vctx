from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from vctx.ai import AiTask, CredentialRef, read_credential, select_ai_route
from vctx.app.auth import AuthError, system_keyring
from vctx.app.models import model_status, select_asr_model_id
from vctx.config import (
    CapabilityPolicy,
    PrepareRequest,
    ResolvedConfig,
    WorkflowProfile,
    load_resolved_config,
)


def doctor_report(
    *,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
    workflow: WorkflowProfile | None = None,
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
            workflow=workflow,
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
    report: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "vctx": _package_version("vctx"),
        "yt-dlp": _package_version("yt-dlp"),
        "profile": _installed_profile(),
        "workflow": resolved.runtime.workflow.value,
        "offline": resolved.runtime.offline,
        "retention": "retain" if resolved.output.retain_media else "omit",
        "cache": _cache_status(resolved.cache.source_dir),
        "capabilities": {
            "asr": _capability(
                resolved.transforms.asr,
                models["asr"].state if resolved.transforms.asr.enabled else "disabled",
            ),
            "ocr": _capability(
                resolved.evidence.ocr,
                models["ocr"].state if resolved.evidence.ocr.enabled else "disabled",
            ),
            "vision": _capability(
                resolved.evidence.vision,
                _read_ai_readiness(resolved, "vision_description", resolved.evidence.vision),
            ),
            "planner": _capability(
                resolved.evidence.planner,
                _read_ai_readiness(resolved, "evidence_plan", resolved.evidence.planner),
            ),
        },
    }
    if json_output:
        return json.dumps(report, indent=2) + "\n"
    lines = [
        *(f"{key}: {report[key]}" for key in ("python", "vctx", "yt-dlp", "profile", "workflow")),
        f"offline: {str(report['offline']).lower()}",
        f"retention: {report['retention']}",
        f"cache: {report['cache']}",
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


def _read_ai_readiness(resolved: ResolvedConfig, task: AiTask, policy: CapabilityPolicy) -> str:
    if policy.disabled():
        return "disabled"
    if resolved.runtime.offline:
        return "unavailable-offline"
    try:
        keyring = system_keyring()
    except AuthError:
        keyring = None
    reference = None
    if policy.auto():
        for candidate in (
            CredentialRef("env:OPENROUTER_API_KEY"),
            CredentialRef("keyring:openrouter"),
        ):
            try:
                read_credential(candidate, env_files=resolved.runtime.env_files, keyring=keyring)
            except ValueError:
                continue
            reference = candidate
            break
    route = select_ai_route(
        task=task,
        instance_name=policy.instance_name(),
        auto=policy.auto(),
        instances=resolved.instances.ai,
        offline=False,
        auto_credential=reference,
    )
    if route is None:
        return "unavailable-auth"
    if route.instance.credential is not None and not policy.auto():
        try:
            read_credential(
                route.instance.credential,
                env_files=resolved.runtime.env_files,
                keyring=keyring,
            )
        except ValueError:
            return "unavailable-auth"
    source = route.credential.partition(":")[0] if route.credential else "none"
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


def _cache_status(cache_dir: Path) -> str:
    if not cache_dir.exists():
        return f"missing ({cache_dir})"
    if not cache_dir.is_dir():
        return f"error: not a directory ({cache_dir})"
    return f"present ({cache_dir})"
