from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from vctx.ai import AiTask, CredentialRef, Keyring, select_ai_route
from vctx.app.auth import AuthError, probe_credential_presence, system_keyring
from vctx.asr import AsrReadinessFacts, decide_asr_readiness
from vctx.asr_faster_whisper import bundled_cuda_state
from vctx.config import (
    AsrInstanceConfig,
    CapabilityPolicy,
    PrepareRequest,
    PrepareTarget,
    ResolvedConfig,
    load_resolved_config,
)
from vctx.model_store import model_status, package_version


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
    asr_instance = _asr_instance(resolved)
    asr_readiness = decide_asr_readiness(
        resolved.asr,
        asr_instance,
        _observe_asr_facts(resolved, asr_instance),
    )
    models = {
        item.capability: item
        for item in model_status(
            ["ocr"],
            cache_dir=resolved.cache.model_dir,
        )
    }
    try:
        keyring: Keyring | None = system_keyring()
    except AuthError:
        keyring = None
    report: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "vctx": package_version("vctx"),
        "yt-dlp": package_version("yt-dlp"),
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
                asr_readiness.state,
                runtime=asr_readiness.runtime.model_dump(mode="json"),
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


def _capability(
    policy: CapabilityPolicy,
    readiness: str,
    *,
    runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"selector": _selector(policy), "readiness": readiness}
    if runtime is not None:
        result["runtime"] = runtime
    return result


def _asr_instance(resolved: ResolvedConfig) -> AsrInstanceConfig:
    name = resolved.asr.instance_name()
    if name is not None:
        return resolved.instances.asr[name]
    reference = resolved.asr.model_ref()
    return AsrInstanceConfig(type="local-faster-whisper", model=reference or "small")


def _observe_asr_facts(resolved: ResolvedConfig, instance: AsrInstanceConfig) -> AsrReadinessFacts:
    package_state = "missing" if package_version("faster-whisper") == "missing" else "ready"
    cuda_libraries = bundled_cuda_state()
    if resolved.asr.disabled():
        return AsrReadinessFacts(
            model_kind="disabled", package_state=package_state, cuda_libraries=cuda_libraries
        )
    reference = resolved.asr.model_ref() or instance.model or "small"
    if reference.startswith("hf:"):
        return AsrReadinessFacts(
            model_kind="unsupported",
            model_reference=reference,
            package_state=package_state,
            cuda_libraries=cuda_libraries,
        )
    explicit = reference.startswith("path:")
    value = reference.removeprefix("path:").removeprefix("local:")
    path = Path(value)
    if explicit or path.is_absolute() or path.exists():
        if not path.is_dir():
            state = "missing"
        elif all((path / name).is_file() for name in ("model.bin", "config.json")):
            state = "ready"
        else:
            state = "corrupt"
        return AsrReadinessFacts(
            model_kind="explicit",
            model_reference=str(path),
            model_state=state,
            package_state=package_state,
            cuda_libraries=cuda_libraries,
        )
    receipt = model_status(["asr"], cache_dir=resolved.cache.model_dir, asr_model_id=value)[0]
    return AsrReadinessFacts(
        model_kind="managed",
        model_reference=value,
        model_state=receipt.state,
        package_state=package_state,
        cuda_libraries=cuda_libraries,
    )


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
    inaccessible = False
    auto_reference = None
    if policy.auto():
        for reference in (
            CredentialRef("env:OPENROUTER_API_KEY"),
            CredentialRef("keyring:openrouter"),
        ):
            fact = probe_credential_presence(
                reference,
                env_files=resolved.runtime.env_files,
                keyring=keyring,
            )
            inaccessible = inaccessible or fact.status == "inaccessible"
            if fact.status == "present":
                auto_reference = reference
                break
    route = select_ai_route(
        task=task,
        instance_name=policy.instance_name(),
        auto=policy.auto(),
        instances=resolved.instances.ai,
        offline=resolved.runtime.offline,
        auto_credential=auto_reference,
    )
    if route is None:
        return "unavailable-keyring" if inaccessible else "unavailable-auth"
    reference = route.instance.credential
    if reference is None:
        return "configured-none"
    fact = probe_credential_presence(
        reference,
        env_files=resolved.runtime.env_files,
        keyring=keyring,
    )
    if fact.status == "present":
        return f"configured-{fact.kind}"
    return f"unavailable-{fact.kind}" if fact.status == "missing" else "unavailable-keyring"


def _installed_profile() -> str:
    has_asr = package_version("faster-whisper") != "missing"
    has_visual = all(
        package_version(package) != "missing" for package in ("av", "onnxruntime", "rapidocr")
    )
    if has_asr and has_visual:
        return "full"
    if has_asr:
        return "asr"
    if has_visual:
        return "visual"
    return "core"


def _config_line(config: dict[str, str | None]) -> str:
    path = f" ({config['path']})" if config["path"] else ""
    return f"config: {config['origin']}{path}"


def _cache_state(cache_dir: Path) -> str:
    if not cache_dir.exists():
        return "missing"
    if not cache_dir.is_dir():
        return "error-not-directory"
    return "present"
