from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

from vctx.ai import (
    AiBinding,
    AiClient,
    AiRoute,
    AiRuntimePool,
    AiTask,
    CredentialRef,
    Keyring,
    read_credential,
    select_ai_route,
)
from vctx.app.auth import AuthError, system_keyring
from vctx.app.models import ModelLifecycleError, require_prepared_model
from vctx.artifact.manifest import ArtifactRef, ManifestBuilder, source_key
from vctx.asr import AsrEnvironment, AsrRuntimePool
from vctx.config import AsrInstanceConfig, CapabilityPolicy, PrepareRequest, ResolvedConfig
from vctx.errors import NoTranscriptError
from vctx.net import HttpxNetRuntime, NetRuntime
from vctx.source.admission import open_source, select_source
from vctx.source.session import (
    MediaAsset,
    MediaPermit,
    ObservePermit,
    SourceSession,
    SubtitlePermit,
    VideoMetadata,
    VisualVideoRequest,
)
from vctx.source.store import SourceStore
from vctx.transcript import TranscriptPayload
from vctx.visual.ocr import OcrAdmission, OcrRuntimePool

logger = logging.getLogger(__name__)


@dataclass
class RunRuntimes:
    net: NetRuntime = field(default_factory=lambda: HttpxNetRuntime())
    asr: AsrRuntimePool = field(default_factory=AsrRuntimePool)
    ai: AiRuntimePool = field(default_factory=AiRuntimePool)
    ocr: OcrRuntimePool = field(default_factory=OcrRuntimePool)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        close = getattr(self.net, "close", None)
        if close is not None:
            close()


@dataclass
class Run:
    request: PrepareRequest
    resolved: ResolvedConfig
    manifest: ManifestBuilder
    source_cache: SourceStore
    model_root: Path
    source: SourceSession
    metadata: VideoMetadata
    subtitle_permit: SubtitlePermit
    media_permit: MediaPermit
    runtimes: RunRuntimes
    ai_bindings: dict[AiTask, AiBinding] = field(default_factory=dict)
    ocr_runtime: OcrAdmission | None = None
    asr_environment: AsrEnvironment | None = None
    media: MediaAsset | None = None
    subtitle: TranscriptPayload | None = None
    artifacts: list[ArtifactRef] = field(default_factory=list)

    def visual_ai_routes(self) -> list[AiRoute]:
        binding = self.ai_bindings.get("vision_description")
        return [binding.route] if binding is not None else []

    def planner_ai_route(self) -> AiRoute | None:
        binding = self.ai_bindings.get("evidence_plan")
        return binding.route if binding is not None else None

    def ai_client(self, route: AiRoute) -> AiClient:
        return AiClient(
            self.ai_bindings[route.task],
            net=self.runtimes.net,
            runtimes=self.runtimes.ai,
        )

    def visual_media_request(self) -> VisualVideoRequest:
        return VisualVideoRequest(
            temp_dir=self.source_cache.root / "tmp" / "yt-dlp",
            profile=self.resolved.source.media_quality.value,
            refresh=self.request.overwrite,
        )

    def load_asr_environment(self) -> AsrEnvironment:
        if self.asr_environment is None:
            self.asr_environment = _load_asr_environment(self.resolved)
        return self.asr_environment


def open_run(
    request: PrepareRequest,
    resolved: ResolvedConfig,
    occupied: dict[str, str],
    runtimes: RunRuntimes,
) -> Run:
    network = "denied" if resolved.runtime.offline else "allowed"
    permit = ObservePermit(operation="prepare", network=network)
    cache = SourceStore(resolved.cache.source_dir)
    selection = select_source(request.inputs[0])
    source = open_source(
        selection,
        request.inputs[0],
        permit=permit,
        options=resolved.source.yt_dlp,
        net=runtimes.net,
        store=cache,
    )
    if source.record.lifecycle != "finite":
        raise NoTranscriptError(
            f"source is {source.record.lifecycle}; prepare requires finite or archived media"
        )
    key = source_key(source.record.source_id, occupied)
    occupied[key.casefold()] = source.record.source_id
    lane_request = request.model_copy(update={"out_dir": request.out_dir / key})
    manifest = ManifestBuilder.start(source.record, key, offline=resolved.runtime.offline)
    logger.info(
        "prepare.config workflow=%s cache=%s config=%s offline=%s",
        resolved.runtime.workflow,
        cache.root,
        request.config_path or "built-in defaults + CLI",
        resolved.runtime.offline,
    )
    logger.debug("prepare.output formats=%s", ",".join(resolved.output.formats))
    manifest.add_step("source.detect", "ok", source.name)
    logger.info("source.detect adapter=%s", source.name)
    metadata = source.record.metadata
    manifest.add_step("metadata.extract", "ok")
    logger.info(
        "metadata.extract status=ok id=%s source_type=%s",
        metadata.id,
        metadata.source_type,
    )
    return Run(
        request=lane_request,
        resolved=resolved,
        manifest=manifest,
        source_cache=cache,
        model_root=resolved.cache.model_dir,
        source=source,
        metadata=metadata,
        subtitle_permit=SubtitlePermit(network=permit.network),
        media_permit=MediaPermit(network=permit.network),
        runtimes=runtimes,
        ai_bindings=_ai_bindings(resolved),
        ocr_runtime=(
            runtimes.ocr.load_rapid(resolved.cache.model_dir)
            if resolved.evidence.ocr.enabled and resolved.evidence.ocr.auto()
            else None
        ),
    )


def _ai_bindings(resolved: ResolvedConfig) -> dict[AiTask, AiBinding]:
    if resolved.runtime.offline:
        return {}
    try:
        keyring: Keyring | None = system_keyring()
    except AuthError:
        keyring = None
    bindings: dict[AiTask, AiBinding] = {}
    selections: tuple[tuple[AiTask, CapabilityPolicy], ...] = (
        ("evidence_plan", resolved.evidence.planner),
        ("vision_description", resolved.evidence.vision),
    )
    for task, policy in selections:
        binding = _ai_binding(task, policy, resolved, keyring)
        if binding is not None:
            bindings[task] = binding
    return bindings


def _ai_binding(
    task: AiTask,
    policy: CapabilityPolicy,
    resolved: ResolvedConfig,
    keyring: Keyring | None,
) -> AiBinding | None:
    if policy.disabled():
        return None
    credential = None
    auto_ref = None
    if policy.auto():
        for candidate in (
            CredentialRef("env:OPENROUTER_API_KEY"),
            CredentialRef("keyring:openrouter"),
        ):
            try:
                credential = read_credential(
                    candidate,
                    env_files=resolved.runtime.env_files,
                    keyring=keyring,
                )
            except ValueError:
                continue
            auto_ref = candidate
            break
    route = select_ai_route(
        task=task,
        instance_name=policy.instance_name(),
        auto=policy.auto(),
        instances=resolved.instances.ai,
        offline=False,
        auto_credential=auto_ref,
    )
    if route is None:
        return None
    if route.instance.credential is not None and credential is None:
        try:
            credential = read_credential(
                route.instance.credential,
                env_files=resolved.runtime.env_files,
                keyring=keyring,
            )
        except ValueError:
            return None
    return AiBinding(route, credential)


def _load_asr_environment(resolved: ResolvedConfig) -> AsrEnvironment:
    instance_name = resolved.transforms.asr.instance_name()
    instance = select_asr_instance(resolved)
    if instance is None:
        return AsrEnvironment(offline=resolved.runtime.offline)
    if instance.type == "local-faster-whisper":
        prepared = instance_name is not None or _builtin_asr_ready(resolved.cache.model_dir)
        return AsrEnvironment(
            offline=resolved.runtime.offline,
            installed=prepared,
            model_id=instance.model or "small",
        )
    return AsrEnvironment(offline=resolved.runtime.offline)


def select_asr_instance(resolved: ResolvedConfig) -> AsrInstanceConfig | None:
    name = resolved.transforms.asr.instance_name()
    if name is not None:
        return resolved.instances.asr.get(name)
    if resolved.transforms.asr.enabled:
        return AsrInstanceConfig(type="local-faster-whisper", model="small")
    return None


def _builtin_asr_ready(model_root: Path) -> bool:
    try:
        require_prepared_model("asr", model_root, asr_model_id="small")
    except (ModelLifecycleError, OSError):
        return False
    return True
