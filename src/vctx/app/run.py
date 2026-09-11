from __future__ import annotations

import logging
from contextlib import suppress
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
    Keyring,
    admit_ai_binding,
)
from vctx.app.auth import AuthError, system_keyring
from vctx.app.models import select_asr_model_id
from vctx.artifact.bundle import retain_source_files
from vctx.artifact.manifest import (
    ArtifactRef,
    ManifestBuilder,
    ManifestSource,
    ProductOutcome,
    source_key,
)
from vctx.asr import AsrEnvironment, AsrRuntimePool
from vctx.config import AsrInstanceConfig, CapabilityPolicy, PrepareRequest, ResolvedConfig
from vctx.errors import CacheError, NoTranscriptError, ProviderError, SourceAssetsError
from vctx.model.store import ModelLifecycleError, ModelStore
from vctx.net import HttpxNetRuntime, NetRuntime
from vctx.source.admission import open_source, select_source
from vctx.source.session import (
    AsrAudioRequest,
    MediaAsset,
    MediaPermit,
    MediaRegistry,
    ObservePermit,
    SourceCapability,
    SourceSession,
    SubtitlePermit,
    VideoMetadata,
    VisualVideoRequest,
)
from vctx.source.store import SourceStore
from vctx.transcript import TranscriptPayload
from vctx.visual.processors import OcrAdmission, OcrRuntimePool

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
        self.asr.close()
        close = getattr(self.net, "close", None)
        if close is not None:
            close()


@dataclass
class PrepareRun:
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
    media: MediaRegistry = field(default_factory=MediaRegistry)
    subtitle: TranscriptPayload | None = None
    artifacts: list[ArtifactRef] = field(default_factory=list)

    def visual_ai_routes(self) -> list[AiRoute]:
        binding = self.ai_bindings.get("vision_description")
        return [binding.route] if binding is not None else []

    def planner_ai_route(self) -> AiRoute | None:
        binding = self.ai_bindings.get("evidence_plan")
        return binding.route if binding is not None else None

    def summary_ai_route(self) -> AiRoute | None:
        binding = self.ai_bindings.get("summary")
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

    def ensure_media(self, request: AsrAudioRequest | VisualVideoRequest) -> MediaAsset:
        cached = self.media.find(request)
        if cached is not None:
            return cached
        asset = self.source.media(request=request, permit=self.media_permit)
        required = "audio" if request.kind == "asr_audio" else "video"
        if required not in asset.capabilities:
            raise NoTranscriptError(f"source media lacks required {required} capability")
        return self.media.adopt(asset)

    def load_asr_environment(self) -> AsrEnvironment:
        if self.asr_environment is None:
            self.asr_environment = _load_asr_environment(self.resolved)
        return self.asr_environment

    def finish(
        self, artifacts: list[ArtifactRef], outcomes: list[ProductOutcome]
    ) -> ManifestSource:
        for outcome in outcomes:
            self.manifest.add_outcome(outcome)
        requested = next(item for item in outcomes if item.product == self.resolved.target)
        status = (
            "ok"
            if requested.status == "ready" and all(item.status == "ready" for item in outcomes)
            else "partial"
        )
        return self.manifest.finish(status, artifacts, self.source.receipts)

    def retain(
        self, artifacts: list[ArtifactRef], required: set[SourceCapability] | None = None
    ) -> None:
        complete = self.resolved.output.source_assets == "complete"
        if complete:
            try:
                capabilities = (
                    self.source.record.source_capabilities if required is None else required
                )
                if capabilities is None:
                    raise ProviderError("source capabilities are unknown")
                if "subtitle" in capabilities and self.subtitle is None:
                    self.subtitle = self.source.transcript(permit=self.subtitle_permit)
                if "audio" in capabilities:
                    self.ensure_media(
                        AsrAudioRequest(temp_dir=self.source_cache.path_for("tmp/retention"))
                    )
                if "video" in capabilities:
                    self.ensure_media(self.visual_media_request())
            except (CacheError, ProviderError) as exc:
                raise SourceAssetsError(exc) from exc
        media = list(self.media.assets.values())
        if (
            not media
            and self.resolved.output.source_assets == "consumed"
            and Path(self.request.inputs[0]).is_file()
        ):
            with suppress(NoTranscriptError):
                self.ensure_media(
                    request=AsrAudioRequest(temp_dir=self.source_cache.path_for("tmp/retention"))
                )
            media = list(self.media.assets.values())
        try:
            retained, omissions = retain_source_files(
                media,
                self.subtitle,
                self.request.out_dir,
                retain=self.resolved.output.source_assets != "omitted",
            )
        except CacheError as exc:
            self.manifest.add_step("source.asset_retention", "error", "integrity or copy failure")
            if complete:
                raise SourceAssetsError(exc) from exc
            raise
        artifacts.extend(retained)
        if retained:
            detail = ", ".join(artifact.path for artifact in retained)
            self.manifest.add_step("source.asset_retention", "ok", detail)
        elif omissions:
            self.manifest.add_step("source.asset_retention", "skipped", omissions[0])


def open_prepare_run(
    request: PrepareRequest,
    resolved: ResolvedConfig,
    occupied: dict[str, str],
    runtimes: RunRuntimes,
) -> PrepareRun:
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
    manifest = ManifestBuilder(
        source.record,
        key,
        offline=resolved.runtime.offline,
        asset_scope=resolved.output.source_assets,
    )
    logger.info(
        "prepare.config target=%s cache=%s config=%s offline=%s",
        resolved.target,
        cache.root,
        resolved.config_file.path or "built-in defaults + CLI",
        resolved.runtime.offline,
    )
    manifest.add_step("source.detect", "ok", source.name)
    metadata = source.record.metadata
    manifest.add_step("metadata.extract", "ok")
    return PrepareRun(
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
        ("summary", resolved.summary.policy),
    )
    for task, policy in selections:
        binding = admit_ai_binding(
            task=task,
            instance_name=policy.instance_name(),
            auto=policy.auto(),
            instances=resolved.instances.ai,
            offline=resolved.runtime.offline,
            env_files=resolved.runtime.env_files,
            keyring=keyring,
        )
        if binding is not None:
            bindings[task] = binding
    return bindings


def _load_asr_environment(resolved: ResolvedConfig) -> AsrEnvironment:
    instance_name = resolved.asr.instance_name()
    instance = select_asr_instance(resolved)
    if instance is None:
        return AsrEnvironment()
    if instance.type == "local-faster-whisper":
        model_id = instance.model or select_asr_model_id(resolved)
        prepared = instance_name is not None or _builtin_asr_ready(
            resolved.cache.model_dir, model_id
        )
        return AsrEnvironment(
            installed=prepared,
            model_id=model_id,
        )
    return AsrEnvironment()


def select_asr_instance(resolved: ResolvedConfig) -> AsrInstanceConfig | None:
    name = resolved.asr.instance_name()
    if name is not None:
        return resolved.instances.asr.get(name)
    if resolved.asr.enabled:
        return AsrInstanceConfig(type="local-faster-whisper", model=select_asr_model_id(resolved))
    return None


def _builtin_asr_ready(model_root: Path, model_id: str) -> bool:
    try:
        ModelStore(model_root).require("asr", asr_model_id=model_id)
    except ModelLifecycleError, OSError:
        return False
    return True
