from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from vctx.app.credentials import (
    CredentialError,
    env_with_credential_presence,
    resolve_env_credential,
)
from vctx.app.media_retention import materialize_source_assets
from vctx.app.progress import phase
from vctx.app.visual import Visuals, visual_products
from vctx.artifact.content import Artifact
from vctx.artifact.manifest import (
    ArtifactRef,
    ManifestBuilder,
    ManifestSource,
    source_key,
)
from vctx.config import (
    AsrInstanceConfig,
    PrepareRequest,
    ResolvedConfig,
    WorkflowProfile,
)
from vctx.errors import CacheError, NoTranscriptError, ProviderError, VctxError
from vctx.io import (
    model_to_json,
    write_artifact,
    write_artifact_bundle,
)
from vctx.models.knowledge_flow import KnowledgeFlow
from vctx.render.bundle import render_artifact_bundle
from vctx.source.admission import admit_source
from vctx.source.session import (
    AsrAudioRequest,
    MediaAsset,
    MediaPermit,
    ObservePermit,
    SourceSession,
    SubtitlePermit,
    VideoMetadata,
    VisualVideoRequest,
)
from vctx.source.store import SourceStore
from vctx.transcript import (
    ChunkOptions,
    ChunkSet,
    Transcript,
    TranscriptPayload,
    chunk_transcript,
    normalize_transcript,
    parse_transcript_payload,
)
from vctx.transforms.ai_routes import AiRoute, AiTaskKind, resolve_openrouter_ai_route
from vctx.transforms.asr import AsrExecutionError, run_asr
from vctx.transforms.model_resolution import (
    OPENROUTER_API_KEY_ENV,
    ModelCapability,
)
from vctx.transforms.planning import RoutePlan, SourceState, TransformEnvironment, plan_asr
from vctx.transforms.text_ai import OpenAiCompatibleTextAdapter

logger = logging.getLogger(__name__)


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
    runtime_cache: dict[str, object]
    media: MediaAsset | None = None
    subtitle: TranscriptPayload | None = None
    artifacts: list[ArtifactRef] = field(default_factory=list)

    def openrouter_env(self) -> dict[str, str]:
        return env_with_credential_presence(
            [OPENROUTER_API_KEY_ENV],
            env_files=self.resolved.runtime.env_files,
            base_env=os.environ,
        )

    def visual_ai_routes(self) -> list[AiRoute]:
        route = resolve_openrouter_ai_route(
            self.resolved.transforms.visual_context,
            task="vision_description",
            capability=ModelCapability.VISION_DESCRIPTION,
            cache_root=self.model_root,
            env=self.openrouter_env(),
            offline=self.resolved.runtime.offline,
        )
        return [route] if route is not None else []

    def text_product_ai_route(self, task: AiTaskKind) -> AiRoute | None:
        return resolve_openrouter_ai_route(
            self.resolved.transforms.knowledge_flow,
            task=task,
            capability=ModelCapability.ESSENTIAL_CASES,
            cache_root=self.model_root,
            env=self.openrouter_env(),
            offline=self.resolved.runtime.offline,
        )

    def text_ai_adapter(self, route: AiRoute) -> OpenAiCompatibleTextAdapter:
        return OpenAiCompatibleTextAdapter(
            route=route,
            api_key=resolve_env_credential(
                route.api_key_env,
                env_files=self.resolved.runtime.env_files,
            ),
        )

    def visual_media_request(self) -> VisualVideoRequest:
        return VisualVideoRequest(
            temp_dir=self.source_cache.root / "tmp" / "yt-dlp",
            profile=self.resolved.source.media_quality.value,
            refresh=self.request.overwrite,
        )


@dataclass(frozen=True)
class Prepared:
    transcript: Transcript
    chunks: ChunkSet


@dataclass(frozen=True)
class AsrReady:
    plan: RoutePlan
    media: MediaAsset
    instance: AsrInstanceConfig
    api_key: str | None


@dataclass(frozen=True)
class SourcePrepared:
    source: ManifestSource
    artifacts: list[ArtifactRef]
    request: PrepareRequest
    resolved: ResolvedConfig
    error: VctxError | None = None


def prepare_source(
    request: PrepareRequest,
    resolved: ResolvedConfig,
    occupied: dict[str, str],
    completed: set[str],
    runtime_cache: dict[str, object],
    previous: dict[str, ManifestSource] | None = None,
    reset_lane: Callable[[str], None] | None = None,
    rollback_lane: Callable[[str], None] | None = None,
) -> SourcePrepared | None:
    logger.info("prepare.start input=%s out=%s", request.inputs[0], request.out_dir)
    run = _start(request, resolved, occupied, runtime_cache)
    if run.source.record.source_id in completed:
        return None
    completed.add(run.source.record.source_id)
    prior = (previous or {}).get(run.source.record.source_id)
    if prior is not None and prior.revision == run.source.record.revision and not request.overwrite:
        return SourcePrepared(prior, prior.artifacts, run.request, resolved)
    if reset_lane is not None:
        reset_lane(run.manifest.key)
    try:
        if run.resolved.runtime.workflow == WorkflowProfile.METADATA:
            run.manifest.add_step("transcript.extract", "skipped", "metadata workflow selected")
            run.manifest.warn("metadata workflow selected; transcript pipeline skipped")
            return _partial(run)
        transcript = _transcript(run)
        if isinstance(transcript, SourcePrepared):
            return transcript
        prepared = _prepared(run, transcript)
        visuals, flow = visual_products(run, prepared)
        return _finish(run, prepared, visuals, flow)
    except (CacheError, ProviderError) as exc:
        return _error_result(run, exc)
    except VctxError:
        if rollback_lane is not None:
            rollback_lane(run.manifest.key)
        raise


def _start(
    request: PrepareRequest,
    resolved: ResolvedConfig,
    occupied: dict[str, str],
    runtime_cache: dict[str, object],
) -> Run:
    network = "denied" if resolved.runtime.offline else "allowed"
    permit = ObservePermit(operation="prepare", network=network)
    cache = SourceStore(resolved.cache.source_dir)
    source = admit_source(
        request.inputs[0],
        permit=permit,
        options=resolved.source.yt_dlp,
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
        runtime_cache=runtime_cache,
    )


def _transcript(run: Run) -> TranscriptPayload | SourcePrepared:
    with phase(logger, "transcript.extract"):
        logger.info("transcript.extract start")
        try:
            payload = run.source.transcript(permit=run.subtitle_permit)
        except NoTranscriptError as exc:
            logger.info("transcript.extract status=missing reason=%s", exc)
            return _asr_transcript(run, exc)

    run.manifest.add_step("transcript.extract", "ok", _transcript_detail(payload))
    run.subtitle = payload
    logger.info("transcript.extract status=ok provenance=%s", payload.provenance_label())
    asr_plan = plan_asr(
        run.resolved.transforms.asr,
        TransformEnvironment(offline=run.resolved.runtime.offline),
        SourceState(has_transcript=True, has_media=False),
    )
    run.manifest.add_transform_evidence(asr_plan.evidence_seed)
    run.manifest.add_step(
        "transform.asr",
        "skipped" if asr_plan.selected == "skipped" else "ok",
        asr_plan.reason,
    )
    logger.info("asr.route selected=%s reason=%s", asr_plan.selected, asr_plan.reason)
    return payload


def _asr_transcript(
    run: Run,
    transcript_error: NoTranscriptError,
) -> TranscriptPayload | SourcePrepared:
    run.manifest.add_step("transcript.extract", "warning", str(transcript_error))
    source = _asr_source(run, transcript_error)
    if isinstance(source, SourcePrepared):
        return source

    ready = _asr_ready(run, source)
    if isinstance(ready, SourcePrepared):
        return ready

    try:
        with phase(logger, "asr.execute"):
            logger.info(
                "asr.execute start route=%s provider=%s model=%s",
                ready.plan.selected,
                ready.plan.provider_id,
                ready.plan.model_id,
            )
            payload = run_asr(
                ready.plan,
                ready.media,
                instance=ready.instance,
                cache_root=run.model_root,
                offline=run.resolved.runtime.offline,
                api_key=ready.api_key,
                runtime_cache=run.runtime_cache,
            )
    except AsrExecutionError as asr_exc:
        run.manifest.add_step("transform.asr", "warning", str(asr_exc))
        run.manifest.warn(str(asr_exc))
        logger.warning("asr.execute status=warning reason=%s", asr_exc)
        return _partial(run)

    run.manifest.add_step("transform.asr", "ok", payload.provenance_label())
    logger.info("asr.execute status=ok provenance=%s", payload.provenance_label())
    return payload


def _asr_source(run: Run, transcript_error: NoTranscriptError) -> RoutePlan | SourcePrepared:
    pre_media_asr_plan = plan_asr(
        run.resolved.transforms.asr,
        _asr_environment(run.resolved),
        SourceState(has_transcript=False, has_media=True),
    )
    if pre_media_asr_plan.selected not in {"local", "configured-online"}:
        run.manifest.add_transform_evidence(pre_media_asr_plan.evidence_seed)
        run.manifest.add_step("source.media", "skipped", "no executable ASR route selected")
        run.manifest.add_step("transform.asr", "warning", pre_media_asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(_asr_missing_input_hint())
        logger.warning("asr.route status=unavailable reason=%s", pre_media_asr_plan.reason)
        return _partial(run)

    try:
        logger.info("source.media start purpose=asr")
        run.media = run.source.media(
            request=AsrAudioRequest(
                temp_dir=run.source_cache.root / "tmp" / "yt-dlp",
                refresh=run.request.overwrite,
            ), permit=run.media_permit,
        )
    except NoTranscriptError as media_exc:
        asr_plan = plan_asr(
            run.resolved.transforms.asr,
            _asr_environment(run.resolved),
            SourceState(has_transcript=False, has_media=False),
        )
        run.manifest.add_step("source.media", "warning", str(media_exc))
        run.manifest.add_transform_evidence(asr_plan.evidence_seed)
        run.manifest.add_step("transform.asr", "warning", asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(_asr_missing_input_hint())
        logger.warning("source.media status=warning purpose=asr reason=%s", media_exc)
        return _partial(run)

    run.manifest.add_step("source.media", "ok", _media_detail(run.media))
    logger.info("source.media status=ok purpose=asr path=%s", run.media.local_path)
    asr_plan = plan_asr(
        run.resolved.transforms.asr,
        _asr_environment(run.resolved),
        SourceState(has_transcript=False, has_media=True),
    )
    if asr_plan.selected not in {"local", "configured-online"}:
        run.manifest.add_transform_evidence(asr_plan.evidence_seed)
        run.manifest.add_step("transform.asr", "warning", asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(asr_plan.reason)
        logger.warning("asr.route status=unavailable reason=%s", asr_plan.reason)
        return _partial(run)
    logger.info("asr.route selected=%s reason=%s", asr_plan.selected, asr_plan.reason)
    return asr_plan


def _asr_ready(run: Run, asr_plan: RoutePlan) -> AsrReady | SourcePrepared:
    instance_name = run.resolved.transforms.asr.instance_name()
    instance = run.resolved.instances.asr.get(instance_name) if instance_name else None
    if instance is None:
        run.manifest.add_step("transform.asr", "warning", "ASR instance is not configured")
        run.manifest.warn("ASR instance is not configured")
        logger.warning("asr.ready status=warning reason=missing-instance")
        return _partial(run)

    run.manifest.add_transform_evidence(asr_plan.evidence_seed)
    api_key: str | None = None
    if asr_plan.selected == "configured-online":
        try:
            api_key = resolve_env_credential(
                instance.api_key_env,
                env_files=run.resolved.runtime.env_files,
            )
        except CredentialError as credential_exc:
            run.manifest.add_step("transform.asr", "warning", str(credential_exc))
            run.manifest.warn(str(credential_exc))
            logger.warning("asr.ready status=warning reason=%s", credential_exc)
            return _partial(run)
    assert run.media is not None
    logger.info(
        "asr.ready status=ok provider=%s model=%s credential=%s",
        asr_plan.provider_id,
        asr_plan.model_id,
        instance.api_key_env if asr_plan.selected == "configured-online" else "not-required",
    )
    return AsrReady(plan=asr_plan, media=run.media, instance=instance, api_key=api_key)


def _asr_missing_input_hint() -> str:
    return (
        "Provide a transcript file, install the default ASR extra, "
        "configure an online fallback, or use metadata-only output."
    )


def _prepared(run: Run, payload: TranscriptPayload) -> Prepared:
    raw = parse_transcript_payload(payload, video_id=run.metadata.id)
    run.manifest.add_step("transcript.parse", "ok", raw.provenance.format)
    logger.info("transcript.parse status=ok format=%s", raw.provenance.format)

    clean = normalize_transcript(raw)
    run.manifest.add_step("transcript.normalize", "ok", f"{len(clean.segments)} segments")
    logger.info("transcript.normalize status=ok segments=%s", len(clean.segments))

    chunks = chunk_transcript(
        clean,
        ChunkOptions(
            max_chars=run.resolved.output.chunk_max_chars,
            max_seconds=run.resolved.output.chunk_max_seconds,
        ),
    )
    run.manifest.add_step("chunk", "ok", f"{len(chunks.chunks)} chunks")
    logger.info("chunk status=ok chunks=%s", len(chunks.chunks))
    return Prepared(transcript=clean, chunks=chunks)


def _finish(run: Run, prepared: Prepared, visuals: Visuals, flow: KnowledgeFlow) -> SourcePrepared:
    with phase(logger, "prepare.finish"):
        bundle = render_artifact_bundle(
            metadata=run.metadata,
            transcript=prepared.transcript,
            chunks=prepared.chunks,
            formats=run.resolved.output.formats,
            visual_records=visuals.records,
            visual_scores=visuals.scores,
            knowledge_flow=flow,
            output_language=run.resolved.output.language,
        )
        artifact_refs = write_artifact_bundle(run.request.out_dir, bundle)
        artifact_refs.extend(visuals.frames)
        run.artifacts = artifact_refs
        _retain_or_error(run, artifact_refs)
        final_manifest = run.manifest.finish(
            status="ok", artifacts=artifact_refs, receipts=run.source.receipts
        )
    logger.info("prepare.finish status=ok artifacts=%s", len(artifact_refs))
    return SourcePrepared(
        source=final_manifest,
        artifacts=artifact_refs,
        request=run.request,
        resolved=run.resolved,
    )


def _partial(run: Run) -> SourcePrepared:
    run.request.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_ref = write_artifact(
        run.request.out_dir,
        Artifact(
            name="metadata.json",
            kind="metadata",
            media_type="application/json",
            content=model_to_json(run.metadata),
        ),
    )
    artifact_refs = [artifact_ref]
    run.artifacts = artifact_refs
    _retain_or_error(run, artifact_refs)
    final_manifest = run.manifest.finish(
        status="partial", artifacts=artifact_refs, receipts=run.source.receipts
    )
    logger.info("prepare.finish status=partial artifacts=%s", len(artifact_refs))
    return SourcePrepared(
        source=final_manifest,
        artifacts=artifact_refs,
        request=run.request,
        resolved=run.resolved,
    )


def _retain_assets(run: Run) -> None:
    media = run.media
    if media is None and Path(run.request.inputs[0]).is_file():
        try:
            media = run.source.media(
                request=AsrAudioRequest(
                    temp_dir=run.source_cache.path_for("tmp/retention"),
                ), permit=run.media_permit,
            )
        except NoTranscriptError:
            media = None
    assets = materialize_source_assets(
        media,
        run.subtitle,
        run.request.out_dir,
        retain=run.resolved.output.retain_media,
    )
    for asset in assets:
        run.manifest.add_source_asset(asset)
    if not assets:
        return
    retained = [asset for asset in assets if asset.retained]
    if retained:
        run.manifest.add_step(
            "source.asset_retention",
            "ok",
            ", ".join(asset.path or "" for asset in retained),
        )
    else:
        run.manifest.add_step(
            "source.asset_retention",
            "skipped",
            assets[0].omission_reason,
        )


def _retain_or_error(run: Run, artifact_refs: list[ArtifactRef]) -> None:
    try:
        _retain_assets(run)
    except CacheError:
        run.manifest.add_step("source.asset_retention", "error", "integrity or copy failure")
        raise


def _error_result(run: Run, exc: CacheError | ProviderError) -> SourcePrepared:
    run.request.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = run.artifacts
    if not artifacts:
        artifacts = [
            write_artifact(
                run.request.out_dir,
                Artifact(
                    name="metadata.json",
                    kind="metadata",
                    media_type="application/json",
                    content=model_to_json(run.metadata),
                ),
            )
        ]
    detail = "provider failure" if isinstance(exc, ProviderError) else "storage failure"
    run.manifest.add_step("source.failure", "error", detail)
    failed = run.manifest.finish(
        status="error", artifacts=artifacts, receipts=run.source.receipts
    )
    return SourcePrepared(
        source=failed,
        artifacts=artifacts,
        request=run.request,
        resolved=run.resolved,
        error=exc,
    )


def _asr_environment(resolved: ResolvedConfig) -> TransformEnvironment:
    instance_name = resolved.transforms.asr.instance_name()
    instance = resolved.instances.asr.get(instance_name) if instance_name else None
    if instance is None:
        return TransformEnvironment(offline=resolved.runtime.offline)
    if instance.type == "local-faster-whisper":
        return TransformEnvironment(
            offline=resolved.runtime.offline,
            installed_asr=True,
            configured_asr_model_id=instance.model or instance.model_policy,
            configured_asr_cost_mode="local",
        )
    if instance.type == "openai-compatible-audio":
        return TransformEnvironment(
            offline=resolved.runtime.offline,
            configured_asr=True,
            configured_asr_provider_id=instance_name,
            configured_asr_model_id=instance.model,
            configured_asr_cost_mode="paid",
        )
    return TransformEnvironment(offline=resolved.runtime.offline)


def _transcript_detail(payload: TranscriptPayload) -> str:
    provenance = payload.provenance
    if provenance.method == "local_file":
        return f"local transcript: {payload.format}"
    if provenance.method == "official_subtitles" or provenance.method == "automatic_subtitles":
        language = provenance.language or provenance.language_evidence.kind
        return (
            f"{provenance.provider or 'source'}:{provenance.method}:"
            f"{language}:{payload.format}"
        )
    return payload.provenance_label()


def _media_detail(media: MediaAsset) -> str:
    origin = "local" if media.source.kind == "file" else "source"
    return f"{origin} media: {media.media_type}"


def _capitalize_warning(message: str) -> str:
    if not message:
        return message
    return message[:1].upper() + message[1:]
