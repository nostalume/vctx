from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

from vctx.app.credentials import (
    CredentialError,
    env_with_credential_presence,
    resolve_env_credential,
)
from vctx.app.result import PrepareResult, PrepareSummary
from vctx.chunking import ChunkOptions, ChunkSet, chunk_transcript
from vctx.config import (
    AsrInstanceConfig,
    CapabilityEnabled,
    PrepareRequest,
    ResolvedConfig,
    WorkflowProfile,
    resolve_config,
)
from vctx.errors import NoTranscriptError
from vctx.io import (
    Cache,
    build_cache,
    model_to_json,
    validate_output_policy,
    write_artifact,
    write_artifact_bundle,
    write_manifest,
)
from vctx.models.acquisition import (
    LocalMedia,
    LocalTranscript,
    SourceMedia,
    SourceSubtitle,
    media_acquisition_detail,
    transcript_acquisition_detail,
)
from vctx.models.artifacts import Artifact
from vctx.models.knowledge_flow import KnowledgeFlow
from vctx.models.manifest import ArtifactRef, ManifestBuilder
from vctx.models.media import AsrAudioFetchRequest, MediaAsset, VisualVideoFetchRequest
from vctx.models.metadata import VideoMetadata
from vctx.models.visual import (
    EssentialVisualCase,
    SourceAccess,
    VisualRecordSet,
    VisualScoreReport,
)
from vctx.render.bundle import render_artifact_bundle
from vctx.sources.detect import SourceAdapter, detect_source_adapter
from vctx.subtitles import parse_transcript_payload
from vctx.transcript import Transcript, TranscriptPayload, normalize_transcript
from vctx.transforms.ai_routes import AiRoute, AiTaskKind, resolve_openrouter_ai_route
from vctx.transforms.asr import AsrExecutionError, run_asr
from vctx.transforms.knowledge_flow import (
    extract_knowledge_flow,
    merge_knowledge_flow_supplement,
)
from vctx.transforms.model_resolution import (
    OPENROUTER_API_KEY_ENV,
    ModelCapability,
)
from vctx.transforms.planning import RoutePlan, SourceState, TransformEnvironment, plan_asr
from vctx.transforms.text_ai import OpenAiCompatibleTextAdapter, TextAiExecutionError
from vctx.transforms.visual_cases import (
    deterministic_essential_cases,
    merge_essential_case_supplement,
    uncertain_visual_segments,
)
from vctx.transforms.visual_evidence import score_visual_records
from vctx.transforms.visual_execute import VisualExecutionError, run_visual_context
from vctx.transforms.visual_planning import (
    VisualAssessment,
    VisualPlan,
    plan_visual_motives,
    visual_motives_from_cases,
)
from vctx.transforms.visual_routes import discover_visual_actions
from vctx.util import vctx_version

logger = logging.getLogger(__name__)


@contextmanager
def _phase(name: str) -> Iterator[None]:
    start = perf_counter()
    logger.debug("%s start", name)
    try:
        yield
    finally:
        logger.info("%s duration_ms=%s", name, int((perf_counter() - start) * 1000))


@dataclass
class Run:
    request: PrepareRequest
    resolved: ResolvedConfig
    manifest: ManifestBuilder
    cache: Cache
    adapter: SourceAdapter
    metadata: VideoMetadata
    media: MediaAsset | None = None

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
            cache_root=self.cache.root,
            env=self.openrouter_env(),
            offline=self.resolved.runtime.offline,
        )
        return [route] if route is not None else []

    def text_product_ai_route(self, task: AiTaskKind) -> AiRoute | None:
        return resolve_openrouter_ai_route(
            self.resolved.transforms.knowledge_flow,
            task=task,
            capability=ModelCapability.ESSENTIAL_CASES,
            cache_root=self.cache.root,
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


@dataclass(frozen=True)
class Prepared:
    raw: Transcript
    clean: Transcript
    chunks: ChunkSet


@dataclass(frozen=True)
class Visuals:
    records: VisualRecordSet | None
    frames: list[ArtifactRef]
    scores: VisualScoreReport | None = None


@dataclass(frozen=True)
class AsrReady:
    plan: RoutePlan
    media: MediaAsset
    instance: AsrInstanceConfig
    api_key: str | None


def prepare_context_pack(request: PrepareRequest) -> PrepareResult:
    with _phase("prepare.total"):
        logger.info("prepare.start input=%s out=%s", request.input, request.out_dir)
        run = _start(request)
        if run.resolved.runtime.workflow == WorkflowProfile.METADATA:
            run.manifest.add_step("transcript.extract", "skipped", "metadata workflow selected")
            run.manifest.warn("metadata workflow selected; transcript pipeline skipped")
            return _partial(run)

        transcript = _transcript(run)
        if isinstance(transcript, PrepareResult):
            return transcript

        prepared = _prepared(run, transcript)
        visuals = _visuals(run, prepared)
        flow = _flow(run, prepared, visuals)
        return _finish(run, prepared, visuals, flow)


def _start(request: PrepareRequest) -> Run:
    resolved = resolve_config(request)
    manifest = ManifestBuilder.start(input=request.input, tool_version=vctx_version())

    validate_output_policy(request.out_dir, overwrite=request.overwrite)
    cache = build_cache(resolved.runtime.cache_dir)
    logger.info(
        "prepare.config workflow=%s cache=%s config=%s offline=%s",
        resolved.runtime.workflow,
        cache.root,
        request.config_path or "built-in defaults + CLI",
        resolved.runtime.offline,
    )
    logger.debug("prepare.output formats=%s", ",".join(resolved.output.formats))

    adapter = detect_source_adapter(request.input)
    manifest.add_step("source.detect", "ok", adapter.name)
    logger.info("source.detect adapter=%s", adapter.name)

    metadata = adapter.extract_metadata(request.input)
    manifest.add_step("metadata.extract", "ok")
    logger.info(
        "metadata.extract status=ok id=%s source_type=%s",
        metadata.id,
        metadata.source_type,
    )

    return Run(
        request=request,
        resolved=resolved,
        manifest=manifest,
        cache=cache,
        adapter=adapter,
        metadata=metadata,
    )


def _transcript(run: Run) -> TranscriptPayload | PrepareResult:
    with _phase("transcript.extract"):
        logger.info("transcript.extract start")
        try:
            payload = run.adapter.extract_transcript(
                run.request.input,
                cache=run.cache,
                source_options=run.resolved.source.yt_dlp,
            )
        except NoTranscriptError as exc:
            logger.info("transcript.extract status=missing reason=%s", exc)
            return _asr_transcript(run, exc)

    run.manifest.add_step("transcript.extract", "ok", _transcript_detail(payload))
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
) -> TranscriptPayload | PrepareResult:
    run.manifest.add_step("transcript.extract", "warning", str(transcript_error))
    source = _asr_source(run, transcript_error)
    if isinstance(source, PrepareResult):
        return source

    ready = _asr_ready(run, source)
    if isinstance(ready, PrepareResult):
        return ready

    try:
        with _phase("asr.execute"):
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
                cache_root=run.cache.root,
                offline=run.resolved.runtime.offline,
                api_key=ready.api_key,
            )
    except AsrExecutionError as asr_exc:
        run.manifest.add_step("transform.asr", "warning", str(asr_exc))
        run.manifest.warn(str(asr_exc))
        logger.warning("asr.execute status=warning reason=%s", asr_exc)
        return _partial(run)

    run.manifest.add_step("transform.asr", "ok", payload.provenance_label())
    logger.info("asr.execute status=ok provenance=%s", payload.provenance_label())
    return payload


def _asr_source(run: Run, transcript_error: NoTranscriptError) -> RoutePlan | PrepareResult:
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
        run.media = run.adapter.extract_media(
            run.request.input,
            request=AsrAudioFetchRequest(
                source_url=run.request.input,
                output_dir=run.request.out_dir / "media",
                temp_dir=run.cache.root / "tmp" / "yt-dlp",
                source_options=run.resolved.source.yt_dlp,
                reuse=not run.request.overwrite,
            ),
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


def _asr_ready(run: Run, asr_plan: RoutePlan) -> AsrReady | PrepareResult:
    instance_name = run.resolved.transforms.asr.instance
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
    return Prepared(raw=raw, clean=clean, chunks=chunks)


def _visuals(run: Run, prepared: Prepared) -> Visuals:
    if not _visual_enabled(run.resolved):
        logger.info("visual.status skipped reason=disabled")
        return Visuals(records=None, frames=[])

    cases = _visual_cases(run, prepared)
    plan = _visual_plan(run, prepared, cases)
    if plan.kind == "skipped":
        return Visuals(records=None, frames=[])

    if not _video_media(run):
        run.manifest.add_step("transform.visual_plan", "skipped", "no video media asset")
        logger.info("visual.status skipped reason=no-video-media")
        return Visuals(records=None, frames=[])

    return _visual_capture(run, prepared, plan.assessment)


def _video_media(run: Run) -> bool:
    if run.media is not None:
        return run.media.media_type == "video"
    try:
        logger.info("source.media start purpose=visual")
        run.media = run.adapter.extract_media(
            run.request.input,
            request=VisualVideoFetchRequest(
                source_url=run.request.input,
                output_dir=run.request.out_dir / "media",
                temp_dir=run.cache.root / "tmp" / "yt-dlp",
                profile=run.resolved.source.yt_dlp.media_profile,
                source_options=run.resolved.source.yt_dlp,
                reuse=not run.request.overwrite,
            ),
        )
    except NoTranscriptError as media_exc:
        run.manifest.add_step("source.media", "skipped", str(media_exc))
        logger.info("source.media status=skipped purpose=visual reason=%s", media_exc)
        return False
    run.manifest.add_step("source.media", "ok", _media_detail(run.media))
    logger.info("source.media status=ok purpose=visual path=%s", run.media.local_path)
    return run.media.media_type == "video"


def _visual_cases(run: Run, prepared: Prepared) -> list[EssentialVisualCase]:
    cases = deterministic_essential_cases(prepared.clean)
    logger.info("visual.cases deterministic=%s", len(cases))
    route = run.text_product_ai_route("essential_case_extraction")
    if route is None:
        return cases
    uncertain = uncertain_visual_segments(prepared.clean, cases)
    if not uncertain.segments:
        run.manifest.add_step(
            "visual_cases.llm_extract",
            "skipped",
            "no uncertain visual transcript segments",
        )
        return cases
    try:
        supplement = run.text_ai_adapter(route).essential_case_supplement(
            uncertain
        )
    except (CredentialError, TextAiExecutionError) as text_ai_exc:
        run.manifest.add_step("visual_cases.llm_extract", "warning", str(text_ai_exc))
        run.manifest.warn(str(text_ai_exc))
        logger.warning("visual.cases.llm status=warning reason=%s", text_ai_exc)
        return cases

    run.manifest.add_transform_evidence(route.transform_evidence("essential_cases"))
    run.manifest.add_step(
        "visual_cases.llm_extract",
        "ok",
        route.detail(),
    )
    merged = merge_essential_case_supplement(cases, supplement, prepared.clean)
    logger.info("visual.cases.llm status=ok cases=%s route=%s", len(merged), route.provider_id)
    return merged


def _visual_plan(
    run: Run,
    prepared: Prepared,
    cases: list[EssentialVisualCase],
) -> VisualPlan:
    plan = plan_visual_motives(
        source=_source_access(run, prepared),
        duration_seconds=run.metadata.duration_seconds,
        motives=visual_motives_from_cases(cases),
        available_actions=discover_visual_actions(
            run.resolved.transforms.visual_context,
            vision_instance_configs=run.resolved.instances.vision,
            ai_routes=run.visual_ai_routes(),
        ),
    )
    if plan.kind == "skipped":
        run.manifest.add_step(
            "transform.visual_plan",
            "skipped",
            f"{plan.reason}: {plan.rationale}",
        )
        logger.info("visual.plan status=skipped reason=%s", plan.reason)
        return plan
    run.manifest.add_step("transform.visual_plan", "ok", _visual_plan_detail(plan.assessment))
    logger.info(
        "visual.plan status=ok actions=%s rationale=%s",
        ",".join(action.name for action in plan.assessment.recipe) or "none",
        plan.assessment.rationale,
    )
    return plan


def _source_access(run: Run, prepared: Prepared) -> SourceAccess:
    media_type = run.media.media_type if run.media is not None else None
    return SourceAccess.from_flags(
        transcript=bool(prepared.clean.segments),
        audio=media_type in {"audio", "video"},
        video=media_type == "video" or run.metadata.source.kind == "url",
    )



def _visual_capture(run: Run, prepared: Prepared, assessment: VisualAssessment) -> Visuals:
    assert run.media is not None
    try:
        with _phase("visual.capture"):
            records = run_visual_context(
                assessment,
                run.media,
                run.request.out_dir,
                env_files=run.resolved.runtime.env_files,
            )
    except VisualExecutionError as visual_exc:
        run.manifest.add_step("transform.visual_capture", "warning", str(visual_exc))
        logger.warning("visual.capture status=warning reason=%s", visual_exc)
        return Visuals(records=None, frames=[])

    scored = score_visual_records(
        records.records,
        prepared.clean,
        motives=assessment.motives,
    )
    visual_records = VisualRecordSet(records=scored.records)
    visual_scores = VisualScoreReport(satisfaction=scored.satisfaction)
    _add_visual_satisfaction_step(run, visual_scores)
    run.manifest.add_step(
        "transform.visual_capture",
        "ok",
        _visual_capture_detail(visual_records),
    )
    logger.info("visual.capture status=ok records=%s", len(visual_records.records))
    return Visuals(
        records=visual_records,
        frames=_visual_frame_refs(visual_records),
        scores=visual_scores,
    )


def _flow(run: Run, prepared: Prepared, visuals: Visuals) -> KnowledgeFlow:
    with _phase("knowledge_flow.extract"):
        knowledge_flow = extract_knowledge_flow(prepared.clean, visuals.records)
    logger.info(
        "knowledge_flow.extract deterministic nodes=%s edges=%s",
        len(knowledge_flow.nodes),
        len(knowledge_flow.edges),
    )
    knowledge_flow_route = run.text_product_ai_route("knowledge_flow_extraction")
    if knowledge_flow_route is not None:
        try:
            supplement = run.text_ai_adapter(
                knowledge_flow_route
            ).knowledge_flow_supplement(prepared.clean)
        except (CredentialError, TextAiExecutionError) as text_ai_exc:
            run.manifest.add_step("knowledge_flow.llm_extract", "warning", str(text_ai_exc))
            run.manifest.warn(str(text_ai_exc))
            logger.warning("knowledge_flow.llm status=warning reason=%s", text_ai_exc)
        else:
            knowledge_flow = merge_knowledge_flow_supplement(
                knowledge_flow,
                supplement,
                prepared.clean,
            )
            run.manifest.add_transform_evidence(
                knowledge_flow_route.transform_evidence("knowledge_flow")
            )
            run.manifest.add_step(
                "knowledge_flow.llm_extract",
                "ok",
                knowledge_flow_route.detail(),
            )
            logger.info(
                "knowledge_flow.llm status=ok nodes=%s edges=%s route=%s",
                len(knowledge_flow.nodes),
                len(knowledge_flow.edges),
                knowledge_flow_route.provider_id,
            )
    if knowledge_flow.nodes:
        run.manifest.add_step(
            "knowledge_flow.extract",
            "ok",
            f"{len(knowledge_flow.nodes)} nodes, {len(knowledge_flow.edges)} edges",
        )
    return knowledge_flow


def _finish(run: Run, prepared: Prepared, visuals: Visuals, flow: KnowledgeFlow) -> PrepareResult:
    with _phase("prepare.finish"):
        bundle = render_artifact_bundle(
            metadata=run.metadata,
            raw_transcript=prepared.raw,
            clean_transcript=prepared.clean,
            chunks=prepared.chunks,
            formats=run.resolved.output.formats,
            visual_records=visuals.records,
            visual_scores=visuals.scores,
            knowledge_flow=flow,
            output_language=run.resolved.output.language,
        )
        artifact_refs = write_artifact_bundle(run.request.out_dir, bundle)
        artifact_refs.extend(visuals.frames)
        final_manifest = run.manifest.finish(status="ok", artifacts=artifact_refs)
        write_manifest(run.request.out_dir, final_manifest)
    logger.info("prepare.finish status=ok artifacts=%s", len(artifact_refs))

    return PrepareResult(
        out_dir=run.request.out_dir,
        manifest=final_manifest,
        artifacts=artifact_refs,
        summary=PrepareSummary.from_run(
            request=run.request,
            resolved=run.resolved,
            manifest=final_manifest,
            artifacts=artifact_refs,
        ),
    )


def _partial(run: Run) -> PrepareResult:
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
    final_manifest = run.manifest.finish(status="partial", artifacts=[artifact_ref])
    write_manifest(run.request.out_dir, final_manifest)
    logger.info("prepare.finish status=partial artifacts=1")
    return PrepareResult(
        out_dir=run.request.out_dir,
        manifest=final_manifest,
        artifacts=[artifact_ref],
        summary=PrepareSummary.from_run(
            request=run.request,
            resolved=run.resolved,
            manifest=final_manifest,
            artifacts=[artifact_ref],
        ),
    )


def _asr_environment(resolved: ResolvedConfig) -> TransformEnvironment:
    instance_name = resolved.transforms.asr.instance
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


def _visual_enabled(resolved: ResolvedConfig) -> bool:
    return resolved.transforms.visual_context.enabled == CapabilityEnabled.TRUE


def _transcript_detail(payload: TranscriptPayload) -> str:
    provenance = payload.provenance
    if provenance.method == "local_file":
        return transcript_acquisition_detail(LocalTranscript(format=payload.format))
    if provenance.method == "official_subtitles" or provenance.method == "automatic_subtitles":
        subtitle_kind = provenance.method
        return transcript_acquisition_detail(
            SourceSubtitle(
                subtitle_kind=subtitle_kind,
                language=provenance.language_evidence,
                format=payload.format,
                provider=provenance.provider or "source",
            )
        )
    return payload.provenance_label()


def _media_detail(media: MediaAsset) -> str:
    if media.source.kind == "file":
        return media_acquisition_detail(
            LocalMedia(path=media.local_path, media_type=media.media_type)
        )
    return media_acquisition_detail(
        SourceMedia(
            cache_path=media.local_path,
            media_type=media.media_type,
            provider=media.provider,
        )
    )


def _visual_capture_detail(visual_records: VisualRecordSet) -> str:
    kept = sum(
        1 for record in visual_records.records if record.score is None or record.score.keep
    )
    dropped = len(visual_records.records) - kept
    if dropped:
        return f"{len(visual_records.records)} records ({kept} kept, {dropped} low-novelty)"
    return f"{len(visual_records.records)} records"


def _add_visual_satisfaction_step(run: Run, scores: VisualScoreReport) -> None:
    if not scores.satisfaction:
        return
    missed = [check for check in scores.satisfaction if check.status == "missed"]
    if missed:
        detail = f"{len(missed)} visual satisfaction missed, {len(scores.satisfaction)} checked"
        run.manifest.add_step("transform.visual_satisfaction", "warning", detail)
        run.manifest.warn(detail)
        return
    run.manifest.add_step(
        "transform.visual_satisfaction",
        "ok",
        f"{len(scores.satisfaction)} checked",
    )


def _visual_frame_refs(visual_records: VisualRecordSet) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    seen: set[str] = set()
    for record in visual_records.records:
        if record.kind != "capture" or record.artifact_path is None:
            continue
        if record.artifact_path in seen:
            continue
        seen.add(record.artifact_path)
        refs.append(
            ArtifactRef(
                kind="visual_frame",
                path=record.artifact_path,
                media_type=_visual_frame_media_type(record.artifact_path),
            )
        )
    return refs


def _visual_frame_media_type(path: str) -> str:
    if path.lower().endswith(".png"):
        return "image/png"
    if path.lower().endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return "application/octet-stream"


def _visual_plan_detail(assessment: VisualAssessment) -> str:
    route_details = []
    for action in assessment.recipe:
        provider_id = action.provider_id
        if action.name == "ocr" and provider_id is not None:
            route_details.append(f"local OCR: {provider_id}")
        if action.name == "describe" and provider_id is not None:
            label = "free VLM" if action.route == "free-online" else "configured VLM"
            route_details.append(f"{label}: {provider_id}")
    if route_details:
        return "; ".join(route_details)
    return assessment.rationale


def _capitalize_warning(message: str) -> str:
    if not message:
        return message
    return message[:1].upper() + message[1:]
