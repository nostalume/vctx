from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from vctx.app.evidence import EvidenceOutcome, evidence_products
from vctx.app.progress import phase
from vctx.app.run import (
    PrepareRun,
    RunRuntimes,
    open_prepare_run,
    select_asr_instance,
)
from vctx.artifact.bundle import Artifact, product_bundle, write_artifact, write_bundle
from vctx.artifact.manifest import (
    ArtifactRef,
    ManifestSource,
    ProductOutcome,
)
from vctx.asr import (
    AsrEnvironment,
    AsrOutcome,
    AsrPlan,
    plan_asr,
    run_asr,
)
from vctx.asr_cache import AsrTransformStore, asr_transform_key
from vctx.config import PrepareRequest, PrepareTarget, ResolvedConfig
from vctx.errors import CacheError, NoTranscriptError, ProviderError, SourceConflictError, VctxError
from vctx.source.session import (
    AsrAudioRequest,
    MediaAsset,
    Revision,
)
from vctx.summary import SummaryOutcome, SummaryPacket, SummaryWriter
from vctx.transcript import (
    ChunkOptions,
    ChunkSet,
    Transcript,
    TranscriptPayload,
    chunk_transcript,
    normalize_transcript,
    parse_transcript_payload,
)

logger = logging.getLogger(__name__)
_ASR_MISSING_HINT = "Prepare the small ASR model with: vctx models pull asr"


@dataclass(frozen=True)
class TranscriptProducts:
    transcript: Transcript
    chunks: ChunkSet


@dataclass(frozen=True)
class SourcePrepared:
    source: ManifestSource
    input_value: str
    error: VctxError | None = None


def prepare_source(
    request: PrepareRequest,
    resolved: ResolvedConfig,
    occupied: dict[str, str],
    completed: dict[str, Revision | None],
    runtimes: RunRuntimes,
    previous: dict[str, ManifestSource] | None = None,
    reset_lane: Callable[[str], None] | None = None,
    rollback_lane: Callable[[str], None] | None = None,
) -> SourcePrepared | None:
    run = open_prepare_run(request, resolved, occupied, runtimes)
    source_id = run.source.record.source_id
    revision = run.source.record.revision
    if source_id in completed:
        admitted = completed[source_id]
        if admitted is not None and admitted == revision:
            return None
        completed[source_id] = None
        raise SourceConflictError(source_id, run.manifest.key)
    completed[source_id] = revision
    prior = (previous or {}).get(run.source.record.source_id)
    if (
        prior is not None
        and prior.revision == run.source.record.revision
        and not request.overwrite
        and _satisfies(prior, resolved)
    ):
        return SourcePrepared(prior, run.request.inputs[0])
    if reset_lane is not None:
        reset_lane(run.manifest.key)
    try:
        transcript = _transcript(run)
        if transcript is None:
            return _partial(run)
        prepared = _prepared(run, transcript)
        evidence = (
            evidence_products(run, prepared)
            if run.resolved.target != PrepareTarget.TRANSCRIPT
            else None
        )
        summary = _summary_products(run, prepared, evidence)
        return _finish(run, prepared, evidence, summary)
    except (CacheError, ProviderError) as exc:
        return _error_result(run, exc)
    except VctxError:
        if rollback_lane is not None:
            rollback_lane(run.manifest.key)
        raise


def _satisfies(source: ManifestSource, resolved: ResolvedConfig) -> bool:
    outcomes = {item.product: item.status for item in source.outcomes}
    kinds = {item.kind for item in source.artifacts}
    return (
        outcomes.get(resolved.target.value) == "ready"
        and resolved.output.projections <= kinds
        and (not resolved.output.retain_media or outcomes.get("retained-media") == "ready")
    )


def _transcript(run: PrepareRun) -> TranscriptPayload | Transcript | None:
    with phase(logger, "transcript.extract"):
        try:
            payload = run.source.transcript(permit=run.subtitle_permit)
        except NoTranscriptError as exc:
            return _asr_transcript(run, exc)

    run.manifest.add_step("transcript.extract", "ok", _transcript_detail(payload))
    run.subtitle = payload
    asr_plan = plan_asr(
        run.resolved.asr,
        AsrEnvironment(offline=run.resolved.runtime.offline),
        has_transcript=True,
        has_media=False,
    )
    run.manifest.add_effect(asr_plan.effect_seed)
    run.manifest.add_step(
        "transform.asr",
        "skipped" if asr_plan.selected == "skipped" else "ok",
        asr_plan.reason,
    )
    return payload


def _asr_transcript(
    run: PrepareRun,
    transcript_error: NoTranscriptError,
) -> Transcript | None:
    run.manifest.add_step("transcript.extract", "warning", str(transcript_error))
    source = _asr_source(run, transcript_error)
    if source is None:
        return None

    instance = select_asr_instance(run.resolved)
    if instance is None:
        detail = "ASR instance is not configured"
        run.manifest.add_step("transform.asr", "warning", detail)
        run.manifest.warn(detail)
        return None
    run.manifest.add_effect(source.effect_seed)
    media = run.media.find(AsrAudioRequest())
    assert media is not None and "audio" in media.capabilities
    interval = (
        (run.request.start_seconds or 0, run.request.end_seconds)
        if run.request.start_seconds is not None
        else None
    )

    key = asr_transform_key(
        media,
        instance,
        model_root=run.model_root,
        model_id=source.model_id or instance.model or "small",
        interval=interval,
    )
    transforms = AsrTransformStore(run.source_cache.root / "transforms")
    cached = None if key is None or run.request.overwrite else transforms.get(key)
    if cached is not None:
        return _asr_outcome(run, cached)
    with phase(logger, "asr.execute"):
        outcome = run_asr(
            source,
            media,
            instance=instance,
            cache_root=run.model_root,
            runtimes=run.runtimes.asr,
            progress=logger.isEnabledFor(logging.INFO),
            interval=interval,
        )
    if outcome.kind == "ready" and key is not None:
        try:
            transforms.put(key, outcome)
        except OSError:
            logger.warning("asr.cache status=write-failed")
    return _asr_outcome(run, outcome)


def _asr_outcome(run: PrepareRun, outcome: AsrOutcome) -> Transcript | None:
    if outcome.kind == "ready":
        detail = f"{outcome.receipt.provider}:{outcome.receipt.model}"
        if outcome.receipt.cache_hit:
            detail = f"cache-hit {detail}"
        run.manifest.add_step("transform.asr", "ok", detail)
        return outcome.transcript
    detail = outcome.reason
    if outcome.receipt.failure is not None:
        detail = f"{outcome.receipt.failure}: {detail}"
    run.manifest.add_step("transform.asr", "warning", detail)
    run.manifest.warn(detail)
    return None


def _asr_source(run: PrepareRun, transcript_error: NoTranscriptError) -> AsrPlan | None:
    environment = run.load_asr_environment()
    pre_media_asr_plan = plan_asr(
        run.resolved.asr,
        environment,
        has_transcript=False,
        has_media=True,
    )
    if pre_media_asr_plan.selected != "local":
        run.manifest.add_effect(pre_media_asr_plan.effect_seed)
        run.manifest.add_step("source.media", "skipped", "no executable ASR route selected")
        run.manifest.add_step("transform.asr", "warning", pre_media_asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(_ASR_MISSING_HINT)
        return None

    try:
        request = AsrAudioRequest(
            temp_dir=run.source_cache.root / "tmp" / "yt-dlp",
            refresh=run.request.overwrite,
        )
        media = run.ensure_media(request)
    except NoTranscriptError as media_exc:
        asr_plan = plan_asr(
            run.resolved.asr,
            environment,
            has_transcript=False,
            has_media=False,
        )
        run.manifest.add_step("source.media", "warning", str(media_exc))
        run.manifest.add_effect(asr_plan.effect_seed)
        run.manifest.add_step("transform.asr", "warning", asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(_ASR_MISSING_HINT)
        return None

    run.manifest.add_step("source.media", "ok", _media_detail(media))
    return pre_media_asr_plan


def _prepared(run: PrepareRun, payload: TranscriptPayload | Transcript) -> TranscriptProducts:
    raw = (
        payload
        if isinstance(payload, Transcript)
        else parse_transcript_payload(payload, source_id=run.metadata.id)
    )
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
    return TranscriptProducts(transcript=clean, chunks=chunks)


def _summary_products(
    run: PrepareRun, prepared: TranscriptProducts, evidence: EvidenceOutcome | None
) -> SummaryOutcome | None:
    if run.resolved.target != PrepareTarget.SUMMARY:
        return None
    if evidence is None or evidence.status == "unavailable":
        return SummaryOutcome(status="unavailable", omissions=["evidence stage did not complete"])
    route = run.summary_ai_route()
    if route is None:
        detail = "summary model unavailable; publishing earlier products"
        run.manifest.warn(detail)
        return SummaryOutcome(status="unavailable", omissions=[detail])
    run.manifest.add_effect(route.effect("summary"))
    packet = SummaryPacket.from_products(
        prepared.transcript, evidence.evidence_plan, evidence.evidence
    )
    return SummaryWriter(run.ai_client(route)).write(packet, language=run.resolved.summary.language)


def _finish(
    run: PrepareRun,
    prepared: TranscriptProducts,
    evidence: EvidenceOutcome | None,
    summary: SummaryOutcome | None,
) -> SourcePrepared:
    with phase(logger, "prepare.finish"):
        evidence_value = evidence.evidence if evidence else None
        plan = evidence.evidence_plan if evidence else None
        bundle = product_bundle(
            metadata=run.metadata,
            transcript=prepared.transcript,
            chunks=prepared.chunks,
            projections=run.resolved.output.projections,
            evidence=evidence_value,
            evidence_plan=plan,
            summary=summary.summary if summary else None,
        )
        artifact_refs = write_bundle(run.request.out_dir, bundle)
        if evidence:
            artifact_refs.extend(evidence.frames)
        outcomes = [
            ProductOutcome(
                product="transcript",
                status="ready",
                artifacts=["transcript.json", "chunks.json"],
            )
        ]
        if evidence:
            outcomes.append(_product_outcome("evidence", evidence, artifact_refs))
        if summary:
            outcomes.append(_product_outcome("summary", summary, artifact_refs))
        run.artifacts = artifact_refs
        run.retain(artifact_refs)
        final_manifest = run.finish(artifact_refs, outcomes)
    logger.info("prepare.finish status=ok artifacts=%s", len(artifact_refs))
    return SourcePrepared(final_manifest, run.request.inputs[0])


def _partial(run: PrepareRun) -> SourcePrepared:
    run.request.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_ref = write_artifact(
        run.request.out_dir,
        Artifact.json("metadata.json", "metadata", run.metadata),
    )
    artifact_refs = [artifact_ref]
    run.artifacts = artifact_refs
    run.retain(artifact_refs)
    unavailable = ProductOutcome(
        product=run.resolved.target,
        status="unavailable",
        omissions=["transcript unavailable; later products were not started"],
    )
    run.manifest.add_outcome(unavailable)
    final_manifest = run.manifest.finish(
        status="partial", artifacts=artifact_refs, receipts=run.source.receipts
    )
    logger.info("prepare.finish status=partial artifacts=%s", len(artifact_refs))
    return SourcePrepared(final_manifest, run.request.inputs[0])


def _product_outcome(
    product: str, outcome: EvidenceOutcome | SummaryOutcome, artifacts: list[ArtifactRef]
) -> ProductOutcome:
    kinds = {"evidence", "evidence_plan", "visual_frame"} if product == "evidence" else {"summary"}
    return ProductOutcome(
        product=product,
        status=outcome.status,
        artifacts=[item.path for item in artifacts if item.kind in kinds],
        omissions=list(outcome.omissions),
    )


def _error_result(run: PrepareRun, exc: CacheError | ProviderError) -> SourcePrepared:
    run.request.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = run.artifacts
    if not artifacts:
        artifacts = [
            write_artifact(
                run.request.out_dir,
                Artifact.json("metadata.json", "metadata", run.metadata),
            )
        ]
    detail = "provider failure" if isinstance(exc, ProviderError) else "storage failure"
    run.manifest.add_step("source.failure", "error", detail)
    failed = run.manifest.finish(status="error", artifacts=artifacts, receipts=run.source.receipts)
    return SourcePrepared(failed, run.request.inputs[0], error=exc)


def _transcript_detail(payload: TranscriptPayload) -> str:
    provenance = payload.provenance
    if provenance.method == "local_file":
        return f"local transcript: {payload.format}"
    if provenance.method == "official_subtitles" or provenance.method == "automatic_subtitles":
        language = provenance.language or provenance.language_evidence.kind
        return f"{provenance.provider or 'source'}:{provenance.method}:{language}:{payload.format}"
    return payload.provenance_label()


def _media_detail(media: MediaAsset) -> str:
    origin = "local" if media.source.kind == "file" else "source"
    return f"{origin} media: {media.media_type}"


def _capitalize_warning(message: str) -> str:
    return message[:1].upper() + message[1:]
