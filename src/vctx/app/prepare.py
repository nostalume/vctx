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
from vctx.config import AsrInstanceConfig, PrepareRequest, PrepareTarget, ResolvedConfig
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
class AsrReady:
    plan: AsrPlan
    media: MediaAsset
    instance: AsrInstanceConfig


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
    logger.info("asr.route selected=%s reason=%s", asr_plan.selected, asr_plan.reason)
    return payload


def _asr_transcript(
    run: PrepareRun,
    transcript_error: NoTranscriptError,
) -> Transcript | None:
    run.manifest.add_step("transcript.extract", "warning", str(transcript_error))
    source = _asr_source(run, transcript_error)
    if source is None:
        return None

    ready = _asr_ready(run, source)
    if ready is None:
        return None

    with phase(logger, "asr.execute"):
        logger.info(
            "asr.execute start route=%s provider=%s model=%s",
            ready.plan.selected,
            ready.plan.provider_id,
            ready.plan.model_id,
        )
        outcome = run_asr(
            ready.plan,
            ready.media,
            instance=ready.instance,
            cache_root=run.model_root,
            runtimes=run.runtimes.asr,
        )
    return _asr_outcome(run, outcome)


def _asr_outcome(run: PrepareRun, outcome: AsrOutcome) -> Transcript | None:
    if outcome.kind == "ready":
        detail = f"{outcome.receipt.provider}:{outcome.receipt.model}"
        run.manifest.add_step("transform.asr", "ok", detail)
        logger.info("asr.execute status=ok provenance=%s", detail)
        return outcome.transcript
    detail = outcome.reason
    if outcome.receipt.failure is not None:
        detail = f"{outcome.receipt.failure}: {detail}"
    run.manifest.add_step("transform.asr", "warning", detail)
    run.manifest.warn(detail)
    logger.warning("asr.execute status=%s reason=%s", outcome.kind, detail)
    return None


def _asr_source(run: PrepareRun, transcript_error: NoTranscriptError) -> AsrPlan | None:
    pre_media_asr_plan = plan_asr(
        run.resolved.asr,
        run.load_asr_environment(),
        has_transcript=False,
        has_media=True,
    )
    if pre_media_asr_plan.selected != "local":
        run.manifest.add_effect(pre_media_asr_plan.effect_seed)
        run.manifest.add_step("source.media", "skipped", "no executable ASR route selected")
        run.manifest.add_step("transform.asr", "warning", pre_media_asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(_ASR_MISSING_HINT)
        logger.warning("asr.route status=unavailable reason=%s", pre_media_asr_plan.reason)
        return None

    try:
        logger.info("source.media start purpose=asr")
        request = (
            run.visual_media_request()
            if run.resolved.target != PrepareTarget.TRANSCRIPT
            else AsrAudioRequest(
                temp_dir=run.source_cache.root / "tmp" / "yt-dlp",
                refresh=run.request.overwrite,
            )
        )
        run.media = run.source.media(
            request=request,
            permit=run.media_permit,
        )
    except NoTranscriptError as media_exc:
        asr_plan = plan_asr(
            run.resolved.asr,
            run.load_asr_environment(),
            has_transcript=False,
            has_media=False,
        )
        run.manifest.add_step("source.media", "warning", str(media_exc))
        run.manifest.add_effect(asr_plan.effect_seed)
        run.manifest.add_step("transform.asr", "warning", asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(_ASR_MISSING_HINT)
        logger.warning("source.media status=warning purpose=asr reason=%s", media_exc)
        return None

    run.manifest.add_step("source.media", "ok", _media_detail(run.media))
    logger.info("source.media status=ok purpose=asr path=%s", run.media.local_path)
    asr_plan = plan_asr(
        run.resolved.asr,
        run.load_asr_environment(),
        has_transcript=False,
        has_media=True,
    )
    if asr_plan.selected != "local":
        run.manifest.add_effect(asr_plan.effect_seed)
        run.manifest.add_step("transform.asr", "warning", asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(asr_plan.reason)
        logger.warning("asr.route status=unavailable reason=%s", asr_plan.reason)
        return None
    logger.info("asr.route selected=%s reason=%s", asr_plan.selected, asr_plan.reason)
    return asr_plan


def _asr_ready(run: PrepareRun, asr_plan: AsrPlan) -> AsrReady | None:
    instance = select_asr_instance(run.resolved)
    if instance is None:
        run.manifest.add_step("transform.asr", "warning", "ASR instance is not configured")
        run.manifest.warn("ASR instance is not configured")
        logger.warning("asr.ready status=warning reason=missing-instance")
        return None

    run.manifest.add_effect(asr_plan.effect_seed)
    assert run.media is not None
    logger.info(
        "asr.ready status=ok provider=%s model=%s credential=%s",
        asr_plan.provider_id,
        asr_plan.model_id,
        "not-required",
    )
    return AsrReady(plan=asr_plan, media=run.media, instance=instance)


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
        return SummaryOutcome(
            status="unavailable", omissions=["evidence stage did not complete"]
        )
    route = run.summary_ai_route()
    if route is None:
        detail = "summary model unavailable; publishing earlier products"
        run.manifest.warn(detail)
        return SummaryOutcome(status="unavailable", omissions=[detail])
    run.manifest.add_effect(route.effect("summary"))
    packet = SummaryPacket.from_products(
        prepared.transcript, evidence.evidence_plan, evidence.evidence
    )
    return SummaryWriter(run.ai_client(route)).write(
        packet, language=run.resolved.summary.language
    )


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
    kinds = (
        {"evidence", "evidence_plan", "visual_frame"}
        if product == "evidence"
        else {"summary"}
    )
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
