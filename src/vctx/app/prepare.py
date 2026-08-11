from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vctx.app.media_retention import materialize_source_assets
from vctx.app.progress import phase
from vctx.app.run import (
    Run,
    RunRuntimes,
    open_run,
    select_asr_instance,
)
from vctx.app.visual import VisualProducts, visual_products
from vctx.artifact.content import Artifact
from vctx.artifact.manifest import (
    ArtifactRef,
    AsrStepReceipt,
    ManifestSource,
)
from vctx.asr import (
    AsrEnvironment,
    AsrOutcome,
    AsrPlan,
    AsrReceipt,
    plan_asr,
    run_asr,
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
from vctx.render.bundle import render_artifact_bundle
from vctx.source.session import (
    AsrAudioRequest,
    MediaAsset,
)
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
class Prepared:
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
    artifacts: list[ArtifactRef]
    request: PrepareRequest
    resolved: ResolvedConfig
    error: VctxError | None = None


def prepare_source(
    request: PrepareRequest,
    resolved: ResolvedConfig,
    occupied: dict[str, str],
    completed: set[str],
    runtimes: RunRuntimes,
    previous: dict[str, ManifestSource] | None = None,
    reset_lane: Callable[[str], None] | None = None,
    rollback_lane: Callable[[str], None] | None = None,
) -> SourcePrepared | None:
    logger.info("prepare.start input=%s out=%s", request.inputs[0], request.out_dir)
    run = open_run(request, resolved, occupied, runtimes)
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
        products = visual_products(run, prepared)
        return _finish(run, prepared, products)
    except (CacheError, ProviderError) as exc:
        return _error_result(run, exc)
    except VctxError:
        if rollback_lane is not None:
            rollback_lane(run.manifest.key)
        raise


def _transcript(run: Run) -> TranscriptPayload | Transcript | SourcePrepared:
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
        AsrEnvironment(offline=run.resolved.runtime.offline),
        has_transcript=True,
        has_media=False,
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
) -> Transcript | SourcePrepared:
    run.manifest.add_step("transcript.extract", "warning", str(transcript_error))
    source = _asr_source(run, transcript_error)
    if isinstance(source, SourcePrepared):
        return source

    ready = _asr_ready(run, source)
    if isinstance(ready, SourcePrepared):
        return ready

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


def _asr_outcome(run: Run, outcome: AsrOutcome) -> Transcript | SourcePrepared:
    if outcome.kind == "ready":
        detail = f"{outcome.receipt.provider}:{outcome.receipt.model}"
        run.manifest.add_step("transform.asr", "ok", detail, _asr_step_receipt(outcome.receipt))
        logger.info("asr.execute status=ok provenance=%s", detail)
        return outcome.transcript
    detail = outcome.reason
    if outcome.receipt.failure is not None:
        detail = f"{outcome.receipt.failure}: {detail}"
    run.manifest.add_step("transform.asr", "warning", detail, _asr_step_receipt(outcome.receipt))
    run.manifest.warn(detail)
    logger.warning("asr.execute status=%s reason=%s", outcome.kind, detail)
    return _partial(run)


def _asr_step_receipt(receipt: AsrReceipt) -> AsrStepReceipt:
    return AsrStepReceipt.model_validate(receipt, from_attributes=True)


def _asr_source(run: Run, transcript_error: NoTranscriptError) -> AsrPlan | SourcePrepared:
    pre_media_asr_plan = plan_asr(
        run.resolved.transforms.asr,
        run.load_asr_environment(),
        has_transcript=False,
        has_media=True,
    )
    if pre_media_asr_plan.selected != "local":
        run.manifest.add_transform_evidence(pre_media_asr_plan.evidence_seed)
        run.manifest.add_step("source.media", "skipped", "no executable ASR route selected")
        run.manifest.add_step("transform.asr", "warning", pre_media_asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(_ASR_MISSING_HINT)
        logger.warning("asr.route status=unavailable reason=%s", pre_media_asr_plan.reason)
        return _partial(run)

    try:
        logger.info("source.media start purpose=asr")
        run.media = run.source.media(
            request=AsrAudioRequest(
                temp_dir=run.source_cache.root / "tmp" / "yt-dlp",
                refresh=run.request.overwrite,
            ),
            permit=run.media_permit,
        )
    except NoTranscriptError as media_exc:
        asr_plan = plan_asr(
            run.resolved.transforms.asr,
            run.load_asr_environment(),
            has_transcript=False,
            has_media=False,
        )
        run.manifest.add_step("source.media", "warning", str(media_exc))
        run.manifest.add_transform_evidence(asr_plan.evidence_seed)
        run.manifest.add_step("transform.asr", "warning", asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(_ASR_MISSING_HINT)
        logger.warning("source.media status=warning purpose=asr reason=%s", media_exc)
        return _partial(run)

    run.manifest.add_step("source.media", "ok", _media_detail(run.media))
    logger.info("source.media status=ok purpose=asr path=%s", run.media.local_path)
    asr_plan = plan_asr(
        run.resolved.transforms.asr,
        run.load_asr_environment(),
        has_transcript=False,
        has_media=True,
    )
    if asr_plan.selected != "local":
        run.manifest.add_transform_evidence(asr_plan.evidence_seed)
        run.manifest.add_step("transform.asr", "warning", asr_plan.reason)
        run.manifest.warn(_capitalize_warning(str(transcript_error)))
        run.manifest.warn(asr_plan.reason)
        logger.warning("asr.route status=unavailable reason=%s", asr_plan.reason)
        return _partial(run)
    logger.info("asr.route selected=%s reason=%s", asr_plan.selected, asr_plan.reason)
    return asr_plan


def _asr_ready(run: Run, asr_plan: AsrPlan) -> AsrReady | SourcePrepared:
    instance = select_asr_instance(run.resolved)
    if instance is None:
        run.manifest.add_step("transform.asr", "warning", "ASR instance is not configured")
        run.manifest.warn("ASR instance is not configured")
        logger.warning("asr.ready status=warning reason=missing-instance")
        return _partial(run)

    run.manifest.add_transform_evidence(asr_plan.evidence_seed)
    assert run.media is not None
    logger.info(
        "asr.ready status=ok provider=%s model=%s credential=%s",
        asr_plan.provider_id,
        asr_plan.model_id,
        "not-required",
    )
    return AsrReady(plan=asr_plan, media=run.media, instance=instance)


def _prepared(run: Run, payload: TranscriptPayload | Transcript) -> Prepared:
    raw = (
        payload
        if isinstance(payload, Transcript)
        else parse_transcript_payload(payload, video_id=run.metadata.id)
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
    return Prepared(transcript=clean, chunks=chunks)


def _finish(run: Run, prepared: Prepared, products: VisualProducts) -> SourcePrepared:
    with phase(logger, "prepare.finish"):
        bundle = render_artifact_bundle(
            metadata=run.metadata,
            transcript=prepared.transcript,
            chunks=prepared.chunks,
            formats=run.resolved.output.formats,
            evidence=products.evidence,
            evidence_plan=products.evidence_plan,
            output_language=run.resolved.output.language,
        )
        artifact_refs = write_artifact_bundle(run.request.out_dir, bundle)
        artifact_refs.extend(products.frames)
        run.artifacts = artifact_refs
        _retain_or_error(run, artifact_refs)
        final_manifest = run.manifest.finish(
            status="partial" if products.partial else "ok",
            artifacts=artifact_refs,
            receipts=run.source.receipts,
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
                ),
                permit=run.media_permit,
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
    failed = run.manifest.finish(status="error", artifacts=artifacts, receipts=run.source.receipts)
    return SourcePrepared(
        source=failed,
        artifacts=artifacts,
        request=run.request,
        resolved=run.resolved,
        error=exc,
    )


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
    if not message:
        return message
    return message[:1].upper() + message[1:]
