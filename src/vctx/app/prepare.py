from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from vctx.app.evidence import EvidenceOutcome, EvidencePipeline
from vctx.app.progress import phase
from vctx.app.run import (
    PrepareRun,
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
)
from vctx.asr.cache import AsrTransformStore
from vctx.config import PrepareTarget, ResolvedConfig
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


def _satisfies(source: ManifestSource, resolved: ResolvedConfig) -> bool:
    outcomes = {item.product: item.status for item in source.outcomes}
    kinds = {item.kind for item in source.artifacts}
    return (
        outcomes.get(resolved.target.value) == "ready"
        and resolved.output.projections <= kinds
        and (not resolved.output.retain_media or outcomes.get("source-assets") == "ready")
    )


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


class PreparePipeline:
    def __init__(self, run: PrepareRun) -> None:
        self.run = run

    def prepare(
        self,
        completed: dict[str, Revision | None],
        previous: dict[str, ManifestSource] | None = None,
        reset_lane: Callable[[str], None] | None = None,
        rollback_lane: Callable[[str], None] | None = None,
    ) -> SourcePrepared | None:
        source_id = self.run.source.record.source_id
        revision = self.run.source.record.revision
        if source_id in completed:
            admitted = completed[source_id]
            if admitted is not None and admitted == revision:
                return None
            completed[source_id] = None
            raise SourceConflictError(source_id, self.run.manifest.key)
        completed[source_id] = revision
        prior = (previous or {}).get(self.run.source.record.source_id)
        if (
            prior is not None
            and prior.revision == self.run.source.record.revision
            and not self.run.request.overwrite
            and _satisfies(prior, self.run.resolved)
        ):
            return SourcePrepared(prior, self.run.request.inputs[0])
        if reset_lane is not None:
            reset_lane(self.run.manifest.key)
        try:
            transcript = self._transcript()
            if transcript is None:
                return self._partial()
            prepared = self._prepared(transcript)
            evidence = (
                EvidencePipeline(self.run).produce(prepared)
                if self.run.resolved.target != PrepareTarget.TRANSCRIPT
                else None
            )
            summary = self._summary_products(prepared, evidence)
            return self._finish(prepared, evidence, summary)
        except (CacheError, ProviderError) as exc:
            return self._error_result(exc)
        except VctxError:
            if rollback_lane is not None:
                rollback_lane(self.run.manifest.key)
            raise

    def _transcript(self) -> TranscriptPayload | Transcript | None:
        with phase(logger, "transcript.extract"):
            try:
                payload = self.run.source.transcript(permit=self.run.subtitle_permit)
            except NoTranscriptError as exc:
                return self._asr_transcript(exc)

        self.run.manifest.add_step("transcript.extract", "ok", _transcript_detail(payload))
        self.run.subtitle = payload
        asr_plan = plan_asr(
            self.run.resolved.asr,
            AsrEnvironment(offline=self.run.resolved.runtime.offline),
            has_transcript=True,
            has_media=False,
        )
        self.run.manifest.add_effect(asr_plan.effect_seed)
        self.run.manifest.add_step(
            "transform.asr",
            "skipped" if asr_plan.selected == "skipped" else "ok",
            asr_plan.reason,
        )
        return payload

    def _asr_transcript(
        self,
        transcript_error: NoTranscriptError,
    ) -> Transcript | None:
        self.run.manifest.add_step("transcript.extract", "warning", str(transcript_error))
        source = self._asr_source(transcript_error)
        if source is None:
            return None

        instance = select_asr_instance(self.run.resolved)
        if instance is None:
            detail = "ASR instance is not configured"
            self.run.manifest.add_step("transform.asr", "warning", detail)
            self.run.manifest.warn(detail)
            return None
        self.run.manifest.add_effect(source.effect_seed)
        media = self.run.media.find(AsrAudioRequest())
        assert media is not None and "audio" in media.capabilities
        interval = (
            (self.run.request.start_seconds or 0, self.run.request.end_seconds)
            if self.run.request.start_seconds is not None
            else None
        )

        transforms = AsrTransformStore(self.run.source_cache.root / "transforms")
        key = transforms.key(
            media,
            instance,
            model_root=self.run.model_root,
            model_id=source.model_id or instance.model or "small",
            interval=interval,
        )
        cached = None if key is None or self.run.request.overwrite else transforms.get(key)
        if cached is not None:
            return self._asr_outcome(cached)
        with phase(logger, "asr.execute"):
            outcome = self.run.runtimes.asr.run(
                source,
                media,
                instance=instance,
                cache_root=self.run.model_root,
                progress=logger.isEnabledFor(logging.INFO),
                interval=interval,
                temp_root=self.run.source_cache.root / "tmp" / "asr",
            )
        if outcome.kind == "ready" and key is not None:
            try:
                transforms.put(key, outcome)
            except OSError:
                logger.warning("asr.cache status=write-failed")
        return self._asr_outcome(outcome)

    def _asr_outcome(self, outcome: AsrOutcome) -> Transcript | None:
        if outcome.kind == "ready":
            detail = f"{outcome.receipt.provider}:{outcome.receipt.model}"
            if outcome.receipt.cache_hit:
                detail = f"cache-hit {detail}"
            self.run.manifest.add_step("transform.asr", "ok", detail)
            return outcome.transcript
        detail = outcome.reason
        if outcome.receipt.failure is not None:
            detail = f"{outcome.receipt.failure}: {detail}"
        self.run.manifest.add_step("transform.asr", "warning", detail)
        self.run.manifest.warn(detail)
        return None

    def _asr_source(self, transcript_error: NoTranscriptError) -> AsrPlan | None:
        environment = self.run.load_asr_environment()
        pre_media_asr_plan = plan_asr(
            self.run.resolved.asr,
            environment,
            has_transcript=False,
            has_media=True,
        )
        if pre_media_asr_plan.selected != "local":
            self.run.manifest.add_effect(pre_media_asr_plan.effect_seed)
            self.run.manifest.add_step(
                "source.media", "skipped", "no executable ASR route selected"
            )
            self.run.manifest.add_step("transform.asr", "warning", pre_media_asr_plan.reason)
            self.run.manifest.warn(_capitalize_warning(str(transcript_error)))
            self.run.manifest.warn(_ASR_MISSING_HINT)
            return None

        try:
            request = AsrAudioRequest(
                temp_dir=self.run.source_cache.root / "tmp" / "yt-dlp",
                refresh=self.run.request.overwrite,
            )
            media = self.run.ensure_media(request)
        except NoTranscriptError as media_exc:
            asr_plan = plan_asr(
                self.run.resolved.asr,
                environment,
                has_transcript=False,
                has_media=False,
            )
            self.run.manifest.add_step("source.media", "warning", str(media_exc))
            self.run.manifest.add_effect(asr_plan.effect_seed)
            self.run.manifest.add_step("transform.asr", "warning", asr_plan.reason)
            self.run.manifest.warn(_capitalize_warning(str(transcript_error)))
            self.run.manifest.warn(_ASR_MISSING_HINT)
            return None

        self.run.manifest.add_step("source.media", "ok", _media_detail(media))
        return pre_media_asr_plan

    def _prepared(self, payload: TranscriptPayload | Transcript) -> TranscriptProducts:
        raw = (
            payload
            if isinstance(payload, Transcript)
            else parse_transcript_payload(payload, source_id=self.run.metadata.id)
        )
        self.run.manifest.add_step("transcript.parse", "ok", raw.provenance.format)
        logger.info("transcript.parse status=ok format=%s", raw.provenance.format)

        clean = normalize_transcript(raw)
        self.run.manifest.add_step("transcript.normalize", "ok", f"{len(clean.segments)} segments")
        logger.info("transcript.normalize status=ok segments=%s", len(clean.segments))

        chunks = chunk_transcript(
            clean,
            ChunkOptions(
                max_chars=self.run.resolved.output.chunk_max_chars,
                max_seconds=self.run.resolved.output.chunk_max_seconds,
            ),
        )
        self.run.manifest.add_step("chunk", "ok", f"{len(chunks.chunks)} chunks")
        logger.info("chunk status=ok chunks=%s", len(chunks.chunks))
        return TranscriptProducts(transcript=clean, chunks=chunks)

    def _summary_products(
        self, prepared: TranscriptProducts, evidence: EvidenceOutcome | None
    ) -> SummaryOutcome | None:
        if self.run.resolved.target != PrepareTarget.SUMMARY:
            return None
        if evidence is None or evidence.status == "unavailable":
            return SummaryOutcome(
                status="unavailable", omissions=["evidence stage did not complete"]
            )
        route = self.run.summary_ai_route()
        if route is None:
            detail = "summary model unavailable; publishing earlier products"
            self.run.manifest.warn(detail)
            return SummaryOutcome(status="unavailable", omissions=[detail])
        self.run.manifest.add_effect(route.effect("summary"))
        packet = SummaryPacket.from_products(
            prepared.transcript, evidence.evidence_plan, evidence.evidence
        )
        return SummaryWriter(self.run.ai_client(route)).write(
            packet, language=self.run.resolved.summary.language
        )

    def _finish(
        self,
        prepared: TranscriptProducts,
        evidence: EvidenceOutcome | None,
        summary: SummaryOutcome | None,
    ) -> SourcePrepared:
        with phase(logger, "prepare.finish"):
            evidence_value = evidence.evidence if evidence else None
            plan = evidence.evidence_plan if evidence else None
            bundle = product_bundle(
                metadata=self.run.metadata,
                transcript=prepared.transcript,
                chunks=prepared.chunks,
                projections=self.run.resolved.output.projections,
                evidence=evidence_value,
                evidence_plan=plan,
                summary=summary.summary if summary else None,
            )
            artifact_refs = write_bundle(self.run.request.out_dir, bundle)
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
            self.run.artifacts = artifact_refs
            self.run.retain(artifact_refs)
            final_manifest = self.run.finish(artifact_refs, outcomes)
        logger.info("prepare.finish status=ok artifacts=%s", len(artifact_refs))
        return SourcePrepared(final_manifest, self.run.request.inputs[0])

    def _partial(self) -> SourcePrepared:
        self.run.request.out_dir.mkdir(parents=True, exist_ok=True)
        artifact_ref = write_artifact(
            self.run.request.out_dir,
            Artifact.json("metadata.json", "metadata", self.run.metadata),
        )
        artifact_refs = [artifact_ref]
        self.run.artifacts = artifact_refs
        self.run.retain(artifact_refs)
        unavailable = ProductOutcome(
            product=self.run.resolved.target,
            status="unavailable",
            omissions=["transcript unavailable; later products were not started"],
        )
        self.run.manifest.add_outcome(unavailable)
        final_manifest = self.run.manifest.finish(
            status="partial", artifacts=artifact_refs, receipts=self.run.source.receipts
        )
        logger.info("prepare.finish status=partial artifacts=%s", len(artifact_refs))
        return SourcePrepared(final_manifest, self.run.request.inputs[0])

    def _error_result(self, exc: CacheError | ProviderError) -> SourcePrepared:
        self.run.request.out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = self.run.artifacts
        if not artifacts:
            artifacts = [
                write_artifact(
                    self.run.request.out_dir,
                    Artifact.json("metadata.json", "metadata", self.run.metadata),
                )
            ]
        detail = "provider failure" if isinstance(exc, ProviderError) else "storage failure"
        self.run.manifest.add_step("source.failure", "error", detail)
        failed = self.run.manifest.finish(
            status="error", artifacts=artifacts, receipts=self.run.source.receipts
        )
        return SourcePrepared(failed, self.run.request.inputs[0], error=exc)


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
