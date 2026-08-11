from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vctx.app.credentials import CredentialError
from vctx.app.progress import phase
from vctx.artifact.manifest import ArtifactRef
from vctx.errors import NoTranscriptError
from vctx.models.knowledge_flow import KnowledgeFlow
from vctx.models.visual import (
    EssentialVisualCase,
    SourceAccess,
    VisualRecordSet,
    VisualScoreReport,
)
from vctx.transforms.knowledge_flow import (
    extract_knowledge_flow,
    merge_knowledge_flow_supplement,
)
from vctx.transforms.text_ai import TextAiExecutionError
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

if TYPE_CHECKING:
    from vctx.app.prepare import Prepared, Run

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Visuals:
    records: VisualRecordSet | None
    frames: list[ArtifactRef]
    scores: VisualScoreReport | None = None


def visual_products(run: Run, prepared: Prepared) -> tuple[Visuals, KnowledgeFlow]:
    visuals = _visuals(run, prepared)
    return visuals, _flow(run, prepared, visuals)


def _visuals(run: Run, prepared: Prepared) -> Visuals:
    if not run.resolved.transforms.visual_context.enabled:
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
        run.media = run.source.media(
            request=run.visual_media_request(), permit=run.media_permit
        )
    except NoTranscriptError as exc:
        run.manifest.add_step("source.media", "skipped", str(exc))
        logger.info("source.media status=skipped purpose=visual reason=%s", exc)
        return False
    origin = "local" if run.media.source.kind == "file" else "source"
    run.manifest.add_step("source.media", "ok", f"{origin} media: {run.media.media_type}")
    logger.info("source.media status=ok purpose=visual path=%s", run.media.local_path)
    return run.media.media_type == "video"


def _visual_cases(run: Run, prepared: Prepared) -> list[EssentialVisualCase]:
    cases = deterministic_essential_cases(prepared.transcript)
    logger.info("visual.cases deterministic=%s", len(cases))
    route = run.text_product_ai_route("essential_case_extraction")
    if route is None:
        return cases
    uncertain = uncertain_visual_segments(prepared.transcript, cases)
    if not uncertain.segments:
        run.manifest.add_step(
            "visual_cases.llm_extract", "skipped", "no uncertain visual transcript segments"
        )
        return cases
    try:
        supplement = run.text_ai_adapter(route).essential_case_supplement(uncertain)
    except (CredentialError, TextAiExecutionError) as exc:
        run.manifest.add_step("visual_cases.llm_extract", "warning", str(exc))
        run.manifest.warn(str(exc))
        logger.warning("visual.cases.llm status=warning reason=%s", exc)
        return cases
    run.manifest.add_transform_evidence(route.transform_evidence("essential_cases"))
    run.manifest.add_step("visual_cases.llm_extract", "ok", route.detail())
    merged = merge_essential_case_supplement(cases, supplement, prepared.transcript)
    logger.info("visual.cases.llm status=ok cases=%s route=%s", len(merged), route.provider_id)
    return merged


def _visual_plan(
    run: Run, prepared: Prepared, cases: list[EssentialVisualCase]
) -> VisualPlan:
    plan = plan_visual_motives(
        source=_source_access(run, prepared),
        duration_seconds=run.metadata.duration_seconds,
        motives=visual_motives_from_cases(cases),
        available_actions=discover_visual_actions(
            run.resolved.transforms.visual_context,
            ocr_policy=run.resolved.transforms.ocr,
            vision_instance_configs=run.resolved.instances.vision,
            ai_routes=run.visual_ai_routes(),
            offline=run.resolved.runtime.offline,
        ),
    )
    if plan.kind == "skipped":
        run.manifest.add_step(
            "transform.visual_plan", "skipped", f"{plan.reason}: {plan.rationale}"
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
        transcript=bool(prepared.transcript.segments),
        audio=media_type in {"audio", "video"},
        video=media_type == "video" or run.metadata.source.kind == "url",
    )


def _visual_capture(run: Run, prepared: Prepared, assessment: VisualAssessment) -> Visuals:
    assert run.media is not None
    try:
        with phase(logger, "visual.capture"):
            records = run_visual_context(
                assessment,
                run.media,
                run.request.out_dir,
                cache_root=run.model_root,
                env_files=run.resolved.runtime.env_files,
                runtime_cache=run.runtime_cache,
            )
    except VisualExecutionError as exc:
        run.manifest.add_step("transform.visual_capture", "warning", str(exc))
        logger.warning("visual.capture status=warning reason=%s", exc)
        return Visuals(records=None, frames=[])
    scored = score_visual_records(records.records, prepared.transcript, motives=assessment.motives)
    visual_records = VisualRecordSet(records=scored.records)
    visual_scores = VisualScoreReport(satisfaction=scored.satisfaction)
    _add_visual_satisfaction_step(run, visual_scores)
    run.manifest.add_step(
        "transform.visual_capture", "ok", _visual_capture_detail(visual_records)
    )
    logger.info("visual.capture status=ok records=%s", len(visual_records.records))
    return Visuals(
        records=visual_records,
        frames=_visual_frame_refs(visual_records, run.request.out_dir),
        scores=visual_scores,
    )


def _flow(run: Run, prepared: Prepared, visuals: Visuals) -> KnowledgeFlow:
    with phase(logger, "knowledge_flow.extract"):
        flow = extract_knowledge_flow(prepared.transcript, visuals.records)
    route = run.text_product_ai_route("knowledge_flow_extraction")
    if route is not None:
        try:
            supplement = run.text_ai_adapter(route).knowledge_flow_supplement(prepared.transcript)
        except (CredentialError, TextAiExecutionError) as exc:
            run.manifest.add_step("knowledge_flow.llm_extract", "warning", str(exc))
            run.manifest.warn(str(exc))
            logger.warning("knowledge_flow.llm status=warning reason=%s", exc)
        else:
            flow = merge_knowledge_flow_supplement(flow, supplement, prepared.transcript)
            run.manifest.add_transform_evidence(route.transform_evidence("knowledge_flow"))
            run.manifest.add_step("knowledge_flow.llm_extract", "ok", route.detail())
    if flow.nodes:
        run.manifest.add_step(
            "knowledge_flow.extract", "ok", f"{len(flow.nodes)} nodes, {len(flow.edges)} edges"
        )
    return flow


def _visual_capture_detail(records: VisualRecordSet) -> str:
    kept = sum(1 for item in records.records if item.score is None or item.score.keep)
    dropped = len(records.records) - kept
    if dropped:
        return f"{len(records.records)} records ({kept} kept, {dropped} low-novelty)"
    return f"{len(records.records)} records"


def _add_visual_satisfaction_step(run: Run, scores: VisualScoreReport) -> None:
    if not scores.satisfaction:
        return
    missed = [check for check in scores.satisfaction if check.status == "missed"]
    if missed:
        detail = f"{len(missed)} visual satisfaction missed, {len(scores.satisfaction)} checked"
        run.manifest.add_step("transform.visual_satisfaction", "warning", detail)
        run.manifest.warn(detail)
    else:
        run.manifest.add_step(
            "transform.visual_satisfaction", "ok", f"{len(scores.satisfaction)} checked"
        )


def _visual_frame_refs(records: VisualRecordSet, out_dir: Path) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    seen: set[str] = set()
    for record in records.records:
        if record.kind != "capture" or record.artifact_path is None:
            continue
        if record.artifact_path in seen:
            continue
        seen.add(record.artifact_path)
        body = (out_dir / record.artifact_path).read_bytes()
        refs.append(
            ArtifactRef(
                kind="visual_frame",
                path=record.artifact_path,
                media_type=_visual_frame_media_type(record.artifact_path),
                bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
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
    details = []
    for action in assessment.recipe:
        if action.name == "ocr" and action.provider_id is not None:
            details.append(f"local OCR: {action.provider_id}")
        if action.name == "describe" and action.provider_id is not None:
            label = "free VLM" if action.route == "free-online" else "configured VLM"
            details.append(f"{label}: {action.provider_id}")
    return "; ".join(details) if details else assessment.rationale
