from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vctx.ai import AiRoute
from vctx.app.progress import phase
from vctx.artifact.manifest import (
    ArtifactRef,
    FrameCaptureReceipt,
    FrameMissReceipt,
    FrameStepReceipt,
)
from vctx.errors import NoTranscriptError
from vctx.visual.evidence import Evidence, assemble
from vctx.visual.frame import FrameBatch, FrameError, capture
from vctx.visual.ocr import OcrOutcome, RapidOcr
from vctx.visual.plan import EvidencePlan, execution_plan, plan_evidence
from vctx.visual.vlm import VisionProcessor, VlmOutcome

if TYPE_CHECKING:
    from vctx.app.prepare import Prepared
    from vctx.app.run import Run

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualProducts:
    evidence: Evidence | None
    evidence_plan: EvidencePlan | None
    frames: list[ArtifactRef]
    partial: bool = False


def visual_products(run: Run, prepared: Prepared) -> VisualProducts:
    if not run.resolved.evidence.planner.enabled:
        return VisualProducts(None, None, [])
    planner = run.planner_ai_route()
    if planner is None:
        return _partial(run, "evidence planner unavailable; publishing transcript-only lane")
    try:
        with phase(logger, "evidence.plan"):
            plan = plan_evidence(prepared.transcript, run.ai_client(planner))
    except ValueError as exc:
        return _partial(run, str(exc))
    run.manifest.add_transform_evidence(planner.transform_evidence("evidence_plan"))
    failed_plans = [item for item in plan.receipts if item.status in {"failed", "unavailable"}]
    run.manifest.add_step(
        "evidence.plan",
        "warning" if failed_plans else "ok",
        f"{len(plan.claims)} claims, {len(plan.frames)} frame targets",
    )
    if failed_plans:
        run.manifest.warn(f"{len(failed_plans)} evidence planning windows failed")
    if not plan.frames:
        return VisualProducts(None, plan, [], partial=bool(failed_plans))
    if not _video_media(run):
        return VisualProducts(None, plan, [], partial=True)

    vision_routes = run.visual_ai_routes()
    assessment = execution_plan(
        plan,
        ocr_available=isinstance(run.ocr_runtime, RapidOcr),
        vision_route=vision_routes[0] if vision_routes else None,
    )
    if assessment.missing_processors:
        detail = "unavailable visual processors: " + ", ".join(assessment.missing_processors)
        run.manifest.add_step("transform.visual_plan", "warning", detail)
        run.manifest.warn(detail)
    else:
        run.manifest.add_step("transform.visual_plan", "ok", assessment.rationale)
    try:
        assert run.media is not None
        with phase(logger, "visual.capture"):
            batch = capture(run.media, assessment.frames, run.request.out_dir)
    except FrameError as exc:
        run.manifest.add_step("transform.visual_capture", "warning", str(exc))
        run.manifest.warn(str(exc))
        return VisualProducts(None, plan, [], partial=True)

    ocr = _observe_ocr(run, batch)
    vision = _observe_vision(run, batch, vision_routes, ocr)
    evidence = assemble(run.metadata.id, batch, run.request.out_dir, ocr=ocr, vision=vision)
    partial = bool(
        failed_plans or assessment.missing_processors or batch.misses or evidence.partial
    )
    if batch.misses:
        run.manifest.warn(f"{len(batch.misses)} frame targets were unavailable")
    run.manifest.add_step(
        "transform.visual_capture",
        "warning" if partial else "ok",
        f"{len(batch.frames)} captures, {len(batch.misses)} misses",
        _frame_receipt(batch, run.request.out_dir),
    )
    return VisualProducts(
        evidence, plan, _visual_frame_refs(batch, run.request.out_dir), partial=partial
    )


def _observe_ocr(run: Run, batch: FrameBatch) -> dict[str, OcrOutcome]:
    if not isinstance(run.ocr_runtime, RapidOcr):
        return {}
    return {
        frame.id: run.ocr_runtime.observe(frame)
        for frame in batch.frames
        if "ocr" in frame.processors
    }


def _observe_vision(
    run: Run,
    batch: FrameBatch,
    routes: list[AiRoute],
    ocr: dict[str, OcrOutcome],
) -> dict[str, VlmOutcome]:
    if not routes:
        return {}
    processor = VisionProcessor(client=run.ai_client(routes[0]))
    outcomes: dict[str, VlmOutcome] = {}
    for frame in batch.frames:
        if "describe" not in frame.processors:
            continue
        ocr_outcome = ocr.get(frame.id)
        outcomes[frame.id] = processor.observe(
            frame, ocr_text=ocr_outcome.text if ocr_outcome is not None else None
        )
    return outcomes


def _partial(run: Run, detail: str) -> VisualProducts:
    run.manifest.add_step("evidence.plan", "warning", detail)
    run.manifest.warn(detail)
    return VisualProducts(None, None, [], partial=True)


def _video_media(run: Run) -> bool:
    if run.media is None:
        try:
            run.media = run.source.media(
                request=run.visual_media_request(), permit=run.media_permit
            )
        except NoTranscriptError as exc:
            run.manifest.add_step("source.media", "warning", str(exc))
            return False
    return run.media.media_type == "video"


def _visual_frame_refs(batch: FrameBatch, out_dir: Path) -> list[ArtifactRef]:
    return [
        ArtifactRef(
            kind="visual_frame",
            path=frame.path.relative_to(out_dir).as_posix(),
            media_type="image/png",
            bytes=frame.bytes,
            sha256=frame.sha256,
        )
        for frame in batch.frames
    ]


def _frame_receipt(batch: FrameBatch, out_dir: Path) -> FrameStepReceipt:
    return FrameStepReceipt(
        captures=[
            FrameCaptureReceipt(
                id=frame.id,
                path=frame.path.relative_to(out_dir).as_posix(),
                requested_seconds=frame.requested_seconds,
                actual_seconds=frame.actual_seconds,
                original_width=frame.original_size[0],
                original_height=frame.original_size[1],
                orientation=frame.orientation,
                width=frame.size[0],
                height=frame.size[1],
                sha256=frame.sha256,
                request_ids=list(frame.request_ids),
                segment_ids=list(frame.segment_ids),
                claim_ids=list(frame.claim_ids),
                processors=list(frame.processors),
            )
            for frame in batch.frames
        ],
        misses=[
            FrameMissReceipt(
                id=miss.id,
                requested_seconds=miss.requested_seconds,
                reason=miss.reason,
                request_ids=list(miss.request_ids),
                segment_ids=list(miss.segment_ids),
                claim_ids=list(miss.claim_ids),
            )
            for miss in batch.misses
        ],
    )
