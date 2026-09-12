from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from vctx.ai import AiRoute
from vctx.app.progress import phase
from vctx.artifact.manifest import ArtifactRef
from vctx.errors import NoTranscriptError
from vctx.visual.evidence import Evidence, Observation
from vctx.visual.frame import FrameBatch, FrameError, capture
from vctx.visual.plan import EvidencePlan, EvidencePlanner, FrameProcessor, PlanReceipt
from vctx.visual.processors import OcrUnavailable, RapidOcr, VisionProcessor

if TYPE_CHECKING:
    from vctx.app.prepare import TranscriptProducts
    from vctx.app.run import PrepareRun

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceOutcome:
    status: Literal["ready", "partial", "unavailable"]
    evidence: Evidence | None
    evidence_plan: EvidencePlan | None
    frames: tuple[ArtifactRef, ...] = ()
    omissions: tuple[str, ...] = ()
    receipts: tuple[PlanReceipt, ...] = ()


class EvidencePipeline:
    def __init__(self, run: PrepareRun) -> None:
        self.run = run

    def produce(self, prepared: TranscriptProducts) -> EvidenceOutcome:
        if not self.run.resolved.evidence.planner.enabled:
            return EvidenceOutcome("ready", None, None)
        planner = self.run.planner_ai_route()
        if planner is None:
            return self._unavailable(
                "evidence planner unavailable; publishing transcript-only lane"
            )
        processors = self._enabled_processors()
        try:
            with phase(logger, "evidence.plan"):
                plan = EvidencePlanner(self.run.ai_client(planner), processors=processors).plan(
                    prepared.transcript
                )
        except ValueError as exc:
            return self._unavailable(str(exc))
        self.run.manifest.add_effect(planner.effect("evidence_plan"))
        failed_plans = [item for item in plan.receipts if item.status in {"failed", "unavailable"}]
        self.run.manifest.add_step(
            "evidence.plan",
            "warning" if failed_plans else "ok",
            f"{len(plan.claims)} claims, {len(plan.frames)} frame targets",
        )
        omissions = [
            f"planning window {receipt.window_id}: {receipt.detail}" for receipt in failed_plans
        ]
        if failed_plans:
            self.run.manifest.warn(f"{len(failed_plans)} evidence planning windows failed")
        if not plan.frames:
            return EvidenceOutcome(
                "partial" if failed_plans else "ready",
                None,
                plan,
                omissions=tuple(omissions),
                receipts=tuple(plan.receipts),
            )
        if not self._video_media():
            detail = "visual media unavailable"
            return EvidenceOutcome(
                "partial",
                None,
                plan,
                omissions=(*omissions, detail),
                receipts=tuple(plan.receipts),
            )

        vision_routes = self.run.visual_ai_routes()
        assessment = plan.recipe(
            ocr_available=isinstance(self.run.ocr_runtime, RapidOcr),
            vision_route=vision_routes[0] if vision_routes else None,
        )
        if assessment.missing_processors:
            detail = "unavailable visual processors: " + ", ".join(assessment.missing_processors)
            omissions.append(detail)
            self.run.manifest.add_step("transform.visual_plan", "warning", detail)
            self.run.manifest.warn(detail)
        else:
            self.run.manifest.add_step("transform.visual_plan", "ok", assessment.rationale)
        try:
            media = self.run.media.find(self.run.visual_media_request())
            assert media is not None
            with phase(logger, "visual.capture"):
                batch = capture(media, assessment.frames, self.run.request.out_dir)
        except FrameError as exc:
            self.run.manifest.add_step("transform.visual_capture", "warning", str(exc))
            self.run.manifest.warn(str(exc))
            return EvidenceOutcome(
                "partial",
                None,
                plan,
                omissions=(*omissions, str(exc)),
                receipts=tuple(plan.receipts),
            )

        ocr = self._observe_ocr(batch)
        vision = self._observe_vision(batch, vision_routes, ocr)
        evidence = Evidence.from_observations(
            plan,
            batch,
            self.run.request.out_dir,
            ocr=ocr,
            vision=vision,
        )
        if batch.misses:
            detail = f"{len(batch.misses)} frame targets were unavailable"
            omissions.append(detail)
            self.run.manifest.warn(detail)
        if evidence.status == "partial" and not omissions:
            omissions.append("one or more visual observations were incomplete")
        partial = bool(
            failed_plans or assessment.missing_processors or evidence.status == "partial"
        )
        self.run.manifest.add_step(
            "transform.visual_capture",
            "warning" if partial else "ok",
            f"{len(batch.frames)} captures, {len(batch.misses)} misses",
        )
        return EvidenceOutcome(
            "partial" if partial else "ready",
            evidence,
            plan,
            tuple(_visual_frame_refs(batch, self.run.request.out_dir)),
            tuple(omissions),
            tuple(plan.receipts),
        )

    def _enabled_processors(self) -> tuple[FrameProcessor, ...]:
        processors: list[FrameProcessor] = []
        if self.run.resolved.evidence.ocr.enabled:
            processors.append("ocr")
        if self.run.resolved.evidence.vision.enabled:
            processors.append("describe")
        return tuple(processors)

    def _observe_ocr(self, batch: FrameBatch) -> dict[str, Observation]:
        requested = [frame for frame in batch.frames if "ocr" in frame.processors]
        if isinstance(self.run.ocr_runtime, RapidOcr):
            return {frame.id: self.run.ocr_runtime.observe(frame) for frame in requested}
        detail = (
            self.run.ocr_runtime.detail
            if isinstance(self.run.ocr_runtime, OcrUnavailable)
            else "OCR processor unavailable"
        )
        return {
            frame.id: Observation(status="unavailable", detail=detail, provider="rapidocr")
            for frame in requested
        }

    def _observe_vision(
        self,
        batch: FrameBatch,
        routes: list[AiRoute],
        ocr: dict[str, Observation],
    ) -> dict[str, Observation]:
        requested = [frame for frame in batch.frames if "describe" in frame.processors]
        if not routes:
            return {
                frame.id: Observation(status="unavailable", detail="vision processor unavailable")
                for frame in requested
            }
        processor = VisionProcessor(client=self.run.ai_client(routes[0]))
        outcomes: dict[str, Observation] = {}
        for frame in requested:
            ocr_outcome = ocr.get(frame.id)
            outcomes[frame.id] = processor.observe(
                frame, ocr_text=ocr_outcome.text if ocr_outcome is not None else None
            )
        return outcomes

    def _unavailable(self, detail: str) -> EvidenceOutcome:
        self.run.manifest.add_step("evidence.plan", "warning", detail)
        self.run.manifest.warn(detail)
        return EvidenceOutcome("unavailable", None, None, omissions=(detail,))

    def _video_media(self) -> bool:
        try:
            return "video" in self.run.ensure_media(self.run.visual_media_request()).capabilities
        except NoTranscriptError as exc:
            self.run.manifest.add_step("source.media", "warning", str(exc))
            return False


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
