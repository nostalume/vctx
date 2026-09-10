from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vctx.visual.frame import Frame, FrameBatch, FrameMiss
from vctx.visual.plan import EvidencePlan, FrameProcessor, PlannedFrame


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Observation(ClosedModel):
    status: Literal["not_requested", "unavailable", "empty", "failed", "ok"]
    text: str | None = None
    detail: str | None = None
    provider: str | None = None

    @model_validator(mode="after")
    def authority_matches_state(self) -> Observation:
        if self.status == "not_requested":
            if self.provider is not None or self.text is not None:
                raise ValueError("a processor that was not requested cannot report output")
        elif self.status in {"ok", "empty", "failed"} and not self.provider:
            raise ValueError(f"{self.status} processor observations require provider identity")
        return self


class CaptureEvidence(ClosedModel):
    id: str
    requested_seconds: float
    actual_seconds: float
    artifact_path: str
    request_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    ocr: Observation
    vision: Observation


class EvidenceMiss(ClosedModel):
    id: str
    requested_seconds: float
    reason: Literal["target_out_of_range"]
    request_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class Evidence(ClosedModel):
    source_id: str
    captures: list[CaptureEvidence] = Field(default_factory=list)
    misses: list[EvidenceMiss] = Field(default_factory=list)

    @property
    def status(self) -> Literal["ready", "partial"]:
        degraded = bool(self.misses) or any(
            observation.status in {"unavailable", "failed"}
            for capture in self.captures
            for observation in (capture.ocr, capture.vision)
        )
        return "partial" if degraded else "ready"

    @classmethod
    def from_observations(
        cls,
        plan: EvidencePlan,
        batch: FrameBatch,
        lane: Path,
        *,
        ocr: Mapping[str, Observation],
        vision: Mapping[str, Observation],
    ) -> Evidence:
        planned = _planned(plan)
        _validate_batch(planned, batch)
        captured_ids = {frame.id for frame in batch.frames}
        if not set(ocr) <= captured_ids or not set(vision) <= captured_ids:
            raise ValueError("processor outcome refers to an unknown capture")
        evidence = cls(
            source_id=plan.source_id,
            captures=[
                _capture(frame, planned[frame.id], lane, ocr.get(frame.id), vision.get(frame.id))
                for frame in batch.frames
            ],
            misses=[_miss(miss) for miss in batch.misses],
        )
        evidence.validate_against(plan)
        return evidence

    def captures_for_claim(self, claim_id: str) -> list[CaptureEvidence]:
        return [capture for capture in self.captures if claim_id in capture.claim_ids]

    def validate_against(self, plan: EvidencePlan) -> None:
        planned = _planned(plan)
        observed = {capture.id for capture in self.captures} | {miss.id for miss in self.misses}
        if observed != set(planned):
            raise ValueError("evidence does not cover every planned frame")
        for capture in self.captures:
            expected = planned.get(capture.id)
            if (
                expected is None
                or not _same_links(capture, expected)
                or not _observation_links(capture, expected)
            ):
                raise ValueError(f"capture {capture.id} does not match evidence plan")
        for miss in self.misses:
            expected = planned.get(miss.id)
            if expected is None or not _same_links(miss, expected):
                raise ValueError(f"capture miss {miss.id} does not match evidence plan")


def _planned(plan: EvidencePlan) -> dict[str, PlannedFrame]:
    planned = {frame.id: frame for frame in plan.frames}
    if len(planned) != len(plan.frames):
        raise ValueError("evidence plan frame ids must be unique")
    return planned


def _validate_batch(planned: Mapping[str, PlannedFrame], batch: FrameBatch) -> None:
    observed = [frame.id for frame in batch.frames] + [miss.id for miss in batch.misses]
    if len(observed) != len(set(observed)) or set(observed) != set(planned):
        raise ValueError("frame batch does not cover every evidence plan frame exactly once")
    for frame in batch.frames:
        if not _same_frame(frame, planned[frame.id]):
            raise ValueError(f"capture {frame.id} does not match evidence plan")
    for miss in batch.misses:
        if not _same_frame(miss, planned[miss.id]):
            raise ValueError(f"capture miss {miss.id} does not match evidence plan")


def _capture(
    frame: Frame,
    planned: PlannedFrame,
    lane: Path,
    ocr: Observation | None,
    vision: Observation | None,
) -> CaptureEvidence:
    return CaptureEvidence(
        id=frame.id,
        requested_seconds=frame.requested_seconds,
        actual_seconds=frame.actual_seconds,
        artifact_path=frame.path.relative_to(lane).as_posix(),
        request_ids=list(frame.request_ids),
        segment_ids=list(frame.segment_ids),
        claim_ids=list(frame.claim_ids),
        ocr=_observation("ocr", planned.processors, ocr),
        vision=_observation("describe", planned.processors, vision),
    )


def _observation(
    processor: FrameProcessor,
    requested: Sequence[FrameProcessor],
    value: Observation | None,
) -> Observation:
    if processor not in requested:
        if value is not None:
            raise ValueError(f"{processor} outcome was supplied but not requested")
        return Observation(status="not_requested")
    if value is None:
        return Observation(status="unavailable", detail="processor produced no outcome")
    if value.status == "not_requested":
        raise ValueError(f"{processor} was requested but reported not_requested")
    return value


def _miss(miss: FrameMiss) -> EvidenceMiss:
    return EvidenceMiss(
        id=miss.id,
        requested_seconds=miss.requested_seconds,
        reason=miss.reason,
        request_ids=list(miss.request_ids),
        segment_ids=list(miss.segment_ids),
        claim_ids=list(miss.claim_ids),
    )


def _same_frame(actual: Frame | FrameMiss, expected: PlannedFrame) -> bool:
    return (
        actual.requested_seconds == expected.target_seconds
        and tuple(actual.request_ids) == tuple(expected.request_ids)
        and tuple(actual.segment_ids) == tuple(expected.segment_ids)
        and tuple(actual.claim_ids) == tuple(expected.claim_ids)
    )


def _same_links(actual: CaptureEvidence | EvidenceMiss, expected: PlannedFrame) -> bool:
    return (
        actual.requested_seconds == expected.target_seconds
        and actual.request_ids == expected.request_ids
        and actual.segment_ids == expected.segment_ids
        and actual.claim_ids == expected.claim_ids
    )


def _observation_links(actual: CaptureEvidence, expected: PlannedFrame) -> bool:
    return (actual.ocr.status != "not_requested") == ("ocr" in expected.processors) and (
        actual.vision.status != "not_requested"
    ) == ("describe" in expected.processors)
