from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vctx.visual.frame import Frame, FrameBatch


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Observation(ClosedModel):
    status: Literal["ok", "empty", "unavailable", "failed"]
    text: str | None = None
    detail: str | None = None
    provider: str


class CaptureEvidence(ClosedModel):
    id: str
    requested_seconds: float
    actual_seconds: float
    artifact_path: str
    request_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    ocr: Observation | None = None
    vision: Observation | None = None


class Evidence(ClosedModel):
    video_id: str
    captures: list[CaptureEvidence] = Field(default_factory=list)

    @property
    def partial(self) -> bool:
        return any(
            observation is not None and observation.status in {"unavailable", "failed"}
            for capture in self.captures
            for observation in (capture.ocr, capture.vision)
        )


def assemble(
    video_id: str,
    batch: FrameBatch,
    out_dir: Path,
    *,
    ocr: Mapping[str, Observation],
    vision: Mapping[str, Observation],
) -> Evidence:
    return Evidence(
        video_id=video_id,
        captures=[
            _capture(frame, out_dir, ocr.get(frame.id), vision.get(frame.id))
            for frame in batch.frames
        ],
    )


def _capture(
    frame: Frame,
    out_dir: Path,
    ocr: Observation | None,
    vision: Observation | None,
) -> CaptureEvidence:
    requested = set(frame.processors)
    return CaptureEvidence(
        id=frame.id,
        requested_seconds=frame.requested_seconds,
        actual_seconds=frame.actual_seconds,
        artifact_path=frame.path.relative_to(out_dir).as_posix(),
        request_ids=list(frame.request_ids),
        segment_ids=list(frame.segment_ids),
        claim_ids=list(frame.claim_ids),
        ocr=_observation(ocr, "rapidocr") if "ocr" in requested else None,
        vision=_observation(vision, "vision") if "describe" in requested else None,
    )


def _observation(value: Observation | None, provider: str) -> Observation:
    if value is None:
        return Observation(status="unavailable", detail="processor unavailable", provider=provider)
    return value
