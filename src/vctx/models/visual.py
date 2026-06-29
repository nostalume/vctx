from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

EvidenceKind = Literal["metadata", "transcript", "frame", "probe"]
FrameSource = Literal["cover", "scene_change", "transcript_anchor", "probe"]
VisualRecordKind = Literal["ocr", "description", "capture"]
SatisfactionStatus = Literal["satisfied", "missed"]
EssentialCaseType = Literal[
    "diagram",
    "formula",
    "screen_demo",
    "table",
    "code",
    "slide_title",
    "visual_summary",
    "other",
]
EssentialCaseAction = Literal["ocr", "describe", "capture"]
VisualOperationName = Literal["capture", "ocr", "describe"]
VisualOperationReason = Literal[
    "visual_reference",
    "action_demonstration",
    "environment_context",
    "shown_text",
    "formula_or_equation",
    "table_or_code",
    "chart_visual_reference",
    "human_inspection",
]




class SourceAccess(IntEnum):
    NONE = 0b000
    VIDEO = 0b001
    AUDIO = 0b010
    AUDIO_VIDEO = 0b011
    TRANSCRIPT = 0b100
    TRANSCRIPT_VIDEO = 0b101
    TRANSCRIPT_AUDIO = 0b110
    TRANSCRIPT_AUDIO_VIDEO = 0b111

    @property
    def bits(self) -> str:
        return f"{self.value:03b}"

    @property
    def has_transcript(self) -> bool:
        return bool(self.value & 0b100)

    @property
    def has_audio(self) -> bool:
        return bool(self.value & 0b010)

    @property
    def has_video(self) -> bool:
        return bool(self.value & 0b001)

    @classmethod
    def from_flags(cls, *, transcript: bool, audio: bool, video: bool) -> SourceAccess:
        value = (0b100 if transcript else 0) | (0b010 if audio else 0) | (0b001 if video else 0)
        return cls(value)


class Evidence(BaseModel):
    kind: EvidenceKind
    name: str
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def clamp_weight(cls, value: float) -> float:
        return round(min(max(value, 0.0), 1.0), 3)


class EssentialVisualCase(BaseModel):
    segment_id: str
    timestamp_seconds: float
    case_type: EssentialCaseType
    priority: float = 0.5
    reason: str
    actions: list[EssentialCaseAction] = Field(default_factory=list)

    @field_validator("priority")
    @classmethod
    def clamp_priority(cls, value: float) -> float:
        return round(min(max(value, 0.0), 1.0), 3)




class VisualOperationMotive(BaseModel):
    operation: VisualOperationName
    reason: VisualOperationReason
    segment_id: str
    timestamp_seconds: float
    priority: float = 0.5
    explanation: str

    @field_validator("priority")
    @classmethod
    def clamp_priority(cls, value: float) -> float:
        return round(min(max(value, 0.0), 1.0), 3)


class EssentialCaseSupplementEvidence(BaseModel):
    route_provider_id: str | None = None
    route_model: str | None = None
    source_segment_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EssentialCaseSupplement(BaseModel):
    cases: list[EssentialVisualCase] = Field(default_factory=list)
    evidence: EssentialCaseSupplementEvidence = Field(
        default_factory=EssentialCaseSupplementEvidence
    )


class FrameAsset(BaseModel):
    id: str
    timestamp_seconds: float | None = None
    path: Path
    source: FrameSource
    evidence: list[Evidence] = Field(default_factory=list)


class VisualUncertainty(BaseModel):
    prior_uncertainty: float = 0.0
    posterior_uncertainty: float = 0.0
    reduction: float = 0.0
    missing_referents: list[str] = Field(default_factory=list)
    resolved_referents: list[str] = Field(default_factory=list)


class VisualEvidenceScore(BaseModel):
    keep: bool
    novelty_score: float = 0.0
    overlap_score: float = 0.0
    grounding_score: float = 0.0
    reason: str
    uncertainty: VisualUncertainty = Field(default_factory=VisualUncertainty)


class VisualRecord(BaseModel):
    id: str
    timestamp_seconds: float | None = None
    frame_id: str
    kind: VisualRecordKind
    text: str | None = None
    artifact_path: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    score: VisualEvidenceScore | None = None


class SatisfactionChecked(BaseModel):
    operation: VisualOperationName
    reason: VisualOperationReason
    segment_id: str
    frame_id: str | None = None
    status: SatisfactionStatus
    detail: str


class VisualRecordSet(BaseModel):
    records: list[VisualRecord] = Field(default_factory=list)


class VisualScoreReport(BaseModel):
    satisfaction: list[SatisfactionChecked] = Field(default_factory=list)


class VisualScoredRecordSet(BaseModel):
    records: list[VisualRecord] = Field(default_factory=list)
    satisfaction: list[SatisfactionChecked] = Field(default_factory=list)
