from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from vctx.models import SourceRef
from vctx.models.artifacts import ArtifactKind

StepStatus = Literal["ok", "skipped", "warning", "error"]
RunStatus = Literal["ok", "partial", "error"]
SelectedRoute = Literal[
    "skipped",
    "deterministic",
    "local",
    "free-online",
    "configured-online",
    "unavailable",
]
CapabilityName = Literal[
    "asr",
    "visual_context",
    "knowledge_flow",
    "essential_cases",
]


class ManifestStep(BaseModel):
    name: str
    status: StepStatus
    detail: str | None = None


class ArtifactRef(BaseModel):
    kind: ArtifactKind
    path: str
    media_type: str


class SourceMediaReceipt(BaseModel):
    source: SourceRef
    media_id: str
    purpose: Literal["input", "asr", "visual"]
    selected_format: str
    retained: bool
    path: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    omission_reason: str | None = None


class TransformEvidence(BaseModel):
    capability: CapabilityName
    selected_route: SelectedRoute
    provider_id: str | None = None
    model_id: str | None = None
    requires_user_config: bool = False
    uploaded: bool = False
    cost_may_apply: bool = False
    deterministic: bool = False
    source_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    reason: str
    warnings: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    schema_version: str = "0.1"
    tool: str = "vctx"
    tool_version: str
    status: RunStatus
    input: str
    created_at: str
    artifacts: list[ArtifactRef]
    steps: list[ManifestStep]
    warnings: list[str] = Field(default_factory=list)
    transform_evidence: list[TransformEvidence] = Field(default_factory=list)
    source_media: list[SourceMediaReceipt] = Field(default_factory=list)


class ManifestBuilder:
    def __init__(self, input: str, tool_version: str) -> None:
        self.input = input
        self.tool_version = tool_version
        self.steps: list[ManifestStep] = []
        self.warnings: list[str] = []
        self.transform_evidence: list[TransformEvidence] = []
        self.source_media: list[SourceMediaReceipt] = []

    @classmethod
    def start(cls, input: str, tool_version: str) -> ManifestBuilder:
        return cls(input=input, tool_version=tool_version)

    def add_step(self, name: str, status: StepStatus, detail: str | None = None) -> None:
        self.steps.append(ManifestStep(name=name, status=status, detail=detail))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def add_transform_evidence(self, evidence: TransformEvidence) -> None:
        self.transform_evidence.append(evidence)

    def add_source_media(self, receipt: SourceMediaReceipt) -> None:
        self.source_media.append(receipt)

    def finish(self, status: RunStatus, artifacts: list[ArtifactRef]) -> Manifest:
        return Manifest(
            tool_version=self.tool_version,
            status=status,
            input=self.input,
            created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            artifacts=artifacts,
            steps=self.steps,
            warnings=self.warnings,
            transform_evidence=self.transform_evidence,
            source_media=self.source_media,
        )
