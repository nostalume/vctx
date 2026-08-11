from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vctx.artifact.content import ArtifactKind
from vctx.source.session import EffectReceipt, Revision, SourceRecord
from vctx.transcript import AsrProvenance

StepStatus = Literal["ok", "skipped", "warning", "error"]
RunStatus = Literal["ok", "partial", "error"]
SelectedRoute = Literal[
    "skipped", "deterministic", "local", "free-online", "configured-online", "unavailable"
]
CapabilityName = Literal["asr", "visual_context", "evidence_plan"]
Freshness = Literal["immutable", "observed-online", "unverified-offline"]
_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AsrStepReceipt(AsrProvenance):
    kind: Literal["asr"] = "asr"
    failure: (
        Literal[
            "missing_package",
            "missing_model",
            "corrupt_model",
            "unwritable_cache",
            "unsupported_hardware",
            "inference_failed",
            "confirmation_failed",
            "invalid_timestamps",
            "invalid_response",
        ]
        | None
    ) = None


class FrameCaptureReceipt(ClosedModel):
    id: str
    path: str
    requested_seconds: float = Field(ge=0)
    actual_seconds: float = Field(ge=0)
    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    orientation: Literal[0, 90, 180, 270]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(min_length=64, max_length=64)
    request_ids: list[str]
    segment_ids: list[str]
    claim_ids: list[str]
    processors: list[Literal["ocr", "describe"]]


class FrameMissReceipt(ClosedModel):
    id: str
    requested_seconds: float = Field(ge=0)
    reason: Literal["target_out_of_range"]
    request_ids: list[str]
    segment_ids: list[str]
    claim_ids: list[str]


class FrameStepReceipt(ClosedModel):
    kind: Literal["frames"] = "frames"
    recipe: Literal["pyav-display-v1"] = "pyav-display-v1"
    captures: list[FrameCaptureReceipt]
    misses: list[FrameMissReceipt]


type StepReceipt = Annotated[AsrStepReceipt | FrameStepReceipt, Field(discriminator="kind")]


class ManifestStep(ClosedModel):
    name: str
    status: StepStatus
    detail: str | None = None
    receipt: StepReceipt | None = None


class ArtifactRef(ClosedModel):
    kind: ArtifactKind
    path: str
    media_type: str
    bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def contained_lane_path(self) -> ArtifactRef:
        path = PurePosixPath(self.path)
        valid_frame = (
            self.kind == "visual_frame"
            and len(path.parts) == 2
            and path.parts[0] == "frames"
            and path.suffix == ".png"
        )
        valid_direct = self.kind != "visual_frame" and len(path.parts) == 1
        if path.is_absolute() or path.name in {"", ".", ".."} or not (valid_frame or valid_direct):
            raise ValueError("artifact path is outside its canonical source-lane location")
        return self


class SourceEffect(ClosedModel):
    operation: Literal["observe", "subtitle", "media"]
    status: Literal["cache_hit", "succeeded", "denied", "failed"]
    attempts: int = Field(ge=0)
    purpose: Literal["transcript", "asr", "visual", "input"] | None = None
    requested_policy: str | None = None
    selected_policy: str | None = None

    @classmethod
    def from_receipt(cls, receipt: EffectReceipt) -> SourceEffect:
        return cls(**receipt.model_dump(exclude={"detail"}))


class SourceAssetCore(ClosedModel):
    retained: bool
    path: str | None = None
    media_type: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    omission_reason: str | None = None

    @model_validator(mode="after")
    def retention_fields_are_consistent(self) -> SourceAssetCore:
        if self.retained:
            if self.path is None or self.bytes is None or self.sha256 is None:
                raise ValueError("retained source assets require path, bytes, and sha256")
            path = PurePosixPath(self.path)
            if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
                raise ValueError("source asset path must be one direct source-lane child")
            if self.bytes < 0 or len(self.sha256) != 64:
                raise ValueError("source asset integrity fields are invalid")
            if self.omission_reason is not None:
                raise ValueError("retained source assets cannot have an omission reason")
        elif any(value is not None for value in (self.path, self.bytes, self.sha256)):
            raise ValueError("omitted source assets cannot claim retained bytes")
        elif self.omission_reason is None:
            raise ValueError("omitted source assets require an omission reason")
        return self


class SubtitleSourceAsset(SourceAssetCore):
    kind: Literal["subtitle"] = "subtitle"
    purpose: Literal["transcript"] = "transcript"
    format: Literal["vtt", "srt", "json", "plain", "unknown"]
    language: str | None = None


class AudioSourceAsset(SourceAssetCore):
    kind: Literal["audio"] = "audio"
    purpose: Literal["asr"] = "asr"
    requested_profile: None = None
    selected_format: str


class VideoSourceAsset(SourceAssetCore):
    kind: Literal["video"] = "video"
    purpose: Literal["visual"] = "visual"
    requested_profile: Literal["auto", "fast", "balanced", "high"]
    selected_format: str


class InputSourceAsset(SourceAssetCore):
    kind: Literal["input"] = "input"
    purpose: Literal["input"] = "input"
    selected_format: str


SourceAsset = Annotated[
    SubtitleSourceAsset | AudioSourceAsset | VideoSourceAsset | InputSourceAsset,
    Field(discriminator="kind"),
]


class TransformEvidence(ClosedModel):
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


class ManifestSource(ClosedModel):
    id: str
    key: str
    path: str
    kind: Literal["url", "file"]
    revision: Revision
    freshness: Freshness
    observed_at: datetime
    title: str | None = None
    duration_seconds: float | None = None
    status: RunStatus
    artifacts: list[ArtifactRef]
    effects: list[SourceEffect]
    assets: list[SourceAsset]
    steps: list[ManifestStep]
    warnings: list[str] = Field(default_factory=list)
    transform_evidence: list[TransformEvidence] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def key_is_portable(cls, value: str) -> str:
        if len(value) > 64 or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", value) is None:
            raise ValueError("source key must be bounded lowercase portable text")
        if value.casefold() in _RESERVED:
            raise ValueError("source key is reserved on a supported filesystem")
        return value

    @model_validator(mode="after")
    def path_matches_key(self) -> ManifestSource:
        if self.path != self.key or PurePosixPath(self.path).parts != (self.key,):
            raise ValueError("source path must equal its direct-child key")
        return self


class Manifest(ClosedModel):
    schema_version: Literal["2"] = "2"
    tool: Literal["vctx"] = "vctx"
    tool_version: str
    pack_id: UUID
    updated_run_id: UUID
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    sources: list[ManifestSource]

    @model_validator(mode="after")
    def sources_are_unique(self) -> Manifest:
        for field in ("id", "key", "path"):
            values = [getattr(source, field) for source in self.sources]
            if len(values) != len(set(values)):
                raise ValueError(f"manifest source {field} values must be unique")
        return self


class ManifestBuilder:
    def __init__(self, source: SourceRecord, key: str, *, offline: bool) -> None:
        self.source = source
        self.key = key
        self.freshness: Freshness = (
            "immutable"
            if source.revision.kind == "immutable"
            else "unverified-offline"
            if offline
            else "observed-online"
        )
        self.steps: list[ManifestStep] = []
        self.warnings: list[str] = []
        self.transform_evidence: list[TransformEvidence] = []
        self.source_assets: list[SourceAsset] = []

    @classmethod
    def start(cls, source: SourceRecord, key: str, *, offline: bool) -> ManifestBuilder:
        return cls(source, key, offline=offline)

    def add_step(
        self,
        name: str,
        status: StepStatus,
        detail: str | None = None,
        receipt: StepReceipt | None = None,
    ) -> None:
        self.steps.append(ManifestStep(name=name, status=status, detail=detail, receipt=receipt))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def add_transform_evidence(self, evidence: TransformEvidence) -> None:
        self.transform_evidence.append(evidence)

    def add_source_asset(self, asset: SourceAsset) -> None:
        self.source_assets.append(asset)

    def finish(
        self, status: RunStatus, artifacts: list[ArtifactRef], receipts: list[EffectReceipt]
    ) -> ManifestSource:
        metadata = self.source.metadata
        return ManifestSource(
            id=self.source.source_id,
            key=self.key,
            path=self.key,
            kind=metadata.source.kind,
            revision=self.source.revision,
            freshness=self.freshness,
            observed_at=self.source.observed_at,
            title=metadata.title,
            duration_seconds=metadata.duration_seconds,
            status=status,
            artifacts=artifacts,
            effects=[SourceEffect.from_receipt(receipt) for receipt in receipts],
            assets=self.source_assets,
            steps=self.steps,
            warnings=self.warnings,
            transform_evidence=self.transform_evidence,
        )


def source_key(source_id: str, occupied: dict[str, str] | None = None) -> str:
    provider, _, identity = source_id.partition("__")
    provider = _token(provider)[:20] or "source"
    identity = _token(identity or source_id)[:36] or "item"
    key = f"{provider}-{identity}"[:57].rstrip("-.")
    if key.casefold() in _RESERVED:
        key = f"source-{key}"
    occupied = occupied or {}
    owner = occupied.get(key.casefold())
    if owner is not None and owner != source_id:
        suffix = hashlib.sha256(source_id.encode()).hexdigest()[:10]
        key = f"{key[:46].rstrip('-')}-{suffix}"
    return key


def build_manifest(
    tool_version: str,
    sources: list[ManifestSource],
    *,
    incomplete: bool = False,
    previous: Manifest | None = None,
) -> Manifest:
    now = datetime.now(UTC)
    statuses = {source.status for source in sources}
    if statuses == {"error"}:
        status: RunStatus = "error"
    elif incomplete or "error" in statuses or "partial" in statuses:
        status = "partial"
    else:
        status = "ok"
    return Manifest(
        tool_version=tool_version,
        pack_id=previous.pack_id if previous else uuid4(),
        updated_run_id=uuid4(),
        status=status,
        created_at=previous.created_at if previous else now,
        updated_at=now,
        sources=sources,
    )


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-._")
