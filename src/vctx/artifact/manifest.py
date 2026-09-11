from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vctx.options import SourceAssetScope
from vctx.source.session import EffectReceipt, Revision, SourceCapability, SourceRecord

StepStatus = Literal["ok", "skipped", "warning", "error"]
RunStatus = Literal["ok", "partial", "error"]
ProductStatus = Literal["ready", "partial", "unavailable"]
Freshness = Literal["immutable", "observed-online", "unverified-offline"]
_TOKEN = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_OPERATION = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_KEY = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")
_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_SOURCE_CAPABILITIES: dict[str, set[SourceCapability]] = {
    "source_audio": {"audio"},
    "source_video": {"video"},
    "source_media": {"audio", "video"},
    "subtitle": {"subtitle"},
}


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _bounded_token(value: str) -> str:
    if _TOKEN.fullmatch(value) is None:
        raise ValueError("value must be bounded lowercase portable text")
    return value


def _portable_path(value: str) -> str:
    if "\\" in value or not value or value.startswith(("/", "//")):
        raise ValueError("artifact path must be a contained POSIX path")
    path = PurePosixPath(value)
    if path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path must be a contained POSIX path")
    for part in path.parts:
        reserved = part.split(".", 1)[0].casefold() in _RESERVED
        if ":" in part or part.endswith((" ", ".")) or reserved:
            raise ValueError("artifact path is not portable")
    return value


class ArtifactRef(ClosedModel):
    kind: str
    path: str
    media_type: str = Field(min_length=1, max_length=127)
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    _kind_is_portable = field_validator("kind")(_bounded_token)
    _path_is_portable = field_validator("path")(_portable_path)


class ProductOutcome(ClosedModel):
    product: str
    status: ProductStatus
    artifacts: list[str] = Field(default_factory=list, max_length=256)
    omissions: list[str] = Field(default_factory=list, max_length=64)

    _product_is_portable = field_validator("product")(_bounded_token)
    _artifact_paths_are_portable = field_validator("artifacts")(
        lambda values: [_portable_path(value) for value in values]
    )

    @field_validator("omissions")
    @classmethod
    def omissions_are_bounded(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 500 for value in values):
            raise ValueError("outcome omissions must contain 1..500 characters")
        return values

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> ProductOutcome:
        if len(self.artifacts) != len({path.casefold() for path in self.artifacts}):
            raise ValueError("outcome artifact paths must be unique")
        if self.status == "ready" and self.omissions:
            raise ValueError("ready outcomes cannot contain omissions")
        return self


class ManifestEffect(ClosedModel):
    operation: str
    status: str
    attempts: int = Field(default=0, ge=0, le=100)
    route: str | None = Field(default=None, max_length=128)
    provider: str | None = Field(default=None, max_length=128)
    model: str | None = Field(default=None, max_length=256)
    uploaded: bool = False
    cost_may_apply: bool = False
    diagnostic: str | None = Field(default=None, max_length=500)

    @field_validator("operation")
    @classmethod
    def operation_is_bounded(cls, value: str) -> str:
        if _OPERATION.fullmatch(value) is None:
            raise ValueError("operation must be bounded lowercase portable text")
        return value

    @field_validator("status")
    @classmethod
    def status_is_bounded(cls, value: str) -> str:
        return _bounded_token(value.replace("-", "_"))

    @classmethod
    def from_receipt(cls, receipt: EffectReceipt) -> ManifestEffect:
        return cls(
            operation=receipt.operation,
            status=receipt.status,
            attempts=receipt.attempts,
            route=receipt.selected_policy or receipt.requested_policy,
            diagnostic=receipt.detail,
        )


class RunFailure(ClosedModel):
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    category: str
    diagnostic: str = Field(min_length=1, max_length=500)

    _category_is_portable = field_validator("category")(_bounded_token)


class PackRun(ClosedModel):
    id: UUID
    requested_target: str
    failures: list[RunFailure] = Field(default_factory=list, max_length=256)

    _target_is_portable = field_validator("requested_target")(_bounded_token)


class ManifestSource(ClosedModel):
    id: str = Field(min_length=1, max_length=512)
    key: str
    path: str
    kind: Literal["url", "file"]
    revision: Revision
    freshness: Freshness
    observed_at: datetime
    title: str | None = Field(default=None, max_length=1000)
    duration_seconds: float | None = Field(default=None, ge=0)
    status: RunStatus
    artifacts: list[ArtifactRef] = Field(max_length=1024)
    outcomes: list[ProductOutcome] = Field(max_length=128)
    effects: list[ManifestEffect] = Field(max_length=1024)
    asset_scope: SourceAssetScope | None = None
    source_capabilities: set[SourceCapability] | None = None

    @field_validator("key")
    @classmethod
    def key_is_portable(cls, value: str) -> str:
        if len(value) > 64 or _KEY.fullmatch(value) is None:
            raise ValueError("source key must be bounded lowercase portable text")
        if value.casefold() in _RESERVED:
            raise ValueError("source key is reserved on a supported filesystem")
        return value

    @model_validator(mode="after")
    def source_index_is_consistent(self) -> ManifestSource:
        _portable_path(self.path)
        paths = [artifact.path.casefold() for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be unique within a source lane")
        indexed = set(paths)
        products = [outcome.product for outcome in self.outcomes]
        if len(products) != len(set(products)):
            raise ValueError("product outcomes must be unique")
        for outcome in self.outcomes:
            if any(path.casefold() not in indexed for path in outcome.artifacts):
                raise ValueError("product outcome references an unlisted artifact")
        return self

    def effective_asset_scope(self) -> SourceAssetScope:
        retained = any(
            item.product == "source-assets" and item.status == "ready" for item in self.outcomes
        )
        return self.asset_scope or ("consumed" if retained else "omitted")


class Manifest(ClosedModel):
    schema_version: Literal["3", "4", "5"] = "5"
    tool: Literal["vctx"] = "vctx"
    tool_version: str = Field(min_length=1, max_length=64)
    pack_id: UUID
    run: PackRun
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    sources: list[ManifestSource] = Field(max_length=1024)

    @model_validator(mode="after")
    def sources_are_unique(self) -> Manifest:
        for source in self.sources:
            expected = source.key if self.schema_version in {"3", "5"} else f"sources/{source.key}"
            if source.path != expected:
                raise ValueError(f"schema-{self.schema_version} source path must equal {expected}")
            scope_fields = {"asset_scope", "source_capabilities"}
            present = scope_fields & source.model_fields_set
            if self.schema_version == "5" and present != scope_fields:
                raise ValueError("schema-5 sources require asset scope and capability evidence")
            if self.schema_version != "5" and present:
                raise ValueError("schema-3/4 sources cannot contain schema-5 fields")
            if source.asset_scope == "complete" and (
                source.source_capabilities is None
                or not source.source_capabilities <= retained_capabilities(source.artifacts)
            ):
                raise ValueError("complete source assets do not cover source capabilities")
        for field in ("id", "key", "path"):
            values = [str(getattr(source, field)).casefold() for source in self.sources]
            if len(values) != len(set(values)):
                raise ValueError(f"manifest source {field} values must be unique")
        revisions = [
            (source.id, source.revision.kind, source.revision.value) for source in self.sources
        ]
        if len(revisions) != len(set(revisions)):
            raise ValueError("manifest source identity/revision pairs must be unique")
        return self


class ManifestBuilder:
    def __init__(
        self, source: SourceRecord, key: str, *, offline: bool, asset_scope: SourceAssetScope
    ) -> None:
        self.source = source
        self.key = key
        self.asset_scope = asset_scope
        self.freshness: Freshness = (
            "immutable"
            if source.revision.kind == "immutable"
            else "unverified-offline"
            if offline
            else "observed-online"
        )
        self.effects: list[ManifestEffect] = []
        self.outcomes: dict[str, ProductOutcome] = {}
        self.omissions: list[str] = []

    def add_step(
        self, name: str, status: StepStatus, detail: str | None = None, receipt: object = None
    ) -> None:
        del receipt
        mapped = {"ok": "succeeded", "warning": "warning", "error": "failed"}.get(status, status)
        self.effects.append(ManifestEffect(operation=name, status=mapped, diagnostic=detail))

    def warn(self, message: str) -> None:
        self.omissions.append(message[:500])

    def add_effect(self, effect: ManifestEffect) -> None:
        self.effects.append(effect)

    def add_outcome(self, outcome: ProductOutcome) -> None:
        self.outcomes[outcome.product] = outcome

    def finish(
        self, status: RunStatus, artifacts: list[ArtifactRef], receipts: list[EffectReceipt]
    ) -> ManifestSource:
        unique = {artifact.path.casefold(): artifact for artifact in artifacts}
        if len(unique) != len(artifacts):
            raise ValueError("each retained file must have exactly one artifact record")
        outcomes = dict(self.outcomes)
        if self.omissions:
            outcomes["prepare"] = ProductOutcome(
                product="prepare", status="partial", omissions=self.omissions[:64]
            )
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
            outcomes=list(outcomes.values()),
            effects=[
                *(ManifestEffect.from_receipt(receipt) for receipt in receipts),
                *self.effects,
            ],
            asset_scope=self.asset_scope,
            source_capabilities=self.source.source_capabilities,
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


def retained_capabilities(artifacts: list[ArtifactRef]) -> set[SourceCapability]:
    return set().union(*(_SOURCE_CAPABILITIES.get(item.kind, set()) for item in artifacts))


def schema_five_source(source: ManifestSource) -> ManifestSource:
    paths = {artifact.path: artifact.path.removeprefix("assets/") for artifact in source.artifacts}
    outcomes = [
        outcome.model_copy(update={"artifacts": [paths[path] for path in outcome.artifacts]})
        for outcome in source.outcomes
        if outcome.product != "source-assets"
    ]
    return source.model_copy(
        update={
            "path": source.key,
            "artifacts": [
                artifact.model_copy(update={"path": paths[artifact.path]})
                for artifact in source.artifacts
            ],
            "outcomes": outcomes,
            "asset_scope": source.effective_asset_scope(),
            "source_capabilities": source.source_capabilities,
        }
    )


def build_manifest(
    tool_version: str,
    sources: list[ManifestSource],
    *,
    incomplete: bool = False,
    previous: Manifest | None = None,
    failures: list[RunFailure] | None = None,
    requested_target: str = "prepare",
) -> Manifest:
    now = datetime.now(UTC)
    statuses = {source.status for source in sources}
    if statuses == {"error"}:
        status: RunStatus = "error"
    elif incomplete or "error" in statuses or "partial" in statuses:
        status = "partial"
    else:
        status = "ok"
    normalized = [schema_five_source(source) for source in sources]
    return Manifest(
        tool_version=tool_version,
        pack_id=previous.pack_id if previous else uuid4(),
        run=PackRun(id=uuid4(), requested_target=requested_target, failures=failures or []),
        status=status,
        created_at=previous.created_at if previous else now,
        updated_at=now,
        sources=normalized,
    )


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-._")
