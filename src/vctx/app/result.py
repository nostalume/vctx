from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from vctx.config import PrepareRequest, ResolvedConfig
from vctx.models.artifacts import ArtifactKind
from vctx.models.manifest import ArtifactRef, Manifest, TransformEvidence


class PrepareRouteSummary(BaseModel):
    capability: str
    route: str
    provider_id: str | None = None
    model_id: str | None = None
    uploaded: bool = False
    cost_may_apply: bool = False
    reason: str

    def render(self) -> str:
        parts = [f"{self.capability}: {self.route}"]
        if self.provider_id is not None:
            parts.append(f"provider={self.provider_id}")
        if self.model_id is not None:
            parts.append(f"model={self.model_id}")
        if self.uploaded:
            parts.append("upload=required")
        if self.cost_may_apply:
            parts.append("cost=may-apply")
        parts.append(self.reason)
        return "; ".join(parts)


class PrepareSummary(BaseModel):
    input: str
    status: str
    workflow: str
    out_dir: Path
    cache_dir: Path
    config_path: Path | None
    artifacts: list[str]
    warnings: list[str]
    routes: list[PrepareRouteSummary]

    @classmethod
    def from_run(
        cls,
        *,
        request: PrepareRequest,
        resolved: ResolvedConfig,
        manifest: Manifest,
        artifacts: list[ArtifactRef],
    ) -> PrepareSummary:
        return cls(
            input=request.input,
            status=manifest.status,
            workflow=resolved.runtime.workflow,
            out_dir=request.out_dir,
            cache_dir=resolved.runtime.cache_dir,
            config_path=request.config_path,
            artifacts=[artifact.path for artifact in artifacts],
            warnings=manifest.warnings,
            routes=[_route_summary(evidence) for evidence in manifest.transform_evidence],
        )

    def render_cli_lines(self) -> list[str]:
        config_label = (
            str(self.config_path) if self.config_path is not None else "built-in defaults + CLI"
        )
        lines = [
            f"Workflow: {self.workflow}",
            f"Status: {self.status}",
            f"Output: {self.out_dir}",
            f"Cache: {self.cache_dir}",
            f"Config: {config_label}",
        ]
        if self.artifacts:
            lines.append("Artifacts:")
            lines.extend(f"  - {artifact}" for artifact in self.artifacts)
        if self.routes:
            lines.append("Routes:")
            lines.extend(f"  - {route.render()}" for route in self.routes)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return lines


class PrepareResult(BaseModel):
    out_dir: Path
    manifest: Manifest
    artifacts: list[ArtifactRef]
    summary: PrepareSummary

    def artifact_path(self, kind: ArtifactKind) -> Path | None:
        for artifact in self.artifacts:
            if artifact.kind == kind:
                return self.out_dir / artifact.path
        return None


def _route_summary(evidence: TransformEvidence) -> PrepareRouteSummary:
    return PrepareRouteSummary(
        capability=evidence.capability,
        route=evidence.selected_route,
        provider_id=evidence.provider_id,
        model_id=evidence.model_id,
        uploaded=evidence.uploaded,
        cost_may_apply=evidence.cost_may_apply,
        reason=evidence.reason,
    )
