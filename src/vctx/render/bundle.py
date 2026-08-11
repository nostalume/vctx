from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from vctx.artifact.content import Artifact, ArtifactBundle, ArtifactKind
from vctx.io import model_to_json
from vctx.render.markdown import render_context_markdown, render_readable_markdown
from vctx.source.session import VideoMetadata
from vctx.transcript import ChunkSet, Transcript
from vctx.visual.evidence import Evidence
from vctx.visual.plan import EvidencePlan

OutputFormat = Literal["json", "context", "readable"]
DEFAULT_FORMATS: set[OutputFormat] = {"json", "context", "readable"}


def json_artifact(name: str, kind: ArtifactKind, model: BaseModel) -> Artifact:
    return Artifact(
        name=name,
        kind=kind,
        media_type="application/json",
        content=model_to_json(model),
    )


def markdown_artifact(name: str, kind: ArtifactKind, content: str) -> Artifact:
    return Artifact(
        name=name,
        kind=kind,
        media_type="text/markdown",
        content=content,
    )


def render_artifact_bundle(
    *,
    metadata: VideoMetadata,
    transcript: Transcript,
    chunks: ChunkSet,
    formats: set[OutputFormat],
    evidence: Evidence | None = None,
    evidence_plan: EvidencePlan | None = None,
    output_language: str = "native",
) -> ArtifactBundle:
    del output_language
    artifacts: list[Artifact] = []
    if "json" in formats:
        artifacts.extend(
            [
                json_artifact("metadata.json", "metadata", metadata),
                json_artifact("transcript.json", "transcript", transcript),
                json_artifact("chunks.json", "chunks", chunks),
            ]
        )
        if evidence is not None and evidence.captures:
            artifacts.append(json_artifact("evidence.json", "evidence", evidence))
        if evidence_plan is not None:
            artifacts.append(json_artifact("evidence-plan.json", "evidence_plan", evidence_plan))
    if "context" in formats:
        artifacts.append(
            markdown_artifact(
                "context.md",
                "context",
                render_context_markdown(
                    metadata,
                    transcript,
                    chunks,
                    evidence,
                    evidence_plan,
                ),
            )
        )
    if "readable" in formats:
        artifacts.append(
            markdown_artifact(
                "read.md",
                "readable",
                render_readable_markdown(
                    metadata,
                    transcript,
                    chunks,
                    evidence,
                    evidence_plan,
                ),
            )
        )
    return ArtifactBundle(artifacts=artifacts)
