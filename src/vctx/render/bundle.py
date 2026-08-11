from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from vctx.artifact.content import Artifact, ArtifactBundle, ArtifactKind
from vctx.io import model_to_json
from vctx.models.knowledge_flow import KnowledgeFlow
from vctx.models.visual import VisualRecordSet, VisualScoreReport
from vctx.render.markdown import render_context_markdown, render_readable_markdown
from vctx.source.session import VideoMetadata
from vctx.transcript import ChunkSet, Transcript

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
    visual_records: VisualRecordSet | None = None,
    visual_scores: VisualScoreReport | None = None,
    knowledge_flow: KnowledgeFlow | None = None,
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
        if visual_records is not None and visual_records.records:
            artifacts.append(json_artifact("visual_records.json", "visual_records", visual_records))
        if visual_scores is not None and visual_scores.satisfaction:
            artifacts.append(json_artifact("visual_scores.json", "visual_scores", visual_scores))
        if knowledge_flow is not None and knowledge_flow.nodes:
            artifacts.append(json_artifact("knowledge_flow.json", "knowledge_flow", knowledge_flow))
    if "context" in formats:
        artifacts.append(
            markdown_artifact(
                "context.md",
                "context",
                render_context_markdown(
                    metadata,
                    transcript,
                    chunks,
                    visual_records,
                    knowledge_flow,
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
                    visual_records,
                    knowledge_flow,
                ),
            )
        )
    return ArtifactBundle(artifacts=artifacts)
