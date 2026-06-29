from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from vctx.chunking import ChunkSet
from vctx.io import model_to_json
from vctx.models.artifacts import Artifact, ArtifactBundle, ArtifactKind
from vctx.models.knowledge_flow import KnowledgeFlow
from vctx.models.metadata import VideoMetadata
from vctx.models.visual import VisualRecordSet, VisualScoreReport
from vctx.render.markdown import (
    render_context_markdown,
    render_readable_markdown,
    render_transcript_markdown,
)
from vctx.transcript import Transcript

OutputFormat = Literal["json", "context", "readable", "transcript"]
DEFAULT_FORMATS: set[OutputFormat] = {"json", "context", "readable", "transcript"}


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
    raw_transcript: Transcript,
    clean_transcript: Transcript,
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
                json_artifact("transcript.raw.json", "transcript_raw", raw_transcript),
                json_artifact("transcript.clean.json", "transcript_clean", clean_transcript),
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
                    clean_transcript,
                    chunks,
                    visual_records,
                    knowledge_flow,
                ),
            )
        )
    if "readable" in formats:
        artifacts.append(
            markdown_artifact(
                "readable.md",
                "readable",
                render_readable_markdown(
                    metadata,
                    clean_transcript,
                    chunks,
                    visual_records,
                    knowledge_flow,
                ),
            )
        )
    if "transcript" in formats:
        artifacts.append(
            markdown_artifact(
                "transcript.md",
                "transcript_md",
                render_transcript_markdown(metadata, clean_transcript),
            )
        )
    return ArtifactBundle(artifacts=artifacts)
