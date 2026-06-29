from __future__ import annotations

from vctx.chunking import ChunkSet, TranscriptChunk
from vctx.models import SourceRef
from vctx.models.knowledge_flow import KnowledgeFlow, KnowledgeFlowEdge, KnowledgeFlowNode
from vctx.models.metadata import VideoMetadata
from vctx.render.markdown import render_context_markdown, render_readable_markdown
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment


def test_render_context_markdown_includes_knowledge_flow_chains_with_evidence() -> None:
    markdown = render_context_markdown(
        _metadata(),
        _transcript(),
        _chunks(),
        knowledge_flow=_knowledge_flow(),
    )

    assert "## Knowledge flow" in markdown
    assert "## Knowledge-flow summary" in markdown
    assert "- Workflow: URL acquisition -> transcript extraction -> knowledge flow." in markdown
    assert "- URL acquisition -> transcript extraction -> knowledge flow" in markdown
    assert "Evidence: seg_000001, ocr-0001" in markdown


def test_render_readable_markdown_includes_knowledge_flow_chains_with_evidence() -> None:
    markdown = render_readable_markdown(
        _metadata(),
        _transcript(),
        _chunks(),
        knowledge_flow=_knowledge_flow(),
    )

    assert "## Knowledge flow" in markdown
    assert "## Knowledge-flow summary" in markdown
    assert "- Workflow: URL acquisition -> transcript extraction -> knowledge flow." in markdown
    assert "- URL acquisition -> transcript extraction -> knowledge flow" in markdown
    assert "Evidence: seg_000001, ocr-0001" in markdown


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        id="video-1",
        source_type="file",
        source=SourceRef(kind="file", value="video.mp4"),
        title="Video",
        duration_seconds=30.0,
    )


def _transcript() -> Transcript:
    return Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text="URL acquisition -> transcript extraction -> knowledge flow",
            )
        ],
    )


def _chunks() -> ChunkSet:
    return ChunkSet(
        video_id="video-1",
        strategy="test",
        chunks=[
            TranscriptChunk(
                id="chunk-1",
                start=0.0,
                end=5.0,
                text="URL acquisition -> transcript extraction -> knowledge flow",
                segment_ids=["seg_000001"],
                char_count=61,
                approx_token_count=8,
            )
        ],
    )


def _knowledge_flow() -> KnowledgeFlow:
    return KnowledgeFlow(
        nodes=[
            KnowledgeFlowNode(
                id="url-acquisition",
                label="URL acquisition",
                evidence=["seg_000001"],
            ),
            KnowledgeFlowNode(
                id="transcript-extraction",
                label="transcript extraction",
                evidence=["seg_000001"],
            ),
            KnowledgeFlowNode(
                id="knowledge-flow",
                label="knowledge flow",
                evidence=["seg_000001", "ocr-0001"],
            ),
        ],
        edges=[
            KnowledgeFlowEdge(
                id="url-acquisition__transcript-extraction",
                source="url-acquisition",
                target="transcript-extraction",
                evidence=["seg_000001"],
            ),
            KnowledgeFlowEdge(
                id="transcript-extraction__knowledge-flow",
                source="transcript-extraction",
                target="knowledge-flow",
                evidence=["seg_000001", "ocr-0001"],
            ),
        ],
    )
