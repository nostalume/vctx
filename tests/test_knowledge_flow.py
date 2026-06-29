from __future__ import annotations

from vctx.models.knowledge_flow import KnowledgeFlow
from vctx.models.visual import VisualEvidenceScore, VisualRecord, VisualRecordSet
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment
from vctx.transforms.knowledge_flow import extract_knowledge_flow


def test_knowledge_flow_extracts_transcript_and_kept_visual_arrow_chains() -> None:
    transcript = Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text="URL acquisition -> transcript extraction -> summary rendering",
            )
        ],
    )
    visual_records = VisualRecordSet(
        records=[
            VisualRecord(
                id="ocr-0001",
                timestamp_seconds=2.0,
                frame_id="frame-0001",
                kind="ocr",
                text="Ponytail -> Evidence Pack -> Knowledge Flow",
                score=VisualEvidenceScore(
                    keep=True,
                    novelty_score=0.8,
                    overlap_score=0.1,
                    reason="new visual tokens add grounded evidence",
                ),
            ),
            VisualRecord(
                id="description-0001",
                timestamp_seconds=2.0,
                frame_id="frame-0001",
                kind="description",
                text="Dropped Secret -> Unsupported Claim",
                score=VisualEvidenceScore(
                    keep=False,
                    novelty_score=0.1,
                    overlap_score=0.9,
                    reason="mostly overlaps nearby transcript",
                ),
            ),
        ]
    )

    flow = extract_knowledge_flow(transcript, visual_records)

    labels = {node.label for node in flow.nodes}
    assert labels >= {
        "URL acquisition",
        "transcript extraction",
        "summary rendering",
        "Ponytail",
        "Evidence Pack",
        "Knowledge Flow",
    }
    assert "Dropped Secret" not in labels
    assert "Unsupported Claim" not in labels
    assert _node(flow, "Ponytail").evidence == ["ocr-0001"]
    assert _edge(flow, "Ponytail", "Evidence Pack").evidence == ["ocr-0001"]
    assert _edge(flow, "URL acquisition", "transcript extraction").evidence == [
        "seg_000001"
    ]


def test_knowledge_flow_merges_duplicate_labels_and_combines_evidence() -> None:
    transcript = Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text="Transcript -> Knowledge Flow",
            )
        ],
    )
    visual_records = VisualRecordSet(
        records=[
            VisualRecord(
                id="ocr-0001",
                timestamp_seconds=2.0,
                frame_id="frame-0001",
                kind="ocr",
                text="Frames -> Knowledge Flow",
                score=VisualEvidenceScore(
                    keep=True,
                    novelty_score=0.7,
                    overlap_score=0.2,
                    reason="new visual tokens add grounded evidence",
                ),
            )
        ]
    )

    flow = extract_knowledge_flow(transcript, visual_records)

    knowledge_flow = _node(flow, "Knowledge Flow")
    assert knowledge_flow.evidence == ["seg_000001", "ocr-0001"]
    assert len([node for node in flow.nodes if node.label == "Knowledge Flow"]) == 1


def test_knowledge_flow_extracts_ordered_prose_steps() -> None:
    transcript = Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text=(
                    "First acquire the URL. Then extract subtitles. "
                    "Finally summarize the content."
                ),
            )
        ],
    )

    flow = extract_knowledge_flow(transcript)

    assert _edge(flow, "acquire the URL", "extract subtitles").evidence == [
        "seg_000001"
    ]
    assert _edge(flow, "extract subtitles", "summarize the content").evidence == [
        "seg_000001"
    ]


def test_knowledge_flow_extracts_numbered_steps() -> None:
    transcript = Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text=(
                    "Step 1: download media. Step 2: transcribe audio. "
                    "Step 3: extract frames."
                ),
            )
        ],
    )

    flow = extract_knowledge_flow(transcript)

    assert _edge(flow, "download media", "transcribe audio").evidence == [
        "seg_000001"
    ]
    assert _edge(flow, "transcribe audio", "extract frames").evidence == [
        "seg_000001"
    ]


def test_knowledge_flow_extracts_input_output_phrase() -> None:
    transcript = Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text="The input is video URL. The output is knowledge-flow pack.",
            )
        ],
    )

    flow = extract_knowledge_flow(transcript)

    assert _edge(flow, "video URL", "knowledge-flow pack").evidence == [
        "seg_000001"
    ]


def test_knowledge_flow_extracts_before_after_dependencies() -> None:
    transcript = Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text=(
                    "Before summarizing, extract the transcript. "
                    "After extracting frames, run OCR."
                ),
            )
        ],
    )

    flow = extract_knowledge_flow(transcript)

    assert _edge(flow, "extract the transcript", "summarizing").evidence == [
        "seg_000001"
    ]
    assert _edge(flow, "extracting frames", "run OCR").evidence == ["seg_000001"]


def test_knowledge_flow_extracts_takes_produces_io_variant() -> None:
    transcript = Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text=(
                    "The workflow takes a video URL and produces "
                    "a knowledge-flow pack."
                ),
            )
        ],
    )

    flow = extract_knowledge_flow(transcript)

    assert _edge(flow, "video URL", "knowledge-flow pack").evidence == [
        "seg_000001"
    ]


def test_knowledge_flow_extracts_pipeline_consists_of_list() -> None:
    transcript = Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text=(
                    "The pipeline consists of download media, "
                    "transcribe audio, and extract frames."
                ),
            )
        ],
    )

    flow = extract_knowledge_flow(transcript)

    assert _edge(flow, "download media", "transcribe audio").evidence == [
        "seg_000001"
    ]
    assert _edge(flow, "transcribe audio", "extract frames").evidence == [
        "seg_000001"
    ]


def _node(flow: KnowledgeFlow, label: str):
    for node in flow.nodes:
        if node.label == label:
            return node
    raise AssertionError(f"missing node {label}")


def _edge(flow: KnowledgeFlow, source: str, target: str):
    for edge in flow.edges:
        source_node = next(node for node in flow.nodes if node.id == edge.source)
        target_node = next(node for node in flow.nodes if node.id == edge.target)
        if source_node.label == source and target_node.label == target:
            return edge
    raise AssertionError(f"missing edge {source} -> {target}")
