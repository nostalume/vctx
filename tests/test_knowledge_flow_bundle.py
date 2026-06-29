from __future__ import annotations

import json

from vctx.chunking import ChunkSet, TranscriptChunk
from vctx.models import SourceRef
from vctx.models.metadata import VideoMetadata
from vctx.models.visual import VisualEvidenceScore, VisualRecord, VisualRecordSet
from vctx.render.bundle import render_artifact_bundle
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment
from vctx.transforms.knowledge_flow import extract_knowledge_flow


def test_render_bundle_writes_knowledge_flow_json_with_visual_evidence() -> None:
    transcript = _transcript()
    visual_records = VisualRecordSet(
        records=[
            VisualRecord(
                id="ocr-0001",
                timestamp_seconds=2.0,
                frame_id="frame-0001",
                kind="ocr",
                text="Ponytail -> Evidence Pack",
                score=VisualEvidenceScore(
                    keep=True,
                    novelty_score=0.8,
                    overlap_score=0.1,
                    reason="new visual tokens add grounded evidence",
                ),
            )
        ]
    )
    knowledge_flow = extract_knowledge_flow(transcript, visual_records)

    bundle = render_artifact_bundle(
        metadata=_metadata(),
        raw_transcript=transcript,
        clean_transcript=transcript,
        chunks=_chunks(),
        formats={"json"},
        visual_records=visual_records,
        knowledge_flow=knowledge_flow,
    )

    artifact = bundle.get("knowledge_flow")
    assert artifact is not None
    assert artifact.name == "knowledge_flow.json"
    payload = json.loads(artifact.content)
    assert {node["label"] for node in payload["nodes"]} >= {"Ponytail", "Evidence Pack"}
    assert payload["edges"][0]["evidence"] == ["seg_000001"]


def test_render_bundle_keeps_native_text_when_output_language_is_requested() -> None:
    transcript = Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="ja", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text="入力データを読み込み、要約を生成します。",
            )
        ],
    )
    chunks = ChunkSet(
        video_id="video-1",
        strategy="test",
        chunks=[
            TranscriptChunk(
                id="chunk-1",
                start=0.0,
                end=5.0,
                text="入力データを読み込み、要約を生成します。",
                segment_ids=["seg_000001"],
                char_count=22,
                approx_token_count=11,
            )
        ],
    )

    bundle = render_artifact_bundle(
        metadata=_metadata(),
        raw_transcript=transcript,
        clean_transcript=transcript,
        chunks=chunks,
        formats={"readable"},
        output_language="en",
    )

    artifact = bundle.get("readable")
    assert artifact is not None
    assert "入力データを読み込み、要約を生成します。" in artifact.content
    assert "Input data" not in artifact.content
    assert "Translated" not in artifact.content


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
                text="URL -> Transcript",
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
                text="URL -> Transcript",
                segment_ids=["seg_000001"],
                char_count=17,
                approx_token_count=3,
            )
        ],
    )
