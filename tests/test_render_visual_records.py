from __future__ import annotations

from vctx.chunking import ChunkSet, TranscriptChunk
from vctx.models import SourceRef
from vctx.models.metadata import VideoMetadata
from vctx.models.visual import VisualEvidenceScore, VisualRecord, VisualRecordSet
from vctx.render.markdown import render_context_markdown
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment


def test_render_context_omits_low_novelty_visual_records() -> None:
    markdown = render_context_markdown(
        _metadata(),
        _transcript(),
        _chunks(),
        VisualRecordSet(
            records=[
                VisualRecord(
                    id="description-0001",
                    timestamp_seconds=12.0,
                    frame_id="frame-0001",
                    kind="description",
                    text="The image shows the same pipeline.",
                    score=VisualEvidenceScore(
                        keep=False,
                        novelty_score=0.1,
                        overlap_score=0.9,
                        reason="mostly overlaps nearby transcript",
                    ),
                ),
                VisualRecord(
                    id="ocr-0001",
                    timestamp_seconds=12.0,
                    frame_id="frame-0001",
                    kind="ocr",
                    text="URL -> Transcript -> Frames -> Summary",
                    score=VisualEvidenceScore(
                        keep=True,
                        novelty_score=0.72,
                        overlap_score=0.2,
                        reason="new visual tokens add grounded evidence",
                    ),
                ),
            ]
        ),
    )

    assert "The image shows the same pipeline" not in markdown
    assert '<visual_ref id="frame-0001" timestamp="00:00:12">' in markdown
    assert "URL -&gt; Transcript -&gt; Frames -&gt; Summary" in markdown
    assert '<ocr novelty="0.72">' in markdown


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
                id="seg-1",
                start=10.0,
                end=14.0,
                text="This diagram shows the same pipeline.",
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
                start=10.0,
                end=14.0,
                text="This diagram shows the same pipeline.",
                segment_ids=["seg-1"],
                char_count=38,
                approx_token_count=8,
            )
        ],
    )
