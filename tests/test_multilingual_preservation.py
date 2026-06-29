from __future__ import annotations

from vctx.chunking import ChunkOptions, chunk_transcript
from vctx.models import SourceRef
from vctx.models.metadata import VideoMetadata
from vctx.models.visual import VisualRecord, VisualRecordSet
from vctx.render.markdown import render_context_markdown, render_readable_markdown
from vctx.transcript import (
    DetectedLanguage,
    Transcript,
    TranscriptProvenance,
    TranscriptSegment,
    normalize_transcript,
)
from vctx.transforms.knowledge_flow import extract_knowledge_flow


def test_native_non_english_transcript_text_survives_normalize_chunk_and_render() -> None:
    native_text = "这是一个关于流程图的例子：输入数据，然后生成摘要。"
    transcript = Transcript(
        video_id="video-zh",
        provenance=TranscriptProvenance(
            method="official_subtitles",
            language_evidence=DetectedLanguage(code="zh-CN", source="subtitle"),
            format="vtt",
            provider="yt-dlp",
        ),
        segments=[
            TranscriptSegment(
                id="raw-1",
                start=0.0,
                end=4.0,
                text=f"<c>{native_text}</c>",
            )
        ],
    )

    clean = normalize_transcript(transcript)
    chunks = chunk_transcript(clean, ChunkOptions(max_chars=200))
    context = render_context_markdown(_metadata(), clean, chunks)
    readable = render_readable_markdown(_metadata(), clean, chunks)

    assert clean.provenance.language_evidence == DetectedLanguage(
        code="zh-CN",
        source="subtitle",
    )
    assert clean.segments[0].text == native_text
    assert chunks.chunks[0].text == native_text
    assert native_text in context
    assert native_text in readable
    assert "Translated" not in context
    assert "Translated" not in readable


def test_native_visual_text_is_rendered_as_source_evidence_without_translation() -> None:
    transcript = Transcript(
        video_id="video-ja",
        provenance=TranscriptProvenance(
            method="local_file",
            language_evidence=DetectedLanguage(code="ja", source="metadata"),
            format="plain",
        ),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=3.0,
                text="画面の表はモデルの比較を示しています。",
            )
        ],
    )
    chunks = chunk_transcript(transcript, ChunkOptions(max_chars=200))
    visual_text = "精度 速度 メモリ使用量"
    visual_records = VisualRecordSet(
        records=[
            VisualRecord(
                id="ocr-0001",
                timestamp_seconds=1.0,
                frame_id="frame-0001",
                kind="ocr",
                text=visual_text,
            )
        ]
    )

    context = render_context_markdown(_metadata(), transcript, chunks, visual_records)

    assert visual_text in context
    assert "accuracy speed memory" not in context.lower()
    assert "Translated" not in context


def test_deterministic_knowledge_flow_preserves_non_english_text_without_chain() -> None:
    native_text = "模型先读取字幕，再结合画面证据，最后输出上下文包。"
    transcript = Transcript(
        video_id="video-zh",
        provenance=TranscriptProvenance(
            method="local_file",
            language_evidence=DetectedLanguage(code="zh-CN", source="metadata"),
            format="plain",
        ),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=5.0,
                text=native_text,
            )
        ],
    )

    flow = extract_knowledge_flow(transcript)

    assert flow.nodes == []
    assert flow.edges == []
    assert transcript.segments[0].text == native_text


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        id="video-multilingual",
        source_type="file",
        source=SourceRef(kind="file", value="lecture.mp4"),
        title="Native source video",
        duration_seconds=30.0,
    )
