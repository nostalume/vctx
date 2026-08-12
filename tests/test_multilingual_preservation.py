from __future__ import annotations

from vctx.projection import Renderer
from vctx.source.session import SourceRef, VideoMetadata
from vctx.transcript import (
    ChunkOptions,
    DetectedLanguage,
    Transcript,
    TranscriptProvenance,
    TranscriptSegment,
    chunk_transcript,
    normalize_transcript,
)
from vctx.visual.evidence import CaptureEvidence, Evidence, Observation


def test_native_non_english_transcript_text_survives_normalize_chunk_and_render() -> None:
    native_text = "这是一个关于流程图的例子：输入数据，然后生成摘要。"
    transcript = Transcript(
        source_id="video-zh",
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
    renderer = Renderer(_metadata(), clean, chunks)
    context = renderer.context()
    readable = renderer.read()

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
        source_id="video-ja",
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
    evidence = Evidence(
        source_id="video-ja",
        captures=[
            CaptureEvidence(
                id="frame-0001",
                requested_seconds=1.0,
                actual_seconds=1.0,
                artifact_path="frames/frame-0001.png",
                ocr=Observation(status="ok", text=visual_text, provider="rapidocr"),
                vision=Observation(status="not_requested"),
            )
        ],
    )

    context = Renderer(_metadata(), transcript, chunks, evidence).context()

    assert visual_text in context
    assert "accuracy speed memory" not in context.lower()
    assert "Translated" not in context


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        id="video-multilingual",
        source=SourceRef(kind="file", value="lecture.mp4"),
        title="Native source video",
        duration_seconds=30.0,
    )
