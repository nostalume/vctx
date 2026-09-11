from __future__ import annotations

import io
import re
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vctx.errors import EmptyChunksError, InvalidTranscriptError

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
MAX_SUBTITLE_BYTES = 8 * 1024 * 1024


def decode_subtitle(data: bytes, encoding: str = "utf-8") -> str:
    if len(data) > MAX_SUBTITLE_BYTES:
        raise InvalidTranscriptError("subtitle exceeds 8 MiB encoded-size limit")
    return data.decode(encoding)


LanguageEvidenceSource = Literal["asr", "subtitle", "metadata", "media", "ocr", "vlm"]


class AutoLanguage(BaseModel):
    kind: Literal["auto"] = "auto"


class DetectedLanguage(BaseModel):
    kind: Literal["detected"] = "detected"
    code: str
    source: LanguageEvidenceSource
    confidence: float | None = None


class MixedLanguage(BaseModel):
    kind: Literal["mixed"] = "mixed"
    items: list[DetectedLanguage]


class UnknownLanguage(BaseModel):
    kind: Literal["unknown"] = "unknown"
    reason: str


LanguageEvidence = Annotated[
    AutoLanguage | DetectedLanguage | MixedLanguage | UnknownLanguage,
    Field(discriminator="kind"),
]


def detected_language(
    code: str | None, *, source: LanguageEvidenceSource, confidence: float | None = None
) -> LanguageEvidence:
    if code is None or not code.strip():
        return UnknownLanguage(reason="not reported")
    return DetectedLanguage(code=code.strip(), source=source, confidence=confidence)


class TranscriptSegment(BaseModel):
    id: str
    start: float
    end: float | None = None
    text: str
    source_id: str | None = None


class AsrProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    revision: str | None = None
    device: str
    compute_type: str
    attempted_devices: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    cpu_threads: int | None = None
    batch_size: int | None = None
    vad: bool
    confirmation: bool = False
    source_duration: float | None = None
    interval_start: float | None = None
    interval_end: float | None = None
    processed_duration: float | None = None
    speech_duration: float | None = None
    language: str | None = None
    language_confidence: float | None = None
    timestamp_method: Literal["segment-milliseconds"] = "segment-milliseconds"


class TranscriptProvenance(BaseModel):
    method: Literal["official_subtitles", "automatic_subtitles", "local_file", "asr"]
    language: str | None = None
    language_evidence: LanguageEvidence = Field(
        default_factory=lambda: UnknownLanguage(reason="not reported")
    )
    format: Literal["vtt", "srt", "json", "plain", "unknown"] = "unknown"
    provider: str | None = None
    asr: AsrProvenance | None = None

    @model_validator(mode="after")
    def mirror_legacy_language_into_tagged_evidence(self) -> TranscriptProvenance:
        if self.language is None:
            return self
        if self.language_evidence.kind != "unknown":
            return self
        source: LanguageEvidenceSource = "subtitle" if "subtitles" in self.method else "asr"
        self.language_evidence = DetectedLanguage(code=self.language, source=source)
        return self


class Transcript(BaseModel):
    source_id: str
    provenance: TranscriptProvenance
    segments: list[TranscriptSegment]


class TranscriptPayload(BaseModel):
    text: str
    original_bytes: bytes | None = None
    format: Literal["vtt", "srt", "json", "plain", "unknown"]
    provenance: TranscriptProvenance

    @model_validator(mode="after")
    def admit_encoded_size(self) -> TranscriptPayload:
        size = len(self.original_bytes or self.text.encode())
        if size > MAX_SUBTITLE_BYTES:
            raise ValueError("subtitle exceeds 8 MiB encoded-size limit")
        return self

    def provenance_label(self) -> str:
        parts: list[str] = []
        if self.provenance.provider:
            parts.append(self.provenance.provider)
        parts.append(self.provenance.method)
        language_label = _language_label(self.provenance.language_evidence)
        if language_label is not None:
            parts.append(language_label)
        parts.append(self.format)
        return ":".join(parts)


class TranscriptReady(BaseModel):
    kind: Literal["ready"] = "ready"
    transcript: Transcript


class TranscriptUnavailable(BaseModel):
    kind: Literal["unavailable"] = "unavailable"
    reason: str


class TranscriptNoSpeech(BaseModel):
    kind: Literal["no_speech"] = "no_speech"
    reason: str


TranscriptOutcome = Annotated[
    TranscriptReady | TranscriptUnavailable | TranscriptNoSpeech,
    Field(discriminator="kind"),
]


class ChunkOptions(BaseModel):
    max_chars: int = 6000
    max_seconds: int | None = None


class TranscriptChunk(BaseModel):
    id: str
    start: float
    end: float | None
    text: str
    segment_ids: list[str]
    char_count: int
    approx_token_count: int


class ChunkSet(BaseModel):
    source_id: str
    strategy: str
    chunks: list[TranscriptChunk]


def _language_label(evidence: LanguageEvidence) -> str | None:
    if evidence.kind == "detected":
        return evidence.code
    if evidence.kind == "mixed" and evidence.items:
        return "+".join(item.code for item in evidence.items)
    return None


def reassign_segment_ids(segments: Sequence[TranscriptSegment]) -> list[TranscriptSegment]:
    return [
        segment.model_copy(update={"id": f"seg_{index:06d}"})
        for index, segment in enumerate(segments, start=1)
    ]


def strip_subtitle_markup(text: str) -> str:
    return _TAG_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def clean_subtitle_text(text: str) -> str:
    return normalize_whitespace(strip_subtitle_markup(text))


def normalize_transcript(raw: Transcript) -> Transcript:
    cleaned: list[TranscriptSegment] = []
    for segment in raw.segments:
        text = clean_subtitle_text(segment.text)
        if not text:
            continue
        cleaned.append(segment.model_copy(update={"text": text}))
    cleaned.sort(key=lambda segment: segment.start)
    return raw.model_copy(update={"segments": reassign_segment_ids(cleaned)})


def parse_transcript_payload(payload: TranscriptPayload, *, source_id: str) -> Transcript:
    if payload.format == "srt":
        import srt

        items = srt.parse(payload.text)
        segments = [
            TranscriptSegment(
                id=f"seg_{index:06d}",
                start=item.start.total_seconds(),
                end=item.end.total_seconds(),
                text=item.content,
                source_id=str(item.index),
            )
            for index, item in enumerate(items, start=1)
        ]
    elif payload.format == "vtt":
        import webvtt

        captions = webvtt.from_buffer(io.StringIO(payload.text)).captions
        segments = [
            TranscriptSegment(
                id=f"seg_{index:06d}",
                start=_timestamp_to_seconds(caption.start),
                end=_timestamp_to_seconds(caption.end),
                text=caption.text,
                source_id=str(index),
            )
            for index, caption in enumerate(captions, start=1)
        ]
    else:
        raise InvalidTranscriptError(f"unsupported transcript format: {payload.format}")
    return Transcript(source_id=source_id, provenance=payload.provenance, segments=segments)


def chunk_transcript(transcript: Transcript, options: ChunkOptions) -> ChunkSet:
    chunks: list[TranscriptChunk] = []
    pending: list[TranscriptSegment] = []
    for segment in transcript.segments:
        if pending and _should_flush(pending, segment, options):
            chunks.append(_build_chunk(len(chunks) + 1, pending))
            pending = []
        pending.append(segment)
    if pending:
        chunks.append(_build_chunk(len(chunks) + 1, pending))
    if not chunks:
        raise EmptyChunksError(f"chunking produced no chunks for {transcript.source_id}")
    return ChunkSet(source_id=transcript.source_id, strategy="chars-v1", chunks=chunks)


def _timestamp_to_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    return (
        (int(parts[-3]) if len(parts) >= 3 else 0) * 3600
        + (int(parts[-2]) if len(parts) >= 2 else 0) * 60
        + float(parts[-1])
    )


def _should_flush(
    pending: Sequence[TranscriptSegment], next_segment: TranscriptSegment, options: ChunkOptions
) -> bool:
    current_text = " ".join(segment.text for segment in pending)
    if len(current_text) + 1 + len(next_segment.text) > options.max_chars:
        return True
    if options.max_seconds is None:
        return False
    end = next_segment.end if next_segment.end is not None else next_segment.start
    return end - pending[0].start > options.max_seconds


def _build_chunk(index: int, segments: Sequence[TranscriptSegment]) -> TranscriptChunk:
    text = " ".join(segment.text for segment in segments).strip()
    end = segments[-1].end if segments[-1].end is not None else segments[-1].start
    return TranscriptChunk(
        id=f"chunk_{index:04d}",
        start=segments[0].start,
        end=end,
        text=text,
        segment_ids=[segment.id for segment in segments],
        char_count=len(text),
        approx_token_count=max(1, len(text) // 4),
    )
