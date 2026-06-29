from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

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


class TranscriptProvenance(BaseModel):
    method: Literal["official_subtitles", "automatic_subtitles", "local_file", "asr"]
    language: str | None = None
    language_evidence: LanguageEvidence = Field(
        default_factory=lambda: UnknownLanguage(reason="not reported")
    )
    format: Literal["vtt", "srt", "json", "plain", "unknown"] = "unknown"
    provider: str | None = None

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
    video_id: str
    provenance: TranscriptProvenance
    segments: list[TranscriptSegment]


class TranscriptPayload(BaseModel):
    text: str
    format: Literal["vtt", "srt", "json", "plain", "unknown"]
    provenance: TranscriptProvenance

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
