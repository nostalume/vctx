from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from vctx.transcript import LanguageEvidence


class LocalSource(BaseModel):
    kind: Literal["local"] = "local"
    path: Path


class UrlSource(BaseModel):
    kind: Literal["url"] = "url"
    url: str
    network: Literal["allowed", "offline-blocked"]


SourceAcquisition = Annotated[LocalSource | UrlSource, Field(discriminator="kind")]


class LocalTranscript(BaseModel):
    kind: Literal["local_transcript"] = "local_transcript"
    path: Path | None = None
    format: Literal["vtt", "srt", "json", "plain", "unknown"]


class SourceSubtitle(BaseModel):
    kind: Literal["source_subtitle"] = "source_subtitle"
    subtitle_kind: Literal["official_subtitles", "automatic_subtitles"]
    language: LanguageEvidence
    format: Literal["vtt", "srt", "json", "plain", "unknown"]
    provider: str


class AsrNeeded(BaseModel):
    kind: Literal["asr_needed"] = "asr_needed"
    reason: str


class TranscriptUnavailable(BaseModel):
    kind: Literal["transcript_unavailable"] = "transcript_unavailable"
    reason: str


TranscriptAcquisition = Annotated[
    LocalTranscript | SourceSubtitle | AsrNeeded | TranscriptUnavailable,
    Field(discriminator="kind"),
]


class LocalMedia(BaseModel):
    kind: Literal["local_media"] = "local_media"
    path: Path
    media_type: Literal["audio", "video", "unknown"]


class SourceMedia(BaseModel):
    kind: Literal["source_media"] = "source_media"
    cache_path: Path
    media_type: Literal["audio", "video", "unknown"]
    provider: str | None = None


class MediaNotNeeded(BaseModel):
    kind: Literal["media_not_needed"] = "media_not_needed"
    reason: str


class MediaUnavailable(BaseModel):
    kind: Literal["media_unavailable"] = "media_unavailable"
    reason: str


MediaAcquisition = Annotated[
    LocalMedia | SourceMedia | MediaNotNeeded | MediaUnavailable,
    Field(discriminator="kind"),
]


def transcript_acquisition_detail(acquisition: TranscriptAcquisition) -> str:
    if acquisition.kind == "local_transcript":
        return f"local transcript: {acquisition.format}"
    if acquisition.kind == "source_subtitle":
        language = _language_label(acquisition.language)
        return f"{acquisition.provider}:{acquisition.subtitle_kind}:{language}:{acquisition.format}"
    if acquisition.kind == "asr_needed":
        return f"ASR needed: {acquisition.reason}"
    return acquisition.reason


def media_acquisition_detail(acquisition: MediaAcquisition) -> str:
    if acquisition.kind == "local_media":
        return f"local media: {acquisition.media_type}: {acquisition.path}"
    if acquisition.kind == "source_media":
        return f"source media: {acquisition.media_type}: {acquisition.cache_path}"
    return acquisition.reason


def _language_label(language: LanguageEvidence) -> str:
    if language.kind == "detected":
        return language.code
    if language.kind == "mixed" and language.items:
        return "+".join(item.code for item in language.items)
    return language.kind
