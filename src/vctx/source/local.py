from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from vctx.config import YtDlpSourceOptions
from vctx.errors import NoTranscriptError
from vctx.source.session import (
    EffectReceipt,
    MediaAsset,
    MediaPermit,
    MediaProfile,
    MediaRequest,
    ObservePermit,
    Revision,
    SourceRecord,
    SourceRef,
    SubtitlePermit,
    VideoMetadata,
)
from vctx.transcript import TranscriptPayload, TranscriptProvenance, UnknownLanguage

SUPPORTED_TRANSCRIPT_SUFFIXES: dict[str, Literal["srt", "vtt"]] = {".srt": "srt", ".vtt": "vtt"}
SUPPORTED_MEDIA_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a"}
VIDEO_SUFFIXES = {".mp4", ".webm"}


class LocalMediaAsset(BaseModel):
    id: str
    source: SourceRef
    local_path: Path
    container: str = "unknown"
    duration_seconds: float | None = None
    media_type: Literal["audio", "video", "unknown"] = "unknown"
    purpose: Literal["input", "asr", "visual"] = "input"
    profile: MediaProfile | None = None
    format_id: str = "local"
    provider: str = "local-file"


@dataclass
class LocalFileSession:
    name = "local-file"
    path: Path
    record: SourceRecord
    receipts: list[EffectReceipt] = field(default_factory=list)

    def transcript(self, *, permit: SubtitlePermit) -> TranscriptPayload:
        del permit
        fmt = SUPPORTED_TRANSCRIPT_SUFFIXES.get(self.path.suffix.lower())
        if fmt is None:
            self.receipts.append(
                EffectReceipt(operation="subtitle", status="failed", purpose="transcript")
            )
            raise NoTranscriptError("no transcript found for media input")
        payload = TranscriptPayload(
            text=self.path.read_text(encoding="utf-8"),
            format=fmt,
            provenance=TranscriptProvenance(
                method="local_file",
                language_evidence=UnknownLanguage(reason="local transcript language not detected"),
                format=fmt,
                provider="local-file",
            ),
        )
        self.receipts.append(
            EffectReceipt(
                operation="subtitle",
                status="succeeded",
                purpose="transcript",
                selected_policy=fmt,
            )
        )
        return payload

    def media(self, *, request: MediaRequest, permit: MediaPermit) -> MediaAsset:
        del request, permit
        suffix = self.path.suffix.lower()
        if suffix not in SUPPORTED_MEDIA_SUFFIXES:
            self.receipts.append(
                EffectReceipt(operation="media", status="failed", purpose="input")
            )
            raise NoTranscriptError("no media found for input")
        media_type: Literal["audio", "video", "unknown"] = "unknown"
        if suffix in AUDIO_SUFFIXES:
            media_type = "audio"
        elif suffix in VIDEO_SUFFIXES:
            media_type = "video"
        asset = LocalMediaAsset(
            id=f"local__{self.path.stem}",
            source=SourceRef(kind="file", value=str(self.path)),
            local_path=self.path,
            media_type=media_type,
            container=suffix.removeprefix("."),
        )
        self.receipts.append(
            EffectReceipt(
                operation="media",
                status="succeeded",
                purpose="input",
                selected_policy=asset.container,
            )
        )
        return asset


class LocalFileSourceAdapter:
    name = "local-file"

    def claim(self, value: str) -> Literal["exact", "unsupported"]:
        path = Path(value)
        supported = SUPPORTED_TRANSCRIPT_SUFFIXES.keys() | SUPPORTED_MEDIA_SUFFIXES
        return "exact" if path.exists() and path.suffix.lower() in supported else "unsupported"

    def observe(
        self, value: str, *, permit: ObservePermit, options: YtDlpSourceOptions
    ) -> LocalFileSession:
        del permit, options
        path = Path(value)
        identity = sha256(str(path.resolve()).casefold().encode()).hexdigest()[:12]
        metadata = VideoMetadata(
            id=f"local__{identity}",
            source_type="local-file",
            source=SourceRef(kind="file", value=str(path)),
            title=path.stem,
            raw_provider="local-file",
        )
        return LocalFileSession(
            path=path,
            record=SourceRecord(
                source_id=metadata.id,
                revision=Revision(kind="immutable", value=sha256(path.read_bytes()).hexdigest()),
                observed_at=datetime.now(UTC),
                metadata=metadata,
                has_subtitles=path.suffix.lower() in SUPPORTED_TRANSCRIPT_SUFFIXES,
                has_media=path.suffix.lower() in SUPPORTED_MEDIA_SUFFIXES,
            ),
            receipts=[EffectReceipt(operation="observe", status="succeeded")],
        )
