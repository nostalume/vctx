from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from vctx.config import YtDlpSourceOptions
from vctx.errors import NoTranscriptError
from vctx.source.session import (
    EffectReceipt,
    MediaAsset,
    MediaPermit,
    MediaRequest,
    ObservePermit,
    Revision,
    SourceCapability,
    SourceRecord,
    SourceRef,
    SubtitlePermit,
    VideoMetadata,
)
from vctx.transcript import (
    MAX_SUBTITLE_BYTES,
    TranscriptPayload,
    TranscriptProvenance,
    UnknownLanguage,
    decode_subtitle,
)

SUPPORTED_TRANSCRIPT_SUFFIXES: dict[str, Literal["srt", "vtt"]] = {".srt": "srt", ".vtt": "vtt"}
SUPPORTED_MEDIA_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a"}
VIDEO_SUFFIXES = {".mp4", ".webm"}


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        with self.path.open("rb") as stream:
            original = stream.read(MAX_SUBTITLE_BYTES + 1)
        payload = TranscriptPayload(
            text=decode_subtitle(original),
            original_bytes=original,
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
            self.receipts.append(EffectReceipt(operation="media", status="failed", purpose="input"))
            raise NoTranscriptError("no media found for input")
        capabilities: set[Literal["audio", "video"]] = (
            {"audio"} if suffix in AUDIO_SUFFIXES else {"audio", "video"}
        )
        asset = MediaAsset(
            id=f"local__{self.path.stem}",
            source=SourceRef(kind="file", value=str(self.path)),
            local_path=self.path,
            container=suffix.removeprefix("."),
            purpose="input",
            profile=None,
            format_id="local",
            provider="local-file",
            capabilities=capabilities,
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
        return "exact" if path.suffix.lower() in supported else "unsupported"

    def observe(
        self, value: str, *, permit: ObservePermit, options: YtDlpSourceOptions
    ) -> LocalFileSession:
        del permit, options
        path = Path(value)
        identity = sha256(str(path.resolve()).casefold().encode()).hexdigest()[:12]
        metadata = VideoMetadata(
            id=f"local__{identity}",
            source=SourceRef(kind="file", value=str(path)),
            title=path.stem,
            raw_provider="local-file",
        )
        suffix = path.suffix.lower()
        capabilities: set[SourceCapability] = (
            {"subtitle"}
            if suffix in SUPPORTED_TRANSCRIPT_SUFFIXES
            else {"audio"}
            if suffix in AUDIO_SUFFIXES
            else {"audio", "video"}
        )
        return LocalFileSession(
            path=path,
            record=SourceRecord(
                source_id=metadata.id,
                revision=Revision(kind="immutable", value=_file_digest(path)),
                observed_at=datetime.now(UTC),
                metadata=metadata,
                source_capabilities=capabilities,
            ),
            receipts=[EffectReceipt(operation="observe", status="succeeded")],
        )
