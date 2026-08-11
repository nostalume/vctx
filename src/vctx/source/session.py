from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from vctx.transcript import TranscriptPayload

type MediaProfile = Literal["auto", "fast", "balanced", "high"]

class SourceRef(BaseModel):
    kind: Literal["url", "file"]
    value: str


class VideoMetadata(BaseModel):
    id: str
    source_type: str
    source: SourceRef
    title: str | None = None
    uploader: str | None = None
    duration_seconds: float | None = None
    webpage_url: str | None = None
    language: str | None = None
    extractor: str | None = None
    raw_provider: str | None = None

type SourceId = str
class Revision(BaseModel):
    kind: Literal["immutable", "observed"]
    value: str
class SourceRecord(BaseModel):
    source_id: SourceId
    revision: Revision
    observed_at: datetime
    metadata: VideoMetadata
    lifecycle: Literal["finite", "live", "upcoming"] = "finite"
    has_subtitles: bool = False
    has_media: bool = False

class ObservePermit(BaseModel):
    operation: Literal["prepare", "metadata"]
    network: Literal["allowed", "denied"]
class SubtitlePermit(BaseModel):
    network: Literal["allowed", "denied"]
class MediaPermit(BaseModel):
    network: Literal["allowed", "denied"]


class AsrAudioRequest(BaseModel):
    kind: Literal["asr_audio"] = "asr_audio"
    temp_dir: Path | None = None
    refresh: bool = False


class VisualVideoRequest(BaseModel):
    kind: Literal["visual_video"] = "visual_video"
    profile: MediaProfile
    temp_dir: Path | None = None
    refresh: bool = False


type MediaRequest = AsrAudioRequest | VisualVideoRequest


class MediaAsset(Protocol):
    id: str
    source: SourceRef
    local_path: Path
    container: str
    duration_seconds: float | None
    media_type: Literal["audio", "video", "unknown"]
    purpose: Literal["input", "asr", "visual"]
    profile: MediaProfile | None
    format_id: str
    provider: str


class EffectReceipt(BaseModel):
    operation: Literal["observe", "subtitle", "media"]
    status: Literal["cache_hit", "succeeded", "denied", "failed"]
    attempts: int = 0
    detail: str | None = None
    purpose: Literal["transcript", "asr", "visual", "input"] | None = None
    requested_policy: str | None = None
    selected_policy: str | None = None


class SourceSession(Protocol):
    name: str
    record: SourceRecord
    receipts: list[EffectReceipt]

    def transcript(self, *, permit: SubtitlePermit) -> TranscriptPayload: ...

    def media(self, *, request: MediaRequest, permit: MediaPermit) -> MediaAsset: ...
