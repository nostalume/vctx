from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from vctx.config import MediaProfile, YtDlpSourceOptions
from vctx.models import SourceRef


class ReuseHit(BaseModel):
    kind: Literal["reuse_hit"] = "reuse_hit"
    sidecar_path: Path


class ReuseMiss(BaseModel):
    kind: Literal["reuse_miss"] = "reuse_miss"
    reason: Literal[
        "missing_sidecar",
        "missing_file",
        "source_mismatch",
        "purpose_mismatch",
        "profile_mismatch",
        "overwrite",
        "not_checked",
    ]


ReuseDecision = Annotated[ReuseHit | ReuseMiss, Field(discriminator="kind")]


class FetchRequestCore(BaseModel):
    source_url: str
    output_dir: Path
    temp_dir: Path
    source_options: YtDlpSourceOptions
    reuse: bool = True


class AsrAudioFetchRequest(FetchRequestCore):
    kind: Literal["asr_audio"] = "asr_audio"


class VisualVideoFetchRequest(FetchRequestCore):
    kind: Literal["visual_video"] = "visual_video"
    profile: MediaProfile


MediaFetchRequest = Annotated[
    AsrAudioFetchRequest | VisualVideoFetchRequest,
    Field(discriminator="kind"),
]


class MediaPathCore(BaseModel):
    id: str
    source: SourceRef
    local_path: Path
    container: str = "unknown"
    duration_seconds: float | None = None


class LocalMediaAsset(MediaPathCore):
    kind: Literal["local_media"] = "local_media"
    media_type: Literal["audio", "video", "unknown"] = "unknown"
    provider: Literal["local-file"] = "local-file"


class DownloadedMediaCore(MediaPathCore):
    provider: Literal["yt-dlp"] = "yt-dlp"
    format_id: str
    reuse: ReuseDecision


class DownloadedAsrAudioAsset(DownloadedMediaCore):
    kind: Literal["downloaded_asr_audio"] = "downloaded_asr_audio"
    media_type: Literal["audio"] = "audio"


class DownloadedVisualVideoAsset(DownloadedMediaCore):
    kind: Literal["downloaded_visual_video"] = "downloaded_visual_video"
    media_type: Literal["video"] = "video"
    profile: MediaProfile


MediaAsset = Annotated[
    LocalMediaAsset | DownloadedAsrAudioAsset | DownloadedVisualVideoAsset,
    Field(discriminator="kind"),
]


class SidecarCore(BaseModel):
    schema_version: Literal[1] = 1
    source: SourceRef
    media_id: str
    local_path: Path
    container: str


class LocalMediaSidecar(SidecarCore):
    kind: Literal["local_media"] = "local_media"
    media_type: Literal["audio", "video", "unknown"]


class DownloadedSidecarCore(SidecarCore):
    provider: Literal["yt-dlp"] = "yt-dlp"
    extractor: str
    webpage_url: str
    format_id: str


class DownloadedAsrAudioSidecar(DownloadedSidecarCore):
    kind: Literal["downloaded_asr_audio"] = "downloaded_asr_audio"
    media_type: Literal["audio"] = "audio"


class DownloadedVisualVideoSidecar(DownloadedSidecarCore):
    kind: Literal["downloaded_visual_video"] = "downloaded_visual_video"
    media_type: Literal["video"] = "video"
    profile: MediaProfile


MediaAssetSidecar = Annotated[
    LocalMediaSidecar | DownloadedAsrAudioSidecar | DownloadedVisualVideoSidecar,
    Field(discriminator="kind"),
]
