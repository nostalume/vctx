from __future__ import annotations

from typing import Protocol

from vctx.config import YtDlpSourceOptions
from vctx.errors import UnsupportedSourceError
from vctx.io import Cache
from vctx.models.media import MediaAsset, MediaFetchRequest
from vctx.models.metadata import VideoMetadata
from vctx.sources.local_file_source import LocalFileSourceAdapter
from vctx.sources.ytdlp_source import YtDlpSourceAdapter
from vctx.transcript import TranscriptPayload


class SourceAdapter(Protocol):
    name: str

    def can_handle(self, value: str) -> bool: ...

    def extract_metadata(self, value: str) -> VideoMetadata: ...

    def extract_transcript(
        self,
        value: str,
        *,
        cache: Cache,
        source_options: YtDlpSourceOptions | None = None,
    ) -> TranscriptPayload: ...

    def extract_media(
        self,
        value: str,
        *,
        request: MediaFetchRequest,
    ) -> MediaAsset: ...


def detect_source_adapter(value: str) -> SourceAdapter:
    adapters: list[SourceAdapter] = [LocalFileSourceAdapter(), YtDlpSourceAdapter()]
    for adapter in adapters:
        if adapter.can_handle(value):
            return adapter
    raise UnsupportedSourceError(f"unsupported source: {value}")
