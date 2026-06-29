from __future__ import annotations

from pydantic import BaseModel

from vctx.models import SourceRef


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
