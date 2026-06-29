from __future__ import annotations

from vctx.models.metadata import VideoMetadata
from vctx.sources.detect import detect_source_adapter


def inspect_metadata(value: str) -> VideoMetadata:
    adapter = detect_source_adapter(value)
    return adapter.extract_metadata(value)


def render_metadata_text(metadata: VideoMetadata) -> str:
    lines: list[str] = [
        f"id: {metadata.id}",
        f"source_type: {metadata.source_type}",
        f"source: {metadata.source.value}",
    ]
    optional_fields = {
        "title": metadata.title,
        "uploader": metadata.uploader,
        "duration_seconds": metadata.duration_seconds,
        "webpage_url": metadata.webpage_url,
        "language": metadata.language,
        "extractor": metadata.extractor,
        "raw_provider": metadata.raw_provider,
    }
    for key, value in optional_fields.items():
        if value is not None:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"
