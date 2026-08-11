from __future__ import annotations

from pathlib import Path

from vctx.config import PrepareRequest, WorkflowProfile, load_resolved_config
from vctx.net import HttpxNetRuntime
from vctx.source.admission import open_source, select_source
from vctx.source.session import ObservePermit, VideoMetadata
from vctx.source.store import SourceStore


def inspect_metadata(
    value: str,
    *,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
    offline: bool | None = None,
) -> VideoMetadata:
    resolved = load_resolved_config(
        PrepareRequest(
            inputs=[value],
            out_dir=Path("."),
            workflow=WorkflowProfile.METADATA,
            config_path=config_path,
            cache_dir=cache_dir,
            offline=offline,
        )
    )
    with HttpxNetRuntime() as net:
        session = open_source(
            select_source(value),
            value,
            permit=ObservePermit(
                operation="metadata",
                network="denied" if resolved.runtime.offline else "allowed",
            ),
            options=resolved.source.yt_dlp,
            net=net,
            store=SourceStore(resolved.cache.source_dir),
        )
    return session.record.metadata


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
