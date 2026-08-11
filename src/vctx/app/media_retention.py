from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import shutil
from pathlib import Path

from vctx.artifact.manifest import (
    AudioSourceAsset,
    InputSourceAsset,
    SourceAsset,
    SubtitleSourceAsset,
    VideoSourceAsset,
)
from vctx.errors import CacheError
from vctx.source.session import MediaAsset
from vctx.transcript import TranscriptPayload


def materialize_source_assets(
    media: MediaAsset | None,
    subtitle: TranscriptPayload | None,
    out_dir: Path,
    *,
    retain: bool,
) -> list[SourceAsset]:
    assets: list[SourceAsset] = []
    published: list[Path] = []
    try:
        if subtitle is not None:
            assets.append(_subtitle_asset(subtitle, out_dir, retain=retain, published=published))
        if media is not None:
            assets.append(_media_asset(media, out_dir, retain=retain, published=published))
    except (OSError, CacheError) as exc:
        for path in published:
            path.unlink(missing_ok=True)
        raise CacheError(f"source asset materialization failed: {exc}") from exc
    return assets


def _subtitle_asset(
    payload: TranscriptPayload, out_dir: Path, *, retain: bool, published: list[Path]
) -> SubtitleSourceAsset:
    language = payload.provenance.language
    safe_language = _safe_token(language or "und")
    extension = {"plain": "txt", "unknown": "txt"}.get(payload.format, payload.format)
    relative = Path(f"subtitle.{safe_language}.{extension}")
    if not retain:
        return SubtitleSourceAsset(
            retained=False,
            format=payload.format,
            language=language,
            omission_reason="disabled by --no-retain-media/output.retain_media",
        )
    body = payload.text.encode("utf-8")
    final = _publish_bytes(body, out_dir, relative)
    published.append(final)
    return SubtitleSourceAsset(
        retained=True,
        path=relative.as_posix(),
        media_type=mimetypes.guess_type(final.name)[0] or "text/plain",
        bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        format=payload.format,
        language=language,
    )


def _media_asset(
    media: MediaAsset, out_dir: Path, *, retain: bool, published: list[Path]
) -> SourceAsset:
    kind = {"input": "input", "asr": "audio", "visual": "video"}[media.purpose]
    extension = _safe_token(media.container if media.container != "unknown" else "bin")
    relative = Path(f"{kind}.{extension}")
    path: str | None = None
    media_type: str | None = None
    size: int | None = None
    digest: str | None = None
    omission: str | None = None
    if not retain:
        omission = "disabled by --no-retain-media/output.retain_media"
    else:
        source = media.local_path.resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise CacheError(f"source media is missing or empty: {source}")
        final, size, digest = _copy_verified(source, out_dir, relative)
        published.append(final)
        path = relative.as_posix()
        media_type = mimetypes.guess_type(final.name)[0] or "application/octet-stream"
    common = {
        "retained": retain,
        "path": path,
        "media_type": media_type,
        "bytes": size,
        "sha256": digest,
        "omission_reason": omission,
        "selected_format": media.container if media.format_id == "local" else media.format_id,
    }
    if kind == "audio":
        return AudioSourceAsset.model_validate(common)
    if kind == "video":
        return VideoSourceAsset.model_validate(
            {**common, "requested_profile": media.profile or "auto"}
        )
    return InputSourceAsset.model_validate(common)


def _publish_bytes(body: bytes, out_dir: Path, relative: Path) -> Path:
    final = _contained_path(out_dir, relative)
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = final.with_name(f".{final.name}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        if _sha256(temp) != hashlib.sha256(body).hexdigest():
            raise CacheError("subtitle copy failed integrity verification")
        temp.replace(final)
    finally:
        temp.unlink(missing_ok=True)
    return final


def _copy_verified(source: Path, out_dir: Path, relative: Path) -> tuple[Path, int, str]:
    final = _contained_path(out_dir, relative)
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = final.with_name(f".{final.name}.tmp")
    source_digest = _sha256(source)
    try:
        with source.open("rb") as reader, temp.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if _sha256(temp) != source_digest:
            raise CacheError("media copy failed integrity verification")
        temp.replace(final)
    finally:
        temp.unlink(missing_ok=True)
    return final, final.stat().st_size, source_digest


def _contained_path(out_dir: Path, relative: Path) -> Path:
    root = out_dir.resolve()
    final = (out_dir / relative).resolve()
    if not final.is_relative_to(root):
        raise CacheError("refusing source asset path outside the output pack")
    return final


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return token.lower() or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
