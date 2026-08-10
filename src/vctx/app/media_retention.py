from __future__ import annotations

import hashlib
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Literal

from vctx.models.manifest import ArtifactRef, SourceMediaReceipt
from vctx.models.media import MediaAsset


def materialize_source_media(
    media: MediaAsset | None, out_dir: Path, *, retain: bool
) -> tuple[list[ArtifactRef], list[SourceMediaReceipt]]:
    if media is None:
        return [], []
    purpose = _purpose(media)
    if not retain:
        _remove_output_owned_media(media, out_dir)
        return [], [
            SourceMediaReceipt(
                source=media.source,
                media_id=media.id,
                purpose=purpose,
                selected_format=media.container,
                retained=False,
                omission_reason="disabled by --no-retain-media/output.retain_media",
            )
        ]

    source = media.local_path.resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"source media is missing or empty: {source}")
    out_root = out_dir.resolve()
    if source.is_relative_to(out_root):
        retained = source
    else:
        media_dir = out_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", media.id).strip("._") or "source"
        retained = media_dir / f"{safe_id}__{source.name}"
        if not retained.resolve().is_relative_to(out_root):
            raise RuntimeError("refusing source-media path outside the output pack")
        shutil.copy2(source, retained)
    relative = retained.resolve().relative_to(out_root).as_posix()
    size = retained.stat().st_size
    refs = [
        ArtifactRef(
            kind="source_media",
            path=relative,
            media_type=mimetypes.guess_type(retained.name)[0] or "application/octet-stream",
        )
    ]
    sidecar = retained.with_name(f"{media.id}.vctx-media.json")
    if sidecar.is_file():
        refs.append(
            ArtifactRef(
                kind="source_media_metadata",
                path=sidecar.resolve().relative_to(out_root).as_posix(),
                media_type="application/json",
            )
        )
    return refs, [
        SourceMediaReceipt(
            source=media.source,
            media_id=media.id,
            purpose=purpose,
            selected_format=media.container,
            retained=True,
            path=relative,
            bytes=size,
            sha256=_sha256(retained),
        )
    ]


def _purpose(media: MediaAsset) -> Literal["input", "asr", "visual"]:
    if media.kind == "downloaded_asr_audio":
        return "asr"
    if media.kind == "downloaded_visual_video":
        return "visual"
    return "input"


def _remove_output_owned_media(media: MediaAsset, out_dir: Path) -> None:
    out_root = out_dir.resolve()
    source = media.local_path.resolve()
    if not source.is_relative_to(out_root):
        return
    for path in (source, source.with_name(f"{media.id}.vctx-media.json")):
        if path.is_file() and path.resolve().is_relative_to(out_root):
            path.unlink()
    media_dir = out_dir / "media"
    if media_dir.is_dir() and not any(media_dir.iterdir()):
        media_dir.rmdir()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
