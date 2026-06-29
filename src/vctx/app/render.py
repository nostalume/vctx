from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ValidationError

from vctx.chunking import ChunkSet
from vctx.errors import VctxError
from vctx.models.metadata import VideoMetadata
from vctx.render.markdown import (
    render_context_markdown,
    render_readable_markdown,
    render_transcript_markdown,
)
from vctx.transcript import Transcript


class RenderFormat(StrEnum):
    CONTEXT = "context"
    READABLE = "readable"
    TRANSCRIPT = "transcript"


class RenderInputError(VctxError):
    exit_code = 4


class RenderWriteError(VctxError):
    exit_code = 1


def write_render_file(
    *,
    metadata_path: Path,
    transcript_path: Path,
    chunks_path: Path | None,
    out_path: Path,
    format: RenderFormat,
) -> Path:
    metadata = _load_model(metadata_path, VideoMetadata)
    transcript = _load_model(transcript_path, Transcript)

    if format == RenderFormat.TRANSCRIPT:
        content = render_transcript_markdown(metadata, transcript)
    else:
        if chunks_path is None:
            raise RenderInputError(f"--chunks is required for {format} render")
        chunks = _load_model(chunks_path, ChunkSet)
        if format == RenderFormat.CONTEXT:
            content = render_context_markdown(metadata, transcript, chunks)
        elif format == RenderFormat.READABLE:
            content = render_readable_markdown(metadata, transcript, chunks)
        else:
            raise RenderInputError(f"unsupported render format: {format}")

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RenderWriteError(f"failed to write render: {out_path}: {exc}") from exc
    return out_path


def _load_model[TModel: BaseModel](path: Path, model_type: type[TModel]) -> TModel:
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise RenderInputError(f"invalid {model_type.__name__} file: {path}") from exc
