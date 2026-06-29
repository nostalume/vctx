from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from vctx.chunking import ChunkOptions, chunk_transcript
from vctx.errors import InvalidTranscriptError, VctxError
from vctx.io import model_to_json
from vctx.transcript import Transcript


class ChunkWriteError(VctxError):
    exit_code = 1


def write_chunk_file(
    transcript_path: Path,
    out_path: Path,
    *,
    max_chars: int = 6000,
    max_seconds: int | None = None,
) -> Path:
    try:
        transcript = Transcript.model_validate_json(transcript_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise InvalidTranscriptError(f"invalid transcript file: {transcript_path}") from exc

    chunks = chunk_transcript(
        transcript,
        ChunkOptions(max_chars=max_chars, max_seconds=max_seconds),
    )
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(model_to_json(chunks), encoding="utf-8")
    except OSError as exc:
        raise ChunkWriteError(f"failed to write chunks: {out_path}: {exc}") from exc
    return out_path
