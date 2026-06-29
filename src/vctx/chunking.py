from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from vctx.errors import EmptyChunksError
from vctx.transcript import Transcript, TranscriptSegment


class ChunkOptions(BaseModel):
    max_chars: int = 6000
    max_seconds: int | None = None


class TranscriptChunk(BaseModel):
    id: str
    start: float
    end: float | None
    text: str
    segment_ids: list[str]
    char_count: int
    approx_token_count: int


class ChunkSet(BaseModel):
    video_id: str
    strategy: str
    chunks: list[TranscriptChunk]


def chunk_transcript(transcript: Transcript, options: ChunkOptions) -> ChunkSet:
    chunks: list[TranscriptChunk] = []
    pending: list[TranscriptSegment] = []

    for segment in transcript.segments:
        if pending and should_flush(pending, segment, options):
            chunks.append(build_chunk(len(chunks) + 1, pending))
            pending = []
        pending.append(segment)

    if pending:
        chunks.append(build_chunk(len(chunks) + 1, pending))

    if not chunks:
        raise EmptyChunksError(f"chunking produced no chunks for {transcript.video_id}")

    return ChunkSet(video_id=transcript.video_id, strategy="chars-v1", chunks=chunks)


def should_flush(
    pending: Sequence[TranscriptSegment], next_segment: TranscriptSegment, options: ChunkOptions
) -> bool:
    current_text = " ".join(segment.text for segment in pending)
    if len(current_text) + 1 + len(next_segment.text) > options.max_chars:
        return True
    if options.max_seconds is not None:
        start = pending[0].start
        end = next_segment.end if next_segment.end is not None else next_segment.start
        return end - start > options.max_seconds
    return False


def build_chunk(index: int, segments: Sequence[TranscriptSegment]) -> TranscriptChunk:
    text = " ".join(segment.text for segment in segments).strip()
    end = segments[-1].end if segments[-1].end is not None else segments[-1].start
    return TranscriptChunk(
        id=f"chunk_{index:04d}",
        start=segments[0].start,
        end=end,
        text=text,
        segment_ids=[segment.id for segment in segments],
        char_count=len(text),
        approx_token_count=max(1, len(text) // 4),
    )
