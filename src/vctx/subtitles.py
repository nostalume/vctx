from __future__ import annotations

import io

import srt
import webvtt

from vctx.errors import InvalidTranscriptError
from vctx.transcript import Transcript, TranscriptPayload, TranscriptSegment


def parse_transcript_payload(payload: TranscriptPayload, *, video_id: str) -> Transcript:
    if payload.format == "srt":
        return parse_srt(payload, video_id=video_id)
    if payload.format == "vtt":
        return parse_webvtt(payload, video_id=video_id)
    raise InvalidTranscriptError(f"unsupported transcript format: {payload.format}")


def parse_srt(payload: TranscriptPayload, *, video_id: str) -> Transcript:
    segments = [
        TranscriptSegment(
            id=f"seg_{index:06d}",
            start=item.start.total_seconds(),
            end=item.end.total_seconds(),
            text=item.content,
            source_id=str(item.index),
        )
        for index, item in enumerate(srt.parse(payload.text), start=1)
    ]
    return Transcript(video_id=video_id, provenance=payload.provenance, segments=segments)


def parse_webvtt(payload: TranscriptPayload, *, video_id: str) -> Transcript:
    captions = webvtt.from_buffer(io.StringIO(payload.text)).captions
    segments = [
        TranscriptSegment(
            id=f"seg_{index:06d}",
            start=_timestamp_to_seconds(caption.start),
            end=_timestamp_to_seconds(caption.end),
            text=caption.text,
            source_id=str(index),
        )
        for index, caption in enumerate(captions, start=1)
    ]
    return Transcript(video_id=video_id, provenance=payload.provenance, segments=segments)


def _timestamp_to_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + seconds
