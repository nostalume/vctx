from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

from vctx.errors import InvalidTranscriptError
from vctx.source.session import MediaAsset


@dataclass(frozen=True)
class AsrInput:
    media: MediaAsset
    origin: float
    end: float
    source_duration: float
    temporary: Path | None = None


def admit_interval(
    media: MediaAsset, requested: tuple[float, float | None] | None
) -> tuple[float, float] | None:
    if requested is None:
        return None
    duration = media.duration_seconds
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise InvalidTranscriptError("bounded ASR requires a finite source duration")
    start, requested_end = requested
    end = duration if requested_end is None else requested_end
    if not (0 <= start < end <= duration):
        raise InvalidTranscriptError(
            f"ASR interval must satisfy 0 <= start < end <= {duration:.3f}"
        )
    return start, end


@contextmanager
def open_asr_input(
    media: MediaAsset,
    requested: tuple[float, float | None] | None,
    temp_root: Path,
) -> Iterator[AsrInput]:
    interval = admit_interval(media, requested)
    if interval is None:
        duration = media.duration_seconds or 0.0
        yield AsrInput(media, 0.0, duration, duration)
        return
    start, end = interval
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = temp_root / f"{uuid4().hex}.wav"
    try:
        _decode_interval(media.local_path, temporary, start, end)
        derived = media.model_copy(
            update={
                "id": f"{media.id}:interval:{start:.3f}:{end:.3f}",
                "local_path": temporary,
                "container": "wav",
                "duration_seconds": end - start,
                "media_type": "audio",
                "capabilities": {"audio"},
                "sha256": None,
            }
        )
        yield AsrInput(derived, start, end, media.duration_seconds or end, temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_interval(source: Path, target: Path, start: float, end: float) -> None:
    import av

    samples_written = 0
    total_samples = round((end - start) * 16_000)
    with av.open(str(source)) as incoming, av.open(str(target), "w", format="wav") as outgoing:
        streams = list(incoming.streams.audio)
        if not streams:
            raise InvalidTranscriptError("ASR source contains no audio stream")
        stream = streams[0]
        if stream.time_base is None:
            raise InvalidTranscriptError("ASR audio stream has no time base")
        incoming.seek(
            int(start / float(stream.time_base)),
            stream=stream,
            any_frame=False,
            backward=True,
        )
        encoder = outgoing.add_stream("pcm_s16le", rate=16_000, layout="mono")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
        for frame in incoming.decode(stream):
            for admitted in resampler.resample(frame):
                if admitted.time_base is None:
                    raise InvalidTranscriptError("decoded ASR audio has no time base")
                timestamp = (
                    float(admitted.pts * admitted.time_base)
                    if admitted.pts is not None
                    else start + samples_written / 16_000
                )
                if timestamp >= end or samples_written >= total_samples:
                    break
                left = max(0, math.ceil((start - timestamp) * 16_000))
                right = min(admitted.samples, math.ceil((end - timestamp) * 16_000))
                right = min(right, left + total_samples - samples_written)
                if right <= left or timestamp + admitted.samples / 16_000 <= start:
                    continue
                array = admitted.to_ndarray()[:, left:right]
                output = av.AudioFrame.from_ndarray(array, format="s16", layout="mono")
                output.sample_rate = 16_000
                output.time_base = Fraction(1, 16_000)
                output.pts = samples_written
                samples_written += output.samples
                for packet in encoder.encode(output):
                    outgoing.mux(packet)
            if samples_written >= total_samples:
                break
        for packet in encoder.encode(None):
            outgoing.mux(packet)
    if samples_written == 0:
        target.unlink(missing_ok=True)
        raise InvalidTranscriptError("ASR interval contains no decodable audio")
