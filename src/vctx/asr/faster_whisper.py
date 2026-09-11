from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vctx.transcript import TranscriptSegment

_DLL_HANDLES: list[object] = []
_WINDOWS_DLLS = (
    ("nvidia-cublas-cu12", "nvidia/cublas/bin", "*.dll"),
    ("nvidia-cudnn-cu12", "nvidia/cudnn/bin", "cudnn64_9.dll"),
)


def bundled_cuda_state() -> Literal["bundled", "missing", "not-applicable"]:
    if os.name != "nt":
        return "not-applicable"
    try:
        roots = [
            Path(str(importlib.metadata.distribution(name).locate_file(relative)))
            for name, relative, _pattern in _WINDOWS_DLLS
        ]
    except importlib.metadata.PackageNotFoundError:
        return "missing"
    required = (roots[0] / "cublas64_12.dll", roots[1] / "cudnn64_9.dll")
    return "bundled" if all(path.is_file() for path in required) else "missing"


def load_bundled_cuda() -> bool:
    """Load optional environment-local CUDA DLLs without changing process PATH."""

    if os.name != "nt":
        return False
    if _DLL_HANDLES:
        return True
    loaded: list[object] = []
    for distribution, relative, pattern in _WINDOWS_DLLS:
        try:
            root = Path(str(importlib.metadata.distribution(distribution).locate_file(relative)))
        except importlib.metadata.PackageNotFoundError:
            return False
        for path in root.glob(pattern):
            try:
                loaded.append(ctypes.WinDLL(str(path.resolve())))
            except OSError:
                return False
    _DLL_HANDLES.extend(loaded)
    return bool(loaded)


class WhisperTranscriber(Protocol):
    def transcribe(self, path: str, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class WhisperApi:
    model: Callable[..., object]
    pipeline: Callable[..., object] | None


class WhisperInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    language: str | None
    language_probability: float | None = Field(strict=True, ge=0, le=1)
    duration: float = Field(strict=True, ge=0)
    duration_after_vad: float = Field(strict=True, ge=0)


class _Segment(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    start: float = Field(strict=True)
    end: float = Field(strict=True)
    text: str


@dataclass(frozen=True)
class WhisperPass:
    segments: list[TranscriptSegment]
    info: WhisperInfo
    batch_size: int | None = None


class InvalidVendorResponse(ValueError):
    pass


class InvalidVendorTimestamps(ValueError):
    pass


class InferenceStreamError(RuntimeError):
    def __init__(self, cause: Exception, *, emitted: bool) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.emitted = emitted


def import_api() -> WhisperApi:
    module = importlib.import_module("faster_whisper")
    model = getattr(module, "WhisperModel", None)
    if not callable(model):
        raise InvalidVendorResponse("faster-whisper lacks callable WhisperModel")
    pipeline = getattr(module, "BatchedInferencePipeline", None)
    if pipeline is not None and not callable(pipeline):
        raise InvalidVendorResponse("faster-whisper BatchedInferencePipeline is not callable")
    return WhisperApi(model=model, pipeline=pipeline)


def admit_model(raw: object, *, device: str, compute: str) -> tuple[WhisperTranscriber, str, str]:
    transcriber = admit_transcriber(raw, label="model")
    engine = getattr(raw, "model", None)
    return (
        transcriber,
        str(getattr(engine, "device", device)),
        str(getattr(engine, "compute_type", compute)),
    )


def admit_transcriber(raw: object, *, label: str) -> WhisperTranscriber:
    if not callable(getattr(raw, "transcribe", None)):
        raise InvalidVendorResponse(f"faster-whisper {label} lacks callable transcribe")
    return cast(WhisperTranscriber, raw)


def admit_pass(raw: object) -> WhisperPass:
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise InvalidVendorResponse("faster-whisper transcribe must return segments and info")
    raw_segments, raw_info = raw
    if not isinstance(raw_segments, Iterable):
        raise InvalidVendorResponse("faster-whisper segments are not iterable")
    try:
        info = WhisperInfo.model_validate(raw_info)
    except ValidationError as exc:
        raise InvalidVendorResponse(str(exc)) from exc
    segments: list[TranscriptSegment] = []
    iterator = iter(raw_segments)
    while True:
        try:
            raw_segment = next(iterator)
        except StopIteration:
            return WhisperPass(segments, info)
        except Exception as exc:
            raise InferenceStreamError(exc, emitted=bool(segments)) from exc
        try:
            segment = _Segment.model_validate(raw_segment)
        except ValidationError as exc:
            raise InvalidVendorResponse(str(exc)) from exc
        text = segment.text.strip()
        if not text:
            continue
        start, end = float(segment.start), float(segment.end)
        if not all(math.isfinite(value) and value >= 0 for value in (start, end)):
            raise InvalidVendorTimestamps("ASR timestamps must be finite and non-negative")
        if end < start or segments and start < segments[-1].start:
            raise InvalidVendorTimestamps("ASR timestamps must have ordered bounds")
        segments.append(
            TranscriptSegment(
                id=f"seg_{len(segments) + 1:06d}",
                start=round(start, 3),
                end=round(end, 3),
                text=text,
            )
        )


def is_oom(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "out of memory" in message or "cuda" in message and "memory" in message
