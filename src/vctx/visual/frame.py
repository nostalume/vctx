from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from vctx.source.session import MediaAsset
from vctx.visual.plan import FrameProcessor, PlannedFrame

if TYPE_CHECKING:
    import av
    from PIL.Image import Image as PillowImage

_FRAME_ID = re.compile(r"frame-\d{4}")
_LONG_EDGE = 2560
_RECIPE = "pyav-display-v1"

type FrameMissReason = Literal["target_out_of_range"]
type Orientation = Literal[0, 90, 180, 270]


class FrameError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    id: str
    path: Path
    requested_seconds: float
    actual_seconds: float
    original_size: tuple[int, int]
    orientation: Orientation
    size: tuple[int, int]
    sha256: str
    bytes: int
    request_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    processors: tuple[FrameProcessor, ...]
    priority: float
    recipe: str = _RECIPE


@dataclass(frozen=True)
class FrameMiss:
    id: str
    requested_seconds: float
    reason: FrameMissReason
    request_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrameBatch:
    frames: tuple[Frame, ...]
    misses: tuple[FrameMiss, ...]


def capture(
    media: MediaAsset,
    requests: Sequence[PlannedFrame],
    out_dir: Path,
) -> FrameBatch:
    """Resolve transcript-planned targets into display-corrected PNG captures."""

    ordered = sorted(requests, key=lambda request: (request.target_seconds, request.id))
    _validate_requests(ordered)
    frames_dir = out_dir / "frames"
    try:
        import av

        container = av.open(str(media.local_path))
    except (ImportError, OSError, ValueError) as exc:
        raise FrameError(f"cannot open visual media {media.local_path}: {exc}") from exc
    frames: list[Frame] = []
    misses: list[FrameMiss] = []
    try:
        if not container.streams.video:
            raise FrameError(f"visual media has no video stream: {media.local_path}")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for request in ordered:
            decoded = _visible_frame(container, stream, request.target_seconds)
            if decoded is None:
                misses.append(_miss(request))
                continue
            frames.append(_publish(decoded, request, frames_dir))
    except FrameError:
        _discard(frames, frames_dir)
        raise
    except Exception as exc:  # pragma: no cover - PyAV codec boundary
        _discard(frames, frames_dir)
        raise FrameError(f"cannot decode visual media {media.local_path}: {exc}") from exc
    finally:
        container.close()
    return FrameBatch(tuple(frames), tuple(misses))


def _validate_requests(requests: Sequence[PlannedFrame]) -> None:
    ids = [request.id for request in requests]
    if any(_FRAME_ID.fullmatch(frame_id) is None for frame_id in ids):
        raise FrameError("frame request ids must match frame-NNNN")
    if len(ids) != len(set(ids)):
        raise FrameError("frame request ids must be unique")


def _visible_frame(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    target: float,
) -> av.VideoFrame | None:
    if target < 0 or stream.time_base is None:
        return None
    if stream.duration is not None and target >= float(stream.duration * stream.time_base):
        return None
    offset = max(0, int(target / float(stream.time_base)))
    container.seek(offset, stream=stream, backward=True, any_frame=False)
    undelimited_pts: int | None = None
    for decoded in container.decode(stream):
        start = decoded.time
        if start is None:
            continue
        if start > target:
            if undelimited_pts is None:
                return None
            del decoded
            return _decode_pts(container, stream, undelimited_pts)
        duration = _duration(decoded)
        if duration is not None and start <= target < start + duration:
            return decoded
        if start <= target:
            undelimited_pts = decoded.pts
    if undelimited_pts is None:
        return None
    del decoded
    return _decode_pts(container, stream, undelimited_pts)


def _decode_pts(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    pts: int | None,
) -> av.VideoFrame | None:
    if pts is None:
        return None
    container.seek(pts, stream=stream, backward=True, any_frame=False)
    for decoded in container.decode(stream):
        if decoded.pts == pts:
            return decoded
        if decoded.pts is not None and decoded.pts > pts:
            return None
    return None


def _duration(frame: av.VideoFrame) -> float | None:
    if frame.duration <= 0 or frame.time_base is None:
        return None
    return float(frame.duration * frame.time_base)


def _publish(frame: av.VideoFrame, request: PlannedFrame, frames_dir: Path) -> Frame:
    orientation = _orientation(frame)
    original_size = (frame.width, frame.height)
    image = _bounded(_display_image(frame, orientation))
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"{request.id}.png"
    temporary = frames_dir / f".{request.id}.png.tmp"
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise FrameError(f"cannot publish {path}: {exc}") from exc
    body = path.read_bytes()
    actual = frame.time
    if actual is None:
        raise FrameError(f"decoded frame {request.id} has no presentation timestamp")
    return Frame(
        id=request.id,
        path=path,
        requested_seconds=request.target_seconds,
        actual_seconds=round(actual, 6),
        original_size=original_size,
        orientation=orientation,
        size=image.size,
        sha256=hashlib.sha256(body).hexdigest(),
        bytes=len(body),
        request_ids=tuple(request.request_ids),
        segment_ids=tuple(request.segment_ids),
        claim_ids=tuple(request.claim_ids),
        processors=tuple(request.processors),
        priority=request.priority,
    )


def _orientation(frame: av.VideoFrame) -> Orientation:
    display = frame.side_data.get("DISPLAYMATRIX")
    if display is None:
        return 0
    body = bytes(display)
    if len(body) != 36:
        raise FrameError("unsupported display matrix size")
    matrix = struct.unpack("=9i", body)
    determinant = matrix[0] * matrix[4] - matrix[1] * matrix[3]
    if determinant <= 0:
        raise FrameError("mirrored display matrices are unsupported")
    angle = (-math.degrees(math.atan2(matrix[1], matrix[0]))) % 360
    nearest = int(round(angle / 90) * 90) % 360
    delta = abs((angle - nearest + 180) % 360 - 180)
    if delta > 0.5:
        raise FrameError(f"unsupported display rotation {angle:.3f} degrees")
    return cast(Orientation, nearest)


def _display_image(frame: av.VideoFrame, orientation: Orientation) -> PillowImage:
    from PIL import Image

    image = frame.to_image()
    transpose = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }.get(orientation)
    return image.transpose(transpose) if transpose is not None else image


def _bounded(image: PillowImage) -> PillowImage:
    from PIL import Image

    width, height = image.size
    edge = max(width, height)
    if edge <= _LONG_EDGE:
        return image
    scale = _LONG_EDGE / edge
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _miss(request: PlannedFrame) -> FrameMiss:
    return FrameMiss(
        id=request.id,
        requested_seconds=request.target_seconds,
        reason="target_out_of_range",
        request_ids=tuple(request.request_ids),
        segment_ids=tuple(request.segment_ids),
        claim_ids=tuple(request.claim_ids),
    )


def _discard(frames: Sequence[Frame], frames_dir: Path) -> None:
    for frame in frames:
        frame.path.unlink(missing_ok=True)
    if frames_dir.exists() and not any(frames_dir.iterdir()):
        frames_dir.rmdir()
