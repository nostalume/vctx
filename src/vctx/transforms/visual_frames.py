from __future__ import annotations

import subprocess
from pathlib import Path

from vctx.models.media import MediaAsset
from vctx.models.visual import EssentialVisualCase, FrameAsset
from vctx.transforms.visual_planning import Evidence, VisualAction


def extract_frames(
    media_asset: MediaAsset,
    sample_action: VisualAction,
    frames_dir: Path,
) -> list[FrameAsset]:
    """Extract deterministic frame assets for the current visual capture slice."""

    frames_dir.mkdir(parents=True, exist_ok=True)
    strategy = sample_action.params.strategy or "cover"
    if strategy == "essential_cases":
        cases = _sample_cases(sample_action)
        budget = sample_action.params.budget or len(cases)
        if cases:
            return [
                _extract_case_frame(media_asset, frames_dir, case, index)
                for index, case in enumerate(cases[:budget], start=1)
            ]
    budget = sample_action.params.budget or 1
    if strategy == "cover" or budget <= 1:
        return [_extract_cover_frame(media_asset, frames_dir)]
    return [_extract_cover_frame(media_asset, frames_dir)]


def _sample_cases(sample_action: VisualAction) -> list[EssentialVisualCase]:
    return list(sample_action.params.cases)


def _extract_case_frame(
    media_asset: MediaAsset,
    frames_dir: Path,
    case: EssentialVisualCase,
    index: int,
) -> FrameAsset:
    frame_id = f"frame-{index:04d}"
    frame_path = frames_dir / f"{frame_id}.png"
    _run_ffmpeg_frame_extract(media_asset, frame_path, case.timestamp_seconds)
    return FrameAsset(
        id=frame_id,
        timestamp_seconds=case.timestamp_seconds,
        path=frame_path,
        source="transcript_anchor",
        evidence=[
            Evidence(kind="transcript", name=case.case_type, weight=case.priority),
            Evidence(kind="transcript", name="essential-case", weight=case.priority),
        ],
    )


def _extract_cover_frame(media_asset: MediaAsset, frames_dir: Path) -> FrameAsset:
    timestamp = _cover_timestamp(media_asset.duration_seconds)
    frame_path = frames_dir / "frame-0001.png"
    _run_ffmpeg_frame_extract(media_asset, frame_path, timestamp)
    return FrameAsset(
        id="frame-0001",
        timestamp_seconds=timestamp,
        path=frame_path,
        source="cover",
        evidence=[Evidence(kind="probe", name="cover-frame", weight=1.0)],
    )


def _run_ffmpeg_frame_extract(
    media_asset: MediaAsset, frame_path: Path, timestamp: float
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(media_asset.local_path),
        "-frames:v",
        "1",
        str(frame_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"failed to extract frame with ffmpeg: {exc}") from exc


def _cover_timestamp(duration_seconds: float | None) -> float:
    if duration_seconds is None or duration_seconds <= 2:
        return 0.0
    return min(max(duration_seconds / 2, 0.0), 30.0)
