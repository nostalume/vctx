from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vctx.models import SourceRef
from vctx.models.media import LocalMediaAsset
from vctx.models.visual import EssentialVisualCase
from vctx.transforms.visual_frames import extract_frames
from vctx.transforms.visual_planning import VisualAction, VisualActionParams


def test_extract_frames_uses_essential_case_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, timeout
        commands.append(command)
        Path(command[-1]).write_bytes(b"fake png bytes")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    media = LocalMediaAsset(
        id="media-1",
        source=SourceRef(kind="file", value=str(tmp_path / "lecture.mp4")),
        local_path=tmp_path / "lecture.mp4",
        media_type="video",
        duration_seconds=200.0,
    )
    media.local_path.write_bytes(b"fake video")
    frames = extract_frames(
        media,
        VisualAction.sample(
            params=VisualActionParams(
                strategy="essential_cases",
                cases=[
                    EssentialVisualCase(
                        segment_id="seg_000001",
                        timestamp_seconds=12.0,
                        case_type="diagram",
                        priority=0.9,
                        reason="diagram referenced",
                        actions=["describe", "capture"],
                    ),
                    EssentialVisualCase(
                        segment_id="seg_000002",
                        timestamp_seconds=32.5,
                        case_type="formula",
                        priority=0.8,
                        reason="formula referenced",
                        actions=["ocr", "describe", "capture"],
                    ),
                ],
            ),
        ),
        tmp_path / "frames",
    )

    assert [frame.id for frame in frames] == ["frame-0001", "frame-0002"]
    assert [frame.timestamp_seconds for frame in frames] == [12.0, 32.5]
    assert [frame.source for frame in frames] == ["transcript_anchor", "transcript_anchor"]
    assert [command[3] for command in commands] == ["12.000", "32.500"]
    assert frames[0].evidence[0].name == "diagram"
    assert frames[1].evidence[0].name == "formula"
