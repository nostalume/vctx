from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vctx.visual.evidence import Evidence, Observation
from vctx.visual.frame import Frame, FrameBatch, FrameMiss
from vctx.visual.plan import EvidencePlan, PlannedFrame


def _planned(frame_id: str, target: float) -> PlannedFrame:
    return PlannedFrame(
        id=frame_id,
        target_seconds=target,
        segment_ids=[f"seg-{frame_id[-4:]}"],
        processors=["ocr"],
        request_ids=[f"request-{frame_id[-4:]}"],
        claim_ids=[],
        priority=0.8,
    )


def test_evidence_preserves_misses_states_and_processor_identity(tmp_path: Path) -> None:
    path = tmp_path / "frames" / "frame-0001.png"
    path.parent.mkdir()
    path.write_bytes(b"png")
    first = _planned("frame-0001", 1.0)
    second = _planned("frame-0002", 9.0)
    batch = FrameBatch(
        frames=(
            Frame(
                id=first.id,
                path=path,
                requested_seconds=first.target_seconds,
                actual_seconds=1.1,
                original_size=(16, 9),
                orientation=0,
                size=(16, 9),
                sha256=hashlib.sha256(b"png").hexdigest(),
                bytes=3,
                request_ids=tuple(first.request_ids),
                segment_ids=tuple(first.segment_ids),
                claim_ids=tuple(first.claim_ids),
                processors=tuple(first.processors),
                priority=first.priority,
            ),
        ),
        misses=(
            FrameMiss(
                id=second.id,
                requested_seconds=second.target_seconds,
                reason="target_out_of_range",
                request_ids=tuple(second.request_ids),
                segment_ids=tuple(second.segment_ids),
                claim_ids=tuple(second.claim_ids),
            ),
        ),
    )
    plan = EvidencePlan(source_id="video", frames=[first, second])

    evidence = Evidence.from_observations(
        plan,
        batch,
        tmp_path,
        ocr={
            first.id: Observation(
                status="empty", provider="rapidocr:onnx", detail="nothing visible"
            )
        },
        vision={},
    )

    assert evidence.captures[0].ocr.provider == "rapidocr:onnx"
    assert evidence.captures[0].vision.status == "not_requested"
    assert [(miss.id, miss.reason) for miss in evidence.misses] == [
        ("frame-0002", "target_out_of_range")
    ]
    assert evidence.status == "partial"


def test_evidence_rejects_capture_links_that_differ_from_plan(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    path.write_bytes(b"png")
    planned = _planned("frame-0001", 1.0)
    frame = Frame(
        id=planned.id,
        path=path,
        requested_seconds=planned.target_seconds,
        actual_seconds=1.0,
        original_size=(1, 1),
        orientation=0,
        size=(1, 1),
        sha256=hashlib.sha256(b"png").hexdigest(),
        bytes=3,
        request_ids=("request-wrong",),
        segment_ids=tuple(planned.segment_ids),
        claim_ids=(),
        processors=tuple(planned.processors),
        priority=planned.priority,
    )

    with pytest.raises(ValueError, match="does not match evidence plan"):
        Evidence.from_observations(
            EvidencePlan(source_id="video", frames=[planned]),
            FrameBatch(frames=(frame,), misses=()),
            tmp_path,
            ocr={},
            vision={},
        )
