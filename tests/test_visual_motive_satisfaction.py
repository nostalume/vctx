from __future__ import annotations

from vctx.models.visual import VisualOperationMotive, VisualRecord
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment
from vctx.transforms.visual_evidence import score_visual_records


def test_formula_ocr_motive_is_missed_when_ocr_has_no_math_signal() -> None:
    records = score_visual_records(
        [
            VisualRecord(
                id="ocr-0001",
                timestamp_seconds=10.0,
                frame_id="frame-0001",
                kind="ocr",
                text="hello world",
            )
        ],
        _transcript(),
        motives=[
            VisualOperationMotive(
                operation="ocr",
                reason="formula_or_equation",
                segment_id="seg_000001",
                timestamp_seconds=10.0,
                priority=0.9,
                explanation="formula on slide",
            )
        ],
    )

    assert [(check.operation, check.reason, check.status) for check in records.satisfaction] == [
        ("ocr", "formula_or_equation", "missed")
    ]
    assert "math signal" in records.satisfaction[0].detail


def test_chart_capture_motive_is_satisfied_by_capture_record() -> None:
    records = score_visual_records(
        [
            VisualRecord(
                id="capture-0001",
                timestamp_seconds=10.0,
                frame_id="frame-0001",
                kind="capture",
                artifact_path="visual/frames/frame-0001.png",
            )
        ],
        _transcript(),
        motives=[
            VisualOperationMotive(
                operation="capture",
                reason="chart_visual_reference",
                segment_id="seg_000001",
                timestamp_seconds=10.0,
                priority=0.9,
                explanation="chart visual reference",
            )
        ],
    )

    assert [(check.operation, check.reason, check.status) for check in records.satisfaction] == [
        ("capture", "chart_visual_reference", "satisfied")
    ]
    assert records.records[0].kind == "capture"


def _transcript() -> Transcript:
    return Transcript(
        video_id="fixture-video",
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=8.0,
                end=12.0,
                text="Here is the visual reference.",
            )
        ],
        provenance=TranscriptProvenance(method="local_file", format="vtt"),
    )
