from __future__ import annotations

from vctx.models.visual import (
    EssentialCaseSupplement,
    EssentialCaseSupplementEvidence,
    EssentialVisualCase,
)
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment
from vctx.transforms.visual_cases import merge_essential_case_supplement


def test_merge_essential_case_supplement_keeps_valid_segment_cases_only() -> None:
    deterministic = [
        EssentialVisualCase(
            segment_id="seg_000001",
            timestamp_seconds=1.0,
            case_type="diagram",
            priority=0.9,
            reason="deterministic diagram cue",
            actions=["describe", "capture"],
        )
    ]
    supplement = EssentialCaseSupplement(
        cases=[
            EssentialVisualCase(
                segment_id="seg_000002",
                timestamp_seconds=4.0,
                case_type="screen_demo",
                priority=1.5,
                reason="model saw screen demo cue",
                actions=["ocr", "capture"],
            ),
            EssentialVisualCase(
                segment_id="unknown-seg",
                timestamp_seconds=8.0,
                case_type="table",
                priority=0.7,
                reason="unsupported segment id",
                actions=["ocr", "capture"],
            ),
        ],
        evidence=EssentialCaseSupplementEvidence(source_segment_ids=["seg_000002", "unknown-seg"]),
    )

    merged = merge_essential_case_supplement(deterministic, supplement, _transcript())

    assert [(case.segment_id, case.case_type) for case in merged] == [
        ("seg_000001", "diagram"),
        ("seg_000002", "screen_demo"),
    ]
    assert merged[1].priority == 1.0
    assert all(case.segment_id != "unknown-seg" for case in merged)


def test_merge_essential_case_supplement_dedupes_same_segment_and_case_type() -> None:
    deterministic = [
        EssentialVisualCase(
            segment_id="seg_000001",
            timestamp_seconds=1.0,
            case_type="diagram",
            priority=0.9,
            reason="deterministic diagram cue",
            actions=["describe", "capture"],
        )
    ]
    supplement = EssentialCaseSupplement(
        cases=[
            EssentialVisualCase(
                segment_id="seg_000001",
                timestamp_seconds=1.2,
                case_type="diagram",
                priority=0.8,
                reason="duplicate model cue",
                actions=["describe", "capture"],
            )
        ],
        evidence=EssentialCaseSupplementEvidence(source_segment_ids=["seg_000001"]),
    )

    merged = merge_essential_case_supplement(deterministic, supplement, _transcript())

    assert [(case.segment_id, case.case_type, case.reason) for case in merged] == [
        ("seg_000001", "diagram", "deterministic diagram cue")
    ]


def _transcript() -> Transcript:
    return Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=2.0,
                text="This diagram explains the request flow.",
            ),
            TranscriptSegment(
                id="seg_000002",
                start=3.0,
                end=5.0,
                text="Now I open the terminal for a demo.",
            ),
        ],
    )
