from __future__ import annotations

from vctx.transcript import (
    DetectedLanguage,
    Transcript,
    TranscriptProvenance,
    TranscriptSegment,
)
from vctx.transforms.visual_cases import (
    EssentialVisualCase,
    dedupe_cases_by_window,
    deterministic_essential_cases,
    uncertain_visual_segments,
)


def _transcript(*segments: TranscriptSegment) -> Transcript:
    return Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", format="json"),
        segments=list(segments),
    )


def test_deterministic_essential_cases_extracts_visual_transcript_cues() -> None:
    cases = deterministic_essential_cases(
        _transcript(
            TranscriptSegment(
                id="seg_000001",
                start=10.0,
                end=14.0,
                text="This architecture diagram shows the request flow.",
            ),
            TranscriptSegment(
                id="seg_000002",
                start=30.0,
                end=35.0,
                text="Now derive the formula shown on the slide.",
            ),
            TranscriptSegment(
                id="seg_000003",
                start=60.0,
                end=62.0,
                text="This part is spoken explanation only.",
            ),
        )
    )

    assert [(case.segment_id, case.case_type, case.timestamp_seconds) for case in cases] == [
        ("seg_000001", "diagram", 12.0),
        ("seg_000002", "formula", 32.5),
    ]
    assert cases[0].actions == ["describe", "capture"]
    assert cases[1].actions == ["ocr", "describe", "capture"]
    assert "architecture diagram" in cases[0].reason


def test_deterministic_essential_cases_extracts_shown_text_cue() -> None:
    cases = deterministic_essential_cases(
        _transcript(
            TranscriptSegment(
                id="seg_000001",
                start=4.0,
                end=8.0,
                text="Look at the slide text.",
            )
        )
    )

    assert [(case.case_type, case.actions, case.priority) for case in cases] == [
        ("slide_title", ["ocr", "capture"], 0.6)
    ]
    assert "shown text" in cases[0].reason


def test_deterministic_essential_cases_extracts_slide_title_cue() -> None:
    cases = deterministic_essential_cases(
        _transcript(
            TranscriptSegment(
                id="seg_000001",
                start=4.0,
                end=8.0,
                text="This title slide introduces the topic.",
            )
        )
    )

    assert [(case.case_type, case.actions, case.priority) for case in cases] == [
        ("slide_title", ["ocr", "capture"], 0.6)
    ]
    assert "slide or section title" in cases[0].reason


def test_deterministic_essential_cases_extracts_visual_summary_cue() -> None:
    cases = deterministic_essential_cases(
        _transcript(
            TranscriptSegment(
                id="seg_000001",
                start=4.0,
                end=8.0,
                text="The overview slide shows the main takeaways.",
            )
        )
    )

    assert [(case.case_type, case.actions, case.priority) for case in cases] == [
        ("visual_summary", ["ocr", "describe", "capture"], 0.65)
    ]
    assert "summary or overview" in cases[0].reason




def test_deterministic_essential_cases_treats_chart_as_capture_only() -> None:
    cases = deterministic_essential_cases(
        _transcript(
            TranscriptSegment(
                id="seg_000001",
                start=10.0,
                end=14.0,
                text="This chart compares child mortality across countries.",
            )
        )
    )

    assert [(case.case_type, case.actions, case.priority) for case in cases] == [
        ("other", ["capture"], 0.65)
    ]
    assert "chart visual reference" in cases[0].reason


def test_deterministic_essential_cases_treats_physical_demo_as_describe_not_ocr() -> None:
    cases = deterministic_essential_cases(
        _transcript(
            TranscriptSegment(
                id="seg_000001",
                start=20.0,
                end=24.0,
                text="Watch as I demonstrate how to tie the knot.",
            )
        )
    )

    assert [(case.case_type, case.actions, case.priority) for case in cases] == [
        ("screen_demo", ["describe", "capture"], 0.7)
    ]
    assert "action demonstration" in cases[0].reason


def test_deterministic_essential_cases_keep_structural_precedence_over_slide_cues() -> None:
    cases = deterministic_essential_cases(
        _transcript(
            TranscriptSegment(
                id="seg_000001",
                start=4.0,
                end=8.0,
                text="This summary diagram explains the architecture.",
            )
        )
    )

    assert [case.case_type for case in cases] == ["diagram"]
    assert cases[0].actions == ["describe", "capture"]


def test_dedupe_cases_by_window_keeps_highest_priority_nearby_case() -> None:
    cases = [
        EssentialVisualCase(
            segment_id="low",
            timestamp_seconds=101.0,
            case_type="screen_demo",
            priority=0.5,
            reason="nearby lower priority",
            actions=["ocr", "capture"],
        ),
        EssentialVisualCase(
            segment_id="high",
            timestamp_seconds=100.0,
            case_type="diagram",
            priority=0.9,
            reason="nearby higher priority",
            actions=["describe", "capture"],
        ),
        EssentialVisualCase(
            segment_id="far",
            timestamp_seconds=125.0,
            case_type="table",
            priority=0.7,
            reason="far enough",
            actions=["ocr", "capture"],
        ),
    ]

    selected = dedupe_cases_by_window(cases, min_gap_s=8.0, budget=10)

    assert [case.segment_id for case in selected] == ["high", "far"]



def test_uncertain_visual_segments_excludes_deterministic_covered_segments() -> None:
    transcript = _transcript_with_texts([
        "This formula is shown on the board.",
        "This part is just narration.",
    ])
    deterministic = deterministic_essential_cases(transcript)

    uncertain = uncertain_visual_segments(transcript, deterministic)

    assert [segment.id for segment in uncertain.segments] == []


def test_uncertain_visual_segments_keeps_vague_visual_reference_with_context() -> None:
    transcript = _transcript_with_texts([
        "First we introduce the topic.",
        "Look at this for a second.",
        "Then we continue the explanation.",
    ])

    uncertain = uncertain_visual_segments(transcript, [])

    assert [segment.id for segment in uncertain.segments] == [
        "seg_000001",
        "seg_000002",
        "seg_000003",
    ]


def test_uncertain_visual_segments_selects_unknown_non_latin_script() -> None:
    transcript = _transcript_with_texts([
        "Intro narration.",
        "请看这里的图。",
        "Follow up narration.",
    ])

    uncertain = uncertain_visual_segments(transcript, [])

    assert [segment.id for segment in uncertain.segments] == [
        "seg_000001",
        "seg_000002",
        "seg_000003",
    ]


def test_uncertain_visual_segments_uses_automatic_limit_for_non_english_transcript() -> None:
    transcript = _transcript_with_texts(
        [f"第 {index} 段说明。" for index in range(1, 101)],
        language="zh",
    )

    uncertain = uncertain_visual_segments(transcript, [])

    assert len(uncertain.segments) == 20
    assert [segment.id for segment in uncertain.segments[:2]] == ["seg_000001", "seg_000002"]
    assert uncertain.segments[-1].id == "seg_000020"


def test_uncertain_visual_segments_keeps_explicit_limit_override() -> None:
    transcript = _transcript_with_texts(
        [f"第 {index} 段说明。" for index in range(1, 20)],
        language="zh",
    )

    uncertain = uncertain_visual_segments(transcript, [], max_segments=3)

    assert [segment.id for segment in uncertain.segments] == [
        "seg_000001",
        "seg_000002",
        "seg_000003",
    ]


def _transcript_with_texts(texts: list[str], *, language: str | None = None) -> Transcript:
    provenance = TranscriptProvenance(method="local_file", format="plain")
    if language is not None:
        provenance = TranscriptProvenance(
            method="local_file",
            format="plain",
            language_evidence=DetectedLanguage(code=language, source="subtitle"),
        )
    return Transcript(
        video_id="video-1",
        provenance=provenance,
        segments=[
            TranscriptSegment(
                id=f"seg_{index:06d}",
                start=float(index),
                end=float(index + 1),
                text=text,
            )
            for index, text in enumerate(texts, start=1)
        ],
    )
