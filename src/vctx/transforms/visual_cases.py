from __future__ import annotations

from vctx.models.visual import (
    EssentialCaseAction,
    EssentialCaseSupplement,
    EssentialCaseType,
    EssentialVisualCase,
)
from vctx.transcript import Transcript, TranscriptSegment


def deterministic_essential_cases(transcript: Transcript) -> list[EssentialVisualCase]:
    """Extract bounded visual sampling cases from timestamped transcript cues."""

    cases: list[EssentialVisualCase] = []
    for segment in transcript.segments:
        cue = _segment_case(segment)
        if cue is None:
            continue
        case_type, priority, actions, reason = cue
        cases.append(
            EssentialVisualCase(
                segment_id=segment.id,
                timestamp_seconds=_segment_timestamp(segment),
                case_type=case_type,
                priority=priority,
                reason=reason,
                actions=actions,
            )
        )
    return cases


def dedupe_cases_by_window(
    cases: list[EssentialVisualCase], *, min_gap_s: float, budget: int
) -> list[EssentialVisualCase]:
    """Keep high-priority case anchors with only simple time-window dedup."""

    if budget <= 0:
        return []
    selected: list[EssentialVisualCase] = []
    for case in sorted(cases, key=lambda item: item.priority, reverse=True):
        if len(selected) >= budget:
            break
        if all(
            abs(case.timestamp_seconds - existing.timestamp_seconds) >= min_gap_s
            for existing in selected
        ):
            selected.append(case)
    return sorted(selected, key=lambda item: item.timestamp_seconds)


def merge_essential_case_supplement(
    deterministic: list[EssentialVisualCase],
    supplement: EssentialCaseSupplement,
    transcript: Transcript,
) -> list[EssentialVisualCase]:
    known_segments = {segment.id for segment in transcript.segments}
    merged = list(deterministic)
    seen = {(case.segment_id, case.case_type) for case in merged}
    for case in supplement.cases:
        key = (case.segment_id, case.case_type)
        if case.segment_id not in known_segments or key in seen:
            continue
        merged.append(case)
        seen.add(key)
    return sorted(merged, key=lambda item: item.timestamp_seconds)


def uncertain_visual_segments(
    transcript: Transcript,
    deterministic: list[EssentialVisualCase],
    *,
    max_segments: int | None = None,
) -> Transcript:
    limit = max_segments if max_segments is not None else _automatic_uncertain_limit(transcript)
    covered = {case.segment_id for case in deterministic}
    selected_indexes: set[int] = set()
    for index, segment in enumerate(transcript.segments):
        if segment.id in covered:
            continue
        if _needs_llm_visual_review(segment, transcript):
            selected_indexes.add(index)
            if index > 0:
                selected_indexes.add(index - 1)
            if index + 1 < len(transcript.segments):
                selected_indexes.add(index + 1)
    selected = [
        transcript.segments[index]
        for index in sorted(selected_indexes)
        if transcript.segments[index].id not in covered
    ][:limit]
    return transcript.model_copy(update={"segments": selected})


def _segment_case(
    segment: TranscriptSegment,
) -> tuple[EssentialCaseType, float, list[EssentialCaseAction], str] | None:
    text = segment.text.lower()
    if _contains_any(text, ("diagram", "architecture", "flowchart", "system flow")):
        return (
            "diagram",
            0.9,
            ["describe", "capture"],
            _reason(segment, "diagram or architecture visual reference"),
        )
    if _contains_any(text, ("formula", "equation", "derive", "derivation", "proof")):
        return (
            "formula",
            0.85,
            ["ocr", "describe", "capture"],
            _reason(segment, "formula or derivation visual reference"),
        )
    if _contains_any(text, ("chart", "graph", "plot", "stats", "statistics")):
        return (
            "other",
            0.65,
            ["capture"],
            _reason(segment, "chart visual reference"),
        )
    if _contains_any(text, ("watch", "demonstrate", "show you", "how to")):
        return (
            "screen_demo",
            0.7,
            ["describe", "capture"],
            _reason(segment, "action demonstration visual reference"),
        )
    if _contains_any(text, ("screen", "click", "open", "terminal", "walkthrough")):
        return (
            "screen_demo",
            0.75,
            ["ocr", "describe", "capture"],
            _reason(segment, "screen visual reference"),
        )
    if _contains_any(text, ("table", "row", "column", "compare", "comparison")):
        return (
            "table",
            0.7,
            ["ocr", "capture"],
            _reason(segment, "table or comparison visual reference"),
        )
    if _contains_any(text, ("code", "function", "class", "snippet")):
        return (
            "code",
            0.7,
            ["ocr", "capture"],
            _reason(segment, "code visual reference"),
        )
    if _contains_any(
        text,
        ("summary slide", "overview slide", "recap slide", "key takeaways", "main takeaway"),
    ):
        return (
            "visual_summary",
            0.65,
            ["ocr", "describe", "capture"],
            _reason(segment, "summary or overview visual reference"),
        )
    if _contains_any(
        text,
        (
            "slide text",
            "shown text",
            "text on screen",
            "slide title",
            "section title",
            "title slide",
            "agenda slide",
            "outline slide",
        ),
    ):
        return (
            "slide_title",
            0.6,
            ["ocr", "capture"],
            _reason(segment, "shown text, slide or section title visual reference"),
        )
    return None


def _needs_llm_visual_review(segment: TranscriptSegment, transcript: Transcript) -> bool:
    return (
        _is_non_english_or_mixed(transcript)
        or _has_non_latin_script(segment.text)
        or _has_vague_visual_reference(segment.text)
    )


def _automatic_uncertain_limit(transcript: Transcript) -> int:
    if not transcript.segments:
        return 0
    return min(40, max(8, round(len(transcript.segments) * 0.2)))


def _is_non_english_or_mixed(transcript: Transcript) -> bool:
    evidence = transcript.provenance.language_evidence
    if evidence.kind == "mixed":
        return True
    if evidence.kind == "detected":
        return evidence.code.lower().split("-")[0] != "en"
    return False


def _has_vague_visual_reference(text: str) -> bool:
    lower = text.lower()
    return _contains_any(
        lower,
        (
            "look at this",
            "look here",
            "see this",
            "shown here",
            "as shown",
            "on screen",
            "watch this",
        ),
    )


def _has_non_latin_script(text: str) -> bool:
    return any(
        "\u4e00" <= char <= "\u9fff"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
        or "\u0600" <= char <= "\u06ff"
        or "\u0400" <= char <= "\u04ff"
        for char in text
    )


def _segment_timestamp(segment: TranscriptSegment) -> float:
    if segment.end is None or segment.end <= segment.start:
        return segment.start
    return round((segment.start + segment.end) / 2, 3)


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _reason(segment: TranscriptSegment, cue: str) -> str:
    excerpt = " ".join(segment.text.split())
    if len(excerpt) > 120:
        excerpt = f"{excerpt[:117]}..."
    return f"{cue}: {excerpt}"
