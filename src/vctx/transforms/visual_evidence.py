from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from vctx.models.visual import (
    SatisfactionChecked,
    VisualEvidenceScore,
    VisualOperationMotive,
    VisualRecord,
    VisualScoredRecordSet,
    VisualUncertainty,
)
from vctx.transcript import Transcript, TranscriptSegment

_VISUAL_REFERENT_CUES: tuple[tuple[str, str], ...] = (
    ("diagram", "diagram"),
    ("architecture", "diagram"),
    ("flow", "flow"),
    ("formula", "formula"),
    ("equation", "formula"),
    ("table", "table"),
    ("screen", "screen"),
    ("shown here", "visual referent"),
    ("as shown", "visual referent"),
    ("this", "visual referent"),
)

_STRUCTURAL_PATTERNS = (
    "->",
    "→",
    "=>",
    "=",
    "|",
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "here",
    "image",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "shows",
    "shown",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def score_visual_records(
    records: list[VisualRecord],
    transcript: Transcript,
    *,
    motives: list[VisualOperationMotive] | None = None,
) -> VisualScoredRecordSet:
    scored: list[VisualRecord] = []
    previous_kept_tokens: set[str] = set()
    for record in records:
        score = score_visual_record(record, transcript, previous_kept_tokens)
        if score.keep:
            previous_kept_tokens.update(_tokens(record.text or ""))
        scored.append(record.model_copy(update={"score": score}))
    return VisualScoredRecordSet(
        records=scored,
        satisfaction=_satisfaction_checks(motives or [], scored),
    )


def score_visual_record(
    record: VisualRecord,
    transcript: Transcript,
    previous_kept_tokens: Iterable[str] = (),
) -> VisualEvidenceScore:
    if record.kind == "capture" or not record.text:
        return VisualEvidenceScore(
            keep=True,
            novelty_score=0.0,
            overlap_score=0.0,
            grounding_score=0.2,
            reason="capture record is retained as source evidence",
            uncertainty=VisualUncertainty(),
        )
    nearby_text = _nearby_transcript_text(transcript, record.timestamp_seconds)
    transcript_tokens = _tokens(nearby_text)
    record_tokens = _tokens(record.text)
    previous_tokens = set(previous_kept_tokens)
    comparable_tokens = record_tokens - previous_tokens
    new_tokens = comparable_tokens - transcript_tokens
    overlap_tokens = comparable_tokens & transcript_tokens

    novelty_score = _ratio(len(new_tokens), len(comparable_tokens))
    overlap_score = _ratio(len(overlap_tokens), len(comparable_tokens))
    structural_score = _structural_score(record.text)
    missing_referents = _missing_referents(nearby_text)
    resolved_referents = _resolved_referents(new_tokens, structural_score, record.text)
    prior_uncertainty = _prior_uncertainty(missing_referents)
    grounding_score = min(
        1.0, novelty_score + structural_score + (0.2 if missing_referents else 0.0)
    )
    keep = novelty_score >= 0.35 or structural_score >= 0.25 or (
        bool(missing_referents) and bool(resolved_referents)
    )
    reduction = min(prior_uncertainty, grounding_score * 0.6) if keep else 0.0
    posterior_uncertainty = max(0.0, prior_uncertainty - reduction)
    reason = _reason(
        keep=keep,
        novelty_score=novelty_score,
        overlap_score=overlap_score,
        structural_score=structural_score,
    )
    return VisualEvidenceScore(
        keep=keep,
        novelty_score=round(novelty_score, 3),
        overlap_score=round(overlap_score, 3),
        grounding_score=round(grounding_score, 3),
        reason=reason,
        uncertainty=VisualUncertainty(
            prior_uncertainty=round(prior_uncertainty, 3),
            posterior_uncertainty=round(posterior_uncertainty, 3),
            reduction=round(reduction, 3),
            missing_referents=missing_referents,
            resolved_referents=resolved_referents,
        ),
    )


def _nearby_transcript_text(transcript: Transcript, timestamp_seconds: float | None) -> str:
    if timestamp_seconds is None:
        return " ".join(segment.text for segment in transcript.segments)
    nearby = [
        segment.text
        for segment in transcript.segments
        if _segment_near_timestamp(segment, timestamp_seconds, window_s=20.0)
    ]
    if nearby:
        return " ".join(nearby)
    return " ".join(segment.text for segment in transcript.segments)


def _segment_near_timestamp(
    segment: TranscriptSegment, timestamp_seconds: float, *, window_s: float
) -> bool:
    end = segment.end if segment.end is not None else segment.start
    return segment.start - window_s <= timestamp_seconds <= end + window_s


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]+", text.lower())
        if len(token) >= 2 and token not in _STOPWORDS
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _structural_score(text: str) -> float:
    score = 0.0
    if any(pattern in text for pattern in _STRUCTURAL_PATTERNS):
        score += 0.3
    if re.search(r"\b\d+(?:\.\d+)?\b", text):
        score += 0.15
    if re.search(r"\b[A-Za-z_]+\s*=", text):
        score += 0.2
    return min(0.6, score)


def _missing_referents(text: str) -> list[str]:
    lower = text.lower()
    referents: list[str] = []
    for cue, referent in _VISUAL_REFERENT_CUES:
        if cue in lower and referent not in referents:
            referents.append(referent)
    return referents


def _resolved_referents(new_tokens: set[str], structural_score: float, text: str) -> list[str]:
    resolved: list[str] = []
    if new_tokens:
        preview = ", ".join(sorted(new_tokens)[:5])
        resolved.append(f"new visual tokens: {preview}")
    if structural_score > 0.0:
        resolved.append("visual structure")
    if "=" in text:
        resolved.append("formula")
    return resolved


def _prior_uncertainty(missing_referents: list[str]) -> float:
    if not missing_referents:
        return 0.2
    return min(0.9, 0.35 + 0.12 * len(missing_referents))


def _reason(
    *, keep: bool, novelty_score: float, overlap_score: float, structural_score: float
) -> str:
    if keep and structural_score > 0.0:
        return "new visual tokens and visual structure add grounded evidence"
    if keep:
        return "new visual tokens add grounded evidence"
    if overlap_score >= 0.6:
        return "visual text mostly overlaps nearby transcript or prior visual records"
    return "visual record has low novelty and weak grounding"


def _satisfaction_checks(
    motives: list[VisualOperationMotive], records: list[VisualRecord]
) -> list[SatisfactionChecked]:
    return [_satisfaction_check(motive, records) for motive in motives]


def _satisfaction_check(
    motive: VisualOperationMotive, records: list[VisualRecord]
) -> SatisfactionChecked:
    matching = [record for record in records if record.kind == _record_kind(motive.operation)]
    nearby = [record for record in matching if _record_near_motive(record, motive)]
    candidates = nearby or matching
    if motive.operation == "capture":
        record = next((item for item in candidates if item.artifact_path), None)
        if record is not None:
            return _checked(motive, record, "satisfied", "capture artifact exists")
        return _checked(motive, None, "missed", "no capture artifact found")
    if motive.operation == "describe":
        record = next((item for item in candidates if item.text), None)
        if record is not None:
            return _checked(motive, record, "satisfied", "description text exists")
        return _checked(motive, None, "missed", "no description text found")
    record = next((item for item in candidates if item.text), None)
    if record is None:
        return _checked(motive, None, "missed", "no OCR text found")
    if motive.reason == "formula_or_equation" and not _has_math_signal(record.text or ""):
        return _checked(motive, record, "missed", "OCR text has no math signal")
    return _checked(motive, record, "satisfied", "OCR text exists")


def _checked(
    motive: VisualOperationMotive,
    record: VisualRecord | None,
    status: Literal["satisfied", "missed"],
    detail: str,
) -> SatisfactionChecked:
    return SatisfactionChecked(
        operation=motive.operation,
        reason=motive.reason,
        segment_id=motive.segment_id,
        frame_id=record.frame_id if record is not None else None,
        status=status,
        detail=detail,
    )


def _record_kind(operation: str) -> str:
    if operation == "describe":
        return "description"
    return operation


def _record_near_motive(record: VisualRecord, motive: VisualOperationMotive) -> bool:
    if record.timestamp_seconds is None:
        return False
    return abs(record.timestamp_seconds - motive.timestamp_seconds) <= 1.0


def _has_math_signal(text: str) -> bool:
    return bool(re.search(r"[=+\-*/^∑√≤≥≠]|\b\d+(?:\.\d+)?\b", text))
