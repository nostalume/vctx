from __future__ import annotations

from vctx.models.visual import VisualRecord
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment
from vctx.transforms.visual_evidence import score_visual_records


def test_visual_scoring_keeps_ocr_with_new_flow_labels() -> None:
    transcript = _transcript(
        "Here is the architecture diagram. It shows the knowledge flow."
    )
    records = [
        VisualRecord(
            id="ocr-0001",
            timestamp_seconds=12.0,
            frame_id="frame-0001",
            kind="ocr",
            text=(
                "URL -> yt-dlp -> Transcript -> Essential Cases -> Frames -> "
                "OCR/VLM -> Knowledge Flow"
            ),
        )
    ]

    scored = score_visual_records(records, transcript)

    assert len(scored.records) == 1
    score = scored.records[0].score
    assert score is not None
    assert score.keep is True
    assert score.novelty_score >= 0.5
    assert score.uncertainty.prior_uncertainty > score.uncertainty.posterior_uncertainty
    assert "new visual tokens" in score.reason


def test_visual_scoring_drops_description_that_only_paraphrases_transcript() -> None:
    transcript = _transcript(
        "This diagram shows the data pipeline from URL to transcript to summary."
    )
    records = [
        VisualRecord(
            id="description-0001",
            timestamp_seconds=12.0,
            frame_id="frame-0001",
            kind="description",
            text="The image shows a diagram of a data pipeline from URL to transcript to summary.",
        )
    ]

    scored = score_visual_records(records, transcript)

    assert len(scored.records) == 1
    score = scored.records[0].score
    assert score is not None
    assert score.keep is False
    assert score.novelty_score < 0.35
    assert score.uncertainty.reduction == 0.0
    assert "mostly overlaps" in score.reason


def test_visual_scoring_detects_unresolved_visual_referents() -> None:
    transcript = _transcript("This formula gives the final result as shown here.")
    records = [
        VisualRecord(
            id="ocr-0001",
            timestamp_seconds=12.0,
            frame_id="frame-0001",
            kind="ocr",
            text="loss = - sum_i y_i log p_i",
        )
    ]

    scored = score_visual_records(records, transcript)
    score = scored.records[0].score

    assert score is not None
    assert score.keep is True
    assert "formula" in score.uncertainty.missing_referents
    assert score.uncertainty.resolved_referents
    assert score.grounding_score > 0.0


def _transcript(text: str) -> Transcript:
    return Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", language="en", format="plain"),
        segments=[TranscriptSegment(id="seg-1", start=10.0, end=14.0, text=text)],
    )
