from __future__ import annotations

from html import escape

from vctx.source.session import VideoMetadata
from vctx.transcript import ChunkSet, Transcript
from vctx.util import format_timestamp
from vctx.visual.evidence import CaptureEvidence, Evidence
from vctx.visual.plan import EvidencePlan


def render_context_markdown(
    metadata: VideoMetadata,
    transcript: Transcript,
    chunks: ChunkSet,
    evidence: Evidence | None = None,
    evidence_plan: EvidencePlan | None = None,
) -> str:
    lines = [
        "# Agent Context Pack",
        "",
        "## Metadata",
        "",
        f"- Title: {metadata.title or metadata.id}",
        f"- Source: {metadata.source.value}",
        f"- Duration: {format_timestamp(metadata.duration_seconds)}",
        "- Transcript source: "
        f"{transcript.provenance.method} / {transcript.provenance.language or 'unknown'} / "
        f"{transcript.provenance.format}",
        "",
        "## Usage",
        "",
        "The chunks below are timestamped source text extracted from the video or transcript.",
        "Preserve timestamps when citing claims.",
        "",
    ]
    lines.extend(_render_context_visual_reference_lines(evidence))
    lines.extend(render_evidence_plan_lines(evidence_plan))
    lines.extend(
        [
            "## Chunks",
            "",
        ]
    )
    for chunk in chunks.chunks:
        lines.extend(
            [
                f'<chunk id="{chunk.id}" start="{format_timestamp(chunk.start)}" '
                f'end="{format_timestamp(chunk.end)}">',
                chunk.text,
                "</chunk>",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_readable_markdown(
    metadata: VideoMetadata,
    transcript: Transcript,
    chunks: ChunkSet,
    evidence: Evidence | None = None,
    evidence_plan: EvidencePlan | None = None,
) -> str:
    transcript_source = (
        f"Transcript source: {transcript.provenance.method} / "
        f"{transcript.provenance.language or 'unknown'}"
    )
    lines = [
        f"# {metadata.title or metadata.id}",
        "",
        f"Source: {metadata.source.value}  ",
        f"Duration: {format_timestamp(metadata.duration_seconds)}  ",
        transcript_source,
        "",
    ]
    lines.extend(_render_readable_visual_reference_lines(evidence))
    lines.extend(render_evidence_plan_lines(evidence_plan))
    for chunk in chunks.chunks:
        lines.extend(
            [
                f"## {format_timestamp(chunk.start)}–{format_timestamp(chunk.end)}",
                "",
                chunk.text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_transcript_markdown(metadata: VideoMetadata, transcript: Transcript) -> str:
    lines = [f"# Transcript — {metadata.title or metadata.id}", ""]
    for segment in transcript.segments:
        lines.append(
            f"[{format_timestamp(segment.start)}–{format_timestamp(segment.end)}] {segment.text}"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_evidence_plan_lines(plan: EvidencePlan | None) -> list[str]:
    if plan is None or not plan.claims:
        return []
    lines = ["## Evidence plan", ""]
    for claim in plan.claims:
        lines.append(f"- {claim.text}")
        lines.append(f"  Evidence: {', '.join(claim.segment_ids)}")
    labels = {claim.id: claim.text for claim in plan.claims}
    for relation in plan.relations:
        lines.append(f"- {labels[relation.source]} —{relation.kind}→ {labels[relation.target]}")
    lines.append("")
    return lines


def _render_context_visual_reference_lines(
    evidence: Evidence | None,
) -> list[str]:
    captures = _captures(evidence)
    if not captures:
        return []
    lines = ["## Visual references", ""]
    for capture in captures:
        timestamp = format_timestamp(capture.actual_seconds)
        lines.append(
            f'<visual_ref id="{escape(capture.id)}" timestamp="{timestamp}" '
            f'path="{escape(capture.artifact_path)}">'
        )
        lines.append(f'  <image path="{escape(capture.artifact_path)}" />')
        for name, observation in (("ocr", capture.ocr), ("description", capture.vision)):
            if observation is not None and observation.text:
                lines.append(f"  <{name}>{escape(observation.text)}</{name}>")
        lines.append("</visual_ref>")
        lines.append("")
    return lines


def _render_readable_visual_reference_lines(
    evidence: Evidence | None,
) -> list[str]:
    captures = _captures(evidence)
    if not captures:
        return []
    lines = ["## Visual references", ""]
    for capture in captures:
        timestamp = format_timestamp(capture.actual_seconds)
        lines.extend([f"### {timestamp} — {capture.id}", ""])
        alt = _markdown_alt(f"Frame {capture.id} at {timestamp}")
        lines.extend([f"![{alt}]({capture.artifact_path})", ""])
        for name, observation in (("OCR", capture.ocr), ("DESCRIPTION", capture.vision)):
            if observation is not None and observation.text:
                lines.append(f"- {name}: {observation.text}")
        lines.append("")
    return lines


def _captures(evidence: Evidence | None) -> list[CaptureEvidence]:
    return (
        sorted(evidence.captures, key=lambda item: (item.actual_seconds, item.id))
        if evidence is not None
        else []
    )


def _markdown_alt(value: str) -> str:
    return value.replace("]", ")")
