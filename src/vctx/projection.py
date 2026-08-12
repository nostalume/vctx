from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from html import escape

from vctx.source.session import VideoMetadata
from vctx.summary import Summary
from vctx.transcript import ChunkSet, Transcript
from vctx.visual.evidence import Evidence


@dataclass(frozen=True)
class Renderer:
    metadata: VideoMetadata
    transcript: Transcript
    chunks: ChunkSet
    evidence: Evidence | None = None
    summary: Summary | None = None
    links: Mapping[str, str] = field(default_factory=dict)

    def context(self) -> str:
        lines = [
            "# Agent Context Pack",
            "",
            "## Metadata",
            "",
            f"- Title: {self.metadata.title or self.metadata.id}",
            f"- Source: {self.metadata.source.value}",
            f"- Duration: {_timestamp(self.metadata.duration_seconds)}",
            "- Transcript source: "
            f"{self.transcript.provenance.method} / "
            f"{self.transcript.provenance.language or 'unknown'} / "
            f"{self.transcript.provenance.format}",
            "",
        ]
        lines.extend(self._summary(context=True))
        lines.extend(self._visual(context=True))
        lines.extend(["## Chunks", ""])
        for chunk in self.chunks.chunks:
            tag = (
                f'<chunk id="{chunk.id}" start="{_timestamp(chunk.start)}" '
                f'end="{_timestamp(chunk.end)}">'
            )
            lines.extend([tag, chunk.text, "</chunk>", ""])
        return "\n".join(lines).rstrip() + "\n"

    def read(self) -> str:
        lines = [
            f"# {self.metadata.title or self.metadata.id}",
            "",
            f"Source: {self.metadata.source.value}  ",
            f"Duration: {_timestamp(self.metadata.duration_seconds)}  ",
            f"Transcript source: {self.transcript.provenance.method} / "
            f"{self.transcript.provenance.language or 'unknown'}",
            "",
        ]
        lines.extend(self._summary(context=False))
        lines.extend(self._visual(context=False))
        for chunk in self.chunks.chunks:
            lines.extend([f"## {_timestamp(chunk.start)}–{_timestamp(chunk.end)}", ""])
            lines.extend([chunk.text, ""])
        return "\n".join(lines).rstrip() + "\n"

    def transcript_text(self) -> str:
        lines = [f"# Transcript — {self.metadata.title or self.metadata.id}", ""]
        lines.extend(
            f"[{_timestamp(segment.start)}–{_timestamp(segment.end)}] {segment.text}"
            for segment in self.transcript.segments
        )
        return "\n".join(lines).rstrip() + "\n"

    def _summary(self, *, context: bool) -> list[str]:
        if self.summary is None:
            return ["## Summary", "", "Summary is not available in this pack.", ""]
        lines = ["## Summary", ""]
        if self.summary.overview:
            lines.extend([self.summary.overview, ""])
        for point in self.summary.points:
            citations = [*point.segment_ids, *point.capture_ids]
            if context:
                lines.append(f'- <point id="{escape(point.id)}">{escape(point.text)}</point>')
                lines.append(f"  Citations: {', '.join(citations)}")
            else:
                lines.append(f"- {point.text} ({', '.join(citations)})")
        lines.append("")
        return lines

    def _visual(self, *, context: bool) -> list[str]:
        captures = sorted(
            self.evidence.captures if self.evidence else [],
            key=lambda item: (item.actual_seconds, item.id),
        )
        if not captures:
            return [
                "## Visual references",
                "",
                "Visual evidence is not available in this pack.",
                "",
            ]
        lines = ["## Visual references", ""]
        for capture in captures:
            timestamp = _timestamp(capture.actual_seconds)
            path = self.links.get(capture.artifact_path, capture.artifact_path)
            if context:
                tag = (
                    f'<visual_ref id="{escape(capture.id)}" timestamp="{timestamp}" '
                    f'path="{escape(path)}">'
                )
                lines.extend([tag, f'  <image path="{escape(path)}" />'])
                for name, observation in (("ocr", capture.ocr), ("description", capture.vision)):
                    if observation.text:
                        lines.append(f"  <{name}>{escape(observation.text)}</{name}>")
                lines.extend(["</visual_ref>", ""])
            else:
                alt = f"Frame {capture.id} at {timestamp}".replace("]", ")")
                lines.extend([f"### {timestamp} — {capture.id}", ""])
                lines.extend([f"![{alt}]({path})", ""])
                for name, observation in (("OCR", capture.ocr), ("DESCRIPTION", capture.vision)):
                    if observation.text:
                        lines.append(f"- {name}: {observation.text}")
                lines.append("")
        return lines


def _timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, second = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{second:02d}"
