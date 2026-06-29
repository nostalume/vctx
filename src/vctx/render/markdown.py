from __future__ import annotations

from dataclasses import dataclass
from html import escape

from vctx.chunking import ChunkSet
from vctx.models.knowledge_flow import KnowledgeFlow, KnowledgeFlowEdge
from vctx.models.metadata import VideoMetadata
from vctx.models.visual import VisualRecord, VisualRecordSet
from vctx.transcript import Transcript
from vctx.util import format_timestamp


def render_context_markdown(
    metadata: VideoMetadata,
    transcript: Transcript,
    chunks: ChunkSet,
    visual_records: VisualRecordSet | None = None,
    knowledge_flow: KnowledgeFlow | None = None,
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
    lines.extend(_render_context_visual_reference_lines(visual_records))
    lines.extend(render_knowledge_flow_lines(knowledge_flow))
    lines.extend([
        "## Chunks",
        "",
    ])
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
    visual_records: VisualRecordSet | None = None,
    knowledge_flow: KnowledgeFlow | None = None,
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
    lines.extend(_render_readable_visual_reference_lines(visual_records))
    lines.extend(render_knowledge_flow_lines(knowledge_flow))
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


def render_knowledge_flow_lines(knowledge_flow: KnowledgeFlow | None) -> list[str]:
    if knowledge_flow is None or not knowledge_flow.nodes or not knowledge_flow.edges:
        return []
    node_labels = {node.id: node.label for node in knowledge_flow.nodes}
    chains = _edge_chains(knowledge_flow.edges)
    rendered_chains: list[tuple[list[str], list[str]]] = []
    for chain in chains:
        labels = [node_labels[node_id] for node_id in chain if node_id in node_labels]
        if len(labels) >= 2:
            rendered_chains.append((labels, _chain_evidence(chain, knowledge_flow.edges)))
    if not rendered_chains:
        return []
    lines = ["## Knowledge-flow summary", ""]
    for labels, evidence in rendered_chains:
        lines.append(f"- Workflow: {' -> '.join(labels)}.")
        if evidence:
            lines.append(f"  Evidence: {', '.join(evidence)}")
    lines.extend(["", "## Knowledge flow", ""])
    for labels, evidence in rendered_chains:
        lines.append(f"- {' -> '.join(labels)}")
        if evidence:
            lines.append(f"  Evidence: {', '.join(evidence)}")
    lines.append("")
    return lines


@dataclass(frozen=True)
class VisualFrameGroup:
    frame_id: str
    timestamp_seconds: float | None
    artifact_path: str | None
    records: list[VisualRecord]


def _render_context_visual_reference_lines(
    visual_records: VisualRecordSet | None,
) -> list[str]:
    groups = _visual_frame_groups(visual_records)
    if not groups:
        return []
    lines = ["## Visual references", ""]
    for group in groups:
        timestamp = _visual_timestamp(group.timestamp_seconds)
        path_attr = f' path="{escape(group.artifact_path)}"' if group.artifact_path else ""
        lines.append(
            f'<visual_ref id="{escape(group.frame_id)}" timestamp="{timestamp}"{path_attr}>'
        )
        if group.artifact_path:
            lines.append(f"  <image path=\"{escape(group.artifact_path)}\" />")
        for record in group.records:
            if record.kind == "capture":
                continue
            text = escape(record.text or record.artifact_path or record.id)
            lines.append(f"  <{record.kind}{_record_score_attr(record)}>{text}</{record.kind}>")
        lines.append("</visual_ref>")
        lines.append("")
    return lines


def _render_readable_visual_reference_lines(
    visual_records: VisualRecordSet | None,
) -> list[str]:
    groups = _visual_frame_groups(visual_records)
    if not groups:
        return []
    lines = ["## Visual references", ""]
    for group in groups:
        timestamp = _visual_timestamp(group.timestamp_seconds)
        lines.extend([f"### {timestamp} — {group.frame_id}", ""])
        if group.artifact_path:
            alt = _markdown_alt(f"Frame {group.frame_id} at {timestamp}")
            lines.extend([f"![{alt}]({group.artifact_path})", ""])
        for record in group.records:
            if record.kind == "capture":
                continue
            detail = record.text or record.artifact_path or record.id
            lines.append(f"- {record.kind.upper()}: {detail}{_readable_score_text(record)}")
        lines.append("")
    return lines


def _visual_frame_groups(visual_records: VisualRecordSet | None) -> list[VisualFrameGroup]:
    if visual_records is None or not visual_records.records:
        return []
    renderable_records = [
        record for record in visual_records.records if record.score is None or record.score.keep
    ]
    if not renderable_records:
        return []
    by_frame: dict[str, list[VisualRecord]] = {}
    for record in renderable_records:
        by_frame.setdefault(record.frame_id, []).append(record)
    groups: list[VisualFrameGroup] = []
    for frame_id, records in by_frame.items():
        capture = next((record for record in records if record.kind == "capture"), None)
        timestamp_seconds = next(
            (
                record.timestamp_seconds
                for record in records
                if record.timestamp_seconds is not None
            ),
            None,
        )
        artifact_path = capture.artifact_path if capture is not None else None
        if artifact_path is None:
            artifact_path = next(
                (record.artifact_path for record in records if record.artifact_path is not None),
                None,
            )
        groups.append(
            VisualFrameGroup(
                frame_id=frame_id,
                timestamp_seconds=timestamp_seconds,
                artifact_path=artifact_path,
                records=records,
            )
        )
    return sorted(groups, key=_visual_group_sort_key)


def _visual_group_sort_key(group: VisualFrameGroup) -> tuple[float, str]:
    timestamp = group.timestamp_seconds if group.timestamp_seconds is not None else float("inf")
    return (timestamp, group.frame_id)


def _visual_timestamp(timestamp_seconds: float | None) -> str:
    return format_timestamp(timestamp_seconds) if timestamp_seconds is not None else "unknown"


def _record_score_attr(record: VisualRecord) -> str:
    if record.score is None or record.kind == "capture":
        return ""
    return f' novelty="{record.score.novelty_score:.2f}"'


def _readable_score_text(record: VisualRecord) -> str:
    if record.score is None or record.kind == "capture":
        return ""
    return f" (novelty {record.score.novelty_score:.2f})"


def _markdown_alt(value: str) -> str:
    return value.replace("]", ")")


def _edge_chains(edges: list[KnowledgeFlowEdge]) -> list[list[str]]:
    outgoing: dict[str, list[KnowledgeFlowEdge]] = {}
    targets = {edge.target for edge in edges}
    for edge in edges:
        outgoing.setdefault(edge.source, []).append(edge)
    starts = [edge.source for edge in edges if edge.source not in targets]
    if not starts:
        starts = [edges[0].source]
    chains: list[list[str]] = []
    for start in dict.fromkeys(starts):
        _walk_chains(start, outgoing, [start], chains)
    return chains


def _walk_chains(
    current: str,
    outgoing: dict[str, list[KnowledgeFlowEdge]],
    path: list[str],
    chains: list[list[str]],
) -> None:
    next_edges = outgoing.get(current, [])
    if not next_edges:
        if len(path) >= 2:
            chains.append(path)
        return
    for edge in next_edges:
        if edge.target in path:
            chains.append(path)
            continue
        _walk_chains(edge.target, outgoing, [*path, edge.target], chains)


def _chain_evidence(chain: list[str], edges: list[KnowledgeFlowEdge]) -> list[str]:
    by_pair = {(edge.source, edge.target): edge for edge in edges}
    evidence: list[str] = []
    for source, target in zip(chain, chain[1:], strict=False):
        edge = by_pair.get((source, target))
        if edge is None:
            continue
        for evidence_id in edge.evidence:
            if evidence_id not in evidence:
                evidence.append(evidence_id)
    return evidence
