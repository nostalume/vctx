from __future__ import annotations

import re

from vctx.models.knowledge_flow import (
    KnowledgeFlow,
    KnowledgeFlowEdge,
    KnowledgeFlowNode,
    KnowledgeFlowSupplement,
)
from vctx.models.visual import VisualRecord, VisualRecordSet
from vctx.transcript import Transcript

_ARROW_PATTERN = re.compile(r"\s*(?:->|→|=>)\s*")
_NON_LABEL_BOUNDARY = re.compile(r"[^A-Za-z0-9 _/+-]+")
_ORDERED_STEP = re.compile(
    r"(?:^|[.!?]\s+)(?:first|then|next|finally)\s+([^.!?]+)", re.IGNORECASE
)
_NUMBERED_STEP = re.compile(
    r"(?:^|[.!?]\s+)(?:step\s*\d+|\d+)[.):]\s*([^.!?]+)", re.IGNORECASE
)
_INPUT_OUTPUT = re.compile(
    r"input\s+is\s+([^.!?]+).*?output\s+is\s+([^.!?]+)", re.IGNORECASE
)
_TAKES_PRODUCES = re.compile(
    r"takes\s+(?:an?|the)?\s*([^.!?]+?)\s+and\s+(?:produces|outputs)\s+(?:an?|the)?\s*([^.!?]+)",
    re.IGNORECASE,
)
_BEFORE_DEPENDENCY = re.compile(
    r"before\s+([^,.!?]+),\s*([^.!?]+)", re.IGNORECASE
)
_AFTER_DEPENDENCY = re.compile(
    r"after\s+([^,.!?]+),\s*([^.!?]+)", re.IGNORECASE
)
_PIPELINE_LIST = re.compile(
    r"(?:pipeline|workflow|process)\s+(?:consists\s+of|includes|has\s+these\s+steps:?\s*)([^.!?]+)",
    re.IGNORECASE,
)
_SPACE = re.compile(r"\s+")


def extract_knowledge_flow(
    transcript: Transcript, visual_records: VisualRecordSet | None = None
) -> KnowledgeFlow:
    builder = _KnowledgeFlowBuilder()
    for segment in transcript.segments:
        for chain in _extract_chains(segment.text):
            builder.add_chain(chain, evidence_id=segment.id)
    if visual_records is not None:
        for record in visual_records.records:
            if _record_is_kept_text(record):
                for chain in _extract_chains(record.text or ""):
                    builder.add_chain(chain, evidence_id=record.id)
    return builder.build()


def merge_knowledge_flow_supplement(
    deterministic: KnowledgeFlow,
    supplement: KnowledgeFlowSupplement,
    transcript: Transcript,
) -> KnowledgeFlow:
    allowed_evidence = {segment.id for segment in transcript.segments}
    builder = _KnowledgeFlowBuilder()
    _add_existing_flow(builder, deterministic, allowed_evidence=None)
    _add_existing_flow(builder, supplement.flow, allowed_evidence=allowed_evidence)
    return builder.build()


def _add_existing_flow(
    builder: _KnowledgeFlowBuilder,
    flow: KnowledgeFlow,
    *,
    allowed_evidence: set[str] | None,
) -> None:
    labels_by_id = {node.id: node.label for node in flow.nodes}
    connected = {edge.source for edge in flow.edges} | {edge.target for edge in flow.edges}
    for edge in flow.edges:
        evidence = _filtered_evidence(edge.evidence, allowed_evidence)
        if not evidence:
            continue
        source = labels_by_id.get(edge.source)
        target = labels_by_id.get(edge.target)
        if source is None or target is None:
            continue
        builder.add_chain([source, target], evidence_ids=evidence)
    for node in flow.nodes:
        if node.id in connected:
            continue
        evidence = _filtered_evidence(node.evidence, allowed_evidence)
        if evidence:
            builder.add_node(node.label, evidence_ids=evidence)


def _filtered_evidence(
    evidence: list[str], allowed_evidence: set[str] | None
) -> list[str]:
    if allowed_evidence is None:
        return evidence
    return [evidence_id for evidence_id in evidence if evidence_id in allowed_evidence]


def _record_is_kept_text(record: VisualRecord) -> bool:
    if record.kind == "capture" or not record.text:
        return False
    return record.score is None or record.score.keep


def _extract_arrow_chain(text: str) -> list[str]:
    if not _ARROW_PATTERN.search(text):
        return []
    labels = [_normalize_label(part) for part in _ARROW_PATTERN.split(text)]
    return [label for label in labels if label]


def _extract_chains(text: str) -> list[list[str]]:
    arrow_chain = _extract_arrow_chain(text)
    if arrow_chain:
        return [arrow_chain]

    chains: list[list[str]] = []
    chains.extend(_extract_input_output_chains(text))
    chains.extend(_extract_before_after_chains(text))

    labels = [_normalize_label(match[1]) for match in _NUMBERED_STEP.finditer(text)]
    if not labels:
        labels = [_normalize_label(match[1]) for match in _ORDERED_STEP.finditer(text)]
    labels = [label for label in labels if label]
    if len(labels) >= 2:
        chains.append(labels)

    pipeline_list = _extract_pipeline_list(text)
    if pipeline_list:
        chains.append(pipeline_list)

    return chains


def _extract_input_output_chains(text: str) -> list[list[str]]:
    chains: list[list[str]] = []
    input_output = _INPUT_OUTPUT.search(text)
    if input_output is not None:
        chains.append([_normalize_label(input_output[1]), _normalize_label(input_output[2])])
    for match in _TAKES_PRODUCES.finditer(text):
        chains.append([_normalize_label(match[1]), _normalize_label(match[2])])
    return [_non_empty_chain(chain) for chain in chains if len(_non_empty_chain(chain)) >= 2]


def _extract_before_after_chains(text: str) -> list[list[str]]:
    chains: list[list[str]] = []
    for match in _BEFORE_DEPENDENCY.finditer(text):
        chains.append([_normalize_label(match[2]), _normalize_label(match[1])])
    for match in _AFTER_DEPENDENCY.finditer(text):
        chains.append([_normalize_label(match[1]), _normalize_label(match[2])])
    return [_non_empty_chain(chain) for chain in chains if len(_non_empty_chain(chain)) >= 2]


def _extract_pipeline_list(text: str) -> list[str]:
    match = _PIPELINE_LIST.search(text)
    if match is None:
        return []
    raw_items = re.split(r"\s*,\s*|\s+and\s+", match[1])
    labels = [_normalize_label(item) for item in raw_items]
    labels = [label for label in labels if label]
    return labels if len(labels) >= 3 else []


def _non_empty_chain(labels: list[str]) -> list[str]:
    return [label for label in labels if label]


def _normalize_label(value: str) -> str:
    label = _NON_LABEL_BOUNDARY.sub(" ", value).strip()
    label = _SPACE.sub(" ", label)
    return _strip_leading_article(label)[:80]


def _strip_leading_article(label: str) -> str:
    for prefix in ("and then ", "then ", "and ", "a ", "an ", "the "):
        if label.lower().startswith(prefix):
            return label[len(prefix) :]
    return label


def _node_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "concept"


def _add_evidence(existing: list[str], evidence_id: str) -> None:
    if evidence_id not in existing:
        existing.append(evidence_id)


class _KnowledgeFlowBuilder:
    def __init__(self) -> None:
        self._nodes: dict[str, KnowledgeFlowNode] = {}
        self._edges: dict[tuple[str, str], KnowledgeFlowEdge] = {}

    def add_node(self, label: str, *, evidence_ids: list[str]) -> None:
        for evidence_id in evidence_ids:
            self._ensure_node(label, evidence_id=evidence_id)

    def add_chain(
        self,
        labels: list[str],
        *,
        evidence_id: str | None = None,
        evidence_ids: list[str] | None = None,
    ) -> None:
        if len(labels) < 2:
            return
        ids = evidence_ids or ([evidence_id] if evidence_id is not None else [])
        for item in ids:
            node_ids = [self._ensure_node(label, evidence_id=item) for label in labels]
            for source, target in zip(node_ids, node_ids[1:], strict=False):
                self._ensure_edge(source, target, evidence_id=item)

    def build(self) -> KnowledgeFlow:
        return KnowledgeFlow(
            nodes=list(self._nodes.values()),
            edges=list(self._edges.values()),
        )

    def _ensure_node(self, label: str, *, evidence_id: str) -> str:
        node_id = _node_id(label)
        node = self._nodes.get(node_id)
        if node is None:
            self._nodes[node_id] = KnowledgeFlowNode(
                id=node_id,
                label=label,
                evidence=[evidence_id],
            )
            return node_id
        _add_evidence(node.evidence, evidence_id)
        return node_id

    def _ensure_edge(self, source: str, target: str, *, evidence_id: str) -> None:
        key = (source, target)
        edge = self._edges.get(key)
        if edge is None:
            self._edges[key] = KnowledgeFlowEdge(
                id=f"{source}__{target}",
                source=source,
                target=target,
                evidence=[evidence_id],
            )
            return
        _add_evidence(edge.evidence, evidence_id)
