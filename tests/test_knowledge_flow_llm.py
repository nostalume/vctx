from __future__ import annotations

from vctx.models.knowledge_flow import (
    KnowledgeFlow,
    KnowledgeFlowEdge,
    KnowledgeFlowNode,
    KnowledgeFlowSupplement,
    KnowledgeFlowSupplementEvidence,
)
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment
from vctx.transforms.knowledge_flow import merge_knowledge_flow_supplement


def test_merge_knowledge_flow_supplement_keeps_backbone_and_discards_unknown_evidence() -> None:
    deterministic = KnowledgeFlow(
        nodes=[
            KnowledgeFlowNode(
                id="download",
                label="download media",
                evidence=["seg_000001"],
            ),
            KnowledgeFlowNode(
                id="transcribe",
                label="transcribe audio",
                evidence=["seg_000001"],
            ),
        ],
        edges=[
            KnowledgeFlowEdge(
                id="download__transcribe",
                source="download",
                target="transcribe",
                evidence=["seg_000001"],
            )
        ],
    )
    supplement = KnowledgeFlowSupplement(
        flow=KnowledgeFlow(
            nodes=[
                KnowledgeFlowNode(
                    id="transcribe-alt",
                    label="transcribe audio",
                    evidence=["seg_000001", "unknown-seg"],
                ),
                KnowledgeFlowNode(
                    id="summarize",
                    label="summarize context",
                    evidence=["seg_000002"],
                ),
                KnowledgeFlowNode(
                    id="hallucinated",
                    label="unsupported cleanup",
                    evidence=["unknown-seg"],
                ),
            ],
            edges=[
                KnowledgeFlowEdge(
                    id="transcribe-alt__summarize",
                    source="transcribe-alt",
                    target="summarize",
                    evidence=["seg_000002"],
                ),
                KnowledgeFlowEdge(
                    id="summarize__hallucinated",
                    source="summarize",
                    target="hallucinated",
                    evidence=["unknown-seg"],
                ),
            ],
        ),
        evidence=KnowledgeFlowSupplementEvidence(source_segment_ids=["seg_000001", "seg_000002"]),
    )

    merged = merge_knowledge_flow_supplement(deterministic, supplement, _transcript())

    labels = {node.label for node in merged.nodes}
    assert labels == {"download media", "transcribe audio", "summarize context"}
    transcribe = next(node for node in merged.nodes if node.label == "transcribe audio")
    assert transcribe.evidence == ["seg_000001", "seg_000002"]
    summarize = next(node for node in merged.nodes if node.label == "summarize context")
    assert summarize.evidence == ["seg_000002"]
    assert "unknown-seg" not in transcribe.evidence
    assert "unknown-seg" not in summarize.evidence
    assert any(
        edge.source == transcribe.id and edge.target == summarize.id
        for edge in merged.edges
    )
    assert not any(node.label == "unsupported cleanup" for node in merged.nodes)


def _transcript() -> Transcript:
    return Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=3.0,
                text="Download media then transcribe audio.",
            ),
            TranscriptSegment(
                id="seg_000002",
                start=3.0,
                end=6.0,
                text="Then summarize context.",
            ),
        ],
    )
