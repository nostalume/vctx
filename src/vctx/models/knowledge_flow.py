from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

KnowledgeFlowNodeKind = Literal["concept"]


class KnowledgeFlowNode(BaseModel):
    id: str
    label: str
    kind: KnowledgeFlowNodeKind = "concept"
    evidence: list[str] = Field(default_factory=list)


class KnowledgeFlowEdge(BaseModel):
    id: str
    source: str
    target: str
    evidence: list[str] = Field(default_factory=list)


class KnowledgeFlow(BaseModel):
    nodes: list[KnowledgeFlowNode] = Field(default_factory=list)
    edges: list[KnowledgeFlowEdge] = Field(default_factory=list)


class KnowledgeFlowSupplementEvidence(BaseModel):
    route_provider_id: str | None = None
    route_model: str | None = None
    source_segment_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class KnowledgeFlowSupplement(BaseModel):
    flow: KnowledgeFlow
    evidence: KnowledgeFlowSupplementEvidence = Field(
        default_factory=KnowledgeFlowSupplementEvidence
    )
