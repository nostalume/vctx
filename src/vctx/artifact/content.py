from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ArtifactKind = Literal[
    "metadata",
    "transcript",
    "chunks",
    "context",
    "readable",
    "evidence",
    "evidence_plan",
    "visual_frame",
    "manifest",
]


class Artifact(BaseModel):
    name: str
    kind: ArtifactKind
    media_type: str
    content: str


class ArtifactBundle(BaseModel):
    artifacts: list[Artifact]

    def get(self, kind: ArtifactKind) -> Artifact | None:
        return next((artifact for artifact in self.artifacts if artifact.kind == kind), None)
