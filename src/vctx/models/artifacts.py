from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ArtifactKind = Literal[
    "metadata",
    "transcript_raw",
    "transcript_clean",
    "transcript_md",
    "chunks",
    "context",
    "readable",
    "visual_records",
    "visual_scores",
    "knowledge_flow",
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
        for artifact in self.artifacts:
            if artifact.kind == kind:
                return artifact
        return None
