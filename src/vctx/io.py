from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from vctx.artifact.content import Artifact, ArtifactBundle
from vctx.artifact.manifest import ArtifactRef, Manifest


def model_to_json(model: BaseModel) -> str:
    return model.model_dump_json(indent=2) + "\n"

def write_artifact_bundle(out_dir: Path, bundle: ArtifactBundle) -> list[ArtifactRef]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [write_artifact(out_dir, artifact) for artifact in bundle.artifacts]

def write_artifact(out_dir: Path, artifact: Artifact) -> ArtifactRef:
    final_path = out_dir / artifact.name
    temp_path = out_dir / f".{artifact.name}.tmp"
    body = artifact.content.encode("utf-8")
    temp_path.write_bytes(body)
    temp_path.replace(final_path)
    return ArtifactRef(
        kind=artifact.kind,
        path=artifact.name,
        media_type=artifact.media_type,
        bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )

def write_manifest(out_dir: Path, manifest: Manifest) -> ArtifactRef:
    return write_artifact(
        out_dir,
        Artifact(
            name="manifest.json",
            kind="manifest",
            media_type="application/json",
            content=model_to_json(manifest),
        ),
    )
