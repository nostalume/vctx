from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_path
from pydantic import BaseModel

from vctx.errors import OutputExistsError
from vctx.models.artifacts import Artifact, ArtifactBundle
from vctx.models.manifest import ArtifactRef, Manifest


class Cache(BaseModel):
    root: Path

    def path_for(self, key: str) -> Path:
        return self.root / key


def build_cache(cache_dir: Path | None) -> Cache:
    root = cache_dir or user_cache_path("vctx", appauthor=False)
    root.mkdir(parents=True, exist_ok=True)
    return Cache(root=root)


def model_to_json(model: BaseModel) -> str:
    return model.model_dump_json(indent=2) + "\n"


def validate_output_policy(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise OutputExistsError(f"output directory already exists: {out_dir}")


def write_artifact_bundle(out_dir: Path, bundle: ArtifactBundle) -> list[ArtifactRef]:
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: list[ArtifactRef] = []
    for artifact in bundle.artifacts:
        refs.append(write_artifact(out_dir, artifact))
    return refs


def write_artifact(out_dir: Path, artifact: Artifact) -> ArtifactRef:
    final_path = out_dir / artifact.name
    temp_path = out_dir / f".{artifact.name}.tmp"
    temp_path.write_text(artifact.content, encoding="utf-8")
    temp_path.replace(final_path)
    return ArtifactRef(kind=artifact.kind, path=artifact.name, media_type=artifact.media_type)


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
