from __future__ import annotations

import hashlib
import mimetypes
import os
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from vctx.artifact.manifest import ArtifactRef, Manifest
from vctx.errors import CacheError
from vctx.projection import Renderer
from vctx.source.session import MediaAsset, VideoMetadata
from vctx.summary import Summary
from vctx.transcript import ChunkSet, Transcript, TranscriptPayload
from vctx.visual.evidence import Evidence
from vctx.visual.plan import EvidencePlan


@dataclass(frozen=True)
class Artifact:
    name: str
    kind: str
    media_type: str
    body: bytes

    @classmethod
    def json(cls, name: str, kind: str, model: BaseModel) -> Artifact:
        return cls(name, kind, "application/json", encode_json(model).encode("utf-8"))


type ArtifactBundle = tuple[Artifact, ...]


def encode_json(model: BaseModel) -> str:
    return model.model_dump_json(indent=2) + "\n"


def product_bundle(
    *,
    metadata: VideoMetadata,
    transcript: Transcript,
    chunks: ChunkSet,
    projections: Collection[str],
    evidence: Evidence | None = None,
    evidence_plan: EvidencePlan | None = None,
    summary: Summary | None = None,
) -> ArtifactBundle:
    renderer = Renderer(metadata, transcript, chunks, evidence, summary)
    artifacts = [
        Artifact.json("metadata.json", "metadata", metadata),
        Artifact.json("transcript.json", "transcript", transcript),
        Artifact.json("chunks.json", "chunks", chunks),
    ]
    if evidence_plan is not None:
        artifacts.append(Artifact.json("evidence-plan.json", "evidence_plan", evidence_plan))
    if evidence is not None:
        artifacts.append(Artifact.json("evidence.json", "evidence", evidence))
    if summary is not None:
        artifacts.append(Artifact.json("summary.json", "summary", summary))
    if "context" in projections:
        artifacts.append(
            Artifact("context.md", "context", "text/markdown", renderer.context().encode())
        )
    if "read" in projections:
        artifacts.append(Artifact("read.md", "read", "text/markdown", renderer.read().encode()))
    return tuple(artifacts)


def write_bundle(lane: Path, bundle: ArtifactBundle) -> list[ArtifactRef]:
    lane.mkdir(parents=True, exist_ok=True)
    return [write_artifact(lane, artifact) for artifact in bundle]


def write_artifact(lane: Path, artifact: Artifact) -> ArtifactRef:
    reference = ArtifactRef(
        kind=artifact.kind,
        path=artifact.name,
        media_type=artifact.media_type,
        bytes=len(artifact.body),
        sha256=hashlib.sha256(artifact.body).hexdigest(),
    )
    final = lane / artifact.name
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(f".{final.name}.tmp")
    try:
        temporary.write_bytes(artifact.body)
        temporary.replace(final)
    finally:
        temporary.unlink(missing_ok=True)
    return reference


def write_manifest(root: Path, manifest: Manifest) -> ArtifactRef:
    return write_artifact(root, Artifact.json("manifest.json", "manifest", manifest))


def retain_source_files(
    media: MediaAsset | None,
    subtitle: TranscriptPayload | None,
    lane: Path,
    *,
    retain: bool,
) -> tuple[list[ArtifactRef], list[str]]:
    if not retain:
        omissions = (
            ["disabled by --no-retain-media/output.retain_media"]
            if subtitle is not None or media is not None
            else []
        )
        return [], omissions
    written: list[ArtifactRef] = []
    paths: list[Path] = []
    try:
        if subtitle is not None:
            language = _token(subtitle.provenance.language or "und")
            extension = {"plain": "txt", "unknown": "txt"}.get(subtitle.format, subtitle.format)
            body = subtitle.original_bytes or subtitle.text.encode("utf-8")
            artifact = Artifact(f"subtitle.{language}.{extension}", "subtitle", "text/plain", body)
            written.append(write_artifact(lane, artifact))
            paths.append(lane / artifact.name)
        if media is not None:
            extension = _token(media.container if media.container != "unknown" else "bin")
            name = f"media.{extension}"
            source = media.local_path.resolve()
            if not source.is_file() or source.stat().st_size == 0:
                raise CacheError(f"source media is missing or empty: {source}")
            reference = _copy_file(
                source, lane / name, name, expected=getattr(media, "sha256", None)
            )
            written.append(reference)
            paths.append(lane / name)
    except (OSError, CacheError) as exc:
        for path in paths:
            path.unlink(missing_ok=True)
        raise CacheError(f"source asset materialization failed: {exc}") from exc
    return written, []


def _copy_file(source: Path, final: Path, name: str, *, expected: str | None = None) -> ArtifactRef:
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(f".{final.name}.tmp")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
                digest.update(block)
            writer.flush()
            os.fsync(writer.fileno())
        if expected is not None and digest.hexdigest() != expected:
            raise CacheError("admitted media changed during retention")
        temporary.replace(final)
    finally:
        temporary.unlink(missing_ok=True)
    return ArtifactRef(
        kind="media",
        path=name,
        media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
        bytes=final.stat().st_size,
        sha256=digest.hexdigest(),
    )


def _token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_").lower() or "unknown"
