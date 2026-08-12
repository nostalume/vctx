from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import quote

from vctx.artifact.manifest import ManifestSource
from vctx.artifact.publish import open_pack, verify_required
from vctx.errors import VctxError
from vctx.projection import Renderer
from vctx.source.session import VideoMetadata
from vctx.summary import Summary
from vctx.transcript import ChunkSet, Transcript
from vctx.visual.evidence import Evidence

type RenderFormat = Literal["context", "read", "transcript"]


class RenderInputError(VctxError):
    exit_code = 4


class RenderWriteError(VctxError):
    exit_code = 1


@dataclass(frozen=True)
class RenderResult:
    content: str | None
    path: Path | None = None


def render_pack(
    pack: Path,
    *,
    source_key: str | None,
    format: RenderFormat,
    out: Path | None,
) -> RenderResult:
    root = pack.resolve()
    manifest = open_pack(root)
    source = _select_source(manifest.sources, source_key)
    kinds = _required_kinds(source, format)
    loaded = verify_required(root, source.key, kinds)
    transcript = cast(Transcript, loaded["transcript"])
    chunks = cast(
        ChunkSet,
        loaded.get("chunks")
        or ChunkSet(source_id=transcript.source_id, strategy="none", chunks=[]),
    )
    evidence = cast(Evidence | None, loaded.get("evidence"))
    renderer = Renderer(
        cast(VideoMetadata, loaded["metadata"]),
        transcript,
        chunks,
        evidence,
        cast(Summary | None, loaded.get("summary")),
        _frame_links(root / source.path, evidence, out),
    )
    content = {
        "context": renderer.context,
        "read": renderer.read,
        "transcript": renderer.transcript_text,
    }[format]()
    if out is None:
        return RenderResult(content)
    target = out.resolve()
    if target == root or target.is_relative_to(root):
        raise RenderWriteError("render output must be outside the source pack")
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    except OSError as exc:
        raise RenderWriteError(f"failed to write render: {target}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return RenderResult(None, target)


def _select_source(
    sources: list[ManifestSource], source_key: str | None
) -> ManifestSource:
    if source_key is not None:
        selected = next((source for source in sources if source.key == source_key), None)
        if selected is None:
            raise RenderInputError(f"source is not present in pack: {source_key}")
        return selected
    if len(sources) != 1:
        choices = ", ".join(source.key for source in sources)
        raise RenderInputError(f"multiple sources require --source; choose one of: {choices}")
    return sources[0]


def _required_kinds(source: ManifestSource, format: RenderFormat) -> set[str]:
    kinds = {"metadata", "transcript"}
    if format == "transcript":
        return kinds
    kinds.add("chunks")
    available = {artifact.kind for artifact in source.artifacts}
    return kinds | ({"evidence", "summary"} & available)


def _frame_links(
    lane: Path, evidence: Evidence | None, out: Path | None
) -> dict[str, str]:
    if evidence is None:
        return {}
    base = out.resolve().parent if out is not None else Path.cwd().resolve()
    return {
        capture.artifact_path: _relative_link(lane / capture.artifact_path, base)
        for capture in evidence.captures
    }


def _relative_link(target: Path, base: Path) -> str:
    try:
        relative = os.path.relpath(target.resolve(), start=base)
    except ValueError as exc:
        raise RenderWriteError(
            "render destination cannot express pack assets as relative links"
        ) from exc
    return quote(Path(relative).as_posix(), safe="/-._~")
