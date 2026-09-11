from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from vctx.artifact.manifest import ArtifactRef, Manifest, ManifestSource, schema_five_source
from vctx.errors import OutputExistsError


@dataclass(frozen=True)
class VerificationReport:
    manifest: Manifest
    unchecked_kinds: tuple[str, ...] = ()


class PackPublisher:
    def __init__(self, target: Path) -> None:
        self.target = Path(os.path.abspath(target))
        token = hashlib.sha256(str(self.target).encode()).hexdigest()[:10]
        stem = f".{self.target.name}.vctx-{token}"
        self.stage = self.target.parent / f"{stem}.stage"
        self.backup = self.target.parent / f"{stem}.backup"
        self.marker = self.target.parent / f"{stem}.json"
        self.previous: Manifest | None = None
        self._committed = False

    def __enter__(self) -> PackPublisher:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._recover()
        if _linked(self.target) or (self.target.exists() and not self.target.is_dir()):
            raise OutputExistsError(f"output is not a directory: {self.target}")
        if self.target.exists() and any(self.target.iterdir()):
            self.previous = verify_pack(self.target).manifest
        try:
            self._write_marker(state="building", exclusive=True)
            self.stage.mkdir()
        except BaseException:
            self._discard(self.stage)
            self.marker.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, *_error: object) -> None:
        if not self._committed:
            self._discard(self.stage)
            self.marker.unlink(missing_ok=True)

    def reset_lane(self, key: str) -> None:
        lane = self.stage / key
        if lane.parent != self.stage or _linked(lane):
            raise OutputExistsError(f"unsafe source lane: {key}")
        self._discard(lane)

    def hydrate_lane(self, source: ManifestSource) -> ManifestSource:
        projected = schema_five_source(source)
        self.reset_lane(projected.key)
        self._copy_lane(self.target / source.path, self.stage / projected.path, source, projected)
        return projected

    def commit(self, manifest: Manifest) -> None:
        self._write_marker(state="ready", current=manifest)
        had_target = self.target.exists() and any(self.target.iterdir())
        if self.target.exists() and not had_target:
            self.target.rmdir()
        try:
            if had_target:
                os.replace(self.target, self.backup)
            self._complete_stage(manifest)
            verified = verify_pack(self.stage).manifest
            if verified.pack_id != manifest.pack_id or verified.run.id != manifest.run.id:
                raise OutputExistsError("staged manifest identity changed before publication")
            os.replace(self.stage, self.target)
            published = verify_pack(self.target).manifest
            if published.run.id != manifest.run.id:
                raise OutputExistsError("published manifest identity does not match the run")
        except BaseException:
            if self.backup.exists():
                self._restore_previous(self.target if self.target.exists() else self.stage)
            raise
        self._discard(self.backup)
        self.marker.unlink(missing_ok=True)
        self._committed = True

    def _recover(self) -> None:
        remnants = self.stage.exists() or self.backup.exists()
        if not self.marker.exists():
            if remnants:
                raise OutputExistsError(f"unbound publication remnant beside output: {self.target}")
            return
        try:
            data = json.loads(self.marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OutputExistsError("invalid pack publication marker") from exc
        if data.get("target") != str(self.target):
            raise OutputExistsError("pack publication marker targets another path")
        if self.backup.exists():
            if self.target.exists():
                try:
                    current = verify_pack(self.target).manifest
                except OutputExistsError:
                    self._restore_previous(self.target)
                else:
                    if str(current.run.id) != data.get("new_run_id"):
                        self._restore_previous(self.target)
                    else:
                        self._discard(self.backup)
            else:
                self._restore_previous(self.stage)
        elif data.get("state") == "ready" and self.target.exists():
            current = verify_pack(self.target).manifest
            if str(current.run.id) != data.get("new_run_id"):
                raise OutputExistsError("ambiguous completed publication state")
        self._discard(self.stage)
        self.marker.unlink(missing_ok=True)

    def _complete_stage(self, manifest: Manifest) -> None:
        previous_by_key = (
            {source.key: source for source in self.previous.sources}
            if self.previous is not None
            else {}
        )
        for source in manifest.sources:
            lane = self.stage / source.path
            if lane.exists():
                continue
            prior = previous_by_key.get(source.key)
            previous = self.backup / (prior.path if prior is not None else source.path)
            if not previous.is_dir() or _linked(previous):
                raise OutputExistsError(f"prior source lane is unavailable: {source.key}")
            projected = prior is not None and (
                prior.path != source.path
                or [item.path for item in prior.artifacts]
                != [item.path for item in source.artifacts]
            )
            if projected:
                self._copy_lane(previous, lane, prior, source)
                continue
            try:
                lane.parent.mkdir(parents=True, exist_ok=True)
                os.replace(previous, lane)
            except OSError:
                lane.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(previous, lane, copy_function=shutil.copy2)

    @staticmethod
    def _copy_lane(origin: Path, lane: Path, old: ManifestSource, new: ManifestSource) -> None:
        if len(old.artifacts) != len(new.artifacts) or _linked(origin):
            raise OutputExistsError(f"prior source lane is unavailable: {old.key}")
        try:
            lane.mkdir(parents=True)
            for before, after in zip(old.artifacts, new.artifacts, strict=True):
                source = origin / before.path
                target = lane / after.path
                if _linked(source) or not source.is_file():
                    raise OutputExistsError(f"prior source artifact is unavailable: {before.path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        except OSError as exc:
            raise OutputExistsError(f"prior source lane cannot be copied: {old.key}") from exc

    def _restore_previous(self, container: Path) -> None:
        try:
            previous = Manifest.model_validate_json(
                (self.backup / "manifest.json").read_text(encoding="utf-8")
            )
            for source in previous.sources:
                destination = self.backup / source.path
                candidate = container / source.path
                if not candidate.exists():
                    candidate = container / "sources" / source.key
                if not destination.exists() and candidate.is_dir() and not _linked(candidate):
                    os.replace(candidate, destination)
            self._discard(container)
            os.replace(self.backup, self.target)
        except (OSError, ValidationError, ValueError) as exc:
            raise OutputExistsError("could not recover the previous pack generation") from exc

    def _write_marker(
        self, *, state: str, current: Manifest | None = None, exclusive: bool = False
    ) -> None:
        previous = self.previous
        identity = current or previous
        data = {
            "target": str(self.target),
            "state": state,
            "pack_id": str(identity.pack_id) if identity is not None else None,
            "old_run_id": str(previous.run.id) if previous else None,
            "new_run_id": str(current.run.id) if current else None,
        }
        with self.marker.open("x" if exclusive else "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True)

    @staticmethod
    def _discard(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir() and not _linked(path):
            shutil.rmtree(path)
        else:
            path.unlink()


def open_pack(root: Path) -> Manifest:
    manifest_path = root / "manifest.json"
    try:
        if _linked(root) or _linked(manifest_path) or not manifest_path.is_file():
            raise ValueError("manifest is missing or linked")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") == "2":
            raise OutputExistsError(
                "schema-2 pack is read-only; regenerate it with the current vctx prepare command"
            )
        manifest = Manifest.model_validate(raw)
        for source in manifest.sources:
            lane = root / source.path
            if _linked(lane) or not lane.is_dir():
                raise ValueError("source lane is missing or linked")
            for artifact in source.artifacts:
                path = lane / Path(artifact.path)
                if _linked(path) or not path.is_file() or path.stat().st_nlink != 1:
                    raise ValueError("source artifact is missing or linked")
        return manifest
    except OutputExistsError:
        raise
    except (OSError, ValueError, ValidationError) as exc:
        raise _pack_error(root, exc) from exc


def verify_required(root: Path, source_key: str, kinds: set[str]) -> dict[str, Any]:
    manifest = open_pack(root)
    source = next((item for item in manifest.sources if item.key == source_key), None)
    if source is None:
        raise OutputExistsError(f"source lane is not present in pack: {source_key}")
    refs = {artifact.kind: artifact for artifact in source.artifacts}
    if len(refs) != len(source.artifacts):
        repeated = {
            item.kind
            for item in source.artifacts
            if sum(other.kind == item.kind for other in source.artifacts) > 1
        }
        if repeated - {"visual_frame"}:
            raise OutputExistsError("required product kind is ambiguous")
    loaded: dict[str, Any] = {}
    visiting: set[str] = set()

    def require(kind: str) -> Any:
        if kind in loaded:
            return loaded[kind]
        if kind in visiting:
            raise ValueError("product dependency cycle")
        visiting.add(kind)
        for dependency in _DEPENDENCIES.get(kind, ()):
            if dependency in refs:
                require(dependency)
        ref = refs.get(kind)
        if ref is None:
            raise ValueError(f"required product is missing: {kind}")
        loaded[kind] = _read_artifact(root / source.path, ref, decode=True)
        visiting.remove(kind)
        _validate_relations(kind, loaded, source, root / source.path)
        return loaded[kind]

    try:
        for kind in sorted(kinds):
            require(kind)
        return loaded
    except (OSError, ValueError, ValidationError) as exc:
        raise _pack_error(root, exc) from exc


def verify_pack(root: Path) -> VerificationReport:
    try:
        manifest = open_pack(root)
        expected_root = (
            {"manifest.json", "sources"}
            if manifest.schema_version == "4"
            else {"manifest.json", *(source.path for source in manifest.sources)}
        )
        if {entry.name for entry in root.iterdir()} != expected_root:
            raise ValueError("pack root contains unlisted or missing entries")
        if manifest.schema_version == "4":
            indexed = {source.key for source in manifest.sources}
            actual = {entry.name for entry in (root / "sources").iterdir()}
            if actual != indexed:
                raise ValueError("pack sources directory contains unlisted or missing lanes")
        unchecked: set[str] = set()
        for source in manifest.sources:
            _verify_complete_lane(root / source.path, source)
            loaded: dict[str, Any] = {}
            for artifact in source.artifacts:
                decoded = _read_artifact(root / source.path, artifact, decode=True)
                if decoded is _UNCHECKED:
                    unchecked.add(artifact.kind)
                else:
                    loaded[artifact.kind] = decoded
            for kind in _DEPENDENCIES:
                if kind in loaded:
                    _validate_relations(kind, loaded, source, root / source.path)
        return VerificationReport(manifest, tuple(sorted(unchecked)))
    except OutputExistsError:
        raise
    except (OSError, ValueError, ValidationError) as exc:
        raise _pack_error(root, exc) from exc


_UNCHECKED = object()
_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "chunks": ("transcript",),
    "evidence": ("evidence_plan",),
    "summary": ("transcript", "evidence"),
}


def _read_artifact(lane: Path, ref: ArtifactRef, *, decode: bool) -> Any:
    path = lane / Path(ref.path)
    if _linked(path) or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("source artifact is missing or linked")
    digest, prefix = _hash_file(path)
    if path.stat().st_size != ref.bytes or digest != ref.sha256:
        raise ValueError(f"source artifact failed integrity verification: {ref.path}")
    if not decode:
        return None
    if ref.kind in {"context", "read"}:
        return path.read_text(encoding="utf-8")
    if ref.kind == "visual_frame":
        if prefix != b"\x89PNG\r\n\x1a\n":
            raise ValueError("visual frame is not a PNG")
        return ref.path
    loaders: dict[str, type[BaseModel]] = _json_loaders()
    model = loaders.get(ref.kind)
    if model is None:
        return _UNCHECKED
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _json_loaders() -> dict[str, type[BaseModel]]:
    from vctx.source.session import VideoMetadata
    from vctx.summary import Summary
    from vctx.transcript import ChunkSet, Transcript
    from vctx.visual.evidence import Evidence
    from vctx.visual.plan import EvidencePlan

    return {
        "metadata": VideoMetadata,
        "transcript": Transcript,
        "chunks": ChunkSet,
        "evidence_plan": EvidencePlan,
        "evidence": Evidence,
        "summary": Summary,
    }


def _validate_relations(
    kind: str, loaded: dict[str, Any], source: ManifestSource, lane: Path
) -> None:
    if kind == "chunks" and "transcript" in loaded:
        transcript = loaded["transcript"]
        chunks = loaded["chunks"]
        segment_ids = {segment.id for segment in transcript.segments}
        if chunks.source_id != transcript.source_id or any(
            not set(chunk.segment_ids) <= segment_ids for chunk in chunks.chunks
        ):
            raise ValueError("chunks do not resolve against the transcript")
    elif kind == "evidence":
        if "evidence_plan" not in loaded:
            raise ValueError("evidence plan is missing")
        evidence = loaded["evidence"]
        evidence.validate_against(loaded["evidence_plan"])
        refs = {artifact.path: artifact for artifact in source.artifacts}
        for capture in evidence.captures:
            ref = refs.get(capture.artifact_path)
            if ref is None or ref.kind != "visual_frame":
                raise ValueError("evidence capture references an unlisted frame")
            _read_artifact(lane, ref, decode=True)
    elif kind == "summary" and "transcript" in loaded:
        summary = loaded["summary"]
        transcript = loaded["transcript"]
        if summary.source_id != transcript.source_id:
            raise ValueError("summary belongs to another transcript")
        segments = {segment.id for segment in transcript.segments}
        captures = (
            {capture.id for capture in loaded.get("evidence", ()).captures}
            if "evidence" in loaded
            else set()
        )
        for point in summary.points:
            if not set(point.segment_ids) <= segments or not set(point.capture_ids) <= captures:
                raise ValueError("summary citation does not resolve to canonical products")


def _verify_complete_lane(lane: Path, source: ManifestSource) -> None:
    expected = {artifact.path for artifact in source.artifacts}
    expected_dirs = {
        parent.as_posix()
        for name in expected
        for parent in Path(name).parents
        if parent != Path(".")
    }
    actual: set[str] = set()
    actual_dirs: set[str] = set()
    for entry in lane.rglob("*"):
        if _linked(entry):
            raise ValueError("source lane contains a linked entry")
        relative = entry.relative_to(lane).as_posix()
        if entry.is_file():
            actual.add(relative)
        elif entry.is_dir():
            actual_dirs.add(relative)
        else:
            raise ValueError("source lane contains an unsupported entry")
    if actual != expected or actual_dirs != expected_dirs:
        raise ValueError(
            "source lane contains unlisted or missing files: "
            f"expected {sorted(expected)}, found {sorted(actual)}"
        )


def _hash_file(path: Path) -> tuple[str, bytes]:
    digest = hashlib.sha256()
    prefix = b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if not prefix:
                prefix = block[:8]
            digest.update(block)
    return digest.hexdigest(), prefix


def _pack_error(root: Path, exc: Exception) -> OutputExistsError:
    return OutputExistsError(f"output is not a verified vctx schema-3/4/5 pack: {root} ({exc})")


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())
