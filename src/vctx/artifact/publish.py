from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from pydantic import ValidationError

from vctx.artifact.manifest import Manifest, ManifestSource
from vctx.errors import OutputExistsError


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
        if self.target.is_symlink() or (self.target.exists() and not self.target.is_dir()):
            raise OutputExistsError(f"output is not a directory: {self.target}")
        if self.target.exists() and any(self.target.iterdir()):
            self.previous = verify_pack(self.target)
        try:
            self._write_marker(state="building", exclusive=True)
            if self.previous is None:
                self.stage.mkdir()
            else:
                shutil.copytree(self.target, self.stage, copy_function=shutil.copy2)
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

    def rollback_lane(self, key: str) -> None:
        self.reset_lane(key)
        if self.previous is not None and any(
            source.key == key for source in self.previous.sources
        ):
            shutil.copytree(self.target / key, self.stage / key, copy_function=shutil.copy2)

    def commit(self, manifest: Manifest) -> None:
        verified = verify_pack(self.stage)
        if (
            verified.pack_id != manifest.pack_id
            or verified.updated_run_id != manifest.updated_run_id
        ):
            raise OutputExistsError("staged manifest identity changed before publication")
        self._write_marker(state="ready", current=manifest)
        had_target = self.target.exists() and any(self.target.iterdir())
        if self.target.exists() and not had_target:
            self.target.rmdir()
        try:
            if had_target:
                os.replace(self.target, self.backup)
            os.replace(self.stage, self.target)
            published = verify_pack(self.target)
            if published.updated_run_id != manifest.updated_run_id:
                raise OutputExistsError("published manifest identity does not match the run")
        except BaseException:
            if self.backup.exists():
                self._discard(self.target)
                os.replace(self.backup, self.target)
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
        if self.backup.exists() and not self.target.exists():
            os.replace(self.backup, self.target)
        elif self.backup.exists():
            current = verify_pack(self.target)
            if str(current.updated_run_id) != data.get("new_run_id"):
                raise OutputExistsError("ambiguous pack publication state")
            self._discard(self.backup)
        elif data.get("state") == "ready" and self.target.exists():
            current = verify_pack(self.target)
            if str(current.updated_run_id) != data.get("new_run_id"):
                raise OutputExistsError("ambiguous completed publication state")
        self._discard(self.stage)
        self.marker.unlink(missing_ok=True)

    def _write_marker(
        self, *, state: str, current: Manifest | None = None, exclusive: bool = False
    ) -> None:
        previous = self.previous
        identity = current or previous
        data = {
            "target": str(self.target),
            "state": state,
            "pack_id": str(identity.pack_id) if identity is not None else None,
            "old_run_id": str(previous.updated_run_id) if previous else None,
            "new_run_id": str(current.updated_run_id) if current else None,
        }
        with self.marker.open("x" if exclusive else "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True)

    @staticmethod
    def _discard(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def verify_pack(root: Path) -> Manifest:
    manifest_path = root / "manifest.json"
    try:
        if _linked(root) or _linked(manifest_path) or not manifest_path.is_file():
            raise ValueError("manifest is missing or linked")
        manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        allowed = {"manifest.json", *(source.path for source in manifest.sources)}
        if {entry.name for entry in root.iterdir()} != allowed:
            raise ValueError("pack root has unknown or missing entries")
        for source in manifest.sources:
            _verify_lane(root / source.path, source)
        return manifest
    except (OSError, ValueError, ValidationError) as exc:
        raise OutputExistsError(
            f"output is not a verified vctx 0.3 pack: {root} ({exc})"
        ) from exc


def _verify_lane(lane: Path, source: ManifestSource) -> None:
    refs = [(ref.path, ref.bytes, ref.sha256) for ref in source.artifacts]
    for asset in source.assets:
        if asset.retained:
            assert (
                asset.path is not None
                and asset.bytes is not None
                and asset.sha256 is not None
            )
            refs.append((asset.path, asset.bytes, asset.sha256))
    expected = {path for path, _, _ in refs}
    if _linked(lane) or not lane.is_dir():
        raise ValueError("source lane is missing or linked")
    actual = {entry.name for entry in lane.iterdir()}
    if len(expected) != len(refs) or actual != expected:
        detail = f"expected {sorted(expected)}, found {sorted(actual)}"
        raise ValueError(f"source lane contents differ: {detail}")
    for name, size, digest in refs:
        path = lane / name
        if _linked(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("source artifact is missing or linked")
        body = path.read_bytes()
        if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
            raise ValueError("source artifact failed integrity verification")


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())
