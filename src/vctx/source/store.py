from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import uuid4

from pydantic import BaseModel

from vctx.errors import CacheError, OfflineSourceError
from vctx.source.session import (
    EffectReceipt,
    MediaAsset,
    MediaPermit,
    MediaProfile,
    MediaRequest,
    SourceRecord,
    SourceRef,
    SourceSession,
    SubtitlePermit,
)
from vctx.transcript import TranscriptPayload

_SCHEMA_VERSION = 3

class BlobRef(BaseModel):
    sha256: str
    size: int

class CacheInventory(BaseModel):
    records: int = 0
    assets: int = 0
    blobs: int = 0
    temporary: int = 0
    bytes: int = 0

class PruneReceipt(BaseModel):
    dry_run: bool
    examined: int
    selected: int
    removed: int
    reclaimed_bytes: int
    failures: list[str]

class MediaEntry(BaseModel):
    id: str
    source: SourceRef
    container: str
    duration_seconds: float | None
    media_type: Literal["audio", "video", "unknown"]
    purpose: Literal["input", "asr", "visual"]
    profile: MediaProfile | None
    format_id: str
    provider: str

class StoredMediaAsset(MediaEntry):
    local_path: Path

@dataclass
class CachedSourceSession:
    record: SourceRecord
    subtitle: TranscriptPayload | None
    store: SourceStore
    record_id: str
    name: str = "source-store"
    receipts: list[EffectReceipt] = field(default_factory=lambda: [
        EffectReceipt(operation="observe", status="cache_hit")
    ])

    def transcript(self, *, permit: SubtitlePermit) -> TranscriptPayload:
        del permit
        if self.subtitle is None:
            raise OfflineSourceError("offline subtitle cache miss")
        self.receipts.append(
            EffectReceipt(operation="subtitle", status="cache_hit", purpose="transcript")
        )
        return self.subtitle

    def media(self, *, request: MediaRequest, permit: MediaPermit) -> MediaAsset:
        del permit
        asset = None if request.refresh else self.store.get_media(self.record_id, request)
        if asset is None:
            raise OfflineSourceError("offline media cache miss")
        self.receipts.append(
            _media_receipt(request, status="cache_hit", selected=asset.format_id)
        )
        return asset

@dataclass
class StoredSourceSession:
    inner: SourceSession
    store: SourceStore
    name: str = field(init=False)
    record: SourceRecord = field(init=False)
    receipts: list[EffectReceipt] = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.inner.name
        self.record = self.inner.record
        self.receipts = self.inner.receipts

    def transcript(self, *, permit: SubtitlePermit) -> TranscriptPayload:
        payload = self.inner.transcript(permit=permit)
        with suppress(CacheError):
            self.store.put_subtitle(self.record, payload)
        return payload

    def media(self, *, request: MediaRequest, permit: MediaPermit) -> MediaAsset:
        record_id = _record_id(self.record)
        if not request.refresh:
            cached = self.store.get_media(record_id, request)
            if cached is not None:
                self.receipts.append(
                    _media_receipt(request, status="cache_hit", selected=cached.format_id)
                )
                return cached
        asset = self.inner.media(request=request, permit=permit)
        try:
            stored = self.store.put_media(self.record, request, asset)
        except CacheError:
            return asset
        temp = asset.local_path.resolve()
        with suppress(OSError):
            if temp.is_relative_to((self.store.root / "tmp").resolve()):
                temp.unlink(missing_ok=True)
        return stored

@dataclass(frozen=True)
class SourceStore:
    root: Path

    @property
    def database(self) -> Path:
        return self.root / "index.sqlite3"

    def path_for(self, key: str) -> Path:
        return self.root / key

    def inventory(self) -> CacheInventory:
        if not self.root.exists():
            return CacheInventory()
        if self.root.is_symlink() or not self.root.is_dir():
            raise CacheError(f"invalid source cache root: {self.root}")
        if self.database.is_symlink():
            raise CacheError(f"linked source cache catalog: {self.database}")
        blobs = _owned_files(self.root, "blobs", digests=True)
        temporary = _owned_files(self.root, "tmp", recursive=True)
        records = assets = 0
        if self.database.exists():
            with self._connect(read_only=True) as connection:
                records = connection.execute("SELECT count(*) FROM source_record").fetchone()[0]
                assets = connection.execute("SELECT count(*) FROM asset").fetchone()[0]
        return CacheInventory(
            records=records,
            assets=assets,
            blobs=len(blobs),
            temporary=len(temporary),
            bytes=sum(path.stat().st_size for path in [*blobs, *temporary]),
        )

    def prune(
        self, *, before: datetime | None = None, all_records: bool = False, dry_run: bool = False
    ) -> PruneReceipt:
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise CacheError(f"invalid source cache root: {self.root}")
        if self.database.is_symlink() or (
            self.database.exists() and self.database.stat().st_nlink != 1
        ):
            raise CacheError(f"linked source cache catalog: {self.database}")
        blobs = _owned_files(self.root, "blobs", digests=True) if self.root.exists() else []
        temporary = _owned_files(self.root, "tmp", recursive=True) if self.root.exists() else []
        rows: list[tuple[str, str, str | None]] = []
        selected_ids: set[str] = set()
        referenced: set[str] = set()
        if self.database.is_file():
            try:
                with closing(self._connect(read_only=dry_run)) as connection:
                    if not dry_run:
                        connection.execute("BEGIN IMMEDIATE")
                    version = connection.execute("PRAGMA user_version").fetchone()[0]
                    column = "last_used_at" if version >= 3 else "NULL"
                    rows = connection.execute(
                        f"SELECT record_id, body, {column} FROM source_record"
                    ).fetchall()
                    selected_ids = {
                        key for key, body, used in rows
                        if all_records or (before is not None and _used_at(body, used) < before)
                    }
                    clause, values = "", ()
                    if selected_ids:
                        marks = ",".join("?" for _ in selected_ids)
                        clause, values = f" WHERE record_id NOT IN ({marks})", tuple(selected_ids)
                    referenced = {
                        row[0] for row in connection.execute(
                            f"SELECT DISTINCT digest FROM asset{clause}", values
                        )
                    }
                    if not dry_run:
                        for table in ("locator", "asset", "source_record"):
                            if selected_ids:
                                connection.execute(
                                    f"DELETE FROM {table} WHERE record_id IN ({marks})",
                                    tuple(selected_ids),
                                )
                        connection.commit()
            except sqlite3.Error as exc:
                raise CacheError(f"source cache is busy or invalid: {exc}") from exc
        candidates = [path for path in blobs if path.name not in referenced] + temporary
        selected = len(selected_ids) + len(candidates)
        failures: list[str] = []
        removed = 0 if dry_run else len(selected_ids)
        reclaimed = sum(path.stat().st_size for path in candidates) if dry_run else 0
        if not dry_run:
            for path in candidates:
                try:
                    size = path.stat().st_size
                    path.unlink()
                    reclaimed += size
                    removed += 1
                except OSError as exc:
                    failures.append(f"{path.name}: {exc}")
        return PruneReceipt(
            dry_run=dry_run, examined=len(rows) + len(blobs) + len(temporary),
            selected=selected, removed=removed, reclaimed_bytes=reclaimed, failures=failures
        )

    def get(self, locator: str) -> CachedSourceSession | None:
        if not self.database.is_file():
            return None
        try:
            with self._connect(read_only=True) as connection:
                row = connection.execute(
                    "SELECT r.record_id, r.body FROM locator l "
                    "JOIN source_record r ON r.record_id=l.record_id WHERE l.locator_hash=?",
                    (_digest(locator.encode()),),
                ).fetchone()
                if row is None:
                    return None
                record_id, body = row
                record = SourceRecord.model_validate_json(body)
                asset = connection.execute(
                    "SELECT digest, size FROM asset WHERE record_id=? AND kind='subtitle'",
                    (record_id,),
                ).fetchone()
                subtitle = None
                if asset is not None:
                    subtitle = TranscriptPayload.model_validate_json(
                        self._verified_blob(*asset).read_bytes()
                    )
                cached = CachedSourceSession(
                    record=record, subtitle=subtitle, store=self, record_id=record_id
                )
            with suppress(OSError, sqlite3.Error), sqlite3.connect(
                self.database, timeout=0.05
            ) as connection:
                connection.execute(
                    "UPDATE source_record SET last_used_at=? WHERE record_id=?",
                    (datetime.now(UTC).isoformat(), record_id),
                )
            return cached
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise CacheError(f"invalid source cache: {exc}") from exc

    def wrap(self, locator: str, session: SourceSession) -> StoredSourceSession:
        body = _canonical_json(session.record)
        record_id = _record_id(session.record)
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO source_record(record_id, body, last_used_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(record_id) DO UPDATE SET last_used_at=excluded.last_used_at",
                    (record_id, body.decode(), datetime.now(UTC).isoformat()),
                )
                connection.execute(
                    "INSERT INTO locator(locator_hash, record_id) VALUES (?, ?) "
                    "ON CONFLICT(locator_hash) DO UPDATE SET record_id=excluded.record_id",
                    (_digest(locator.encode()), record_id),
                )
        except (OSError, sqlite3.Error) as exc:
            raise CacheError(f"source cache write failed: {exc}") from exc
        return StoredSourceSession(session, self)

    def put_subtitle(self, record: SourceRecord, payload: TranscriptPayload) -> BlobRef:
        body = _canonical_json(payload)
        try:
            blob = self._put_blob(body)
            record_id = _record_id(record)
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO asset(record_id, kind, digest, size) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(record_id, kind) DO UPDATE SET "
                    "digest=excluded.digest, size=excluded.size",
                    (record_id, "subtitle", blob.sha256, blob.size),
                )
        except (OSError, sqlite3.Error) as exc:
            raise CacheError(f"source asset publication failed: {exc}") from exc
        return blob

    def get_media(self, record_id: str, request: MediaRequest) -> StoredMediaAsset | None:
        if not self.database.is_file():
            return None
        try:
            with self._connect(read_only=True) as connection:
                try:
                    row = connection.execute(
                        "SELECT digest, size, body FROM asset WHERE record_id=? AND kind=?",
                        (record_id, _media_key(request)),
                    ).fetchone()
                except sqlite3.OperationalError:
                    return None
                if row is None:
                    return None
                digest, size, body = row
                path = self._verified_blob(digest, size)
                entry = MediaEntry.model_validate_json(body)
                return StoredMediaAsset(**entry.model_dump(), local_path=path)
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise CacheError(f"invalid source media cache: {exc}") from exc

    def put_media(
        self, record: SourceRecord, request: MediaRequest, asset: MediaAsset
    ) -> StoredMediaAsset:
        purpose = "asr" if request.kind == "asr_audio" else "visual"
        if asset.purpose != purpose:
            raise CacheError("media purpose does not match its request")
        try:
            blob = self._put_path(asset.local_path)
            record_id = _record_id(record)
            entry = MediaEntry(
                id=asset.id,
                source=asset.source,
                container=asset.container,
                duration_seconds=asset.duration_seconds,
                media_type=asset.media_type,
                purpose=asset.purpose,
                profile=asset.profile,
                format_id=asset.format_id,
                provider=asset.provider,
            )
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO asset(record_id, kind, digest, size, body) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(record_id, kind) DO UPDATE SET "
                    "digest=excluded.digest, size=excluded.size, body=excluded.body",
                    (
                        record_id,
                        _media_key(request),
                        blob.sha256,
                        blob.size,
                        _canonical_json(entry).decode(),
                    ),
                )
            return StoredMediaAsset(
                **entry.model_dump(),
                local_path=self.root / "blobs" / blob.sha256,
            )
        except (OSError, sqlite3.Error) as exc:
            raise CacheError(f"source media publication failed: {exc}") from exc

    def _verified_blob(self, digest: str, size: int) -> Path:
        path = self.root / "blobs" / digest
        if not path.is_file() or path.stat().st_size != size:
            raise CacheError("cached blob is missing or has the wrong size")
        if _file_digest(path) != digest:
            raise CacheError("cached blob failed integrity verification")
        return path

    def _put_blob(self, body: bytes) -> BlobRef:
        return self._put_stream(BytesIO(body))

    def _put_path(self, source: Path) -> BlobRef:
        if not source.is_file():
            raise CacheError(f"media asset is missing: {source}")
        with source.open("rb") as reader:
            return self._put_stream(reader)

    def _put_stream(self, reader: BinaryIO) -> BlobRef:
        blobs = self.root / "blobs"
        temp_dir = self.root / "tmp"
        blobs.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp = temp_dir / f"{uuid4().hex}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            with temp.open("xb") as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(block)
                    digest.update(block)
                    size += len(block)
                writer.flush()
                os.fsync(writer.fileno())
            blob = BlobRef(sha256=digest.hexdigest(), size=size)
            final = blobs / blob.sha256
            if final.exists():
                if final.stat().st_size != blob.size or _file_digest(final) != blob.sha256:
                    raise CacheError("content-addressed blob collision failed verification")
            else:
                temp.replace(final)
            return blob
        finally:
            temp.unlink(missing_ok=True)

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            connection = sqlite3.connect(f"{self.database.resolve().as_uri()}?mode=ro", uri=True)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database, timeout=5)
        connection.execute("PRAGMA foreign_keys=ON")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > _SCHEMA_VERSION:
            connection.close()
            raise CacheError(f"source cache schema {version} is newer than {_SCHEMA_VERSION}")
        if not read_only and version == 0:
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS source_record(record_id TEXT PRIMARY KEY, "
                "body TEXT NOT NULL, last_used_at TEXT);"
                "CREATE TABLE IF NOT EXISTS locator(locator_hash TEXT PRIMARY KEY, "
                "record_id TEXT NOT NULL REFERENCES source_record(record_id));"
                "CREATE TABLE IF NOT EXISTS asset(record_id TEXT NOT NULL "
                "REFERENCES source_record(record_id), "
                "kind TEXT NOT NULL, digest TEXT NOT NULL, size INTEGER NOT NULL, "
                "body TEXT NOT NULL DEFAULT '{}', "
                "PRIMARY KEY(record_id, kind));"
                f"PRAGMA user_version={_SCHEMA_VERSION};"
            )
        elif not read_only and version in {1, 2}:
            if version == 1:
                connection.execute("ALTER TABLE asset ADD COLUMN body TEXT NOT NULL DEFAULT '{}'")
            connection.execute("ALTER TABLE source_record ADD COLUMN last_used_at TEXT")
            connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            connection.commit()
        elif version not in ({1, 2, _SCHEMA_VERSION} if read_only else {_SCHEMA_VERSION}):
            connection.close()
            raise CacheError("source cache schema is missing")
        return connection

def _canonical_json(model: BaseModel) -> bytes:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()

def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()

def _record_id(record: SourceRecord) -> str:
    identity = {"source_id": record.source_id, "revision": record.revision.model_dump(mode="json")}
    return _digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())

def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _media_key(request: MediaRequest) -> str:
    return "media:asr" if request.kind == "asr_audio" else f"media:visual:{request.profile}"

def _media_receipt(
    request: MediaRequest,
    *,
    status: Literal["cache_hit", "succeeded", "denied", "failed"],
    selected: str | None = None,
) -> EffectReceipt:
    requested = "audio" if request.kind == "asr_audio" else request.profile
    return EffectReceipt(
        operation="media",
        status=status,
        purpose="asr" if request.kind == "asr_audio" else "visual",
        requested_policy=requested,
        selected_policy=selected,
    )

def _owned_files(
    root: Path, name: str, *, recursive: bool = False, digests: bool = False
) -> list[Path]:
    directory = root / name
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise CacheError(f"invalid source cache directory: {directory}")
    entries = directory.rglob("*") if recursive else directory.iterdir()
    files = []
    for path in entries:
        linked = path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())
        if linked or (not path.is_file() and not (recursive and path.is_dir())):
            raise CacheError(f"invalid source cache entry under: {directory}")
        if path.is_file():
            if path.stat().st_nlink != 1:
                raise CacheError(f"linked source cache file: {path.name}")
            files.append(path)
    invalid_digest = digests and any(
        len(path.name) != 64 or not all(c in "0123456789abcdef" for c in path.name)
        for path in files
    )
    if invalid_digest:
        raise CacheError(f"invalid source blob name under: {directory}")
    return files
def _used_at(body: str, last_used: str | None) -> datetime:
    if last_used:
        return datetime.fromisoformat(last_used)
    return SourceRecord.model_validate_json(body).observed_at
