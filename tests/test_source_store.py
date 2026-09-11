from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from vctx.errors import CacheError, OfflineSourceError
from vctx.source.session import (
    AsrAudioRequest,
    EffectReceipt,
    MediaAsset,
    MediaPermit,
    MediaRequest,
    Revision,
    SourceRecord,
    SourceRef,
    SourceSession,
    SubtitlePermit,
    VideoMetadata,
    VisualVideoRequest,
)
from vctx.source.store import SourceStore
from vctx.transcript import TranscriptPayload, TranscriptProvenance, UnknownLanguage


@dataclass
class _Session:
    record: SourceRecord
    payload: TranscriptPayload
    name: str = "fixture"
    receipts: list[EffectReceipt] = field(default_factory=list)

    def transcript(self, *, permit: object) -> TranscriptPayload:
        del permit
        return self.payload

    media_path: Path | None = None
    media_calls: int = 0

    def media(self, *, request: MediaRequest, permit: object) -> MediaAsset:
        del permit
        assert self.media_path is not None
        self.media_calls += 1
        purpose = "asr" if isinstance(request, AsrAudioRequest) else "visual"
        return MediaAsset(
            id=self.record.source_id,
            source=self.record.metadata.source,
            local_path=self.media_path,
            capabilities={"audio"} if purpose == "asr" else {"video"},
            purpose=purpose,
            profile=request.profile if isinstance(request, VisualVideoRequest) else None,
        )


def _session() -> _Session:
    source = SourceRef(kind="url", value="https://video.example/watch")
    return _Session(
        record=SourceRecord(
            source_id="example__abc",
            revision=Revision(kind="observed", value="revision"),
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            metadata=VideoMetadata(id="example__abc", source=source, title="Lecture"),
        ),
        payload=TranscriptPayload(
            text="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n",
            format="vtt",
            provenance=TranscriptProvenance(
                method="official_subtitles",
                language_evidence=UnknownLanguage(reason="fixture"),
                format="vtt",
                provider="fixture",
            ),
        ),
    )


def test_source_store_miss_is_lazy(tmp_path: Path) -> None:
    root = tmp_path / "source"

    assert SourceStore(root).get("https://video.example/missing") is None
    assert not root.exists()


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_source_store_verifies_bytes_and_never_persists_raw_locator(
    tmp_path: Path, damage: str
) -> None:
    store = SourceStore(tmp_path / "source")
    locator = "https://video.example/watch?v=abc&token=secret"
    session = _session()
    tracked = store.wrap(locator, cast(SourceSession, session))
    tracked.transcript(permit=SubtitlePermit(network="allowed"))

    cached = store.get(locator)
    assert cached is not None
    assert cached.transcript(permit=SubtitlePermit(network="denied")).text == session.payload.text
    assert b"secret" not in store.database.read_bytes()

    blob = next((store.root / "blobs").iterdir())
    if damage == "missing":
        blob.unlink()
    else:
        blob.write_bytes(b"x" * blob.stat().st_size)
    with pytest.raises(CacheError, match="missing|integrity"):
        store.get(locator)


def test_source_store_rejects_newer_schema_without_mutating_it(tmp_path: Path) -> None:
    store = SourceStore(tmp_path / "source")
    store.root.mkdir(parents=True)
    with sqlite3.connect(store.database) as connection:
        connection.execute("PRAGMA user_version=5")

    with pytest.raises(CacheError, match="newer"):
        store.get("https://video.example/watch")
    with sqlite3.connect(store.database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5


def test_source_store_serializes_concurrent_publication(tmp_path: Path) -> None:
    store = SourceStore(tmp_path / "source")
    session = cast(SourceSession, _session())
    locators = [f"https://video.example/watch?v={index}" for index in range(4)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(store.wrap, locator, session) for locator in locators]
        for future in futures:
            future.result()

    assert all(store.get(locator) is not None for locator in locators)


def test_source_store_reuses_exact_media_by_purpose_and_profile_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SourceStore(tmp_path / "source")
    locator = "https://video.example/watch?v=abc"
    session = _session()
    session.media_path = tmp_path / "download.webm"
    session.media_path.write_bytes(b"downloaded-once")
    tracked = store.wrap(locator, cast(SourceSession, session))
    permit = MediaPermit(network="allowed")

    audio = tracked.media(request=AsrAudioRequest(), permit=permit)
    assert audio.local_path.read_bytes() == b"downloaded-once"
    assert session.media_calls == 1
    monkeypatch.setattr(
        "vctx.source.store._file_digest", lambda _path: pytest.fail("unchanged blob was rehashed")
    )
    cached = store.get(locator)
    assert cached is not None
    offline_audio = cached.media(request=AsrAudioRequest(), permit=MediaPermit(network="denied"))
    assert (offline_audio.local_path, offline_audio.purpose) == (audio.local_path, "asr")

    reobserved = _session()
    reobserved.record.observed_at = datetime(2026, 1, 2, tzinfo=UTC)
    reobserved.record.source_capabilities = {"audio", "video"}
    online_again = store.wrap(locator, cast(SourceSession, reobserved))
    reused = online_again.media(request=AsrAudioRequest(), permit=permit)
    assert (reused.local_path, reobserved.media_calls) == (audio.local_path, 0)
    refreshed = store.get(locator)
    assert refreshed is not None and refreshed.record.source_capabilities == {"audio", "video"}

    with pytest.raises(OfflineSourceError, match="offline media cache miss"):
        cached.media(
            request=VisualVideoRequest(profile="fast"),
            permit=MediaPermit(network="denied"),
        )


def test_source_store_prune_age_preserves_shared_blob_until_last_record(tmp_path: Path) -> None:
    store = SourceStore(tmp_path / "source")
    old = _session()
    old.record.revision.value = "old"
    old.record.observed_at = datetime(2025, 1, 1, tzinfo=UTC)
    new = _session()
    new.record.revision.value = "new"
    new.record.observed_at = datetime(2026, 8, 1, tzinfo=UTC)
    for locator, session in (("old", old), ("new", new)):
        tracked = store.wrap(locator, cast(SourceSession, session))
        tracked.transcript(permit=SubtitlePermit(network="allowed"))
    with sqlite3.connect(store.database) as connection:
        connection.execute("UPDATE source_record SET last_used_at=NULL")

    aged = store.prune(before=datetime(2026, 1, 1, tzinfo=UTC))

    assert aged.selected == aged.removed == 1
    assert aged.reclaimed_bytes == 0
    assert len(list((store.root / "blobs").iterdir())) == 1
    assert store.get("old") is None and store.get("new") is not None

    cleared = store.prune(all_records=True)
    assert cleared.selected == cleared.removed == 2
    assert cleared.reclaimed_bytes > 0
    assert not list((store.root / "blobs").iterdir())


def test_source_store_prune_refuses_busy_catalog_without_changes(tmp_path: Path) -> None:
    store = SourceStore(tmp_path / "source")
    store.wrap("source", cast(SourceSession, _session()))
    with sqlite3.connect(store.database) as lock:
        lock.execute("BEGIN IMMEDIATE")
        with pytest.raises(CacheError, match="busy|locked"):
            store.prune(all_records=True)

    assert store.get("source") is not None


def test_store_adopts_controlled_media_without_second_full_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SourceStore(tmp_path / "source")
    session = _session()
    session.media_path = store.root / "tmp" / "lease" / "audio.m4a"
    session.media_path.parent.mkdir(parents=True)
    session.media_path.write_bytes(b"controlled-media")
    tracked = store.wrap("source", cast(SourceSession, session))
    monkeypatch.setattr(
        SourceStore,
        "_put_stream",
        lambda *_args: pytest.fail("controlled media was copied through a second stream"),
    )

    asset = tracked.media(request=AsrAudioRequest(), permit=MediaPermit(network="allowed"))

    assert asset.local_path.read_bytes() == b"controlled-media"
    assert not session.media_path.exists()
