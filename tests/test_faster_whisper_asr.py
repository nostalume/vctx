from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from vctx.config import AsrInstanceConfig
from vctx.models import SourceRef
from vctx.models.media import LocalMediaAsset, MediaAsset
from vctx.transforms.asr import AsrExecutionError, FasterWhisperAsrAdapter


def test_faster_whisper_uses_persistent_model_cache_and_offline_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    class FakeSegment:
        start = 0.0
        end = 1.25
        text = " hello from real boundary "

    class FakeWhisperModel:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs

        def transcribe(self, path: str, **kwargs: object) -> tuple[list[FakeSegment], object]:
            calls["path"] = path
            calls["transcribe_kwargs"] = kwargs
            return [FakeSegment()], object()

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)

    cache_root = tmp_path / "cache"
    media = _media_asset(tmp_path / "lecture.wav")
    media.local_path.write_bytes(b"fake audio")
    adapter = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model="tiny"),
        model_id="tiny",
        cache_root=cache_root,
        offline=True,
    )

    payload = adapter.transcribe(media)

    assert calls["model_id"] == "tiny"
    assert calls["kwargs"] == {
        "compute_type": "default",
        "device": "auto",
        "download_root": str(cache_root / "models" / "faster-whisper"),
        "local_files_only": True,
    }
    assert calls["path"] == str(media.local_path)
    assert calls["transcribe_kwargs"] == {"language": None}
    assert payload.format == "vtt"
    assert payload.provenance.method == "asr"
    assert payload.provenance.provider == "faster-whisper"
    assert "hello from real boundary" in payload.text


def test_faster_whisper_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def raise_import_error(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.import_module", raise_import_error)
    media = _media_asset(tmp_path / "lecture.wav")
    media.local_path.write_bytes(b"fake audio")
    adapter = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model="tiny"),
        model_id="tiny",
        cache_root=tmp_path / "cache",
        offline=False,
    )

    with pytest.raises(AsrExecutionError, match="Install the ASR extra"):
        adapter.transcribe(media)


def test_explicit_local_model_path_disables_managed_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    class FakeSegment:
        start = 0.0
        end = 1.0
        text = "local path"

    class FakeWhisperModel:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs

        def transcribe(self, path: str, **kwargs: object) -> tuple[list[FakeSegment], object]:
            del path, kwargs
            return [FakeSegment()], object()

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)
    model_dir = tmp_path / "models" / "tiny-local"
    model_dir.mkdir(parents=True)
    media = _media_asset(tmp_path / "lecture.wav")
    media.local_path.write_bytes(b"fake audio")

    adapter = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model=str(model_dir)),
        model_id=str(model_dir),
        cache_root=tmp_path / "cache",
        offline=False,
    )

    adapter.transcribe(media)

    assert calls["model_id"] == str(model_dir)
    assert calls["kwargs"] == {
        "compute_type": "default",
        "device": "auto",
        "local_files_only": True,
    }


def test_managed_cache_write_failure_is_actionable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise OSError("No space left on device")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    media = _media_asset(tmp_path / "lecture.wav")
    media.local_path.write_bytes(b"fake audio")
    adapter = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model="tiny"),
        model_id="tiny",
        cache_root=tmp_path / "cache",
        offline=False,
    )

    with pytest.raises(AsrExecutionError, match="ASR model cache is not writable"):
        adapter.transcribe(media)


def test_faster_whisper_reports_model_load_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeWhisperModel:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            del model_id, kwargs
            raise RuntimeError("model cache miss with network disabled")

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)
    media = _media_asset(tmp_path / "lecture.wav")
    media.local_path.write_bytes(b"fake audio")
    adapter = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model="tiny"),
        model_id="tiny",
        cache_root=tmp_path / "cache",
        offline=True,
    )

    with pytest.raises(AsrExecutionError, match="faster-whisper ASR failed"):
        adapter.transcribe(media)


def _media_asset(path: Path) -> MediaAsset:
    return LocalMediaAsset(
        id="local__lecture",
        source=SourceRef(kind="file", value=str(path)),
        local_path=path,
        media_type="audio",
        container="wav",
        provider="local-file",
    )
