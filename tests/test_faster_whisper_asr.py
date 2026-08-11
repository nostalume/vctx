from __future__ import annotations

import types
from pathlib import Path

import pytest

from vctx.config import AsrInstanceConfig
from vctx.source.local import LocalMediaAsset
from vctx.source.session import SourceRef
from vctx.transforms.asr import AsrExecutionError, FasterWhisperAsrAdapter


def _media(path: Path) -> LocalMediaAsset:
    path.write_bytes(b"audio")
    return LocalMediaAsset(
        id="audio",
        source=SourceRef(kind="file", value=str(path)),
        local_path=path,
        container="wav",
        media_type="audio",
    )


def test_asr_uses_explicit_local_model_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    calls: dict[str, object] = {}

    class WhisperModel:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            calls.update(model_id=model_id, **kwargs)

        def transcribe(self, path: str, **kwargs: object) -> tuple[list[object], object]:
            del path, kwargs
            return [], object()

    monkeypatch.setattr(
        "importlib.import_module",
        lambda _name: types.SimpleNamespace(WhisperModel=WhisperModel),
    )
    adapter = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model=str(model)),
        model_id=str(model),
        cache_root=tmp_path / "cache",
    )

    adapter.transcribe(_media(tmp_path / "audio.wav"))
    assert calls["model_id"] == str(model)
    assert calls["local_files_only"] is True


def test_asr_rejects_unprepared_named_model_before_loading_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "importlib.import_module",
        lambda _name: pytest.fail("adapter construction could trigger a download"),
    )
    adapter = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model="base"),
        model_id="base",
        cache_root=tmp_path / "cache",
    )

    with pytest.raises(AsrExecutionError, match="vctx models pull asr"):
        adapter.transcribe(_media(tmp_path / "audio.wav"))
