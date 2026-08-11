from __future__ import annotations

import types
from pathlib import Path

import pytest

from vctx.asr import FasterWhisperAsrAdapter
from vctx.config import AsrInstanceConfig
from vctx.source.local import LocalMediaAsset
from vctx.source.session import SourceRef


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
    model = _model(tmp_path)
    calls: dict[str, object] = {}

    class WhisperModel:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            calls.update(model_id=model_id, **kwargs)

        def transcribe(self, path: str, **kwargs: object) -> tuple[list[object], object]:
            del path, kwargs
            return [], _info()

    monkeypatch.setattr(
        "importlib.import_module",
        lambda _name: types.SimpleNamespace(WhisperModel=WhisperModel),
    )
    adapter = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model=str(model)),
        model_id=str(model),
        cache_root=tmp_path / "cache",
    )

    outcome = adapter.transcribe(_media(tmp_path / "audio.wav"))
    assert calls["model_id"] == str(model)
    assert calls["local_files_only"] is True
    assert outcome.kind == "no_speech"


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

    outcome = adapter.transcribe(_media(tmp_path / "audio.wav"))
    assert outcome.kind == "unavailable"
    assert outcome.receipt.failure == "missing_model"


def test_asr_rejects_vendor_model_without_transcribe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class WhisperModel:
        def __init__(self, _model_id: str, **_kwargs: object) -> None:
            pass

    outcome = _adapter(monkeypatch, tmp_path, WhisperModel).transcribe(
        _media(tmp_path / "audio.wav")
    )
    assert outcome.kind == "unavailable" and outcome.receipt.failure == "invalid_response"


def test_asr_verifies_no_speech_with_vad_and_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _model(tmp_path)
    passes: list[bool] = []

    class WhisperModel:
        def __init__(self, _model_id: str, **_kwargs: object) -> None:
            pass

        def transcribe(self, _path: str, **kwargs: object) -> tuple[list[object], object]:
            passes.append(bool(kwargs["vad_filter"]))
            return [], types.SimpleNamespace(
                language=None,
                language_probability=None,
                duration=4.0,
                duration_after_vad=0.0,
            )

    monkeypatch.setattr(
        "importlib.import_module",
        lambda _name: types.SimpleNamespace(WhisperModel=WhisperModel),
    )
    outcome = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model=str(model)),
        model_id=str(model),
        cache_root=tmp_path / "cache",
    ).transcribe(_media(tmp_path / "audio.wav"))

    assert outcome.kind == "no_speech"
    assert passes == [True, False]


def test_asr_failed_confirmation_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class WhisperModel:
        def __init__(self, _model_id: str, **_kwargs: object) -> None:
            self.calls = 0

        def transcribe(self, _path: str, **_kwargs: object) -> tuple[list[object], object]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("decoder failed")
            return [], _info()

    outcome = _adapter(monkeypatch, tmp_path, WhisperModel).transcribe(
        _media(tmp_path / "audio.wav")
    )
    assert outcome.kind == "unavailable"
    assert outcome.receipt.failure == "confirmation_failed"


def test_asr_rejects_invalid_timestamps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class WhisperModel:
        def __init__(self, _model_id: str, **_kwargs: object) -> None:
            pass

        def transcribe(self, _path: str, **_kwargs: object) -> tuple[list[object], object]:
            return [types.SimpleNamespace(start=-1.0, end=2.0, text="invalid")], _info()

    outcome = _adapter(monkeypatch, tmp_path, WhisperModel).transcribe(
        _media(tmp_path / "audio.wav")
    )
    assert outcome.kind == "unavailable"
    assert outcome.receipt.failure == "invalid_timestamps"


@pytest.mark.parametrize("fault", ["segment", "info"])
def test_asr_rejects_malformed_vendor_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    class WhisperModel:
        def __init__(self, _model_id: str, **_kwargs: object) -> None:
            pass

        def transcribe(self, _path: str, **_kwargs: object) -> tuple[list[object], object]:
            if fault == "segment":
                return [types.SimpleNamespace(start=0.0, end=1.0)], _info()
            segment = types.SimpleNamespace(start=0.0, end=1.0, text="speech")
            return [segment], types.SimpleNamespace(language_probability="certain")

    outcome = _adapter(monkeypatch, tmp_path, WhisperModel).transcribe(
        _media(tmp_path / "audio.wav")
    )
    assert outcome.kind == "unavailable" and outcome.receipt.failure == "invalid_response"


def test_asr_auto_device_falls_back_once_to_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    devices: list[str] = []

    class WhisperModel:
        def __init__(self, _model_id: str, **kwargs: object) -> None:
            devices.append(str(kwargs["device"]))
            if kwargs["device"] == "auto":
                raise RuntimeError("GPU unavailable")

        def transcribe(self, _path: str, **_kwargs: object) -> tuple[list[object], object]:
            return [], _info()

    outcome = _adapter(monkeypatch, tmp_path, WhisperModel).transcribe(
        _media(tmp_path / "audio.wav")
    )
    assert outcome.kind == "no_speech"
    assert outcome.receipt.device == "cpu"
    assert devices == ["auto", "cpu"]


def test_asr_explicit_cuda_never_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    devices: list[str] = []

    class WhisperModel:
        def __init__(self, _model_id: str, **kwargs: object) -> None:
            devices.append(str(kwargs["device"]))
            raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(
        "importlib.import_module",
        lambda _name: types.SimpleNamespace(WhisperModel=WhisperModel),
    )
    model = _model(tmp_path)
    outcome = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model=str(model), device="cuda"),
        model_id=str(model),
        cache_root=tmp_path / "cache",
    ).transcribe(_media(tmp_path / "audio.wav"))
    assert outcome.kind == "unavailable"
    assert devices == ["cuda"]


def test_asr_cuda_retries_pre_output_oom_with_smaller_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batches: list[int] = []

    class WhisperModel:
        def __init__(self, _model_id: str, **_kwargs: object) -> None:
            pass

        def transcribe(self, _path: str, **_kwargs: object) -> tuple[list[object], object]:
            raise AssertionError("CUDA execution must use the batched pipeline")

    class BatchedInferencePipeline:
        def __init__(self, *, model: object) -> None:
            del model

        def transcribe(
            self, _path: str, *, batch_size: int, **_kwargs: object
        ) -> tuple[list[object], object]:
            batches.append(batch_size)
            if batch_size == 8:
                raise RuntimeError("CUDA out of memory")
            segment = types.SimpleNamespace(start=0.0, end=1.2345, text=" speech ")
            return [segment], _info(language="en")

    module = types.SimpleNamespace(
        WhisperModel=WhisperModel, BatchedInferencePipeline=BatchedInferencePipeline
    )
    monkeypatch.setattr("importlib.import_module", lambda _name: module)
    model = _model(tmp_path)
    outcome = FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model=str(model), device="cuda"),
        model_id=str(model),
        cache_root=tmp_path / "cache",
    ).transcribe(_media(tmp_path / "audio.wav"))
    assert outcome.kind == "ready"
    assert outcome.receipt.batch_size == 4
    assert outcome.transcript.provenance.asr is not None
    assert outcome.transcript.provenance.asr.batch_size == 4
    assert outcome.transcript.segments[0].end == 1.234
    assert batches == [8, 4]


def _adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_type: type[object]
) -> FasterWhisperAsrAdapter:
    monkeypatch.setattr(
        "importlib.import_module",
        lambda _name: types.SimpleNamespace(WhisperModel=model_type),
    )
    model = _model(tmp_path)
    return FasterWhisperAsrAdapter(
        instance=AsrInstanceConfig(type="local-faster-whisper", model=str(model)),
        model_id=str(model),
        cache_root=tmp_path / "cache",
    )


def _info(*, language: str | None = None) -> object:
    return types.SimpleNamespace(
        language=language,
        language_probability=0.9 if language else None,
        duration=4.0,
        duration_after_vad=1.0,
    )


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").write_bytes(b"model")
    (model / "config.json").write_text("{}", encoding="utf-8")
    return model
