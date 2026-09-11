from __future__ import annotations

import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.support import local_asr_model
from vctx.asr import AsrOutcome, AsrRuntimePool, FasterWhisperAsrAdapter
from vctx.config import AsrInstanceConfig
from vctx.source.local import LocalMediaAsset
from vctx.source.session import SourceRef

VendorInit = Callable[[Any, str, dict[str, object]], None]
VendorTranscribe = Callable[[Any, str, dict[str, object]], tuple[object, object]]


def _vendor(
    *, init: VendorInit | None = None, transcribe: VendorTranscribe | None = None
) -> type[object]:
    class WhisperModel:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            if init is not None:
                init(self, model_id, kwargs)

        def transcribe(self, path: str, **kwargs: object) -> tuple[object, object]:
            if transcribe is None:
                raise AttributeError("transcribe")
            return transcribe(self, path, kwargs)

    if transcribe is None:
        delattr(WhisperModel, "transcribe")
    return WhisperModel


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_type: type[object],
    *,
    instance: AsrInstanceConfig | None = None,
    pipeline: object | None = None,
) -> AsrOutcome:
    exports: dict[str, object] = {"WhisperModel": model_type}
    if pipeline is not None:
        exports["BatchedInferencePipeline"] = pipeline
    monkeypatch.setattr("importlib.import_module", lambda _name: types.SimpleNamespace(**exports))
    model = local_asr_model(tmp_path)
    selected = instance or AsrInstanceConfig(type="local-faster-whisper", model=str(model))
    return FasterWhisperAsrAdapter(
        instance=selected, model_id=str(model), cache_root=tmp_path / "cache"
    ).transcribe(_media(tmp_path / "audio.wav"))


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
    calls: dict[str, object] = {}

    def capture(_self: object, model_id: str, options: dict[str, object]) -> None:
        calls.update(model_id=model_id, **options)

    outcome = _run(
        monkeypatch,
        tmp_path,
        _vendor(init=capture, transcribe=lambda *_args: ([], _info())),
    )
    assert calls["model_id"] == str(tmp_path / "model")
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
    outcome = _run(monkeypatch, tmp_path, _vendor())
    assert outcome.kind == "unavailable" and outcome.receipt.failure == "invalid_response"


def test_asr_verifies_no_speech_with_vad_and_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    passes: list[bool] = []

    def transcribe(_self: object, _path: str, options: dict[str, object]) -> tuple[object, object]:
        passes.append(bool(options["vad_filter"]))
        return [], _info(duration_after_vad=0.0)

    outcome = _run(monkeypatch, tmp_path, _vendor(transcribe=transcribe))
    assert outcome.kind == "no_speech"
    assert passes == [True, False]


def test_asr_failed_confirmation_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def transcribe(model: Any, _path: str, _options: dict[str, object]) -> tuple[object, object]:
        calls = getattr(model, "calls", 0) + 1
        model.calls = calls
        if calls == 2:
            raise RuntimeError("decoder failed")
        return [], _info()

    outcome = _run(monkeypatch, tmp_path, _vendor(transcribe=transcribe))
    assert outcome.kind == "unavailable"
    assert outcome.receipt.failure == "confirmation_failed"


def test_asr_rejects_invalid_timestamps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = ([types.SimpleNamespace(start=-1.0, end=2.0, text="invalid")], _info())
    outcome = _run(monkeypatch, tmp_path, _vendor(transcribe=lambda *_args: result))
    assert outcome.kind == "unavailable"
    assert outcome.receipt.failure == "invalid_timestamps"


@pytest.mark.parametrize("fault", ["segment", "info"])
def test_asr_rejects_malformed_vendor_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    segment = types.SimpleNamespace(start=0.0, end=1.0, text="speech")
    result = (
        ([types.SimpleNamespace(start=0.0, end=1.0)], _info())
        if fault == "segment"
        else ([segment], types.SimpleNamespace(language_probability="certain"))
    )
    outcome = _run(monkeypatch, tmp_path, _vendor(transcribe=lambda *_args: result))
    assert outcome.kind == "unavailable" and outcome.receipt.failure == "invalid_response"


def test_asr_auto_device_falls_back_once_to_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    devices: list[str] = []

    def load(_self: object, _model_id: str, options: dict[str, object]) -> None:
        device = str(options["device"])
        devices.append(device)
        if device == "cuda":
            raise RuntimeError("GPU unavailable")

    outcome = _run(
        monkeypatch,
        tmp_path,
        _vendor(init=load, transcribe=lambda *_args: ([], _info())),
    )
    assert outcome.kind == "no_speech"
    assert outcome.receipt.device == "cpu"
    assert devices == ["cuda", "cpu"]


def test_asr_auto_device_falls_back_after_lazy_runtime_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    devices: list[str] = []

    def load(model: Any, _model_id: str, options: dict[str, object]) -> None:
        model.device = str(options["device"])
        devices.append(str(options["device"]))

    def transcribe(model: Any, _path: str, _options: dict[str, object]) -> tuple[object, object]:
        if model.device == "cuda":

            def failed_segments() -> object:
                raise RuntimeError("Could not load cublas64_12.dll")
                yield

            return failed_segments(), _info()
        return [], _info()

    outcome = _run(monkeypatch, tmp_path, _vendor(init=load, transcribe=transcribe))
    assert outcome.kind == "no_speech"
    assert outcome.receipt.device == "cpu"
    assert outcome.receipt.attempted_devices == ["cuda", "cpu"]
    assert devices == ["cuda", "cpu"]


def test_asr_explicit_cuda_never_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    devices: list[str] = []

    def fail(_self: object, _model_id: str, options: dict[str, object]) -> None:
        devices.append(str(options["device"]))
        raise RuntimeError("CUDA unavailable")

    model = local_asr_model(tmp_path)
    outcome = _run(
        monkeypatch,
        tmp_path,
        _vendor(init=fail),
        instance=AsrInstanceConfig(type="local-faster-whisper", model=str(model), device="cuda"),
    )
    assert outcome.kind == "unavailable"
    assert devices == ["cuda"]


def test_asr_cuda_retries_pre_output_oom_with_smaller_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batches: list[int] = []

    def direct(*_args: object) -> tuple[object, object]:
        raise AssertionError("CUDA execution must use the batched pipeline")

    class BatchedInferencePipeline:
        def __init__(self, *, model: object) -> None:
            del model

        def transcribe(
            self, _path: str, *, batch_size: int, **_kwargs: object
        ) -> tuple[object, object]:
            batches.append(batch_size)
            if batch_size == 8:
                raise RuntimeError("CUDA out of memory")
            segment = types.SimpleNamespace(start=0.0, end=1.2345, text=" speech ")
            return [segment], _info(language="en")

    model = local_asr_model(tmp_path)
    outcome = _run(
        monkeypatch,
        tmp_path,
        _vendor(transcribe=direct),
        instance=AsrInstanceConfig(type="local-faster-whisper", model=str(model), device="cuda"),
        pipeline=BatchedInferencePipeline,
    )
    assert outcome.kind == "ready"
    assert outcome.receipt.batch_size == 4
    assert outcome.transcript.provenance.asr is not None
    assert outcome.transcript.provenance.asr.batch_size == 4
    assert outcome.transcript.segments[0].end == 1.234
    assert batches == [8, 4]


def test_asr_auto_batch_uses_bounded_direct_cpu_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_options: dict[str, object] = {}
    transcribe_options: dict[str, object] = {}

    def load(_self: object, _model_id: str, options: dict[str, object]) -> None:
        model_options.update(options)

    def transcribe(_self: object, _path: str, options: dict[str, object]) -> tuple[object, object]:
        transcribe_options.update(options)
        return [types.SimpleNamespace(start=0.0, end=1.0, text="speech")], _info(language="en")

    def forbidden_pipeline(*, model: object) -> object:
        del model
        raise AssertionError("automatic CPU execution must not buffer a batch pipeline")

    model = local_asr_model(tmp_path)
    outcome = _run(
        monkeypatch,
        tmp_path,
        _vendor(init=load, transcribe=transcribe),
        instance=AsrInstanceConfig(
            type="local-faster-whisper",
            model=str(model),
            device="cpu",
            cpu_threads=2,
        ),
        pipeline=forbidden_pipeline,
    )
    assert outcome.kind == "ready"
    assert model_options["cpu_threads"] == 2
    assert "batch_size" not in transcribe_options
    assert (outcome.receipt.cpu_threads, outcome.receipt.batch_size) == (2, None)


def test_asr_runtime_pool_keeps_only_one_model_identity(tmp_path: Path) -> None:
    pool = AsrRuntimePool()
    first = AsrInstanceConfig(type="local-faster-whisper", model="small")
    second = AsrInstanceConfig(type="local-faster-whisper", model="tiny")
    pool.faster_whisper(instance=first, model_id="small", cache_root=tmp_path)
    pool.faster_whisper(instance=second, model_id="tiny", cache_root=tmp_path)
    assert len(pool.local) == 1


def _info(*, language: str | None = None, duration_after_vad: float = 1.0) -> object:
    return types.SimpleNamespace(
        language=language,
        language_probability=0.9 if language else None,
        duration=4.0,
        duration_after_vad=duration_after_vad,
    )
