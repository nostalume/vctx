from __future__ import annotations

import importlib
import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vctx.app.models import ModelLifecycleError, require_prepared_model
from vctx.artifact.manifest import TransformEvidence
from vctx.config import AsrInstanceConfig, CapabilityPolicy
from vctx.source.session import MediaAsset
from vctx.transcript import (
    AsrProvenance,
    DetectedLanguage,
    Transcript,
    TranscriptNoSpeech,
    TranscriptProvenance,
    TranscriptReady,
    TranscriptSegment,
    TranscriptUnavailable,
    UnknownLanguage,
)


class AsrPlan(BaseModel):
    selected: Literal["local", "skipped", "unavailable"]
    reason: str
    provider_id: str | None = None
    model_id: str | None = None
    requirements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    deterministic: bool = False

    @property
    def evidence_seed(self) -> TransformEvidence:
        return TransformEvidence(
            capability="asr",
            selected_route=self.selected,
            provider_id=self.provider_id,
            model_id=self.model_id,
            requires_user_config=False,
            uploaded=False,
            cost_may_apply=False,
            deterministic=self.deterministic,
            reason=self.reason,
            warnings=self.warnings,
        )


class AsrEnvironment(BaseModel):
    installed: bool = False
    offline: bool = False
    model_id: str | None = None


def plan_asr(
    policy: CapabilityPolicy,
    environment: AsrEnvironment,
    *,
    has_transcript: bool,
    has_media: bool,
) -> AsrPlan:
    if has_transcript:
        return AsrPlan(
            selected="skipped", reason="transcript already available", deterministic=True
        )
    if policy.disabled():
        return AsrPlan(selected="skipped", reason="ASR disabled by policy")
    if not has_media:
        return AsrPlan(
            selected="unavailable",
            reason=("No transcript found and no media asset is available for ASR."),
            requirements=["media asset"],
        )
    if environment.installed:
        return AsrPlan(
            selected="local",
            provider_id="faster-whisper",
            model_id=policy.model_ref() or environment.model_id or "small",
            reason="default local ASR route available",
        )
    return AsrPlan(
        selected="unavailable",
        reason="No transcript found and the prepared local ASR route is unavailable.",
        requirements=["install ASR extra", "prepare ASR model", "provide transcript file"],
    )


AsrFailureCode = Literal[
    "missing_package",
    "missing_model",
    "corrupt_model",
    "unwritable_cache",
    "unsupported_hardware",
    "inference_failed",
    "confirmation_failed",
    "invalid_timestamps",
    "invalid_response",
]


class AsrReceipt(AsrProvenance):
    failure: AsrFailureCode | None = None

    def provenance(self) -> AsrProvenance:
        return AsrProvenance.model_validate(self, from_attributes=True)


class AsrReady(TranscriptReady):
    receipt: AsrReceipt


class AsrNoSpeech(TranscriptNoSpeech):
    receipt: AsrReceipt


class AsrUnavailable(TranscriptUnavailable):
    receipt: AsrReceipt


type AsrOutcome = AsrReady | AsrNoSpeech | AsrUnavailable


class TimedText(Protocol):
    start: float
    end: float
    text: str


class _WhisperModel(Protocol):
    def transcribe(
        self,
        path: str,
        *,
        language: None,
        task: Literal["transcribe"],
        word_timestamps: Literal[False],
        vad_filter: bool,
        vad_parameters: dict[str, float | int] | None,
    ) -> object: ...


class _WhisperPipeline(Protocol):
    def transcribe(
        self,
        path: str,
        *,
        batch_size: int,
        language: None,
        task: Literal["transcribe"],
        word_timestamps: Literal[False],
        vad_filter: bool,
        vad_parameters: dict[str, float | int] | None,
    ) -> object: ...


class _WhisperModelFactory(Protocol):
    def __call__(
        self,
        model_id: str,
        *,
        device: str,
        compute_type: str,
        local_files_only: Literal[True],
    ) -> object: ...


class _WhisperPipelineFactory(Protocol):
    def __call__(self, *, model: _WhisperModel) -> object: ...


@dataclass(frozen=True)
class _WhisperApi:
    model: _WhisperModelFactory
    pipeline: _WhisperPipelineFactory | None


class _WhisperInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    language: str | None
    language_probability: float | None = Field(strict=True, ge=0, le=1)
    duration: float = Field(strict=True, ge=0)
    duration_after_vad: float = Field(strict=True, ge=0)


class _WhisperSegment(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    start: float = Field(strict=True)
    end: float = Field(strict=True)
    text: str


@dataclass(frozen=True)
class _WhisperPass:
    segments: list[TranscriptSegment]
    info: _WhisperInfo
    batch_size: int | None = None


class _InvalidVendorResponse(ValueError):
    pass


class _InvalidVendorTimestamps(ValueError):
    pass


class _InferenceStreamError(RuntimeError):
    def __init__(self, cause: Exception, *, emitted: bool) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.emitted = emitted


class FasterWhisperAsrAdapter:
    def __init__(
        self, *, instance: AsrInstanceConfig, model_id: str | None, cache_root: Path
    ) -> None:
        self.instance = instance
        self.model_id = model_id or instance.model or "small"
        self.cache_root = cache_root
        self._model: _WhisperModel | None = None
        self._api: _WhisperApi | None = None
        self._device = instance.device
        self._compute = instance.compute
        self._lock = threading.Lock()

    def transcribe(self, media: MediaAsset) -> AsrOutcome:
        model_id, failure = self._local_model()
        if failure is not None:
            return failure
        assert model_id is not None
        loaded = self._load(model_id)
        if isinstance(loaded, AsrUnavailable):
            return loaded
        with self._lock:
            first = self._pass(media, vad=True)
            if isinstance(first, AsrUnavailable) or first.segments:
                return self._finish(media, first, confirmation=False)
            second = self._pass(media, vad=False)
            if isinstance(second, AsrUnavailable):
                return second.model_copy(
                    update={
                        "reason": f"ASR silence confirmation failed: {second.reason}",
                        "receipt": second.receipt.model_copy(
                            update={"failure": "confirmation_failed", "confirmation": True}
                        ),
                    }
                )
            if not second.segments:
                return AsrNoSpeech(
                    reason="two successful ASR passes found no speech",
                    receipt=self._receipt(second.info, vad=False, confirmation=True),
                )
            return self._finish(media, second, confirmation=True)

    def _local_model(self) -> tuple[str | None, AsrUnavailable | None]:
        path = Path(self.model_id)
        if path.is_absolute() or path.exists():
            if not path.is_dir() or any(
                not (path / name).is_file() for name in ("model.bin", "config.json")
            ):
                return None, self._unavailable("corrupt_model", "invalid CTranslate2 model")
            return str(path), None
        if self.instance.cache == "disabled":
            return None, self._unavailable("missing_model", "disabled cache requires model path")
        try:
            prepared = require_prepared_model("asr", self.cache_root, asr_model_id=self.model_id)
        except ModelLifecycleError as exc:
            code: AsrFailureCode = "corrupt_model" if "corrupt" in str(exc) else "missing_model"
            return None, self._unavailable(code, str(exc))
        self._revision = prepared.integrity
        return str(self.cache_root / prepared.cache_path), None

    def _load(self, model_id: str) -> _WhisperModel | AsrUnavailable:
        if self._model is not None:
            return self._model
        try:
            self._api = _admit_whisper_api(importlib.import_module("faster_whisper"))
        except ModuleNotFoundError:
            return self._unavailable("missing_package", "install vctx[asr]")
        except _InvalidVendorResponse as exc:
            return self._unavailable("invalid_response", str(exc))
        try:
            raw_model = self._api.model(
                model_id,
                device=self.instance.device,
                compute_type=self.instance.compute,
                local_files_only=True,
            )
        except Exception as exc:
            if self.instance.device != "auto":
                return self._unavailable("unsupported_hardware", str(exc))
            try:
                raw_model = self._api.model(
                    model_id, device="cpu", compute_type="auto", local_files_only=True
                )
                self._device = "cpu"
            except Exception as fallback:
                return self._unavailable("unsupported_hardware", str(fallback))
        try:
            self._model, self._device, self._compute = _admit_whisper_model(
                raw_model, default_device=self._device, default_compute=self._compute
            )
        except _InvalidVendorResponse as exc:
            return self._unavailable("invalid_response", str(exc))
        return self._model

    def _pass(self, media: MediaAsset, *, vad: bool) -> _WhisperPass | AsrUnavailable:
        assert self._model is not None
        vad_parameters = (
            {
                "threshold": 0.5,
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400,
            }
            if vad
            else None
        )
        if self._device == "cuda":
            return self._batched_pass(media, vad=vad, vad_parameters=vad_parameters)
        try:
            raw = self._model.transcribe(
                str(media.local_path),
                language=None,
                task="transcribe",
                word_timestamps=False,
                vad_filter=vad,
                vad_parameters=vad_parameters,
            )
            return _admit_whisper_pass(raw)
        except _InvalidVendorTimestamps as exc:
            return self._unavailable("invalid_timestamps", str(exc), vad=vad)
        except _InvalidVendorResponse as exc:
            return self._unavailable("invalid_response", str(exc), vad=vad)
        except _InferenceStreamError as exc:
            return self._unavailable("inference_failed", str(exc.cause), vad=vad)
        except Exception as exc:
            return self._unavailable("inference_failed", str(exc), vad=vad)

    def _batched_pass(
        self,
        media: MediaAsset,
        *,
        vad: bool,
        vad_parameters: dict[str, float | int] | None,
    ) -> _WhisperPass | AsrUnavailable:
        assert self._api is not None and self._model is not None
        if self._api.pipeline is None:
            return self._unavailable(
                "invalid_response", "faster-whisper lacks BatchedInferencePipeline", vad=vad
            )
        try:
            pipeline = _admit_whisper_pipeline(self._api.pipeline(model=self._model))
        except _InvalidVendorResponse as exc:
            return self._unavailable("invalid_response", str(exc), vad=vad)
        except Exception as exc:
            return self._unavailable("inference_failed", str(exc), vad=vad)
        for batch_size in (8, 4, 2, 1):
            try:
                raw = pipeline.transcribe(
                    str(media.local_path),
                    batch_size=batch_size,
                    language=None,
                    task="transcribe",
                    word_timestamps=False,
                    vad_filter=vad,
                    vad_parameters=vad_parameters,
                )
                admitted = _admit_whisper_pass(raw)
                return _WhisperPass(admitted.segments, admitted.info, batch_size)
            except _InvalidVendorTimestamps as exc:
                return self._unavailable("invalid_timestamps", str(exc), vad=vad)
            except _InvalidVendorResponse as exc:
                return self._unavailable("invalid_response", str(exc), vad=vad)
            except _InferenceStreamError as exc:
                if exc.emitted or not _is_oom(exc.cause) or batch_size == 1:
                    return self._unavailable("inference_failed", str(exc.cause), vad=vad)
            except Exception as exc:
                if not _is_oom(exc) or batch_size == 1:
                    return self._unavailable("inference_failed", str(exc), vad=vad)
        raise AssertionError("unreachable batch retry state")

    def _finish(
        self,
        media: MediaAsset,
        result: _WhisperPass | AsrUnavailable,
        *,
        confirmation: bool,
    ) -> AsrOutcome:
        if isinstance(result, AsrUnavailable):
            return result
        segments, info, batch = result.segments, result.info, result.batch_size
        if not segments:
            return self._unavailable("invalid_timestamps", "ASR emitted no usable timed text")
        language = info.language
        confidence = info.language_probability
        evidence = (
            DetectedLanguage(code=language, source="asr", confidence=confidence)
            if language
            else UnknownLanguage(reason="ASR language not reported")
        )
        receipt = self._receipt(info, vad=not confirmation, confirmation=confirmation, batch=batch)
        return AsrReady(
            transcript=Transcript(
                video_id=media.id,
                provenance=TranscriptProvenance(
                    method="asr",
                    language=language,
                    language_evidence=evidence,
                    format="json",
                    provider="faster-whisper",
                    asr=receipt.provenance(),
                ),
                segments=segments,
            ),
            receipt=receipt,
        )

    def _receipt(
        self,
        info: _WhisperInfo,
        *,
        vad: bool,
        confirmation: bool = False,
        batch: int | None = None,
    ) -> AsrReceipt:
        return AsrReceipt(
            provider="faster-whisper",
            model=self.model_id,
            revision=getattr(self, "_revision", None),
            device=self._device,
            compute_type=self._compute,
            batch_size=batch,
            vad=vad,
            confirmation=confirmation,
            source_duration=info.duration,
            speech_duration=info.duration_after_vad,
            language=info.language,
            language_confidence=info.language_probability,
        )

    def _unavailable(
        self, code: AsrFailureCode, reason: str, *, vad: bool = True
    ) -> AsrUnavailable:
        return AsrUnavailable(
            reason=reason,
            receipt=AsrReceipt(
                provider="faster-whisper",
                model=self.model_id,
                revision=getattr(self, "_revision", None),
                device=self._device,
                compute_type=self._compute,
                vad=vad,
                failure=code,
            ),
        )


@dataclass
class AsrRuntimePool:
    local: dict[str, FasterWhisperAsrAdapter] = field(default_factory=dict)

    def faster_whisper(
        self, *, instance: AsrInstanceConfig, model_id: str | None, cache_root: Path
    ) -> FasterWhisperAsrAdapter:
        key = f"{instance.model_dump_json()}:{model_id}:{cache_root.resolve()}"
        adapter = self.local.get(key)
        if adapter is None:
            adapter = FasterWhisperAsrAdapter(
                instance=instance, model_id=model_id, cache_root=cache_root
            )
            self.local[key] = adapter
        return adapter


def run_asr(
    plan: AsrPlan,
    media: MediaAsset,
    *,
    instance: AsrInstanceConfig,
    cache_root: Path,
    runtimes: AsrRuntimePool | None = None,
) -> AsrOutcome:
    if plan.selected == "local" and instance.type == "local-faster-whisper":
        pool = runtimes or AsrRuntimePool()
        return pool.faster_whisper(
            instance=instance, model_id=plan.model_id, cache_root=cache_root
        ).transcribe(media)
    raise ValueError(f"ASR plan is not executable: {plan.selected}")


def _admit_whisper_api(module: object) -> _WhisperApi:
    model = getattr(module, "WhisperModel", None)
    if not callable(model):
        raise _InvalidVendorResponse("faster-whisper lacks callable WhisperModel")
    pipeline = getattr(module, "BatchedInferencePipeline", None)
    if pipeline is not None and not callable(pipeline):
        raise _InvalidVendorResponse("faster-whisper BatchedInferencePipeline is not callable")
    return _WhisperApi(
        model=cast(_WhisperModelFactory, model),
        pipeline=cast(_WhisperPipelineFactory, pipeline) if pipeline is not None else None,
    )


def _admit_whisper_model(
    raw: object, *, default_device: str, default_compute: str
) -> tuple[_WhisperModel, str, str]:
    if not callable(getattr(raw, "transcribe", None)):
        raise _InvalidVendorResponse("faster-whisper model lacks callable transcribe")
    engine = getattr(raw, "model", None)
    device = str(getattr(engine, "device", default_device))
    compute = str(getattr(engine, "compute_type", default_compute))
    return cast(_WhisperModel, raw), device, compute


def _admit_whisper_pipeline(raw: object) -> _WhisperPipeline:
    if not callable(getattr(raw, "transcribe", None)):
        raise _InvalidVendorResponse("faster-whisper pipeline lacks callable transcribe")
    return cast(_WhisperPipeline, raw)


def _admit_whisper_pass(raw: object) -> _WhisperPass:
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise _InvalidVendorResponse("faster-whisper transcribe must return segments and info")
    raw_segments, raw_info = raw
    if not isinstance(raw_segments, Iterable):
        raise _InvalidVendorResponse("faster-whisper segments are not iterable")
    try:
        info = _WhisperInfo.model_validate(raw_info)
    except ValidationError as exc:
        raise _InvalidVendorResponse(str(exc)) from exc
    vendor_segments: list[_WhisperSegment] = []
    iterator = iter(raw_segments)
    while True:
        try:
            raw_segment = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            raise _InferenceStreamError(exc, emitted=bool(vendor_segments)) from exc
        try:
            vendor_segments.append(_WhisperSegment.model_validate(raw_segment))
        except ValidationError as exc:
            raise _InvalidVendorResponse(str(exc)) from exc
    try:
        return _WhisperPass(_admit_segments(vendor_segments), info)
    except ValueError as exc:
        raise _InvalidVendorTimestamps(str(exc)) from exc


def _admit_segments(segments: Iterable[TimedText]) -> list[TranscriptSegment]:
    admitted: list[TranscriptSegment] = []
    previous = -1.0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        start, end = float(segment.start), float(segment.end)
        if not all(math.isfinite(value) and value >= 0 for value in (start, end)):
            raise ValueError("ASR timestamps must be finite and non-negative")
        if end < start or start < previous:
            raise ValueError("ASR timestamps must have ordered bounds")
        start, end = round(start, 3), round(end, 3)
        admitted.append(
            TranscriptSegment(id=f"seg_{len(admitted) + 1:06d}", start=start, end=end, text=text)
        )
        previous = start
    return admitted


def _is_oom(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "out of memory" in message or "cuda" in message and "memory" in message
