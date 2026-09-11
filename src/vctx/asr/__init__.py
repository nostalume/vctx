from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

from vctx.artifact.manifest import ManifestEffect
from vctx.asr.faster_whisper import (
    InferenceStreamError,
    InvalidVendorResponse,
    InvalidVendorTimestamps,
    WhisperApi,
    WhisperInfo,
    WhisperPass,
    WhisperTranscriber,
    admit_model,
    admit_pass,
    admit_transcriber,
    import_api,
    is_oom,
    load_bundled_cuda,
)
from vctx.config import AsrInstanceConfig, CapabilityPolicy
from vctx.model.store import ModelLifecycleError, ModelStore
from vctx.source.session import MediaAsset
from vctx.transcript import (
    AsrProvenance,
    DetectedLanguage,
    Transcript,
    TranscriptNoSpeech,
    TranscriptProvenance,
    TranscriptReady,
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
    def effect_seed(self) -> ManifestEffect:
        return ManifestEffect(
            operation="asr",
            status=self.selected,
            route=self.selected,
            provider=self.provider_id,
            model=self.model_id,
            uploaded=False,
            cost_may_apply=False,
            diagnostic="; ".join([self.reason, *self.warnings])[:500],
        )


class AsrEnvironment(BaseModel):
    installed: bool = False
    offline: bool = False
    model_id: str | None = None


class AsrRuntimeReadiness(BaseModel):
    requested_device: Literal["auto", "cpu", "cuda"]
    compute_type: str
    package_state: Literal["ready", "missing"]
    cuda_libraries: Literal["bundled", "missing", "not-applicable"]


AsrReadinessState = Literal[
    "disabled",
    "managed-ready",
    "managed-missing",
    "managed-incomplete",
    "explicit-ready",
    "explicit-missing",
    "corrupt",
    "unsupported-reference",
]


class AsrReadiness(BaseModel):
    state: AsrReadinessState
    model: str | None = None
    runtime: AsrRuntimeReadiness


class AsrReadinessFacts(BaseModel):
    model_kind: Literal["disabled", "managed", "explicit", "unsupported"]
    model_reference: str | None = None
    model_state: Literal["ready", "missing", "incomplete", "changed", "corrupt"] | None = None
    package_state: Literal["ready", "missing"]
    cuda_libraries: Literal["bundled", "missing", "not-applicable"] = "missing"


def decide_asr_readiness(
    policy: CapabilityPolicy,
    instance: AsrInstanceConfig,
    facts: AsrReadinessFacts,
) -> AsrReadiness:
    runtime = AsrRuntimeReadiness(
        requested_device=instance.device,
        compute_type=instance.compute,
        package_state=facts.package_state,
        cuda_libraries=facts.cuda_libraries,
    )
    if policy.disabled():
        return AsrReadiness(state="disabled", runtime=runtime)
    if facts.model_kind == "unsupported":
        state = "unsupported-reference"
    elif facts.model_kind == "explicit":
        state = {
            "ready": "explicit-ready",
            "missing": "explicit-missing",
            "incomplete": "corrupt",
            "changed": "corrupt",
            "corrupt": "corrupt",
        }[facts.model_state or "missing"]
    else:
        state = {
            "ready": "managed-ready",
            "missing": "managed-missing",
            "incomplete": "managed-incomplete",
            "changed": "corrupt",
            "corrupt": "corrupt",
        }[facts.model_state or "missing"]
    return AsrReadiness(
        state=cast(AsrReadinessState, state),
        model=facts.model_reference,
        runtime=runtime,
    )


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
    cache_hit: bool = False

    def provenance(self) -> AsrProvenance:
        return AsrProvenance.model_validate(self, from_attributes=True)


class AsrReady(TranscriptReady):
    receipt: AsrReceipt


class AsrNoSpeech(TranscriptNoSpeech):
    receipt: AsrReceipt


class AsrUnavailable(TranscriptUnavailable):
    receipt: AsrReceipt


type AsrOutcome = AsrReady | AsrNoSpeech | AsrUnavailable


class FasterWhisperAsrAdapter:
    def __init__(
        self, *, instance: AsrInstanceConfig, model_id: str | None, cache_root: Path
    ) -> None:
        self.instance = instance
        self.model_id = model_id or instance.model or "small"
        self.cache_root = cache_root
        self._model: WhisperTranscriber | None = None
        self._api: WhisperApi | None = None
        self._device = instance.device
        self._compute = instance.compute
        self._attempted_devices: list[str] = []
        self._fallback_reason: str | None = None
        self._cpu_threads = (
            instance.cpu_threads
            if isinstance(instance.cpu_threads, int)
            else max(1, min(8, os.process_cpu_count() or 1))
        )
        self._lock = threading.Lock()

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        if callable(close):
            close()
        self._model = None
        self._api = None

    def transcribe(
        self,
        media: MediaAsset,
        *,
        progress: bool = False,
        interval: tuple[float, float | None] | None = None,
    ) -> AsrOutcome:
        model_id, failure = self._local_model()
        if failure is not None:
            return failure
        assert model_id is not None
        loaded = self._load(model_id)
        if isinstance(loaded, AsrUnavailable):
            return loaded
        with self._lock:
            outcome = self._transcribe_loaded(media, progress=progress, interval=interval)
            if (
                isinstance(outcome, AsrUnavailable)
                and outcome.receipt.failure == "inference_failed"
                and self.instance.device == "auto"
                and self._device != "cpu"
            ):
                self._fallback_reason = outcome.reason
                self._model = None
                loaded = self._load_device(model_id, device="cpu", compute="auto")
                if isinstance(loaded, AsrUnavailable):
                    return loaded
                return self._transcribe_loaded(media, progress=progress, interval=interval)
            return outcome

    def _transcribe_loaded(
        self,
        media: MediaAsset,
        *,
        progress: bool,
        interval: tuple[float, float | None] | None,
    ) -> AsrOutcome:
        first = self._pass(media, vad=True, progress=progress, interval=interval)
        if isinstance(first, AsrUnavailable) or first.segments:
            return self._finish(media, first, confirmation=False)
        second = self._pass(media, vad=False, progress=progress, interval=interval)
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
            prepared = ModelStore(self.cache_root).require("asr", asr_model_id=self.model_id)
        except ModelLifecycleError as exc:
            code: AsrFailureCode = "corrupt_model" if "corrupt" in str(exc) else "missing_model"
            return None, self._unavailable(code, str(exc))
        self._revision = prepared.integrity
        return str(self.cache_root / prepared.cache_path), None

    def _load(self, model_id: str) -> WhisperTranscriber | AsrUnavailable:
        if self._model is not None:
            return self._model
        try:
            if self.instance.device != "cpu":
                load_bundled_cuda()
            self._api = import_api()
        except ModuleNotFoundError:
            return self._unavailable("missing_package", "install vctx[asr]")
        except InvalidVendorResponse as exc:
            return self._unavailable("invalid_response", str(exc))
        loaded = self._load_device(
            model_id, device=self.instance.device, compute=self.instance.compute
        )
        if not isinstance(loaded, AsrUnavailable):
            return loaded
        if self.instance.device != "auto":
            return loaded
        self._fallback_reason = loaded.reason
        return self._load_device(model_id, device="cpu", compute="auto")

    def _load_device(
        self, model_id: str, *, device: str, compute: str
    ) -> WhisperTranscriber | AsrUnavailable:
        assert self._api is not None
        self._attempted_devices.append(device)
        try:
            raw_model = self._api.model(
                model_id,
                device=device,
                compute_type=compute,
                local_files_only=True,
                cpu_threads=self._cpu_threads,
            )
        except Exception as exc:
            return self._unavailable("unsupported_hardware", str(exc))
        try:
            self._model, self._device, self._compute = admit_model(
                raw_model, device=device, compute=compute
            )
        except InvalidVendorResponse as exc:
            return self._unavailable("invalid_response", str(exc))
        return self._model

    def _pass(
        self,
        media: MediaAsset,
        *,
        vad: bool,
        progress: bool,
        interval: tuple[float, float | None] | None,
    ) -> WhisperPass | AsrUnavailable:
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
        use_pipeline = isinstance(self.instance.batch_size, int) or self._device != "cpu"
        if self._api is not None and self._api.pipeline is not None and use_pipeline:
            return self._batched_pass(
                media,
                vad=vad,
                vad_parameters=vad_parameters,
                progress=progress,
                interval=interval,
            )
        try:
            options = _transcribe_options(
                vad=vad,
                vad_parameters=vad_parameters,
                progress=progress,
                interval=interval,
            )
            raw = self._model.transcribe(str(media.local_path), **options)
            return admit_pass(raw)
        except InvalidVendorTimestamps as exc:
            return self._unavailable("invalid_timestamps", str(exc), vad=vad)
        except InvalidVendorResponse as exc:
            return self._unavailable("invalid_response", str(exc), vad=vad)
        except InferenceStreamError as exc:
            return self._unavailable("inference_failed", str(exc.cause), vad=vad)
        except Exception as exc:
            return self._unavailable("inference_failed", str(exc), vad=vad)

    def _batched_pass(
        self,
        media: MediaAsset,
        *,
        vad: bool,
        vad_parameters: dict[str, float | int] | None,
        progress: bool,
        interval: tuple[float, float | None] | None,
    ) -> WhisperPass | AsrUnavailable:
        assert self._api is not None and self._model is not None
        if self._api.pipeline is None:
            raise AssertionError("batched pass requires an admitted pipeline")
        try:
            pipeline = admit_transcriber(self._api.pipeline(model=self._model), label="pipeline")
        except InvalidVendorResponse as exc:
            return self._unavailable("invalid_response", str(exc), vad=vad)
        except Exception as exc:
            return self._unavailable("inference_failed", str(exc), vad=vad)
        batch_sizes = (
            (self.instance.batch_size,)
            if isinstance(self.instance.batch_size, int)
            else (8, 4, 2, 1)
        )
        for batch_size in batch_sizes:
            try:
                options = _transcribe_options(
                    vad=vad,
                    vad_parameters=vad_parameters,
                    progress=progress,
                    interval=interval,
                    batch_size=batch_size,
                )
                raw = pipeline.transcribe(str(media.local_path), **options)
                admitted = admit_pass(raw)
                return WhisperPass(admitted.segments, admitted.info, batch_size)
            except InvalidVendorTimestamps as exc:
                return self._unavailable("invalid_timestamps", str(exc), vad=vad)
            except InvalidVendorResponse as exc:
                return self._unavailable("invalid_response", str(exc), vad=vad)
            except InferenceStreamError as exc:
                if exc.emitted or not is_oom(exc.cause) or batch_size == 1:
                    return self._unavailable("inference_failed", str(exc.cause), vad=vad)
            except Exception as exc:
                if not is_oom(exc) or batch_size == 1:
                    return self._unavailable("inference_failed", str(exc), vad=vad)
        raise AssertionError("unreachable batch retry state")

    def _finish(
        self,
        media: MediaAsset,
        result: WhisperPass | AsrUnavailable,
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
                source_id=media.id,
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
        info: WhisperInfo,
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
            attempted_devices=self._attempted_devices,
            fallback_reason=self._fallback_reason,
            cpu_threads=self._cpu_threads,
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
                attempted_devices=self._attempted_devices,
                fallback_reason=self._fallback_reason,
                cpu_threads=self._cpu_threads,
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
            self.close()
            adapter = FasterWhisperAsrAdapter(
                instance=instance, model_id=model_id, cache_root=cache_root
            )
            self.local[key] = adapter
        return adapter

    def run(
        self,
        plan: AsrPlan,
        media: MediaAsset,
        *,
        instance: AsrInstanceConfig,
        cache_root: Path,
        progress: bool = False,
        interval: tuple[float, float | None] | None = None,
    ) -> AsrOutcome:
        if plan.selected != "local" or instance.type != "local-faster-whisper":
            raise ValueError(f"ASR plan is not executable: {plan.selected}")
        adapter = self.faster_whisper(
            instance=instance, model_id=plan.model_id, cache_root=cache_root
        )
        return adapter.transcribe(media, progress=progress, interval=interval)

    def close(self) -> None:
        for adapter in self.local.values():
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
        self.local.clear()


def _clip_timestamps(interval: tuple[float, float | None]) -> str:
    start, end = interval
    return str(start) if end is None else f"{start},{end}"


def _transcribe_options(
    *,
    vad: bool,
    vad_parameters: dict[str, float | int] | None,
    progress: bool,
    interval: tuple[float, float | None] | None,
    batch_size: int | None = None,
) -> dict[str, object]:
    options: dict[str, object] = {
        "language": None,
        "task": "transcribe",
        "word_timestamps": False,
        "vad_filter": vad,
        "vad_parameters": vad_parameters,
        "log_progress": progress,
    }
    if batch_size is not None:
        options["batch_size"] = batch_size
    if interval is not None:
        options.update(
            clip_timestamps=_clip_timestamps(interval),
            vad_filter=False,
            vad_parameters=None,
        )
    return options
