from __future__ import annotations

import importlib
import math
import mimetypes
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, Field

from vctx.app.models import ModelLifecycleError, require_prepared_model
from vctx.config import AsrInstanceConfig
from vctx.net import NetRequest, NetRuntime, UrllibNetRuntime
from vctx.source.session import MediaAsset
from vctx.transcript import (
    DetectedLanguage,
    Transcript,
    TranscriptNoSpeech,
    TranscriptProvenance,
    TranscriptReady,
    TranscriptSegment,
    TranscriptUnavailable,
    UnknownLanguage,
)
from vctx.transforms.planning import RoutePlan

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


class AsrReceipt(BaseModel):
    provider: str
    model: str
    revision: str | None = None
    device: str
    compute_type: str
    batch_size: int | None = None
    vad: bool
    confirmation: bool = False
    source_duration: float | None = None
    speech_duration: float | None = None
    language: str | None = None
    language_confidence: float | None = None
    timestamp_method: Literal["segment-milliseconds"] = "segment-milliseconds"
    failure: AsrFailureCode | None = None


class AsrReady(TranscriptReady):
    receipt: AsrReceipt


class AsrNoSpeech(TranscriptNoSpeech):
    receipt: AsrReceipt


class AsrUnavailable(TranscriptUnavailable):
    receipt: AsrReceipt


type AsrOutcome = AsrReady | AsrNoSpeech | AsrUnavailable


class WhisperSegment(Protocol):
    start: float
    end: float
    text: str


class OpenAiAsrSegment(BaseModel):
    start: float
    end: float
    text: str


class OpenAiAsrResponse(BaseModel):
    segments: list[OpenAiAsrSegment] = Field(default_factory=list)


class FasterWhisperAsrAdapter:
    def __init__(
        self, *, instance: AsrInstanceConfig, model_id: str | None, cache_root: Path
    ) -> None:
        self.instance = instance
        self.model_id = model_id or instance.model or "small"
        self.cache_root = cache_root
        self._model: Any | None = None
        self._module: Any | None = None
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
            if isinstance(first, AsrUnavailable) or first[0]:
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
            if not second[0]:
                return AsrNoSpeech(
                    reason="two successful ASR passes found no speech",
                    receipt=self._receipt(second[1], vad=False, confirmation=True),
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

    def _load(self, model_id: str) -> Any | AsrUnavailable:
        if self._model is not None:
            return self._model
        try:
            self._module = importlib.import_module("faster_whisper")
        except ModuleNotFoundError:
            return self._unavailable("missing_package", "install vctx[asr]")
        try:
            self._model = self._module.WhisperModel(
                model_id,
                device=self.instance.device,
                compute_type=self.instance.compute,
                local_files_only=True,
            )
        except Exception as exc:
            if self.instance.device != "auto":
                return self._unavailable("unsupported_hardware", str(exc))
            try:
                self._model = self._module.WhisperModel(
                    model_id, device="cpu", compute_type="auto", local_files_only=True
                )
                self._device = "cpu"
            except Exception as fallback:
                return self._unavailable("unsupported_hardware", str(fallback))
        self._device = str(getattr(getattr(self._model, "model", None), "device", self._device))
        self._compute = str(
            getattr(getattr(self._model, "model", None), "compute_type", self._compute)
        )
        return self._model

    def _pass(
        self, media: MediaAsset, *, vad: bool
    ) -> tuple[list[WhisperSegment], object, int | None] | AsrUnavailable:
        assert self._model is not None
        kwargs: dict[str, object] = {
            "language": None,
            "task": "transcribe",
            "word_timestamps": False,
            "vad_filter": vad,
        }
        if vad:
            kwargs["vad_parameters"] = {
                "threshold": 0.5,
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400,
            }
        if self._cuda():
            return self._batched_pass(media, kwargs)
        try:
            segments, info = self._model.transcribe(str(media.local_path), **kwargs)
            return list(segments), info, None
        except Exception as exc:
            return self._unavailable("inference_failed", str(exc), vad=vad)

    def _batched_pass(
        self, media: MediaAsset, kwargs: dict[str, object]
    ) -> tuple[list[WhisperSegment], object, int | None] | AsrUnavailable:
        assert self._module is not None
        pipeline = self._module.BatchedInferencePipeline(model=self._model)
        for batch_size in (8, 4, 2, 1):
            emitted: list[WhisperSegment] = []
            try:
                segments, info = pipeline.transcribe(
                    str(media.local_path), batch_size=batch_size, **kwargs
                )
                for segment in segments:
                    emitted.append(segment)
                return emitted, info, batch_size
            except Exception as exc:
                if emitted or not _is_oom(exc) or batch_size == 1:
                    return self._unavailable(
                        "inference_failed", str(exc), vad=bool(kwargs["vad_filter"])
                    )
        raise AssertionError("unreachable batch retry state")

    def _finish(
        self,
        media: MediaAsset,
        result: tuple[list[WhisperSegment], object, int | None] | AsrUnavailable,
        *,
        confirmation: bool,
    ) -> AsrOutcome:
        if isinstance(result, AsrUnavailable):
            return result
        raw, info, batch = result
        try:
            segments = _admit_segments(raw)
        except ValueError as exc:
            return self._unavailable("invalid_timestamps", str(exc), vad=not confirmation)
        if not segments:
            return self._unavailable("invalid_timestamps", "ASR emitted no usable timed text")
        language = getattr(info, "language", None)
        confidence = getattr(info, "language_probability", None)
        evidence = (
            DetectedLanguage(code=language, source="asr", confidence=confidence)
            if language
            else UnknownLanguage(reason="ASR language not reported")
        )
        receipt = self._receipt(
            info, vad=not confirmation, confirmation=confirmation, batch=batch
        )
        return AsrReady(
            transcript=Transcript(
                video_id=media.id,
                provenance=TranscriptProvenance(
                    method="asr",
                    language=language,
                    language_evidence=evidence,
                    format="json",
                    provider="faster-whisper",
                    asr=receipt.model_dump(exclude_none=True),
                ),
                segments=segments,
            ),
            receipt=receipt,
        )

    def _receipt(
        self, info: object, *, vad: bool, confirmation: bool = False, batch: int | None = None
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
            source_duration=getattr(info, "duration", None),
            speech_duration=getattr(info, "duration_after_vad", None),
            language=getattr(info, "language", None),
            language_confidence=getattr(info, "language_probability", None),
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

    def _cuda(self) -> bool:
        if self._device == "cuda":
            return True
        device = getattr(getattr(self._model, "model", None), "device", None)
        return str(device).lower() == "cuda"


class OpenAiCompatibleAsrAdapter:
    def __init__(
        self, *, instance: AsrInstanceConfig, api_key: str, provider_id: str,
        net: NetRuntime | None = None
    ) -> None:
        self.instance = instance
        self.api_key = api_key
        self.provider_id = provider_id
        self.net = net or UrllibNetRuntime()

    def transcribe(self, media: MediaAsset) -> AsrOutcome:
        try:
            response = self.net.request(self._request(media))
        except Exception as exc:
            return self._unavailable("inference_failed", type(exc).__name__)
        if not 200 <= response.status_code < 300:
            return self._unavailable("inference_failed", f"HTTP {response.status_code}")
        try:
            payload = OpenAiAsrResponse.model_validate_json(response.body)
            segments = _admit_segments(payload.segments)
        except ValueError as exc:
            return self._unavailable("invalid_response", str(exc))
        if not segments:
            return self._unavailable("invalid_response", "online ASR returned no timed segments")
        return AsrReady(
            transcript=Transcript(
                video_id=media.id,
                provenance=TranscriptProvenance(
                    method="asr", format="json", provider=self.provider_id
                ),
                segments=segments,
            ),
            receipt=AsrReceipt(
                provider=self.provider_id, model=self.instance.model or "whisper-1",
                device="remote", compute_type="remote", vad=False
            ),
        )

    def _request(self, media: MediaAsset) -> NetRequest:
        if not self.instance.base_url:
            raise ValueError("openai-compatible ASR instance is missing base_url")
        body, content_type = _multipart_audio_payload(
            media.local_path, model=self.instance.model or "whisper-1"
        )
        return NetRequest(
            method="POST", url=self.instance.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": content_type},
            body=body, timeout_s=120, purpose="asr_transcription", provider_id=self.provider_id
        )

    def _unavailable(self, code: AsrFailureCode, reason: str) -> AsrUnavailable:
        return AsrUnavailable(
            reason=reason,
            receipt=AsrReceipt(
                provider=self.provider_id, model=self.instance.model or "whisper-1",
                device="remote", compute_type="remote", vad=False, failure=code
            ),
        )


def run_asr(
    plan: RoutePlan, media: MediaAsset, *, instance: AsrInstanceConfig,
    cache_root: Path, api_key: str | None = None,
    runtime_cache: dict[str, object] | None = None
) -> AsrOutcome:
    if plan.selected == "local" and instance.type == "local-faster-whisper":
        cache = runtime_cache if runtime_cache is not None else {}
        key = f"asr:{instance.model_dump_json()}:{cache_root.resolve()}"
        adapter = cast(FasterWhisperAsrAdapter | None, cache.get(key))
        if adapter is None:
            adapter = FasterWhisperAsrAdapter(
                instance=instance, model_id=plan.model_id, cache_root=cache_root
            )
            cache[key] = adapter
        return adapter.transcribe(media)
    if plan.selected == "configured-online" and instance.type == "openai-compatible-audio":
        if not api_key:
            return AsrUnavailable(
                reason="online ASR credential is missing",
                receipt=AsrReceipt(
                    provider=plan.provider_id or "configured-asr",
                    model=instance.model or "whisper-1", device="remote",
                    compute_type="remote", vad=False, failure="inference_failed"
                ),
            )
        return OpenAiCompatibleAsrAdapter(
            instance=instance, api_key=api_key, provider_id=plan.provider_id or "configured-asr"
        ).transcribe(media)
    raise ValueError(f"ASR plan is not executable: {plan.selected}")


def _admit_segments(segments: Iterable[WhisperSegment]) -> list[TranscriptSegment]:
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
            TranscriptSegment(
                id=f"seg_{len(admitted) + 1:06d}", start=start, end=end, text=text
            )
        )
        previous = start
    return admitted


def _is_oom(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "out of memory" in message or "cuda" in message and "memory" in message


def _multipart_audio_payload(path: Path, *, model: str) -> tuple[bytes, str]:
    boundary = f"----vctx-{uuid.uuid4().hex}"
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = bytearray()
    for name, value in (("model", model),):
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {media_type}\r\n\r\n".encode()
    )
    body.extend(path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"
