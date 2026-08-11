from __future__ import annotations

import importlib
import mimetypes
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from vctx.app.models import ModelLifecycleError, require_prepared_model
from vctx.config import AsrInstanceConfig
from vctx.net import NetRequest, NetResponse, NetRuntime, UrllibNetRuntime
from vctx.source.session import MediaAsset
from vctx.transcript import TranscriptPayload, TranscriptProvenance, UnknownLanguage
from vctx.transforms.planning import RoutePlan


class AsrExecutionError(RuntimeError):
    pass


class WhisperSegment(Protocol):
    start: float
    end: float
    text: str


class OpenAiAsrSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: float
    end: float
    text: str


class OpenAiAsrResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    segments: list[OpenAiAsrSegment] = Field(default_factory=list)


class FasterWhisperAsrAdapter:
    def __init__(
        self,
        *,
        instance: AsrInstanceConfig,
        model_id: str | None,
        cache_root: Path,
        offline: bool = False,
    ) -> None:
        self.instance = instance
        self.model_id = model_id or instance.model or instance.model_policy
        self.cache_root = cache_root
        self.offline = offline
        self._model: Any | None = None

    def transcribe(self, media_asset: MediaAsset) -> TranscriptPayload:
        model_id = self._model_id()
        model_path = Path(model_id)
        if not model_path.is_absolute() and not model_path.exists():
            try:
                receipt = require_prepared_model("asr", self.cache_root, asr_model_id=model_id)
            except ModelLifecycleError as exc:
                raise AsrExecutionError(str(exc)) from exc
            if receipt.model_id != model_id:
                raise AsrExecutionError(
                    f"prepared ASR model is {receipt.model_id}, but {model_id} was selected; "
                    "run: vctx models pull asr"
                )
            model_id = str(self.cache_root / receipt.cache_path)
        model_kwargs = self._model_kwargs(model_id)
        module = self._load_faster_whisper()
        try:
            if self._model is None:
                self._model = module.WhisperModel(model_id, **model_kwargs)
            segments, _info = self._model.transcribe(str(media_asset.local_path), language=None)
        except Exception as exc:
            raise AsrExecutionError(
                "faster-whisper ASR failed. If offline, pre-populate the model cache "
                "or set instance model to path:<local-path>. Original error: "
                f"{exc}"
            ) from exc
        return TranscriptPayload(
            text=_segments_to_vtt(segments),
            format="vtt",
            provenance=TranscriptProvenance(
                method="asr",
                language_evidence=UnknownLanguage(reason="ASR language not reported"),
                format="vtt",
                provider="faster-whisper",
            ),
        )

    def _model_id(self) -> str:
        model_id = self.model_id
        if model_id == "auto":
            return "base"
        return model_id

    def _model_kwargs(self, model_id: str) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "device": "auto",
            "compute_type": "default",
        }
        model_path = Path(model_id)
        if model_path.is_absolute() or model_path.exists():
            kwargs["local_files_only"] = True
            return kwargs
        if self.instance.cache == "disabled":
            raise AsrExecutionError(
                "cache = disabled requires path:<local-path> for local-faster-whisper; "
                "omit cache for managed persistent cache"
            )
        model_cache = self.cache_root / "models" / "faster-whisper"
        try:
            model_cache.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AsrExecutionError(
                "ASR model cache is not writable. Free disk space, choose another "
                "cache.model_dir, or set model to path:<local-path>. "
                f"Cache path: {model_cache}. Original error: {exc}"
            ) from exc
        kwargs["download_root"] = str(model_cache)
        kwargs["local_files_only"] = True
        return kwargs

    def _load_faster_whisper(self) -> Any:
        try:
            return importlib.import_module("faster_whisper")
        except ModuleNotFoundError as exc:
            raise AsrExecutionError(
                "Install the ASR extra to use local faster-whisper ASR: "
                "uv sync --extra asr or uv add 'vctx[asr]'"
            ) from exc


class OpenAiCompatibleAsrAdapter:
    def __init__(
        self,
        *,
        instance: AsrInstanceConfig,
        api_key: str,
        provider_id: str,
        net: NetRuntime | None = None,
    ) -> None:
        self.instance = instance
        self.api_key = api_key
        self.provider_id = provider_id
        self.net = net or UrllibNetRuntime()

    def transcribe(self, media_asset: MediaAsset) -> TranscriptPayload:
        response = self._request_one(self._request_for_media(media_asset))
        if response.status_code < 200 or response.status_code >= 300:
            raise AsrExecutionError(f"online ASR request failed: HTTP {response.status_code}")
        try:
            payload = OpenAiAsrResponse.model_validate_json(response.body)
        except ValueError as exc:
            raise AsrExecutionError("online ASR returned invalid JSON") from exc
        return TranscriptPayload(
            text=_openai_response_to_vtt(payload),
            format="vtt",
            provenance=TranscriptProvenance(
                method="asr",
                language_evidence=UnknownLanguage(reason="online ASR language not reported"),
                format="vtt",
                provider=self.provider_id,
            ),
        )

    def _request_for_media(self, media_asset: MediaAsset) -> NetRequest:
        if not self.instance.base_url:
            raise AsrExecutionError("openai-compatible ASR instance is missing base_url")
        model = self.instance.model or "whisper-1"
        data, content_type = _multipart_audio_payload(media_asset.local_path, model=model)
        return NetRequest(
            method="POST",
            url=self.instance.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": content_type,
            },
            body=data,
            timeout_s=120,
            purpose="asr_transcription",
            provider_id=self.provider_id,
        )

    def _request_one(self, request: NetRequest) -> NetResponse:
        try:
            return self.net.request(request)
        except Exception as exc:
            raise AsrExecutionError(
                "online ASR request failed "
                f"for provider {self.provider_id}: {type(exc).__name__}"
            ) from exc


def run_asr(
    plan: RoutePlan,
    media_asset: MediaAsset,
    *,
    instance: AsrInstanceConfig,
    cache_root: Path,
    offline: bool = False,
    api_key: str | None = None,
    runtime_cache: dict[str, object] | None = None,
) -> TranscriptPayload:
    if plan.selected == "local":
        if instance.type != "local-faster-whisper":
            raise AsrExecutionError(f"unsupported local ASR instance type: {instance.type}")
        cache = runtime_cache if runtime_cache is not None else {}
        key = f"asr:{instance.model_dump_json()}:{cache_root.resolve()}:{offline}"
        adapter = cast(FasterWhisperAsrAdapter | None, cache.get(key))
        if adapter is None:
            adapter = FasterWhisperAsrAdapter(
                instance=instance,
                model_id=_asr_model_id(plan),
                cache_root=cache_root,
                offline=offline,
            )
            cache[key] = adapter
        return adapter.transcribe(media_asset)
    if plan.selected == "configured-online":
        if instance.type != "openai-compatible-audio":
            raise AsrExecutionError(f"unsupported online ASR instance type: {instance.type}")
        if not api_key:
            raise AsrExecutionError("online ASR credential is missing")
        adapter = OpenAiCompatibleAsrAdapter(
            instance=instance,
            api_key=api_key,
            provider_id=_asr_provider_id(plan),
        )
        return adapter.transcribe(media_asset)
    raise AsrExecutionError(f"ASR plan is not executable: {plan.selected}")


def _asr_model_id(plan: RoutePlan) -> str | None:
    if plan.model_id is not None:
        return plan.model_id
    if plan.ai_route is not None:
        return plan.ai_route.model
    return None


def _asr_provider_id(plan: RoutePlan) -> str:
    if plan.provider_id is not None:
        return plan.provider_id
    if plan.ai_route is not None and plan.ai_route.provider_id is not None:
        return plan.ai_route.provider_id
    return "configured-asr"


def _openai_response_to_vtt(payload: OpenAiAsrResponse) -> str:
    if payload.segments:
        return _segments_to_vtt(payload.segments)
    if payload.text and payload.text.strip():
        return "WEBVTT\n\n00:00:00.000 --> 00:00:00.001\n" + payload.text.strip() + "\n"
    raise AsrExecutionError("online ASR returned no transcript text")


def _multipart_audio_payload(path: Path, *, model: str) -> tuple[bytes, str]:
    boundary = f"----vctx-{uuid.uuid4().hex}"
    content_type = f"multipart/form-data; boundary={boundary}"
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = bytearray()
    _append_form_field(body, boundary, "model", model)
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode()
    )
    body.extend(path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), content_type


def _append_form_field(body: bytearray, boundary: str, name: str, value: str) -> None:
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
    body.extend(value.encode())
    body.extend(b"\r\n")


def _segments_to_vtt(segments: Iterable[WhisperSegment]) -> str:
    blocks = ["WEBVTT", ""]
    for segment in segments:
        start = float(segment.start)
        end = float(segment.end)
        text = segment.text.strip()
        if not text:
            continue
        blocks.append(f"{_format_vtt_timestamp(start)} --> {_format_vtt_timestamp(end)}")
        blocks.append(text)
        blocks.append("")
    return "\n".join(blocks)


def _format_vtt_timestamp(seconds: float) -> str:
    milliseconds_total = max(0, round(seconds * 1000))
    milliseconds = milliseconds_total % 1000
    seconds_total = milliseconds_total // 1000
    secs = seconds_total % 60
    minutes_total = seconds_total // 60
    mins = minutes_total % 60
    hours = minutes_total // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{milliseconds:03d}"
