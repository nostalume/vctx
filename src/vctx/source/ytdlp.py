from __future__ import annotations

import importlib
import json
import shutil
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from pydantic import BaseModel

from vctx.config import (
    BrowserSourceSession,
    CookieFileSourceSession,
    PlaylistItemsSelection,
    ProxySourceNetwork,
    YtDlpSourceOptions,
)
from vctx.errors import (
    NoTranscriptError,
    OfflineSourceError,
    OperationCancelledError,
    ProviderError,
)
from vctx.net import NetError, NetRequest, NetRuntime, RetryPolicy
from vctx.source.session import (
    EffectReceipt,
    MediaAsset,
    MediaPermit,
    MediaProfile,
    MediaRequest,
    ObservePermit,
    Revision,
    SourceRecord,
    SourceRef,
    SubtitlePermit,
    VideoMetadata,
)
from vctx.transcript import TranscriptPayload, TranscriptProvenance, detected_language

SubtitleKind = Literal["official_subtitles", "automatic_subtitles"]
YtDlpScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
YtDlpValue: TypeAlias = (  # noqa: UP040
    YtDlpScalar | list["YtDlpValue"] | dict[str, "YtDlpValue"]
)
YtDlpInfo: TypeAlias = dict[str, YtDlpValue]  # noqa: UP040
YtDlpParams: TypeAlias = dict[str, object]  # noqa: UP040
_SUPPORTED_SUBTITLE_EXTS = {"vtt", "srt", "json", "plain"}
_VISUAL_HEIGHT_CAPS: dict[MediaProfile, int] = {
    "auto": 720,
    "fast": 480,
    "balanced": 720,
    "high": 1080,
}


class DownloadedMediaAsset(BaseModel):
    id: str
    source: SourceRef
    local_path: Path
    container: str = "unknown"
    duration_seconds: float | None = None
    media_type: Literal["audio", "video", "unknown"]
    purpose: Literal["input", "asr", "visual"]
    profile: MediaProfile | None = None
    format_id: str
    provider: str = "yt-dlp"


@dataclass(frozen=True)
class SubtitleCandidate:
    kind: SubtitleKind
    language: str
    ext: Literal["vtt", "srt", "json", "plain", "unknown"]
    url: str


@dataclass
class YtDlpSession:
    name = "yt-dlp"
    info: YtDlpInfo
    options: YtDlpSourceOptions
    record: SourceRecord
    net: NetRuntime
    receipts: list[EffectReceipt] = field(default_factory=list)

    def transcript(self, *, permit: SubtitlePermit) -> TranscriptPayload:
        if permit.network == "denied":
            self.receipts.append(
                EffectReceipt(operation="subtitle", status="denied", purpose="transcript")
            )
            raise OfflineSourceError("offline policy denied subtitle fetch")
        candidate = _select_subtitle_candidate(
            self.info, subtitle_languages=self.options.subtitle_languages
        )
        if candidate is None:
            self.receipts.append(
                EffectReceipt(operation="subtitle", status="failed", purpose="transcript")
            )
            raise NoTranscriptError(f"no subtitles found for input: {self.record.metadata.id}")
        try:
            text = _read_subtitle_text(candidate.url, net=self.net)
        except (NetError, UnicodeError, NoTranscriptError) as exc:
            attempts = exc.attempts if isinstance(exc, NetError) else 1
            self.receipts.append(
                EffectReceipt(
                    operation="subtitle", status="failed", attempts=attempts, purpose="transcript"
                )
            )
            raise
        self.receipts.append(
            EffectReceipt(
                operation="subtitle",
                status="succeeded",
                attempts=1,
                purpose="transcript",
                requested_policy=",".join(self.options.subtitle_languages or ["auto"]),
                selected_policy=f"{candidate.language}:{candidate.ext}",
            )
        )
        return TranscriptPayload(
            text=text,
            format=candidate.ext,
            provenance=TranscriptProvenance(
                method=candidate.kind,
                language=candidate.language,
                language_evidence=detected_language(candidate.language, source="subtitle"),
                format=candidate.ext,
                provider="yt-dlp",
            ),
        )

    def media(self, *, request: MediaRequest, permit: MediaPermit) -> MediaAsset:
        if permit.network == "denied":
            self.receipts.append(_media_receipt(request, status="denied"))
            raise OfflineSourceError("offline policy denied media fetch")
        if request.temp_dir is None:
            self.receipts.append(_media_receipt(request, status="failed"))
            raise ProviderError("yt-dlp media fetch requires a cache temp directory")
        request.temp_dir.mkdir(parents=True, exist_ok=True)
        source_url = self.record.metadata.source.value
        try:
            planned, plan_detail = _plan_media(request, self.info)
        except ProviderError:
            self.receipts.append(_media_receipt(request, status="failed"))
            raise
        params = _download_params(planned, self.options)
        yt_dlp = _yt_dlp()
        try:
            with yt_dlp.YoutubeDL(params) as ydl:
                raw_info = ydl.process_ie_result(deepcopy(self.info), download=True)
        except yt_dlp.utils.DownloadError as exc:
            _cleanup_parts(planned.temp_dir)
            self.receipts.append(_media_receipt(request, status="failed", attempts=1))
            raise ProviderError(f"yt-dlp media fetch failed: {exc}") from exc
        except KeyboardInterrupt:
            _cleanup_parts(planned.temp_dir)
            self.receipts.append(_media_receipt(request, status="failed", attempts=1))
            raise OperationCancelledError("media fetch cancelled") from None
        info = _info_dict(raw_info)
        path = _downloaded_media_path(info)
        if path is None or not path.exists():
            raise NoTranscriptError(
                f"yt-dlp did not produce a media file for input: {self.record.metadata.id}"
            )
        asset = _downloaded_asset(planned, info, path, source_url=source_url)
        self.receipts.append(
            _media_receipt(
                request,
                status="succeeded",
                attempts=1,
                selected=asset.format_id,
                detail=plan_detail,
            )
        )
        return asset


class YtDlpSourceAdapter:
    name = "yt-dlp"

    def __init__(self, *, net: NetRuntime) -> None:
        self._net = net

    def claim(self, value: str) -> Literal["fallback", "unsupported"]:
        parsed = urlparse(value)
        return "fallback" if parsed.scheme in {"http", "https"} and parsed.netloc else "unsupported"

    def observe(
        self, value: str, *, permit: ObservePermit, options: YtDlpSourceOptions
    ) -> YtDlpSession:
        if permit.network == "denied":
            raise OfflineSourceError(
                "offline URL cache miss: no verified source cache is available"
            )
        yt_dlp = _yt_dlp()
        try:
            info = _extract_info(value, options)
        except yt_dlp.utils.DownloadError as exc:
            raise ProviderError(f"yt-dlp observation failed: {exc}") from exc
        return YtDlpSession(
            info=info,
            options=options,
            record=_source_record(value, info),
            net=self._net,
            receipts=[EffectReceipt(operation="observe", status="succeeded", attempts=1)],
        )


def _download_params(request: MediaRequest, options: YtDlpSourceOptions) -> YtDlpParams:
    if request.temp_dir is None:
        raise ProviderError("yt-dlp media fetch requires a cache temp directory")
    params: YtDlpParams = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": False,
        "paths": {"home": str(request.temp_dir), "temp": str(request.temp_dir)},
        "outtmpl": "%(extractor)s__%(id)s.%(ext)s",
        "continuedl": True,
        "part": True,
        "overwrites": False,
    }
    _apply_source_options(params, options)
    if request.kind == "asr_audio":
        params["format"] = "bestaudio/best"
        return params
    params["format"] = _visual_format(request.profile)
    return params


def _apply_source_options(params: YtDlpParams, options: YtDlpSourceOptions) -> None:
    if isinstance(options.session, BrowserSourceSession):
        params["cookiesfrombrowser"] = (options.session.browser,)
    if isinstance(options.session, CookieFileSourceSession):
        params["cookiefile"] = str(options.session.path)
    if isinstance(options.network, ProxySourceNetwork):
        params["proxy"] = options.network.url
    if isinstance(options.playlist, PlaylistItemsSelection):
        params["playlist_items"] = options.playlist.spec


def _visual_format(profile: MediaProfile) -> str:
    height = _VISUAL_HEIGHT_CAPS[profile]
    return f"bestvideo[height<={height}]/bestvideo"


def _plan_media(request: MediaRequest, info: YtDlpInfo) -> tuple[MediaRequest, str | None]:
    if request.temp_dir is None:
        return request, None
    estimate = _estimated_media_bytes(request, info)
    if estimate is None or _has_space(request.temp_dir, estimate):
        return request, None
    if request.kind == "visual_video" and request.profile == "auto":
        fast = request.model_copy(update={"profile": "fast"})
        fast_estimate = _estimated_media_bytes(fast, info)
        if fast_estimate is None or _has_space(request.temp_dir, fast_estimate):
            return fast, "media-quality auto selected fast because cache space is constrained"
    raise ProviderError(
        "insufficient cache space for the selected media quality; free space, "
        "choose another --cache-dir, or request --media-quality fast"
    )


def _estimated_media_bytes(request: MediaRequest, info: YtDlpInfo) -> int | None:
    estimates: list[int] = []
    height_cap = (
        _VISUAL_HEIGHT_CAPS.get(request.profile) if request.kind == "visual_video" else None
    )
    for raw in _list_value(info.get("formats")):
        item = _mapping_value(raw)
        if item is None:
            continue
        if request.kind == "asr_audio" and _as_optional_str(item.get("acodec")) == "none":
            continue
        if request.kind == "visual_video":
            if _as_optional_str(item.get("vcodec")) == "none":
                continue
            height = _as_optional_float(item.get("height"))
            if height is not None and height_cap is not None and height > height_cap:
                continue
        size = item.get("filesize") or item.get("filesize_approx")
        if isinstance(size, int | float) and size > 0:
            estimates.append(int(size))
    return max(estimates) if estimates else None


def _has_space(path: Path, estimate: int) -> bool:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    reserve = max(256 * 1024 * 1024, usage.total // 50)
    return estimate <= max(0, usage.free - reserve)


def _cleanup_parts(temp_dir: Path | None) -> None:
    if temp_dir is None or not temp_dir.is_dir():
        return
    for part in temp_dir.glob("*.part"):
        part.unlink(missing_ok=True)


def _media_receipt(
    request: MediaRequest,
    *,
    status: Literal["cache_hit", "succeeded", "denied", "failed"],
    attempts: int = 0,
    selected: str | None = None,
    detail: str | None = None,
) -> EffectReceipt:
    purpose: Literal["asr", "visual"] = "asr" if request.kind == "asr_audio" else "visual"
    requested = "audio" if request.kind == "asr_audio" else request.profile
    return EffectReceipt(
        operation="media",
        status=status,
        attempts=attempts,
        purpose=purpose,
        requested_policy=requested,
        selected_policy=selected,
        detail=detail,
    )


def _downloaded_asset(
    request: MediaRequest,
    info: YtDlpInfo,
    path: Path,
    *,
    source_url: str,
) -> MediaAsset:
    container = path.suffix.lower().lstrip(".") or _as_optional_str(info.get("ext")) or "unknown"
    format_id = _as_optional_str(info.get("format_id")) or "unknown"
    duration = _as_optional_float(info.get("duration"))
    source = SourceRef(kind="url", value=source_url)
    if request.kind == "asr_audio":
        return DownloadedMediaAsset(
            id=_media_id(info),
            source=source,
            local_path=path,
            container=container,
            duration_seconds=duration,
            media_type="audio",
            purpose="asr",
            format_id=format_id,
        )
    return DownloadedMediaAsset(
        id=_media_id(info),
        source=source,
        local_path=path,
        container=container,
        duration_seconds=duration,
        media_type="video",
        purpose="visual",
        profile=request.profile,
        format_id=format_id,
    )


def _downloaded_media_path(info: YtDlpInfo) -> Path | None:
    requested = _list_value(info.get("requested_downloads"))
    for raw_download in requested:
        download = _mapping_value(raw_download)
        if download is None:
            continue
        path = _as_optional_str(download.get("filepath")) or _as_optional_str(
            download.get("filename")
        )
        if path:
            return Path(path)
    filepath = _as_optional_str(info.get("filepath")) or _as_optional_str(info.get("_filename"))
    return Path(filepath) if filepath else None


def _extract_info(value: str, options: YtDlpSourceOptions) -> YtDlpInfo:
    params: YtDlpParams = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 30,
    }
    _apply_source_options(params, options)
    with _yt_dlp().YoutubeDL(params) as ydl:
        raw_info = ydl.extract_info(value, download=False)
    return _info_dict(raw_info)


def _yt_dlp() -> Any:
    return importlib.import_module("yt_dlp")


def _source_record(locator: str, info: YtDlpInfo) -> SourceRecord:
    extractor = _as_optional_str(info.get("extractor"))
    source_id = _media_id(info)
    canonical = _sanitize_url(
        _as_optional_str(info.get("webpage_url")) or locator, extractor=extractor
    )
    metadata = VideoMetadata(
        id=source_id,
        source=SourceRef(kind="url", value=canonical),
        title=_as_optional_str(info.get("title")),
        uploader=_as_optional_str(info.get("uploader")),
        duration_seconds=_as_optional_float(info.get("duration")),
        language=_as_optional_str(info.get("language")),
        extractor=extractor,
        raw_provider="yt-dlp",
    )
    lifecycle = _lifecycle(info)
    fingerprint = {
        "source_id": source_id,
        "duration": metadata.duration_seconds,
        "lifecycle": lifecycle,
        "subtitles": _subtitle_facts(info),
        "formats": _format_facts(info),
    }
    digest = sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SourceRecord(
        source_id=source_id,
        revision=Revision(kind="observed", value=digest),
        observed_at=datetime.now(UTC),
        metadata=metadata,
        lifecycle=lifecycle,
        has_subtitles=bool(fingerprint["subtitles"]),
        has_media=bool(fingerprint["formats"]) or metadata.duration_seconds is not None,
    )


def _sanitize_url(value: str, *, extractor: str | None) -> str:
    parsed = urlparse(value)
    allowed = {"v"} if extractor and "youtube" in extractor.lower() else set()
    query = urlencode([(key, item) for key, item in parse_qsl(parsed.query) if key in allowed])
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme, host, parsed.path, "", query, ""))


def _lifecycle(info: Mapping[str, YtDlpValue]) -> Literal["finite", "live", "upcoming"]:
    status = _as_optional_str(info.get("live_status"))
    if status == "is_upcoming":
        return "upcoming"
    if status == "is_live" or info.get("is_live") is True:
        return "live"
    return "finite"


def _subtitle_facts(info: Mapping[str, YtDlpValue]) -> list[str]:
    facts: list[str] = []
    for key in ("subtitles", "automatic_captions"):
        languages = _mapping_value(info.get(key)) or {}
        for language, entries in languages.items():
            for entry in _list_value(entries):
                item = _mapping_value(entry)
                if item is not None:
                    facts.append(f"{key}:{language}:{_normalize_subtitle_ext(item.get('ext'))}")
    return sorted(set(facts))


def _format_facts(info: Mapping[str, YtDlpValue]) -> list[str]:
    items = (_mapping_value(raw) for raw in _list_value(info.get("formats")))
    values = (_as_optional_str(item.get("format_id")) for item in items if item)
    return sorted({value for value in values if value})


def _info_dict(raw_info: YtDlpValue) -> YtDlpInfo:
    if not isinstance(raw_info, dict):
        raise NoTranscriptError("yt-dlp returned no metadata for input")
    return cast(YtDlpInfo, raw_info)


def _media_id(info: Mapping[str, YtDlpValue]) -> str:
    extractor = _as_optional_str(info.get("extractor"))
    identity = _as_optional_str(info.get("id")) or "unknown"
    return f"{extractor}__{identity}" if extractor else f"url__{identity}"


def _select_subtitle_candidate(
    info: Mapping[str, YtDlpValue],
    *,
    subtitle_languages: list[str] | None = None,
) -> SubtitleCandidate | None:
    language_order = _language_order(info, subtitle_languages=subtitle_languages or [])
    subtitle_maps: list[tuple[SubtitleKind, YtDlpValue | None]] = [
        ("official_subtitles", info.get("subtitles")),
        ("automatic_subtitles", info.get("automatic_captions")),
    ]
    for kind, raw_subtitle_map in subtitle_maps:
        subtitle_entries = _mapping_value(raw_subtitle_map)
        if subtitle_entries is None:
            continue
        for language in language_order:
            candidate = _candidate_from_entries(kind, language, subtitle_entries.get(language))
            if candidate is not None:
                return candidate
        for language, entries in subtitle_entries.items():
            candidate = _candidate_from_entries(kind, language, entries)
            if candidate is not None:
                return candidate
    return None


def _language_order(
    info: Mapping[str, YtDlpValue],
    *,
    subtitle_languages: list[str],
) -> list[str]:
    values: list[str] = []
    info_language = _as_optional_str(info.get("language"))
    if info_language:
        values.append(info_language)
    values.extend(subtitle_languages)
    values.extend(["en", "zh", "zh-Hans", "zh-CN"])
    return list(dict.fromkeys(values))


def _candidate_from_entries(
    kind: SubtitleKind, language: str, entries: YtDlpValue | None
) -> SubtitleCandidate | None:
    raw_entries = _list_value(entries)
    fallback: SubtitleCandidate | None = None
    for entry in raw_entries:
        subtitle = _mapping_value(entry)
        if subtitle is None:
            continue
        url = _as_optional_str(subtitle.get("url"))
        if not url:
            continue
        ext = _normalize_subtitle_ext(subtitle.get("ext"))
        candidate = SubtitleCandidate(kind=kind, language=language, ext=ext, url=url)
        if ext in {"vtt", "srt"}:
            return candidate
        if fallback is None:
            fallback = candidate
    return fallback


def _normalize_subtitle_ext(
    value: YtDlpValue | None,
) -> Literal["vtt", "srt", "json", "plain", "unknown"]:
    normalized = value.lower() if isinstance(value, str) else ""
    if normalized == "vtt":
        return "vtt"
    if normalized == "srt":
        return "srt"
    if normalized == "json":
        return "json"
    if normalized == "plain":
        return "plain"
    return "unknown"


def _read_subtitle_text(url: str, *, net: NetRuntime) -> str:
    text = _fetch_text(url, net=net)
    if _is_hls_playlist(text):
        return _read_hls_vtt_playlist(url, text, net=net)
    return text


def _fetch_text(url: str, *, net: NetRuntime) -> str:
    response = net.request(
        NetRequest(
            method="GET",
            url=url,
            timeout_s=30,
            purpose="subtitle_fetch",
            provider_id="yt-dlp",
            retry=RetryPolicy(
                max_attempts=3,
                statuses=(429, 500, 502, 503, 504),
                retry_connect=True,
                retry_timeouts=True,
            ),
        )
    )
    if response.status_code < 200 or response.status_code >= 300:
        raise NoTranscriptError(f"subtitle fetch failed: HTTP {response.status_code}")
    return response.body.decode("utf-8-sig")


def _is_hls_playlist(text: str) -> bool:
    return text.lstrip().startswith("#EXTM3U")


def _read_hls_vtt_playlist(playlist_url: str, playlist_text: str, *, net: NetRuntime) -> str:
    segment_urls = _hls_segment_urls(playlist_url, playlist_text)
    segments = [
        _strip_vtt_header(_fetch_text(segment_url, net=net)) for segment_url in segment_urls
    ]
    cues = [segment.strip() for segment in segments if segment.strip()]
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


def _hls_segment_urls(playlist_url: str, playlist_text: str) -> list[str]:
    urls: list[str] = []
    for raw_line in playlist_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(urljoin(playlist_url, line))
    return urls


def _strip_vtt_header(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip("\ufeff").strip() == "WEBVTT":
        return "\n".join(lines[1:]).strip()
    return text.strip()


def _mapping_value(value: YtDlpValue | None) -> Mapping[str, YtDlpValue] | None:
    return value if isinstance(value, dict) else None


def _list_value(value: YtDlpValue | None) -> list[YtDlpValue]:
    return value if isinstance(value, list) else []


def _as_optional_str(value: YtDlpValue | None) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_optional_float(value: YtDlpValue | None) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
