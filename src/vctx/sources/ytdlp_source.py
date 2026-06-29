from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast
from urllib.parse import urljoin, urlparse

import yt_dlp
from pydantic import BaseModel, ConfigDict

from vctx.config import (
    BrowserSourceSession,
    CookieFileSourceSession,
    MediaProfile,
    PlaylistItemsSelection,
    ProxySourceNetwork,
    YtDlpSourceOptions,
)
from vctx.errors import NoTranscriptError
from vctx.io import Cache, model_to_json
from vctx.models import SourceRef
from vctx.models.media import (
    DownloadedAsrAudioAsset,
    DownloadedAsrAudioSidecar,
    DownloadedVisualVideoAsset,
    DownloadedVisualVideoSidecar,
    MediaAsset,
    MediaAssetSidecar,
    MediaFetchRequest,
    ReuseDecision,
    ReuseHit,
    ReuseMiss,
)
from vctx.models.metadata import VideoMetadata
from vctx.net import NetRequest, NetRuntime, UrllibNetRuntime
from vctx.transcript import TranscriptPayload, TranscriptProvenance, detected_language

SubtitleKind = Literal["official_subtitles", "automatic_subtitles"]
SourceNetFactory = Callable[[], NetRuntime]
YtDlpScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
YtDlpValue: TypeAlias = (  # noqa: UP040
    YtDlpScalar | list["YtDlpValue"] | dict[str, "YtDlpValue"]
)
YtDlpInfo: TypeAlias = dict[str, YtDlpValue]  # noqa: UP040
YtDlpParams: TypeAlias = dict[str, str | bool | int | dict[str, str]]  # noqa: UP040
_SUPPORTED_SUBTITLE_EXTS = {"vtt", "srt", "json", "plain"}
_VISUAL_HEIGHT_CAPS: dict[MediaProfile, int] = {
    MediaProfile.FAST: 480,
    MediaProfile.BALANCED: 720,
    MediaProfile.HIGH: 1080,
}


class _SidecarEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: str
    media_id: str
    source: SourceRef
    local_path: Path


@dataclass(frozen=True)
class SubtitleCandidate:
    kind: SubtitleKind
    language: str
    ext: Literal["vtt", "srt", "json", "plain", "unknown"]
    url: str


class YtDlpSourceAdapter:
    name = "yt-dlp"

    def __init__(self, *, net_factory: SourceNetFactory | None = None) -> None:
        self._net_factory = net_factory or UrllibNetRuntime
        self._net: NetRuntime | None = None

    def _net_runtime(self) -> NetRuntime:
        if self._net is None:
            self._net = self._net_factory()
        return self._net

    def can_handle(self, value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def extract_metadata(self, value: str) -> VideoMetadata:
        info = _extract_info(value)
        extractor = _as_optional_str(info.get("extractor"))
        video_id = _as_optional_str(info.get("id")) or "unknown"
        normalized_id = f"{extractor}__{video_id}" if extractor else f"url__{video_id}"
        return VideoMetadata(
            id=normalized_id,
            source_type="url",
            source=SourceRef(kind="url", value=value),
            title=_as_optional_str(info.get("title")),
            uploader=_as_optional_str(info.get("uploader")),
            duration_seconds=_as_optional_float(info.get("duration")),
            webpage_url=_as_optional_str(info.get("webpage_url")) or value,
            language=_as_optional_str(info.get("language")),
            extractor=extractor,
            raw_provider="yt-dlp",
        )

    def extract_transcript(
        self,
        value: str,
        *,
        cache: Cache,
        source_options: YtDlpSourceOptions | None = None,
    ) -> TranscriptPayload:
        del cache
        info = _extract_info(value)
        candidate = _select_subtitle_candidate(
            info,
            subtitle_languages=source_options.subtitle_languages if source_options else [],
        )
        if candidate is None:
            raise NoTranscriptError(f"no subtitles found for input: {value}")
        return TranscriptPayload(
            text=_read_subtitle_text(candidate.url, net=self._net_runtime()),
            format=candidate.ext,
            provenance=TranscriptProvenance(
                method=candidate.kind,
                language=candidate.language,
                language_evidence=detected_language(candidate.language, source="subtitle"),
                format=candidate.ext,
                provider="yt-dlp",
            ),
        )

    def extract_media(self, value: str, *, request: MediaFetchRequest) -> MediaAsset:
        if value != request.source_url:
            raise NoTranscriptError("media request source does not match input URL")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        request.temp_dir.mkdir(parents=True, exist_ok=True)
        metadata = _extract_info(value)
        media_id = _media_id(metadata)
        reuse = _reuse_decision(request, media_id)
        if reuse.kind == "reuse_hit":
            return _asset_from_sidecar(reuse.sidecar_path, reuse=reuse)
        params = _download_params(request)
        with yt_dlp.YoutubeDL(params) as ydl:
            raw_info = ydl.extract_info(value, download=True)
        info = _info_dict(raw_info)
        path = _downloaded_media_path(info)
        if path is None or not path.exists():
            raise NoTranscriptError(f"yt-dlp did not produce a media file for input: {value}")
        asset = _downloaded_asset(request, info, path, reuse)
        _write_sidecar(request.output_dir, asset, info)
        return asset


def _download_params(request: MediaFetchRequest) -> YtDlpParams:
    params: YtDlpParams = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": False,
        "paths": {"home": str(request.output_dir), "temp": str(request.temp_dir)},
        "outtmpl": "%(extractor)s__%(id)s.%(ext)s",
        "continuedl": True,
        "part": True,
        "overwrites": False,
    }
    _apply_source_options(params, request)
    if request.kind == "asr_audio":
        params["format"] = "bestaudio/best"
        return params
    params["format"] = _visual_format(request.profile)
    params["merge_output_format"] = "mp4/webm"
    return params


def _apply_source_options(params: YtDlpParams, request: MediaFetchRequest) -> None:
    options = request.source_options
    if isinstance(options.session, BrowserSourceSession):
        params["cookiesfrombrowser"] = options.session.browser
    if isinstance(options.session, CookieFileSourceSession):
        params["cookiefile"] = str(options.session.path)
    if isinstance(options.network, ProxySourceNetwork):
        params["proxy"] = options.network.url
    if isinstance(options.playlist, PlaylistItemsSelection):
        params["playlist_items"] = options.playlist.spec


def _visual_format(profile: MediaProfile) -> str:
    height = _VISUAL_HEIGHT_CAPS[profile]
    return (
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}]/best[height<={height}]/best"
    )


def _reuse_decision(request: MediaFetchRequest, media_id: str) -> ReuseDecision:
    if not request.reuse:
        return ReuseMiss(reason="overwrite")
    sidecar_path = request.output_dir / f"{media_id}.vctx-media.json"
    if not sidecar_path.exists():
        return ReuseMiss(reason="missing_sidecar")
    envelope = _SidecarEnvelope.model_validate_json(sidecar_path.read_text(encoding="utf-8"))
    if envelope.source.value != request.source_url:
        return ReuseMiss(reason="source_mismatch")
    if not envelope.local_path.exists():
        return ReuseMiss(reason="missing_file")
    if request.kind == "asr_audio" and envelope.kind != "downloaded_asr_audio":
        return ReuseMiss(reason="purpose_mismatch")
    if request.kind == "visual_video" and envelope.kind != "downloaded_visual_video":
        return ReuseMiss(reason="purpose_mismatch")
    return ReuseHit(sidecar_path=sidecar_path)


def _asset_from_sidecar(path: Path, *, reuse: ReuseHit) -> MediaAsset:
    envelope = _SidecarEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    if envelope.kind == "downloaded_asr_audio":
        sidecar = DownloadedAsrAudioSidecar.model_validate_json(path.read_text(encoding="utf-8"))
        return DownloadedAsrAudioAsset(
            id=sidecar.media_id,
            source=sidecar.source,
            local_path=sidecar.local_path,
            container=sidecar.container,
            format_id=sidecar.format_id,
            reuse=reuse,
        )
    if envelope.kind == "downloaded_visual_video":
        sidecar = DownloadedVisualVideoSidecar.model_validate_json(path.read_text(encoding="utf-8"))
        return DownloadedVisualVideoAsset(
            id=sidecar.media_id,
            source=sidecar.source,
            local_path=sidecar.local_path,
            container=sidecar.container,
            profile=sidecar.profile,
            format_id=sidecar.format_id,
            reuse=reuse,
        )
    raise NoTranscriptError(f"unsupported media sidecar kind: {envelope.kind}")


def _downloaded_asset(
    request: MediaFetchRequest,
    info: YtDlpInfo,
    path: Path,
    reuse: ReuseDecision,
) -> MediaAsset:
    container = path.suffix.lower().lstrip(".") or _as_optional_str(info.get("ext")) or "unknown"
    format_id = _as_optional_str(info.get("format_id")) or "unknown"
    duration = _as_optional_float(info.get("duration"))
    source = SourceRef(kind="url", value=request.source_url)
    if request.kind == "asr_audio":
        return DownloadedAsrAudioAsset(
            id=_media_id(info),
            source=source,
            local_path=path,
            container=container,
            duration_seconds=duration,
            format_id=format_id,
            reuse=reuse,
        )
    return DownloadedVisualVideoAsset(
        id=_media_id(info),
        source=source,
        local_path=path,
        container=container,
        duration_seconds=duration,
        profile=request.profile,
        format_id=format_id,
        reuse=reuse,
    )


def _write_sidecar(output_dir: Path, asset: MediaAsset, info: YtDlpInfo) -> None:
    if asset.kind == "local_media":
        return
    extractor = _as_optional_str(info.get("extractor")) or "unknown"
    webpage_url = _as_optional_str(info.get("webpage_url")) or asset.source.value
    if asset.kind == "downloaded_asr_audio":
        sidecar: MediaAssetSidecar = DownloadedAsrAudioSidecar(
            source=asset.source,
            media_id=asset.id,
            local_path=asset.local_path,
            container=asset.container,
            extractor=extractor,
            webpage_url=webpage_url,
            format_id=asset.format_id,
        )
    else:
        sidecar = DownloadedVisualVideoSidecar(
            source=asset.source,
            media_id=asset.id,
            local_path=asset.local_path,
            container=asset.container,
            profile=asset.profile,
            extractor=extractor,
            webpage_url=webpage_url,
            format_id=asset.format_id,
        )
    sidecar_path = output_dir / f"{asset.id}.vctx-media.json"
    sidecar_path.write_text(model_to_json(sidecar), encoding="utf-8")


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


def _extract_info(value: str) -> YtDlpInfo:
    params: YtDlpParams = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(params) as ydl:
        raw_info = ydl.extract_info(value, download=False)
    return _info_dict(raw_info)


def _info_dict(raw_info: YtDlpValue) -> YtDlpInfo:
    if not isinstance(raw_info, dict):
        raise NoTranscriptError("yt-dlp returned no metadata for input")
    return cast(YtDlpInfo, raw_info)


def _media_id(info: Mapping[str, YtDlpValue]) -> str:
    extractor = _as_optional_str(info.get("extractor"))
    video_id = _as_optional_str(info.get("id")) or "unknown"
    return f"{extractor}__{video_id}" if extractor else f"url__{video_id}"


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
