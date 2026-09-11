from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode, urlparse, urlunparse

from pydantic import AliasChoices, BaseModel, Field, ValidationError

from vctx.config import YtDlpSourceOptions
from vctx.errors import NoTranscriptError, OfflineSourceError, ProviderError
from vctx.net import NetRequest, NetRuntime, RetryPolicy
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
from vctx.source.transfer import LocatorExpired, RangeTransfer
from vctx.transcript import TranscriptPayload

_BV = re.compile(r"BV[0-9A-Za-z]{10}\Z")
_HEIGHT_CAP: dict[MediaProfile, int] = {
    "auto": 720,
    "fast": 480,
    "balanced": 720,
    "high": 1080,
}
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; vctx/0.3)",
    "Accept": "application/json",
}


class _Owner(BaseModel):
    name: str | None = None


class _ViewData(BaseModel):
    bvid: str
    cid: int
    title: str | None = None
    duration: float | None = None
    owner: _Owner = Field(default_factory=_Owner)


class _ViewEnvelope(BaseModel):
    code: int
    data: _ViewData | None = None


class _DashStream(BaseModel):
    id: int
    base_url: str = Field(validation_alias=AliasChoices("baseUrl", "base_url"))
    bandwidth: int = 0
    height: int | None = None
    mime_type: str | None = Field(
        default=None, validation_alias=AliasChoices("mimeType", "mime_type")
    )


class _Dash(BaseModel):
    audio: list[_DashStream] = Field(default_factory=list)
    video: list[_DashStream] = Field(default_factory=list)


class _PlayData(BaseModel):
    dash: _Dash


class _PlayEnvelope(BaseModel):
    code: int
    data: _PlayData | None = None


@dataclass
class BilibiliSession:
    bvid: str
    cid: int
    source_url: str
    record: SourceRecord
    net: NetRuntime
    name: str = "bilibili"
    receipts: list[EffectReceipt] = field(default_factory=list)

    def transcript(self, *, permit: SubtitlePermit) -> TranscriptPayload:
        del permit
        raise NoTranscriptError("Bilibili source has no admitted subtitle track")

    def media(self, *, request: MediaRequest, permit: MediaPermit) -> MediaAsset:
        if permit.network == "denied":
            raise OfflineSourceError("offline Bilibili media cache miss")
        if request.temp_dir is None:
            raise ProviderError("Bilibili media fetch requires a cache temp directory")
        for attempt in range(2):
            play = self._playback()
            stream = _select_stream(play, request)
            extension = _extension(stream)
            destination = request.temp_dir / (
                f"bilibili__{self.bvid}__{request.kind}__{stream.id}.{extension}"
            )
            try:
                path = RangeTransfer(
                    self.net,
                    stream.base_url,
                    destination,
                    headers={"Referer": self.source_url, "User-Agent": _HEADERS["User-Agent"]},
                    refresh=request.refresh,
                ).download()
                break
            except LocatorExpired:
                if attempt:
                    raise ProviderError("Bilibili media locator refresh exhausted") from None
        else:
            raise AssertionError("unreachable locator refresh state")
        purpose: Literal["asr", "visual"] = "asr" if request.kind == "asr_audio" else "visual"
        profile = request.profile if request.kind == "visual_video" else None
        asset = MediaAsset(
            id=f"bilibili__{self.bvid}__{purpose}",
            source=self.record.metadata.source,
            local_path=path,
            container=extension,
            duration_seconds=self.record.metadata.duration_seconds,
            media_type="audio" if request.kind == "asr_audio" else "video",
            purpose=purpose,
            profile=profile,
            format_id=str(stream.id),
            provider="bilibili",
            capabilities={"audio" if request.kind == "asr_audio" else "video"},
        )
        self.receipts.append(
            EffectReceipt(
                operation="media",
                status="succeeded",
                attempts=1,
                purpose=purpose,
                requested_policy="audio" if request.kind == "asr_audio" else request.profile,
                selected_policy=str(stream.id),
                detail=f"anonymous DASH {purpose} representation {stream.id}",
            )
        )
        return asset

    def _playback(self) -> _PlayData:
        url = "https://api.bilibili.com/x/player/playurl?" + urlencode(
            {"bvid": self.bvid, "cid": self.cid, "qn": 127, "fnval": 4048, "fourk": 1}
        )
        request = _api_request(url).model_copy(
            update={"headers": {**_HEADERS, "Referer": self.source_url}}
        )
        response = self.net.request(request)
        try:
            envelope = _PlayEnvelope.model_validate_json(response.body)
        except ValidationError as exc:
            raise ProviderError("Bilibili playback returned an invalid response") from exc
        if response.status_code != 200 or envelope.code != 0 or envelope.data is None:
            raise ProviderError("Bilibili playback unavailable")
        return envelope.data


class BilibiliSourceAdapter:
    name = "bilibili"

    def __init__(self, *, net: NetRuntime) -> None:
        self.net = net

    def claim(self, value: str) -> Literal["exact", "unsupported"]:
        return "exact" if bilibili_bvid(value) is not None else "unsupported"

    def observe(
        self,
        value: str,
        *,
        permit: ObservePermit,
        options: YtDlpSourceOptions,
    ) -> BilibiliSession:
        del options
        if permit.network == "denied":
            raise OfflineSourceError("offline Bilibili observation cache miss")
        bvid = bilibili_bvid(value)
        if bvid is None:
            raise ProviderError("unsupported Bilibili URL")
        data = self._view(bvid)
        source_url = urlunparse(("https", "www.bilibili.com", f"/video/{bvid}", "", "", ""))
        source = SourceRef(kind="url", value=source_url)
        metadata = VideoMetadata(
            id=f"bilibili__{bvid}",
            source=source,
            title=data.title,
            uploader=data.owner.name,
            duration_seconds=data.duration,
            extractor="bilibili",
            raw_provider="bilibili",
        )
        revision_value = hashlib.sha256(
            json.dumps(
                [data.bvid, data.cid, data.title, data.duration], separators=(",", ":")
            ).encode()
        ).hexdigest()
        record = SourceRecord(
            source_id=metadata.id,
            revision=Revision(kind="observed", value=revision_value),
            observed_at=datetime.now(UTC),
            metadata=metadata,
            has_subtitles=False,
            has_media=True,
        )
        return BilibiliSession(
            bvid=bvid,
            cid=data.cid,
            source_url=source_url,
            record=record,
            net=self.net,
            receipts=[EffectReceipt(operation="observe", status="succeeded", attempts=1)],
        )

    def _view(self, bvid: str) -> _ViewData:
        url = "https://api.bilibili.com/x/web-interface/view?" + urlencode({"bvid": bvid})
        response = self.net.request(_api_request(url))
        try:
            envelope = _ViewEnvelope.model_validate_json(response.body)
        except ValidationError as exc:
            raise ProviderError("Bilibili metadata returned an invalid response") from exc
        if response.status_code != 200 or envelope.code != 0 or envelope.data is None:
            raise ProviderError("Bilibili metadata unavailable")
        return envelope.data


def bilibili_bvid(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "bilibili.com",
        "www.bilibili.com",
    }:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0] != "video" or _BV.fullmatch(parts[1]) is None:
        return None
    return parts[1]


def _api_request(url: str) -> NetRequest:
    return NetRequest(
        method="GET",
        url=url,
        headers=_HEADERS,
        timeout_s=30,
        connect_timeout_s=10,
        purpose="source_observe",
        provider_id="bilibili",
        retry=RetryPolicy(
            max_attempts=3,
            statuses=(408, 429, 500, 502, 503, 504),
            retry_connect=True,
            retry_timeouts=True,
        ),
    )


def _select_stream(data: _PlayData, request: MediaRequest) -> _DashStream:
    if request.kind == "asr_audio":
        candidates = data.dash.audio
        if not candidates:
            raise ProviderError("Bilibili playback has no anonymous audio stream")
        return max(candidates, key=lambda item: (item.bandwidth, item.id))
    cap = _HEIGHT_CAP[request.profile]
    candidates = [item for item in data.dash.video if item.height is None or item.height <= cap]
    if not candidates:
        candidates = data.dash.video
    if not candidates:
        raise ProviderError("Bilibili playback has no anonymous video stream")
    return max(candidates, key=lambda item: (item.height or 0, item.bandwidth, item.id))


def _extension(stream: _DashStream) -> str:
    if stream.mime_type == "video/mp4" or stream.mime_type == "audio/mp4":
        return "m4s"
    return Path(urlparse(stream.base_url).path).suffix.lstrip(".") or "m4s"
