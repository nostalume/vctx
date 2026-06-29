from __future__ import annotations

from pathlib import Path
from types import TracebackType

import pytest

from vctx.config import MediaProfile, YtDlpSourceOptions
from vctx.errors import NoTranscriptError
from vctx.io import Cache
from vctx.models.media import AsrAudioFetchRequest, VisualVideoFetchRequest
from vctx.net import NetRequest, NetResponse
from vctx.sources.detect import detect_source_adapter
from vctx.sources.ytdlp_source import YtDlpInfo, YtDlpParams, YtDlpSourceAdapter


class FakeYoutubeDL:
    calls: list[YtDlpParams] = []
    info: YtDlpInfo = {}

    def __init__(self, params: YtDlpParams) -> None:
        self.params = params
        self.calls.append(params)

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def extract_info(self, value: str, download: bool = False) -> YtDlpInfo:
        assert value == "https://video.example/watch?v=abc"
        assert download is False
        return self.info


class FakeSubtitleNetRuntime:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.requests: list[NetRequest] = []

    def request(self, request: NetRequest) -> NetResponse:
        self.requests.append(request)
        return NetResponse(
            url=request.url,
            status_code=200,
            headers={"content-type": "text/vtt"},
            body=self.responses[request.url].encode("utf-8-sig"),
        )


def _runtime_from_files(*paths: Path) -> FakeSubtitleNetRuntime:
    return FakeSubtitleNetRuntime(
        {path.as_uri(): path.read_text(encoding="utf-8") for path in paths}
    )



def _raising_net_factory() -> FakeSubtitleNetRuntime:
    raise AssertionError("net runtime should be lazy")


def test_detect_source_adapter_does_not_construct_ytdlp_net_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vctx.sources.ytdlp_source as module

    monkeypatch.setattr(module, "UrllibNetRuntime", _raising_net_factory)

    adapter = detect_source_adapter("https://video.example/watch?v=abc")

    assert adapter.name == "yt-dlp"


def test_ytdlp_metadata_does_not_construct_net_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vctx.sources.ytdlp_source as module

    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "title": "Lecture",
        "webpage_url": "https://video.example/watch?v=abc",
        "extractor": "example",
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    metadata = YtDlpSourceAdapter(net_factory=_raising_net_factory).extract_metadata(
        "https://video.example/watch?v=abc"
    )

    assert metadata.id == "example__abc"


def test_ytdlp_no_subtitle_path_does_not_construct_net_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vctx.sources.ytdlp_source as module

    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {"id": "abc", "subtitles": {}, "automatic_captions": {}}
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    with pytest.raises(NoTranscriptError, match="no subtitles found"):
        YtDlpSourceAdapter(net_factory=_raising_net_factory).extract_transcript(
            "https://video.example/watch?v=abc",
            cache=Cache(root=tmp_path / "cache"),
        )

def test_detect_source_adapter_selects_ytdlp_for_url() -> None:
    adapter = detect_source_adapter("https://video.example/watch?v=abc")

    assert adapter.name == "yt-dlp"


def test_ytdlp_metadata_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    import vctx.sources.ytdlp_source as module

    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "title": "Lecture",
        "uploader": "Teacher",
        "duration": 123.4,
        "webpage_url": "https://video.example/watch?v=abc",
        "language": "en",
        "extractor": "example",
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    metadata = YtDlpSourceAdapter().extract_metadata("https://video.example/watch?v=abc")

    assert metadata.id == "example__abc"
    assert metadata.source_type == "url"
    assert metadata.title == "Lecture"
    assert metadata.uploader == "Teacher"
    assert metadata.duration_seconds == 123.4
    assert metadata.webpage_url == "https://video.example/watch?v=abc"
    assert metadata.extractor == "example"


def test_ytdlp_extract_transcript_prefers_official_subtitles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as module

    subtitle_file = tmp_path / "caption.vtt"
    subtitle_file.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "subtitles": {"en": [{"ext": "vtt", "url": subtitle_file.as_uri()}]},
        "automatic_captions": {"en": [{"ext": "vtt", "url": "file:///should/not/use.vtt"}]},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    runtime = _runtime_from_files(subtitle_file)

    payload = YtDlpSourceAdapter(net_factory=lambda: runtime).extract_transcript(
        "https://video.example/watch?v=abc",
        cache=Cache(root=tmp_path / "cache"),
    )

    assert payload.text.startswith("WEBVTT")
    assert payload.format == "vtt"
    assert payload.provenance.method == "official_subtitles"
    assert payload.provenance.language == "en"
    assert payload.provenance.provider == "yt-dlp"
    assert payload.provenance.language_evidence.kind == "detected"
    assert payload.provenance.language_evidence.code == "en"
    assert payload.provenance.language_evidence.source == "subtitle"


def test_ytdlp_extract_transcript_fetches_vtt_through_injected_net_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vctx.sources.ytdlp_source as module

    subtitle_url = "https://cdn.example/subtitles/en.vtt"
    runtime = FakeSubtitleNetRuntime(
        {subtitle_url: "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello via net\n"}
    )
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "subtitles": {"en": [{"ext": "vtt", "url": subtitle_url}]},
        "automatic_captions": {},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    payload = YtDlpSourceAdapter(net_factory=lambda: runtime).extract_transcript(
        "https://video.example/watch?v=abc",
        cache=Cache(root=tmp_path / "cache"),
    )

    assert "Hello via net" in payload.text
    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.method == "GET"
    assert request.url == subtitle_url
    assert request.timeout_s == 30
    assert request.purpose == "subtitle_fetch"
    assert request.provider_id == "yt-dlp"


def test_ytdlp_extract_transcript_resolves_hls_vtt_playlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as module

    segment_a = tmp_path / "segment-a.vtt"
    segment_a.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nFirst segment\n",
        encoding="utf-8",
    )
    segment_b = tmp_path / "segment-b.vtt"
    segment_b.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nSecond segment\n",
        encoding="utf-8",
    )
    playlist = tmp_path / "captions.m3u8"
    playlist.write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-VERSION:4",
                "#EXTINF:1.000,",
                segment_a.name,
                "#EXTINF:1.000,",
                segment_b.name,
                "#EXT-X-ENDLIST",
            ]
        ),
        encoding="utf-8",
    )
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "subtitles": {
            "en": [
                {
                    "ext": "vtt",
                    "protocol": "m3u8_native",
                    "url": playlist.as_uri(),
                }
            ]
        },
        "automatic_captions": {},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    runtime = _runtime_from_files(playlist, segment_a, segment_b)

    payload = YtDlpSourceAdapter(net_factory=lambda: runtime).extract_transcript(
        "https://video.example/watch?v=abc",
        cache=Cache(root=tmp_path / "cache"),
    )

    assert payload.format == "vtt"
    assert payload.text.startswith("WEBVTT")
    assert "First segment" in payload.text
    assert "Second segment" in payload.text
    assert payload.text.count("WEBVTT") == 1


def test_ytdlp_extract_transcript_fetches_hls_segments_through_injected_net_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vctx.sources.ytdlp_source as module

    playlist_url = "https://cdn.example/captions/playlist.m3u8"
    segment_a_url = "https://cdn.example/captions/segment-a.vtt"
    segment_b_url = "https://cdn.example/captions/segment-b.vtt"
    runtime = FakeSubtitleNetRuntime(
        {
            playlist_url: "\n".join(
                [
                    "#EXTM3U",
                    "#EXT-X-VERSION:4",
                    "#EXTINF:1.000,",
                    "segment-a.vtt",
                    "#EXTINF:1.000,",
                    "segment-b.vtt",
                    "#EXT-X-ENDLIST",
                ]
            ),
            segment_a_url: "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nFirst net segment\n",
            segment_b_url: "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nSecond net segment\n",
        }
    )
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "subtitles": {"en": [{"ext": "vtt", "url": playlist_url}]},
        "automatic_captions": {},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    payload = YtDlpSourceAdapter(net_factory=lambda: runtime).extract_transcript(
        "https://video.example/watch?v=abc",
        cache=Cache(root=tmp_path / "cache"),
    )

    assert payload.text.count("WEBVTT") == 1
    assert "First net segment" in payload.text
    assert "Second net segment" in payload.text
    assert [request.url for request in runtime.requests] == [
        playlist_url,
        segment_a_url,
        segment_b_url,
    ]
    assert {request.purpose for request in runtime.requests} == {"subtitle_fetch"}


def test_ytdlp_extract_transcript_uses_source_language_before_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as module

    en_file = tmp_path / "en.vtt"
    zh_file = tmp_path / "zh.vtt"
    en_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nEnglish\n", encoding="utf-8"
    )
    zh_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nChinese\n", encoding="utf-8"
    )
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "language": "en",
        "subtitles": {
            "en": [{"ext": "vtt", "url": en_file.as_uri()}],
            "zh-Hans": [{"ext": "vtt", "url": zh_file.as_uri()}],
        },
        "automatic_captions": {},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    runtime = _runtime_from_files(en_file, zh_file)

    payload = YtDlpSourceAdapter(net_factory=lambda: runtime).extract_transcript(
        "https://video.example/watch?v=abc",
        cache=Cache(root=tmp_path / "cache"),
    )

    assert "English" in payload.text
    assert payload.provenance.language == "en"
    assert payload.provenance.method == "official_subtitles"


def test_ytdlp_extract_transcript_uses_request_subtitle_language_before_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as module

    en_file = tmp_path / "en.vtt"
    ja_file = tmp_path / "ja.vtt"
    en_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nEnglish\n", encoding="utf-8"
    )
    ja_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n日本語字幕\n", encoding="utf-8"
    )
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "subtitles": {
            "en": [{"ext": "vtt", "url": en_file.as_uri()}],
            "ja": [{"ext": "vtt", "url": ja_file.as_uri()}],
        },
        "automatic_captions": {},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    runtime = _runtime_from_files(en_file, ja_file)

    payload = YtDlpSourceAdapter(net_factory=lambda: runtime).extract_transcript(
        "https://video.example/watch?v=abc",
        cache=Cache(root=tmp_path / "cache"),
        source_options=YtDlpSourceOptions(subtitle_languages=["ja"]),
    )

    assert "日本語字幕" in payload.text
    assert payload.provenance.language == "ja"


def test_ytdlp_extract_transcript_keeps_metadata_language_before_request_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as module

    en_file = tmp_path / "en.vtt"
    ja_file = tmp_path / "ja.vtt"
    en_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nEnglish metadata language\n",
        encoding="utf-8",
    )
    ja_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n日本語字幕\n", encoding="utf-8"
    )
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "language": "en",
        "subtitles": {
            "en": [{"ext": "vtt", "url": en_file.as_uri()}],
            "ja": [{"ext": "vtt", "url": ja_file.as_uri()}],
        },
        "automatic_captions": {},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    runtime = _runtime_from_files(en_file, ja_file)

    payload = YtDlpSourceAdapter(net_factory=lambda: runtime).extract_transcript(
        "https://video.example/watch?v=abc",
        cache=Cache(root=tmp_path / "cache"),
        source_options=YtDlpSourceOptions(subtitle_languages=["ja"]),
    )

    assert "English metadata language" in payload.text
    assert payload.provenance.language == "en"


def test_ytdlp_extract_transcript_uses_automatic_caption_when_official_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as module

    auto_file = tmp_path / "auto.vtt"
    auto_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nAutomatic\n", encoding="utf-8"
    )
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "language": "en",
        "subtitles": {},
        "automatic_captions": {"en": [{"ext": "vtt", "url": auto_file.as_uri()}]},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    runtime = _runtime_from_files(auto_file)

    payload = YtDlpSourceAdapter(net_factory=lambda: runtime).extract_transcript(
        "https://video.example/watch?v=abc",
        cache=Cache(root=tmp_path / "cache"),
    )

    assert "Automatic" in payload.text
    assert payload.provenance.method == "automatic_subtitles"
    assert payload.provenance.language == "en"


def test_ytdlp_extract_transcript_uses_source_language_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as module

    fallback_file = tmp_path / "fallback.vtt"
    fallback_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nFallback\n", encoding="utf-8"
    )
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "language": "en",
        "subtitles": {"en": [{"ext": "vtt", "url": fallback_file.as_uri()}]},
        "automatic_captions": {},
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    runtime = _runtime_from_files(fallback_file)

    payload = YtDlpSourceAdapter(net_factory=lambda: runtime).extract_transcript(
        "https://video.example/watch?v=abc",
        cache=Cache(root=tmp_path / "cache"),
    )

    assert "Fallback" in payload.text
    assert payload.provenance.language == "en"
    assert payload.provenance.method == "official_subtitles"


def test_ytdlp_extract_transcript_raises_when_subtitles_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.sources.ytdlp_source as module

    FakeYoutubeDL.calls = []
    FakeYoutubeDL.info = {"id": "abc", "subtitles": {}, "automatic_captions": {}}
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    with pytest.raises(NoTranscriptError, match="no subtitles found"):
        YtDlpSourceAdapter().extract_transcript(
            "https://video.example/watch?v=abc",
            cache=Cache(root=tmp_path / "cache"),
        )



class FakeYoutubeDLDownload:
    calls: list[tuple[bool, YtDlpParams]] = []
    info: YtDlpInfo = {}

    def __init__(self, params: YtDlpParams) -> None:
        self.params = params

    def __enter__(self) -> FakeYoutubeDLDownload:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def extract_info(self, value: str, download: bool = False) -> YtDlpInfo:
        assert value == "https://video.example/watch?v=abc"
        self.calls.append((download, self.params))
        if download:
            paths = self.params["paths"]
            assert isinstance(paths, dict)
            final_dir = Path(paths["home"])
            filename = final_dir / "example__abc.webm"
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_bytes(b"downloaded media")
            return {
                **self.info,
                "requested_downloads": [{"filepath": str(filename)}],
                "filepath": str(filename),
                "ext": "webm",
                "format_id": "format-1",
            }
        return self.info


def test_ytdlp_extract_media_asr_writes_audio_to_output_media_and_temp_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vctx.sources.ytdlp_source as module

    FakeYoutubeDLDownload.calls = []
    FakeYoutubeDLDownload.info = {
        "id": "abc",
        "extractor": "example",
        "webpage_url": "https://video.example/watch?v=abc",
        "duration": 12.0,
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDLDownload)
    request = AsrAudioFetchRequest(
        source_url="https://video.example/watch?v=abc",
        output_dir=tmp_path / "out" / "media",
        temp_dir=tmp_path / "cache" / "tmp" / "yt-dlp",
        source_options=YtDlpSourceOptions(),
    )

    media = YtDlpSourceAdapter().extract_media(
        "https://video.example/watch?v=abc",
        request=request,
    )

    assert media.kind == "downloaded_asr_audio"
    assert media.local_path.parent == tmp_path / "out" / "media"
    assert media.reuse.kind == "reuse_miss"
    download_params = [params for download, params in FakeYoutubeDLDownload.calls if download][0]
    assert download_params["format"] == "bestaudio/best"
    assert download_params["paths"] == {
        "home": str(tmp_path / "out" / "media"),
        "temp": str(tmp_path / "cache" / "tmp" / "yt-dlp"),
    }


def test_ytdlp_extract_media_visual_uses_bounded_video_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vctx.sources.ytdlp_source as module

    FakeYoutubeDLDownload.calls = []
    FakeYoutubeDLDownload.info = {
        "id": "abc",
        "extractor": "example",
        "webpage_url": "https://video.example/watch?v=abc",
    }
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDLDownload)
    request = VisualVideoFetchRequest(
        source_url="https://video.example/watch?v=abc",
        output_dir=tmp_path / "out" / "media",
        temp_dir=tmp_path / "cache" / "tmp" / "yt-dlp",
        profile=MediaProfile.BALANCED,
        source_options=YtDlpSourceOptions(),
    )

    media = YtDlpSourceAdapter().extract_media(
        "https://video.example/watch?v=abc",
        request=request,
    )

    assert media.kind == "downloaded_visual_video"
    assert media.profile == MediaProfile.BALANCED
    download_params = [params for download, params in FakeYoutubeDLDownload.calls if download][0]
    assert "height<=720" in str(download_params["format"])
    assert download_params["merge_output_format"] == "mp4/webm"
