from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import TypeAlias, cast

import pytest
from typer.testing import CliRunner

from tests.support import asr_ready
from vctx.cli import app
from vctx.errors import ProviderError
from vctx.source.session import MediaAsset
from vctx.source.ytdlp import YtDlpInfo, YtDlpParams

runner = CliRunner()

JsonScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]  # noqa: UP040
JsonObject: TypeAlias = dict[str, JsonValue]  # noqa: UP040


class FakeYoutubeDLMedia:
    info: YtDlpInfo = {}
    downloaded_path: Path
    calls: list[tuple[bool, YtDlpParams]] = []

    def __init__(self, params: YtDlpParams) -> None:
        self.params = params

    def __enter__(self) -> FakeYoutubeDLMedia:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def extract_info(self, value: str, download: bool = False) -> YtDlpInfo:
        assert value == "https://video.example/watch?v=no-captions"
        self.calls.append((download, self.params))
        if download:
            paths = cast(dict[str, object], self.params["paths"])
            type(self).downloaded_path = Path(str(paths["home"])) / "example__no-captions.m4a"
            type(self).downloaded_path.parent.mkdir(parents=True, exist_ok=True)
            type(self).downloaded_path.write_bytes(b"fake downloaded audio")
            return {
                **self.info,
                "requested_downloads": [{"filepath": str(self.downloaded_path)}],
            }
        return self.info

    def process_ie_result(self, info: YtDlpInfo, download: bool = False) -> YtDlpInfo:
        assert download is True
        self.calls.append((download, self.params))
        paths = cast(dict[str, object], self.params["paths"])
        type(self).downloaded_path = Path(str(paths["home"])) / "example__no-captions.m4a"
        type(self).downloaded_path.parent.mkdir(parents=True, exist_ok=True)
        type(self).downloaded_path.write_bytes(b"fake downloaded audio")
        return {**info, "requested_downloads": [{"filepath": str(self.downloaded_path)}]}


@pytest.mark.parametrize("retain_media", [True, False])
def test_prepare_url_without_subtitles_downloads_media_and_runs_asr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, retain_media: bool
) -> None:
    import vctx.asr as asr_module
    import vctx.source.ytdlp as ytdlp_module

    FakeYoutubeDLMedia.calls = []
    FakeYoutubeDLMedia.downloaded_path = tmp_path / "downloaded" / "lecture.m4a"
    FakeYoutubeDLMedia.info = {
        "id": "no-captions",
        "title": "No Captions",
        "duration": 2,
        "webpage_url": "https://video.example/watch?v=no-captions",
        "language": "en",
        "extractor": "example",
        "subtitles": {},
        "automatic_captions": {},
        "formats": [
            {"format_id": "audio", "acodec": "aac", "vcodec": "none"},
            {"format_id": "video", "acodec": "none", "vcodec": "avc1", "height": 480},
        ],
    }
    monkeypatch.setattr(ytdlp_module._yt_dlp(), "YoutubeDL", FakeYoutubeDLMedia)

    class FakeAsrAdapter:
        def __init__(self, **kwargs: JsonValue) -> None:
            del kwargs

        def transcribe(self, media_asset: MediaAsset, **options: object) -> object:
            assert options == {"progress": False}
            assert media_asset.local_path.read_bytes() == b"fake downloaded audio"
            assert media_asset.local_path.parent == tmp_path / "cache" / "source" / "blobs"
            return asr_ready(media_asset.id, "URL ASR text.", model="tiny")

    monkeypatch.setattr(asr_module, "FasterWhisperAsrAdapter", FakeAsrAdapter)
    config_path = tmp_path / "vctx.toml"
    config_path.write_text(
        """
[transforms.asr]
use = "instance:local-default"

[instances.asr.local-default]
type = "local-faster-whisper"
model = "tiny"
""".strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    args = [
        "prepare",
        "https://video.example/watch?v=no-captions",
        "--out",
        str(out_dir),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--config",
        str(config_path),
    ]
    if not retain_media:
        args.append("--no-retain-media")
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    manifest = cast(JsonObject, json.loads((out_dir / "manifest.json").read_text(encoding="utf-8")))
    source_entry = cast(JsonObject, cast(list[JsonObject], manifest["sources"])[0])
    lane = out_dir / cast(str, source_entry["path"])
    assert manifest["status"] == "ok"
    artifacts = cast(list[JsonObject], source_entry["artifacts"])
    retained = [item for item in artifacts if item["kind"] == "source_audio"]
    if retain_media:
        assert len(retained) == 1
        assert (lane / cast(str, retained[0]["path"])).is_file()
    else:
        assert retained == []
        assert not (lane / "assets").exists()
    effects = cast(list[JsonObject], source_entry["effects"])
    route = next(item for item in effects if item["operation"] == "asr")
    assert (route["route"], route["provider"], route["model"]) == (
        "local",
        "faster-whisper",
        "tiny",
    )
    clean = cast(
        JsonObject,
        json.loads((lane / "transcript.json").read_text(encoding="utf-8")),
    )
    segments = cast(list[JsonObject], clean["segments"])
    assert segments[0]["text"] == "URL ASR text."
    assert any(download for download, _params in FakeYoutubeDLMedia.calls)
    download_params = [params for download, params in FakeYoutubeDLMedia.calls if download][0]
    assert download_params["skip_download"] is False
    assert download_params["format"] == "bestaudio/best"
    assert download_params["paths"] == {
        "home": str(tmp_path / "cache" / "source" / "tmp" / "yt-dlp"),
        "temp": str(tmp_path / "cache" / "source" / "tmp" / "yt-dlp"),
    }
    assert FakeYoutubeDLMedia.downloaded_path.parent == (
        tmp_path / "cache" / "source" / "tmp" / "yt-dlp"
    )
    if retain_media:
        before = (lane / "transcript.json").read_bytes()
        prior_tree = {
            path.relative_to(out_dir): path.read_bytes()
            for path in out_dir.rglob("*")
            if path.is_file()
        }
        process = FakeYoutubeDLMedia.process_ie_result
        monkeypatch.setattr(
            FakeYoutubeDLMedia,
            "process_ie_result",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ProviderError("video failed")),
        )
        failed = runner.invoke(app, [*args, "--source-assets", "complete"])
        assert failed.exit_code == 7
        assert {
            path.relative_to(out_dir): path.read_bytes()
            for path in out_dir.rglob("*")
            if path.is_file()
        } == prior_tree
        monkeypatch.setattr(FakeYoutubeDLMedia, "process_ie_result", process)
        FakeYoutubeDLMedia.calls = []
        upgraded = runner.invoke(app, [*args, "--source-assets", "complete"])
        assert upgraded.exit_code == 0, upgraded.output
        complete = cast(
            JsonObject,
            cast(
                list[JsonObject],
                cast(JsonObject, json.loads((out_dir / "manifest.json").read_text()))["sources"],
            )[0],
        )
        assert complete["asset_scope"] == "complete"
        assert (lane / "transcript.json").read_bytes() == before
        assert {item["kind"] for item in cast(list[JsonObject], complete["artifacts"])} >= {
            "source_audio",
            "source_video",
        }
        downloads = [params for download, params in FakeYoutubeDLMedia.calls if download]
        assert len(downloads) == 1 and "bestvideo" in cast(str, downloads[0]["format"])
        FakeYoutubeDLMedia.calls = []
        lower = runner.invoke(app, args)
        assert lower.exit_code == 0 and not any(
            download for download, _params in FakeYoutubeDLMedia.calls
        )
        preserved = json.loads((out_dir / "manifest.json").read_text())["sources"][0]
        assert preserved["asset_scope"] == "complete"
        FakeYoutubeDLMedia.calls = []
        offline = runner.invoke(app, [*args, "--source-assets", "complete", "--offline"])
        assert offline.exit_code == 0 and not FakeYoutubeDLMedia.calls
        FakeYoutubeDLMedia.info = {
            key: value for key, value in FakeYoutubeDLMedia.info.items() if key != "formats"
        }
        FakeYoutubeDLMedia.calls = []
        unknown_out = tmp_path / "unknown"
        unknown_args = [str(unknown_out) if item == str(out_dir) else item for item in args]
        unknown = runner.invoke(app, [*unknown_args, "--source-assets", "complete"])
        assert unknown.exit_code == 7 and not unknown_out.exists()
        assert not any(download for download, _params in FakeYoutubeDLMedia.calls)
