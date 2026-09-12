from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.cli import app
from vctx.net import NetRequest, NetResponse

runner = CliRunner()


class FakeYoutubeDL:
    info: dict[str, object] = {}
    observations = 0

    def __init__(self, params: dict[str, object]) -> None:
        self.params = params

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
        assert value.startswith("https://video.example/watch?v=abc")
        assert download is False
        type(self).observations += 1
        return self.info


class FakeSubtitleRuntime:
    response_text: str = ""
    requests: list[NetRequest] = []

    def request(self, request: NetRequest) -> NetResponse:
        self.requests.append(request)
        return NetResponse(
            url=request.url,
            status_code=200,
            headers={"content-type": "text/vtt"},
            body=self.response_text.encode("utf-8"),
        )

    def close(self) -> None:
        pass


class OfflineRuntime:
    def request(self, request: NetRequest) -> NetResponse:
        pytest.fail(f"offline transcript attempted network access: {request.url}")

    def close(self) -> None:
        pass


class ConflictingYoutubeDL(FakeYoutubeDL):
    def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
        duration = 1 if value.endswith("one") else 2
        return {
            "id": "shared",
            "title": f"Revision {duration}",
            "duration": duration,
            "webpage_url": value,
            "extractor": "example",
            "subtitles": {"en": [{"ext": "vtt", "url": "https://cdn.example/shared.vtt"}]},
            "automatic_captions": {},
        }


def test_prepare_url_with_official_subtitles_writes_full_context_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vctx.app.run as run_module
    import vctx.source.ytdlp as module

    subtitle_url = "https://cdn.example/caption.vtt"
    FakeSubtitleRuntime.response_text = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\n"
        "The workflow takes a video URL and produces a knowledge-flow pack.\n\n"
        "00:00:02.000 --> 00:00:04.000\n"
        "The pipeline consists of download media, transcribe audio, and extract frames.\n"
    )
    FakeSubtitleRuntime.requests = []
    FakeYoutubeDL.info = {
        "id": "abc",
        "title": "URL Lecture",
        "uploader": "Teacher",
        "duration": 2,
        "webpage_url": "https://video.example/watch?v=abc",
        "language": "en",
        "extractor": "example",
        "subtitles": {"en": [{"ext": "vtt", "url": subtitle_url}]},
        "automatic_captions": {},
    }
    FakeYoutubeDL.observations = 0
    monkeypatch.setattr(module._yt_dlp(), "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(run_module, "HttpxNetRuntime", FakeSubtitleRuntime)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "prepare",
            "https://video.example/watch?v=abc&token=manifest-secret",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert FakeYoutubeDL.observations == 1
    assert "Wrote context pack" in result.output
    assert "Wrote partial context pack" not in result.output
    assert (out_dir / "manifest.json").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    source_entry = manifest["sources"][0]
    lane = out_dir / source_entry["path"]
    assert {path.name for path in lane.iterdir()} >= {
        "metadata.json",
        "transcript.json",
        "chunks.json",
        "context.md",
        "read.md",
    }
    assert "manifest-secret" not in json.dumps(manifest)
    assert manifest["status"] == "ok"
    assert manifest["schema_version"] == "5"

    subtitle_asset = next(item for item in source_entry["artifacts"] if item["kind"] == "subtitle")
    assert subtitle_asset["kind"] == "subtitle"
    assert subtitle_asset["path"] == "subtitle.en.vtt"
    assert (lane / subtitle_asset["path"]).read_text(encoding="utf-8") == (
        FakeSubtitleRuntime.response_text
    )
    source_effects = [
        effect["operation"]
        for effect in source_entry["effects"]
        if effect["operation"] in {"observe", "subtitle", "media"}
    ]
    assert source_effects == [
        "observe",
        "subtitle",
    ]

    metadata = json.loads((lane / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "example__abc"
    assert metadata["title"] == "URL Lecture"
    assert metadata["source"]["kind"] == "url"

    clean = json.loads((lane / "transcript.json").read_text(encoding="utf-8"))
    assert clean["segments"][0]["text"] == (
        "The workflow takes a video URL and produces a knowledge-flow pack."
    )

    context = (lane / "context.md").read_text(encoding="utf-8")
    assert "# Agent Context Pack" in context
    assert "The workflow takes a video URL" in context


@pytest.mark.parametrize("urls", [("one", "two"), ("two", "one")])
def test_batch_rejects_conflicting_revisions_independently_of_input_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, urls: tuple[str, str]
) -> None:
    import vctx.app.run as run_module
    import vctx.source.ytdlp as module

    FakeSubtitleRuntime.response_text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nShared source.\n"
    monkeypatch.setattr(module._yt_dlp(), "YoutubeDL", ConflictingYoutubeDL)
    monkeypatch.setattr(run_module, "HttpxNetRuntime", FakeSubtitleRuntime)
    local = tmp_path / "independent.srt"
    local.write_text("1\n00:00:00,000 --> 00:00:01,000\nIndependent.\n", encoding="utf-8")
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "prepare",
            f"https://video.example/{urls[0]}",
            str(local),
            f"https://video.example/{urls[1]}",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 3
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [source["title"] for source in manifest["sources"]] == ["independent"]
    assert len(manifest["run"]["failures"]) == 2
    assert "video.example" not in json.dumps(manifest["run"]["failures"])


def test_prepare_url_seeds_verified_cache_for_network_free_offline_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.app.run as run_module
    import vctx.source.ytdlp as module

    url = "https://video.example/watch?v=abc"
    FakeYoutubeDL.info = {
        "id": "abc",
        "title": "Cached Lecture",
        "webpage_url": url,
        "extractor": "example",
        "subtitles": {"en": [{"ext": "vtt", "url": "https://cdn.example/caption.vtt"}]},
        "automatic_captions": {},
    }
    FakeSubtitleRuntime.response_text = (
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nCached transcript survives offline.\n"
    )
    monkeypatch.setattr(module._yt_dlp(), "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(run_module, "HttpxNetRuntime", FakeSubtitleRuntime)
    cache = tmp_path / "cache"

    online = runner.invoke(
        app,
        ["prepare", url, "--out", str(tmp_path / "online"), "--cache-dir", str(cache)],
    )
    assert online.exit_code == 0, online.output

    monkeypatch.setattr(
        module._yt_dlp(),
        "YoutubeDL",
        lambda _params: pytest.fail("offline admission attempted provider discovery"),
    )
    monkeypatch.setattr(
        run_module,
        "HttpxNetRuntime",
        OfflineRuntime,
    )
    offline_out = tmp_path / "offline"
    offline = runner.invoke(
        app,
        [
            "prepare",
            url,
            "--out",
            str(offline_out),
            "--cache-dir",
            str(cache),
            "--offline",
        ],
    )

    assert offline.exit_code == 0, offline.output
    manifest = json.loads((offline_out / "manifest.json").read_text(encoding="utf-8"))
    source_entry = manifest["sources"][0]
    lane = offline_out / source_entry["path"]
    clean = json.loads((lane / "transcript.json").read_text(encoding="utf-8"))
    assert clean["segments"][0]["text"] == "Cached transcript survives offline."
    assert source_entry["freshness"] == "unverified-offline"
    source_effects = [
        (item["operation"], item["status"])
        for item in source_entry["effects"]
        if item["operation"] in {"observe", "subtitle", "media"}
    ]
    assert source_effects == [
        ("observe", "cache_hit"),
        ("subtitle", "cache_hit"),
    ]
    assert (cache / "source" / "index.sqlite3").is_file()


def test_online_prepare_degrades_when_source_cache_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.app.run as run_module
    import vctx.source.ytdlp as module

    FakeYoutubeDL.info = {
        "id": "abc",
        "webpage_url": "https://video.example/watch?v=abc",
        "extractor": "example",
        "subtitles": {"en": [{"ext": "vtt", "url": "https://cdn.example/caption.vtt"}]},
        "automatic_captions": {},
    }
    FakeSubtitleRuntime.response_text = (
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nUncached result remains usable.\n"
    )
    monkeypatch.setattr(module._yt_dlp(), "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(run_module, "HttpxNetRuntime", FakeSubtitleRuntime)
    blocked_cache = tmp_path / "cache"
    blocked_cache.write_text("not a directory", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "prepare",
            "https://video.example/watch?v=abc",
            "--out",
            str(tmp_path / "out"),
            "--cache-dir",
            str(blocked_cache),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert (tmp_path / "out" / manifest["sources"][0]["path"] / "transcript.json").is_file()
