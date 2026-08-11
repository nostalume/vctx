from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

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


def test_prepare_url_with_official_subtitles_writes_full_context_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(module, "UrllibNetRuntime", FakeSubtitleRuntime)
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
        "metadata.json", "transcript.raw.json", "transcript.clean.json", "chunks.json",
        "context.md", "readable.md", "transcript.md", "knowledge_flow.json",
    }
    assert "manifest-secret" not in json.dumps(manifest)
    assert manifest["status"] == "ok"
    assert manifest["schema_version"] == "0.3"
    subtitle_asset = source_entry["assets"][0]
    assert subtitle_asset["kind"] == "subtitle"
    assert subtitle_asset["path"] == "subtitle.en.vtt"
    assert (lane / subtitle_asset["path"]).read_text(encoding="utf-8") == (
        FakeSubtitleRuntime.response_text
    )
    assert [effect["operation"] for effect in source_entry["effects"]] == [
        "observe",
        "subtitle",
    ]
    assert _step_status(manifest, "source.detect") == "ok"
    assert _step_status(manifest, "metadata.extract") == "ok"
    assert _step_status(manifest, "transcript.extract") == "ok"
    assert _step_detail(manifest, "transcript.extract") == "yt-dlp:official_subtitles:en:vtt"

    metadata = json.loads((lane / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "example__abc"
    assert metadata["title"] == "URL Lecture"
    assert metadata["source_type"] == "url"

    clean = json.loads((lane / "transcript.clean.json").read_text(encoding="utf-8"))
    assert clean["segments"][0]["text"] == (
        "The workflow takes a video URL and produces a knowledge-flow pack."
    )

    context = (lane / "context.md").read_text(encoding="utf-8")
    assert "# Agent Context Pack" in context
    assert "The workflow takes a video URL" in context
    assert "## Knowledge-flow summary" in context

    readable = (lane / "readable.md").read_text(encoding="utf-8")
    assert "## Knowledge-flow summary" in readable

    knowledge_flow = json.loads(
        (lane / "knowledge_flow.json").read_text(encoding="utf-8")
    )
    assert _has_edge(knowledge_flow, "video URL", "knowledge-flow pack")
    assert _has_edge(knowledge_flow, "download media", "transcribe audio")
    assert _has_edge(knowledge_flow, "transcribe audio", "extract frames")


def test_prepare_url_seeds_verified_cache_for_network_free_offline_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.source.ytdlp as module

    url = "https://video.example/watch?v=abc"
    FakeYoutubeDL.info = {
        "id": "abc",
        "title": "Cached Lecture",
        "webpage_url": url,
        "extractor": "example",
        "subtitles": {
            "en": [{"ext": "vtt", "url": "https://cdn.example/caption.vtt"}]
        },
        "automatic_captions": {},
    }
    FakeSubtitleRuntime.response_text = (
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nCached transcript survives offline.\n"
    )
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(module, "UrllibNetRuntime", FakeSubtitleRuntime)
    cache = tmp_path / "cache"

    online = runner.invoke(
        app,
        ["prepare", url, "--out", str(tmp_path / "online"), "--cache-dir", str(cache)],
    )
    assert online.exit_code == 0, online.output

    monkeypatch.setattr(
        module.yt_dlp,
        "YoutubeDL",
        lambda _params: pytest.fail("offline admission attempted provider discovery"),
    )
    monkeypatch.setattr(
        module,
        "UrllibNetRuntime",
        lambda: pytest.fail("offline transcript attempted network access"),
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
    clean = json.loads((lane / "transcript.clean.json").read_text(encoding="utf-8"))
    assert clean["segments"][0]["text"] == "Cached transcript survives offline."
    assert source_entry["freshness"] == "unverified-offline"
    assert [(item["operation"], item["status"]) for item in source_entry["effects"]] == [
        ("observe", "cache_hit"),
        ("subtitle", "cache_hit"),
    ]
    assert (cache / "source" / "index.sqlite3").is_file()


def test_online_prepare_degrades_when_source_cache_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.source.ytdlp as module

    FakeYoutubeDL.info = {
        "id": "abc",
        "webpage_url": "https://video.example/watch?v=abc",
        "extractor": "example",
        "subtitles": {
            "en": [{"ext": "vtt", "url": "https://cdn.example/caption.vtt"}]
        },
        "automatic_captions": {},
    }
    FakeSubtitleRuntime.response_text = (
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nUncached result remains usable.\n"
    )
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(module, "UrllibNetRuntime", FakeSubtitleRuntime)
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
    assert (tmp_path / "out" / manifest["sources"][0]["path"] / "transcript.clean.json").is_file()


def _step_status(manifest: dict[str, Any], name: str) -> str:
    step = _step(manifest, name)
    status = step["status"]
    assert isinstance(status, str)
    return status


def _step_detail(manifest: dict[str, Any], name: str) -> str:
    step = _step(manifest, name)
    detail = step["detail"]
    assert isinstance(detail, str)
    return detail


def _step(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    steps = manifest["sources"][0]["steps"]
    assert isinstance(steps, list)
    for raw_step in steps:
        assert isinstance(raw_step, dict)
        step = cast(dict[str, Any], raw_step)
        if step["name"] == name:
            return step
    raise AssertionError(f"missing manifest step: {name}")


def _has_edge(flow: dict[str, Any], source: str, target: str) -> bool:
    nodes = flow["nodes"]
    edges = flow["edges"]
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    node_labels = {
        node["id"]: node["label"] for node in nodes if isinstance(node, dict)
    }
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if node_labels[edge["source"]] == source and node_labels[edge["target"]] == target:
            return True
    return False
