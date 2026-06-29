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

    def __init__(self, params: dict[str, object]) -> None:
        self.params = params

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
        assert value == "https://video.example/watch?v=abc"
        assert download is False
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
    import vctx.sources.ytdlp_source as module

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
    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(module, "UrllibNetRuntime", FakeSubtitleRuntime)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["prepare", "https://video.example/watch?v=abc", "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote context pack" in result.output
    assert "Wrote partial context pack" not in result.output
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "transcript.raw.json").exists()
    assert (out_dir / "transcript.clean.json").exists()
    assert (out_dir / "chunks.json").exists()
    assert (out_dir / "context.md").exists()
    assert (out_dir / "readable.md").exists()
    assert (out_dir / "transcript.md").exists()
    assert (out_dir / "knowledge_flow.json").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert _step_status(manifest, "source.detect") == "ok"
    assert _step_status(manifest, "metadata.extract") == "ok"
    assert _step_status(manifest, "transcript.extract") == "ok"
    assert _step_detail(manifest, "transcript.extract") == "yt-dlp:official_subtitles:en:vtt"

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "example__abc"
    assert metadata["title"] == "URL Lecture"
    assert metadata["source_type"] == "url"

    clean = json.loads((out_dir / "transcript.clean.json").read_text(encoding="utf-8"))
    assert clean["segments"][0]["text"] == (
        "The workflow takes a video URL and produces a knowledge-flow pack."
    )

    context = (out_dir / "context.md").read_text(encoding="utf-8")
    assert "# Agent Context Pack" in context
    assert "The workflow takes a video URL" in context
    assert "## Knowledge-flow summary" in context

    readable = (out_dir / "readable.md").read_text(encoding="utf-8")
    assert "## Knowledge-flow summary" in readable

    knowledge_flow = json.loads(
        (out_dir / "knowledge_flow.json").read_text(encoding="utf-8")
    )
    assert _has_edge(knowledge_flow, "video URL", "knowledge-flow pack")
    assert _has_edge(knowledge_flow, "download media", "transcribe audio")
    assert _has_edge(knowledge_flow, "transcribe audio", "extract frames")


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
    steps = manifest["steps"]
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
