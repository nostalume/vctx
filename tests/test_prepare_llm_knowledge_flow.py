from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from vctx.cli import app
from vctx.models.knowledge_flow import (
    KnowledgeFlow,
    KnowledgeFlowEdge,
    KnowledgeFlowNode,
    KnowledgeFlowSupplement,
    KnowledgeFlowSupplementEvidence,
)
from vctx.net import NetRequest, NetResponse
from vctx.transcript import Transcript

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
        assert value == "https://video.example/watch?v=llm"
        assert download is False
        return self.info


class FakeSubtitleRuntime:
    response_text: str = ""

    def request(self, request: NetRequest) -> NetResponse:
        return NetResponse(
            url=request.url,
            status_code=200,
            headers={"content-type": "text/vtt"},
            body=self.response_text.encode("utf-8"),
        )


class FakeTextAdapter:
    calls: list[Transcript] = []
    seen_api_key: str | None = None

    def __init__(self, *, route: object, api_key: str, net: object | None = None) -> None:
        del route, net
        self.seen_api_key = api_key
        FakeTextAdapter.seen_api_key = api_key

    def knowledge_flow_supplement(self, transcript: Transcript) -> KnowledgeFlowSupplement:
        FakeTextAdapter.calls.append(transcript)
        return KnowledgeFlowSupplement(
            flow=KnowledgeFlow(
                nodes=[
                    KnowledgeFlowNode(
                        id="extract-subtitles",
                        label="extract subtitles",
                        evidence=["seg_000001"],
                    ),
                    KnowledgeFlowNode(
                        id="build-context-pack",
                        label="build context pack",
                        evidence=["seg_000001"],
                    ),
                ],
                edges=[
                    KnowledgeFlowEdge(
                        id="extract-subtitles__build-context-pack",
                        source="extract-subtitles",
                        target="build-context-pack",
                        evidence=["seg_000001"],
                    )
                ],
            ),
            evidence=KnowledgeFlowSupplementEvidence(
                route_provider_id="openrouter",
                route_model="test/text-model",
                source_segment_ids=["seg_000001"],
            ),
        )


def test_prepare_explicit_llm_knowledge_flow_supplement_merges_into_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vctx.app.prepare as prepare_module
    import vctx.sources.ytdlp_source as ytdlp_module

    FakeTextAdapter.calls = []
    FakeTextAdapter.seen_api_key = None
    subtitle_url = "https://cdn.example/caption.vtt"
    FakeSubtitleRuntime.response_text = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:03.000\n"
        "The workflow extracts subtitles and then builds a context pack.\n"
    )
    FakeYoutubeDL.info = {
        "id": "llm",
        "title": "LLM Lecture",
        "uploader": "Teacher",
        "duration": 3,
        "webpage_url": "https://video.example/watch?v=llm",
        "language": "en",
        "extractor": "example",
        "subtitles": {"en": [{"ext": "vtt", "url": subtitle_url}]},
        "automatic_captions": {},
    }
    monkeypatch.setattr(ytdlp_module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(ytdlp_module, "UrllibNetRuntime", FakeSubtitleRuntime)
    monkeypatch.setattr(prepare_module, "OpenAiCompatibleTextAdapter", FakeTextAdapter)
    config_path = tmp_path / "config.toml"
    env_path = tmp_path / ".env"
    env_path.write_text("OPENROUTER_API_KEY=from-dotenv", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "[runtime]",
                'env_files = [".env"]',
                "",
                "[transforms.knowledge_flow]",
                'enabled = "true"',
                'route = "configured-online"',
                'model = "openrouter:test/text-model"',
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "prepare",
            "https://video.example/watch?v=llm",
            "--out",
            str(out_dir),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert FakeTextAdapter.seen_api_key == "from-dotenv"
    assert len(FakeTextAdapter.calls) == 1
    flow = json.loads((out_dir / "knowledge_flow.json").read_text(encoding="utf-8"))
    assert _has_edge(flow, "extract subtitles", "build context pack")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert _step_status(manifest, "knowledge_flow.llm_extract") == "ok"
    assert manifest["transform_evidence"][-1]["capability"] == "knowledge_flow"
    assert manifest["transform_evidence"][-1]["selected_route"] == "configured-online"
    assert manifest["transform_evidence"][-1]["model_id"] == "test/text-model"
    assert "from-dotenv" not in json.dumps(manifest)


def _step_status(manifest: dict[str, Any], name: str) -> str:
    step = _step(manifest, name)
    status = step["status"]
    assert isinstance(status, str)
    return status


def _step(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for raw_step in manifest["steps"]:
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
    node_labels = {node["id"]: node["label"] for node in nodes if isinstance(node, dict)}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if node_labels[edge["source"]] == source and node_labels[edge["target"]] == target:
            return True
    return False
