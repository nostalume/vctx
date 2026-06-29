from __future__ import annotations

import json

import pytest

from vctx.net import NetRequest, NetResponse
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment
from vctx.transforms.ai_routes import AiRoute, AiTaskKind
from vctx.transforms.model_resolution import ModelRef, ResolvedModelRoute
from vctx.transforms.text_ai import OpenAiCompatibleTextAdapter, TextAiExecutionError


class FakeTextNetRuntime:
    def __init__(self, response_body: dict[str, object]) -> None:
        self.response_body = response_body
        self.requests: list[NetRequest] = []

    def request(self, request: NetRequest) -> NetResponse:
        self.requests.append(request)
        return NetResponse(
            url=request.url,
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(self.response_body).encode("utf-8"),
        )


class FailingTextNetRuntime:
    def request(self, request: NetRequest) -> NetResponse:
        del request
        raise OSError("network failed with secret-token Authorization header")


def test_text_adapter_requests_openai_compatible_knowledge_flow_json() -> None:
    net = FakeTextNetRuntime(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "nodes": [
                                    {
                                        "id": "url-acquisition",
                                        "label": "URL acquisition",
                                        "evidence": ["seg_000001"],
                                    },
                                    {
                                        "id": "context-pack",
                                        "label": "context pack",
                                        "evidence": ["seg_000001"],
                                    },
                                ],
                                "edges": [
                                    {
                                        "id": "url-acquisition__context-pack",
                                        "source": "url-acquisition",
                                        "target": "context-pack",
                                        "evidence": ["seg_000001"],
                                    }
                                ],
                            }
                        )
                    }
                }
            ]
        }
    )
    adapter = OpenAiCompatibleTextAdapter(
        route=_openrouter_text_route(task="knowledge_flow_extraction"),
        api_key="secret-token",
        net=net,
    )

    supplement = adapter.knowledge_flow_supplement(_transcript())

    assert supplement.flow.nodes[0].label == "URL acquisition"
    assert supplement.flow.edges[0].evidence == ["seg_000001"]
    assert supplement.evidence.route_provider_id == "openrouter"
    assert supplement.evidence.route_model == "test/text-model"
    request = net.requests[0]
    assert request.method == "POST"
    assert request.url == "https://openrouter.example/api/v1/chat/completions"
    assert request.purpose == "knowledge_flow_extraction"
    assert request.provider_id == "openrouter"
    assert request.headers["Authorization"] == "Bearer secret-token"
    body = json.loads((request.body or b"").decode("utf-8"))
    assert body["model"] == "test/text-model"
    assert body["response_format"] == {"type": "json_object"}
    assert "seg_000001" in body["messages"][1]["content"]
    assert "cleanup" not in body["messages"][1]["content"].lower()
    assert "chapter" not in body["messages"][1]["content"].lower()


def test_text_adapter_requests_openai_compatible_essential_case_json() -> None:
    net = FakeTextNetRuntime(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "cases": [
                                    {
                                        "segment_id": "seg_000001",
                                        "timestamp_seconds": 1.5,
                                        "case_type": "diagram",
                                        "priority": 0.8,
                                        "reason": "architecture diagram cue",
                                        "actions": ["describe", "capture"],
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )
    adapter = OpenAiCompatibleTextAdapter(
        route=_openrouter_text_route(task="essential_case_extraction"),
        api_key="secret-token",
        net=net,
    )

    supplement = adapter.essential_case_supplement(_transcript())

    assert [(case.segment_id, case.case_type) for case in supplement.cases] == [
        ("seg_000001", "diagram")
    ]
    assert supplement.evidence.route_provider_id == "openrouter"
    request = net.requests[0]
    assert request.method == "POST"
    assert request.purpose == "essential_case_extraction"
    body = json.loads((request.body or b"").decode("utf-8"))
    assert body["model"] == "test/text-model"
    assert body["response_format"] == {"type": "json_object"}
    system_prompt = body["messages"][0]["content"]
    assert "Use capture for visual reference/action context" in system_prompt
    assert "Use OCR only for literal shown text" in system_prompt
    assert "Do not mark charts for OCR/VLM" in system_prompt
    assert "seg_000001" in body["messages"][1]["content"]
    assert "cleanup" not in body["messages"][1]["content"].lower()
    assert "chapter" not in body["messages"][1]["content"].lower()


def test_text_adapter_network_errors_do_not_include_api_key_or_raw_error() -> None:
    adapter = OpenAiCompatibleTextAdapter(
        route=_openrouter_text_route(task="knowledge_flow_extraction"),
        api_key="secret-token",
        net=FailingTextNetRuntime(),
    )

    with pytest.raises(TextAiExecutionError) as exc_info:
        adapter.knowledge_flow_supplement(_transcript())

    message = str(exc_info.value)
    assert "text AI request failed for provider openrouter: OSError" in message
    assert "network failed" not in message
    assert "Authorization" not in message
    assert "secret-token" not in message


def _transcript() -> Transcript:
    return Transcript(
        video_id="video-1",
        provenance=TranscriptProvenance(method="local_file", format="plain"),
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=3.0,
                text="A URL acquisition step creates the context pack.",
            )
        ],
    )



def _openrouter_text_route(*, task: AiTaskKind) -> AiRoute:
    return AiRoute.from_model_route(
        task=task,
        selected="configured-online",
        model_route=ResolvedModelRoute(
            ref=ModelRef(prefix="openrouter", value="test/text-model"),
            provider="openrouter",
            model="test/text-model",
            base_url="https://openrouter.example/api/v1/chat/completions",
            api_key_env="OPENROUTER_API_KEY",
            cost="free",
            upload="required",
            available=True,
            reason="test route",
        ),
    )
