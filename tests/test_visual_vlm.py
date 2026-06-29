from __future__ import annotations

import json
from pathlib import Path

import pytest

from vctx.models.visual import Evidence, FrameAsset
from vctx.net import NetRequest, NetResponse
from vctx.transforms.ai_routes import AiRoute
from vctx.transforms.visual_vlm import OpenAiCompatibleVisionAdapter, VisionExecutionError


class _FakeNetRuntime:
    def __init__(self, descriptions: list[str] | None = None) -> None:
        self.descriptions = descriptions or ["Diagram shows ingestion to context."]
        self.requests: list[NetRequest] = []

    def request(self, request: NetRequest) -> NetResponse:
        self.requests.append(request)
        index = len(self.requests) - 1
        description = self.descriptions[min(index, len(self.descriptions) - 1)]
        return NetResponse(
            url=request.url,
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps({"choices": [{"message": {"content": description}}]}).encode("utf-8"),
        )


class _FakeBatchNetRuntime:
    def __init__(self) -> None:
        self.batches: list[list[NetRequest]] = []

    def request(self, request: NetRequest) -> NetResponse:
        del request
        raise AssertionError("describe_many should use request_many when available")

    def request_many(self, requests: list[NetRequest]) -> list[NetResponse]:
        self.batches.append(requests)
        return [
            NetResponse(
                url=request.url,
                status_code=200,
                headers={"content-type": "application/json"},
                body=json.dumps(
                    {"choices": [{"message": {"content": f"Batch description {index}."}}]}
                ).encode("utf-8"),
            )
            for index, request in enumerate(requests, start=1)
        ]


class _FailingVisionNetRuntime:
    def request(self, request: NetRequest) -> NetResponse:
        del request
        raise OSError("network failed with test-secret Authorization header")


def test_openai_compatible_vision_adapter_describes_through_injected_net_runtime(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"fake png bytes")
    runtime = _FakeNetRuntime()

    adapter = _adapter(runtime=runtime)

    text = adapter.describe(_frame(frame_path, "frame-0001"))

    assert text == "Diagram shows ingestion to context."
    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.method == "POST"
    assert request.url == "https://example.invalid/v1/chat/completions"
    assert request.timeout_s == 120
    assert request.purpose == "vision_description"
    assert request.provider_id == "test-vlm"
    assert request.headers == {
        "Authorization": "Bearer test-secret",
        "Content-Type": "application/json",
    }
    payload = json.loads((request.body or b"").decode("utf-8"))
    assert payload["model"] == "vision-test"
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_openai_compatible_vision_adapter_describe_many_keeps_frame_order(
    tmp_path: Path,
) -> None:
    first_frame = tmp_path / "first.png"
    second_frame = tmp_path / "second.png"
    first_frame.write_bytes(b"first png")
    second_frame.write_bytes(b"second png")
    runtime = _FakeNetRuntime(["First frame description.", "Second frame description."])
    adapter = _adapter(runtime=runtime)

    outcomes = adapter.describe_many(
        [_frame(first_frame, "frame-0001"), _frame(second_frame, "frame-0002")]
    )

    assert [(outcome.frame_id, outcome.text, outcome.error) for outcome in outcomes] == [
        ("frame-0001", "First frame description.", None),
        ("frame-0002", "Second frame description.", None),
    ]
    assert [request.provider_id for request in runtime.requests] == ["test-vlm", "test-vlm"]
    assert len(runtime.requests) == 2


def test_openai_compatible_vision_adapter_describe_many_uses_batch_runtime(
    tmp_path: Path,
) -> None:
    first_frame = tmp_path / "first.png"
    second_frame = tmp_path / "second.png"
    first_frame.write_bytes(b"first png")
    second_frame.write_bytes(b"second png")
    runtime = _FakeBatchNetRuntime()
    adapter = _adapter(runtime=runtime)

    outcomes = adapter.describe_many(
        [_frame(first_frame, "frame-0001"), _frame(second_frame, "frame-0002")]
    )

    assert [(outcome.frame_id, outcome.text, outcome.error) for outcome in outcomes] == [
        ("frame-0001", "Batch description 1.", None),
        ("frame-0002", "Batch description 2.", None),
    ]
    assert len(runtime.batches) == 1
    assert [request.provider_id for request in runtime.batches[0]] == ["test-vlm", "test-vlm"]
    assert [request.purpose for request in runtime.batches[0]] == [
        "vision_description",
        "vision_description",
    ]


def test_openai_compatible_vision_errors_do_not_include_api_key_or_raw_error(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"fake png bytes")
    adapter = OpenAiCompatibleVisionAdapter(
        route=_vision_route(),
        api_key="test-secret",
        net=_FailingVisionNetRuntime(),
    )

    with pytest.raises(VisionExecutionError) as exc_info:
        adapter.describe(_frame(frame_path, "frame-0001"))

    message = str(exc_info.value)
    assert "vision request failed for provider test-vlm: OSError" in message
    assert "network failed" not in message
    assert "Authorization" not in message
    assert "test-secret" not in message


def _adapter(
    runtime: _FakeNetRuntime | _FakeBatchNetRuntime,
) -> OpenAiCompatibleVisionAdapter:
    return OpenAiCompatibleVisionAdapter(
        route=_vision_route(),
        api_key="test-secret",
        net=runtime,
    )


def _vision_route() -> AiRoute:
    return AiRoute.configured_alias(
        task="vision_description",
        selected="configured-online",
        provider_id="test-vlm",
        reason="test route",
        base_url="https://example.invalid/v1/chat/completions",
        model="vision-test",
        cost="free",
        upload="required",
    )


def _frame(path: Path, frame_id: str) -> FrameAsset:
    return FrameAsset(
        id=frame_id,
        timestamp_seconds=1.0,
        path=path,
        source="cover",
        evidence=[Evidence(kind="probe", name="test-frame", weight=1.0)],
    )
