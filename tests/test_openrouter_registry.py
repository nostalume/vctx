from __future__ import annotations

import json
from pathlib import Path

from vctx.net import NetRequest, NetResponse
from vctx.transforms.model_resolution import load_openrouter_models


class _FakeNetRuntime:
    def __init__(self, response: NetResponse) -> None:
        self.response = response
        self.requests: list[NetRequest] = []

    def request(self, request: NetRequest) -> NetResponse:
        self.requests.append(request)
        return self.response


class _FailingNetRuntime:
    def request(self, request: NetRequest) -> NetResponse:
        del request
        raise AssertionError("network should not be used when cache exists")


def _registry_response(payload: str) -> NetResponse:
    return NetResponse(
        url="https://openrouter.ai/api/v1/models",
        status_code=200,
        headers={"content-type": "application/json"},
        body=payload.encode("utf-8"),
    )


def test_openrouter_registry_uses_pydantic_owned_payload() -> None:
    source = Path("src/vctx/transforms/model_resolution.py").read_text(encoding="utf-8")

    assert "dict[str, Any]" not in source
    assert "json.loads" not in source
    assert "json.dumps" not in source
    assert "model_validate_json" in source
    assert "model_dump_json" in source


def test_load_openrouter_models_fetches_through_injected_net_runtime_and_writes_cache(
    tmp_path: Path,
) -> None:
    runtime = _FakeNetRuntime(
        _registry_response(
            json.dumps(
                {
                    "data": [
                        {
                            "id": "nex-agi/nex-n2-pro:free",
                            "architecture": {
                                "input_modalities": ["text", "image"],
                                "output_modalities": ["text"],
                            },
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                }
            )
        )
    )

    models = load_openrouter_models(tmp_path, offline=False, net=runtime)

    assert [model.id for model in models] == ["nex-agi/nex-n2-pro:free"]
    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.method == "GET"
    assert request.url == "https://openrouter.ai/api/v1/models"
    assert request.purpose == "model_registry"
    assert request.timeout_s == 20
    assert request.headers == {"Accept": "application/json", "User-Agent": "vctx"}
    cached = json.loads((tmp_path / "openrouter" / "models.json").read_text())
    assert cached["data"][0]["id"] == "nex-agi/nex-n2-pro:free"


def test_load_openrouter_models_uses_cached_registry_without_network(tmp_path: Path) -> None:
    cache_file = tmp_path / "openrouter" / "models.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "cached/vision:free",
                        "architecture": {
                            "input_modalities": ["text", "image"],
                            "output_modalities": ["text"],
                        },
                        "pricing": {"prompt": "0", "completion": "0"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    models = load_openrouter_models(tmp_path, offline=True, net=_FailingNetRuntime())

    assert [model.id for model in models] == ["cached/vision:free"]


def test_load_openrouter_models_returns_empty_on_offline_cache_miss(tmp_path: Path) -> None:
    assert load_openrouter_models(tmp_path, offline=True, net=_FailingNetRuntime()) == []


def test_load_openrouter_models_returns_empty_on_invalid_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "openrouter" / "models.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text('{"data": "not a model list"}', encoding="utf-8")

    assert load_openrouter_models(tmp_path, offline=True, net=_FailingNetRuntime()) == []
