from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import BaseModel

from vctx.ai import (
    AiBinding,
    AiClient,
    AiInstance,
    AiInstanceConfig,
    AiMessage,
    AiRoute,
    AiRuntimePool,
    Credential,
    CredentialRef,
    read_credential,
    select_ai_route,
)
from vctx.net import NetRequest, NetResponse


class Answer(BaseModel):
    answer: str


class FakeNet:
    def __init__(self, responses: list[NetResponse]) -> None:
        self.responses = responses
        self.requests: list[NetRequest] = []

    def request(self, request: NetRequest) -> NetResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _response(status: int, body: dict[str, object]) -> NetResponse:
    return NetResponse(
        url="https://ai.example/v1/chat/completions",
        status_code=status,
        body=json.dumps(body).encode(),
    )


def _instance(*, format: Literal["auto", "schema", "json", "prompt"] = "auto") -> AiInstance:
    return AiInstance(
        name="fixture",
        provider_id="fixture",
        base_url="https://ai.example/v1",
        model="configured-model",
        credential=CredentialRef("env:AI_TOKEN"),
        format=format,
    )


def _binding(instance: AiInstance, credential: str | None = "fixture-secret") -> AiBinding:
    route = AiRoute(
        task="evidence_plan",
        selected="configured-online",
        instance=instance,
        reason="test binding",
    )
    return AiBinding(route, Credential(credential) if credential is not None else None)


def test_structured_call_negotiates_format_and_returns_typed_receipt() -> None:
    net = FakeNet(
        [
            _response(400, {"error": {"message": "json_schema response_format unsupported"}}),
            _response(
                200,
                {
                    "model": "actual-model",
                    "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            ),
        ]
    )
    client = AiClient(
        _binding(_instance()),
        net=net,
        runtimes=AiRuntimePool(),
        clock=lambda: 1.0,
    )

    outcome = client.complete(
        task="evidence_plan",
        request_id="window-1",
        messages=[AiMessage(role="user", content="return an answer")],
        result=Answer,
    )

    assert outcome.kind == "ok" and outcome.value == Answer(answer="ok")
    assert outcome.receipt.format == "json" and outcome.receipt.actual_model == "actual-model"
    assert outcome.receipt.total_tokens == 5 and outcome.receipt.attempts == 2
    bodies = [json.loads(request.body or b"{}") for request in net.requests]
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[1]["response_format"]["type"] == "json_object"
    assert net.requests[0].connect_timeout_s == 10
    assert net.requests[0].timeout_s == 120
    assert "fixture-secret" not in outcome.model_dump_json()


def test_successful_invalid_output_never_falls_back_or_repairs() -> None:
    net = FakeNet(
        [
            _response(
                200,
                {
                    "model": "actual-model",
                    "choices": [{"message": {"content": "not-json"}}],
                },
            )
        ]
    )
    outcome = AiClient(
        _binding(_instance()),
        net=net,
        runtimes=AiRuntimePool(),
        clock=lambda: 1.0,
    ).complete(
        task="evidence_plan",
        request_id="window-1",
        messages=[AiMessage(role="user", content="answer")],
        result=Answer,
    )

    assert outcome.kind == "failed" and outcome.failure == "invalid_output"
    assert len(net.requests) == 1 and "not-json" not in outcome.model_dump_json()


def test_retryable_status_retries_once_with_stable_request_identity() -> None:
    net = FakeNet(
        [
            _response(
                200,
                {
                    "model": "actual-model",
                    "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                },
            ).model_copy(update={"attempts": 2}),
        ]
    )
    outcome = AiClient(
        _binding(_instance(format="json")),
        net=net,
        runtimes=AiRuntimePool(),
        clock=lambda: 1.0,
    ).complete(
        task="evidence_plan",
        request_id="stable-id",
        messages=[AiMessage(role="user", content="answer")],
        result=Answer,
    )

    assert outcome.kind == "ok" and outcome.receipt.attempts == 2
    assert net.requests[0].headers["Idempotency-Key"] == "stable-id"
    assert net.requests[0].retry.max_attempts == 2


class MemoryKeyring:
    priority = 1.0

    def __init__(self, value: str = "stored-secret") -> None:
        self.value = value
        self.reads: list[tuple[str, str]] = []

    def get_password(self, service: str, username: str) -> str | None:
        self.reads.append((service, username))
        return self.value

    def set_password(self, service: str, username: str, password: str) -> None:
        raise AssertionError("credential resolution must not write")

    def delete_password(self, service: str, username: str) -> None:
        raise AssertionError("credential resolution must not delete")


def test_credential_name_is_storage_only_and_does_not_select_request_policy() -> None:
    keyring = MemoryKeyring()
    instance = AiInstanceConfig(
        base_url="https://proxy.example/v1",
        model="proxy-model",
        credential=CredentialRef("keyring:openrouter"),
        format="json",
    ).admit("openrouter")
    net = FakeNet(
        [
            _response(
                200,
                {
                    "model": "proxy-model",
                    "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                },
            )
        ]
    )

    credential = read_credential(instance.credential, keyring=keyring)
    assert credential is not None
    outcome = AiClient(
        _binding(instance, credential.value),
        net=net,
        clock=lambda: 1.0,
    ).complete(
        task="evidence_plan",
        request_id="generic-openrouter-name",
        messages=[AiMessage(role="user", content="answer")],
        result=Answer,
    )

    assert outcome.kind == "ok"
    assert keyring.reads == [("vctx", "openrouter")]
    assert net.requests[0].url == "https://proxy.example/v1/chat/completions"
    assert "provider" not in json.loads(net.requests[0].body or b"{}")


def test_auto_openrouter_policy_is_independent_from_credential_precedence() -> None:
    keyring = MemoryKeyring()
    route = select_ai_route(
        task="evidence_plan",
        instance_name=None,
        auto=True,
        instances={},
        offline=False,
        auto_credential=CredentialRef("env:OPENROUTER_API_KEY"),
    )

    assert route is not None
    assert route.credential == "env:OPENROUTER_API_KEY"
    assert route.instance.request_policy == "openrouter-free-zdr"
    assert keyring.reads == []


def test_auto_keyring_credential_is_read_once_and_never_serialized() -> None:
    keyring = MemoryKeyring()
    credential = read_credential(CredentialRef("keyring:openrouter"), keyring=keyring)
    route = select_ai_route(
        task="evidence_plan",
        instance_name=None,
        auto=True,
        instances={},
        offline=False,
        auto_credential=CredentialRef("keyring:openrouter"),
    )

    assert route is not None and credential is not None
    assert keyring.reads == [("vctx", "openrouter")]
    assert "stored-secret" not in route.model_dump_json()
    assert "stored-secret" not in repr(AiBinding(route, credential))


@pytest.mark.parametrize(
    "reference",
    ["auth:openrouter", "keyring:vctx/openrouter", "env:BAD-NAME", "keyring:"],
)
def test_credential_grammar_rejects_semantic_or_nested_names(reference: str) -> None:
    with pytest.raises(ValueError, match="env:NAME or keyring:NAME"):
        AiInstanceConfig(
            base_url="https://ai.example/v1",
            model="model",
            credential=CredentialRef(reference),
        ).admit("instance")
