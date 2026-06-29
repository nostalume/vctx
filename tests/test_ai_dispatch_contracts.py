from __future__ import annotations

from typing import get_args

from vctx.config import CapabilityEnabled, CapabilityPolicy
from vctx.transforms.ai_routes import (
    AiRoute,
    AiTaskKind,
    model_capability_for_ai_task,
    resolve_openrouter_ai_route,
)
from vctx.transforms.model_resolution import (
    ModelCapability,
    ModelRef,
    OpenRouterModel,
    OpenRouterModelArchitecture,
    OpenRouterModelPricing,
    ResolvedModelRoute,
)


def test_ai_route_preserves_model_route_fields_for_vision_description() -> None:
    model_route = ResolvedModelRoute(
        ref=ModelRef(prefix="openrouter", value="nex-agi/nex-n2-pro:free"),
        provider="openrouter",
        model="nex-agi/nex-n2-pro:free",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="OPENROUTER_API_KEY",
        cost="free",
        upload="required",
        available=True,
        reason="resolved from OpenRouter model reference",
    )

    ai_route = AiRoute.from_model_route(
        task="vision_description",
        selected="free-online",
        model_route=model_route,
    )

    assert ai_route == AiRoute(
        task="vision_description",
        selected="free-online",
        provider="openrouter",
        provider_id="openrouter",
        model="nex-agi/nex-n2-pro:free",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="OPENROUTER_API_KEY",
        cost="free",
        upload="required",
        available=True,
        reason="resolved from OpenRouter model reference",
    )


def test_ai_route_keeps_unavailable_reason_and_no_provider_id_for_disabled_route() -> None:
    model_route = ResolvedModelRoute(
        ref=ModelRef(prefix="none"),
        provider="none",
        available=False,
        reason="model-mediated transform disabled",
    )

    ai_route = AiRoute.from_model_route(
        task="knowledge_flow_extraction",
        selected="skipped",
        model_route=model_route,
    )

    assert ai_route.task == "knowledge_flow_extraction"
    assert ai_route.selected == "skipped"
    assert ai_route.provider == "none"
    assert ai_route.provider_id is None
    assert ai_route.available is False
    assert ai_route.reason == "model-mediated transform disabled"


def test_ai_route_constructors_encode_local_and_alias_defaults() -> None:
    local = AiRoute.local(
        task="asr_transcription",
        provider_id="faster-whisper",
        model="base",
        reason="default local ASR route available",
    )
    alias = AiRoute.configured_alias(
        task="vision_description",
        selected="configured-online",
        provider_id="test-vlm",
        model="vision-test",
        base_url="https://example.invalid/v1/chat/completions",
        api_key_env="TEST_VLM_API_KEY",
        cost="paid",
        upload="required",
        reason="resolved from configured vision provider",
    )

    assert local.provider == "local"
    assert local.selected == "local"
    assert local.provider_id == "faster-whisper"
    assert local.cost == "local"
    assert local.upload == "none"
    assert local.available is True
    assert alias.provider == "alias"
    assert alias.provider_id == "test-vlm"
    assert alias.available is True


def test_ai_task_maps_to_existing_model_capability_without_provider_knowledge() -> None:
    assert model_capability_for_ai_task("vision_description") is ModelCapability.VISION_DESCRIPTION
    assert (
        model_capability_for_ai_task("essential_case_extraction")
        is ModelCapability.ESSENTIAL_CASES
    )
    assert (
        model_capability_for_ai_task("knowledge_flow_extraction")
        is ModelCapability.ESSENTIAL_CASES
    )
    assert model_capability_for_ai_task("asr_transcription") is ModelCapability.ASR
    assert model_capability_for_ai_task("ocr_text") is ModelCapability.OCR


def test_ai_task_kind_names_are_behavior_not_provider_names() -> None:
    task_values = set(get_args(AiTaskKind.__value__))

    assert task_values == {
        "asr_transcription",
        "ocr_text",
        "vision_description",
        "essential_case_extraction",
        "knowledge_flow_extraction",
    }
    assert "openrouter" not in task_values
    assert "free-online" not in task_values


def test_resolve_openrouter_ai_route_returns_free_vision_route_for_auto_model(
    tmp_path,
) -> None:
    route = resolve_openrouter_ai_route(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            route="free-online",
            model="auto",
            allow_network=True,
            allow_upload=True,
        ),
        task="vision_description",
        capability=ModelCapability.VISION_DESCRIPTION,
        cache_root=tmp_path,
        env={"OPENROUTER_API_KEY": "present"},
        offline=True,
        openrouter_models=[_openrouter_model("nex-agi/nex-n2-pro:free", vision=True)],
    )

    assert route is not None
    assert route.task == "vision_description"
    assert route.selected == "free-online"
    assert route.model == "nex-agi/nex-n2-pro:free"
    assert route.cost == "free"


def test_resolve_openrouter_ai_route_rejects_paid_model_for_free_online(
    tmp_path,
) -> None:
    route = resolve_openrouter_ai_route(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            route="free-online",
            model="openrouter:anthropic/claude-sonnet-4",
            allow_network=True,
            allow_upload=True,
        ),
        task="vision_description",
        capability=ModelCapability.VISION_DESCRIPTION,
        cache_root=tmp_path,
        env={"OPENROUTER_API_KEY": "present"},
        offline=True,
    )

    assert route is None


def test_resolve_openrouter_ai_route_returns_configured_text_route_for_paid_model(
    tmp_path,
) -> None:
    route = resolve_openrouter_ai_route(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            route="configured-online",
            model="openrouter:anthropic/claude-sonnet-4",
            allow_network=True,
            allow_upload=True,
        ),
        task="knowledge_flow_extraction",
        capability=ModelCapability.ESSENTIAL_CASES,
        cache_root=tmp_path,
        env={"OPENROUTER_API_KEY": "present"},
        offline=True,
    )

    assert route is not None
    assert route.task == "knowledge_flow_extraction"
    assert route.selected == "configured-online"
    assert route.model == "anthropic/claude-sonnet-4"
    assert route.cost == "paid"


def _openrouter_model(model_id: str, *, vision: bool) -> OpenRouterModel:
    input_modalities = ["text", "image"] if vision else ["text"]
    return OpenRouterModel(
        id=model_id,
        architecture=OpenRouterModelArchitecture(
            input_modalities=input_modalities,
            output_modalities=["text"],
        ),
        pricing=OpenRouterModelPricing(prompt="0", completion="0"),
    )
