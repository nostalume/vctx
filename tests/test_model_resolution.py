from __future__ import annotations

from pathlib import Path

from vctx.transforms.model_resolution import (
    ModelCapability,
    ModelRef,
    OpenRouterModel,
    OpenRouterModelArchitecture,
    OpenRouterModelPricing,
    choose_openrouter_model,
    resolve_model_ref,
)


def test_openrouter_prefix_inferrs_endpoint_key_cost_and_upload() -> None:
    route = resolve_model_ref(
        "openrouter:nex-agi/nex-n2-pro:free",
        capability=ModelCapability.VISION_DESCRIPTION,
        env={"OPENROUTER_API_KEY": "present"},
    )

    assert route.ref == ModelRef(prefix="openrouter", value="nex-agi/nex-n2-pro:free")
    assert route.provider == "openrouter"
    assert route.model == "nex-agi/nex-n2-pro:free"
    assert route.base_url == "https://openrouter.ai/api/v1/chat/completions"
    assert route.api_key_env == "OPENROUTER_API_KEY"
    assert route.cost == "free"
    assert route.upload == "required"
    assert route.available is True


def test_local_prefix_inferrs_no_upload_and_resolves_config_relative_path(
    tmp_path: Path,
) -> None:
    route = resolve_model_ref(
        "local:models/qwen-vl.gguf",
        capability=ModelCapability.VISION_DESCRIPTION,
        env={},
        base_dir=tmp_path,
    )

    assert route.ref == ModelRef(prefix="local", value=str(tmp_path / "models" / "qwen-vl.gguf"))
    assert route.provider == "local"
    assert route.model == str(tmp_path / "models" / "qwen-vl.gguf")
    assert route.base_url is None
    assert route.api_key_env is None
    assert route.cost == "local"
    assert route.upload == "none"
    assert route.available is True


def test_auto_selects_free_openrouter_vlm_from_registry_when_key_exists() -> None:
    free_vlm = OpenRouterModel(
        id="nex-agi/nex-n2-pro:free",
        architecture=OpenRouterModelArchitecture(
            input_modalities=["text", "image"],
            output_modalities=["text"],
        ),
        pricing=OpenRouterModelPricing(prompt="0", completion="0"),
        context_length=262144,
    )
    paid_vlm = OpenRouterModel(
        id="paid/vision-model",
        architecture=OpenRouterModelArchitecture(
            input_modalities=["text", "image"],
            output_modalities=["text"],
        ),
        pricing=OpenRouterModelPricing(prompt="0.1", completion="0.1"),
        context_length=262144,
    )

    route = resolve_model_ref(
        "auto",
        capability=ModelCapability.VISION_DESCRIPTION,
        env={"OPENROUTER_API_KEY": "present"},
        openrouter_models=[paid_vlm, free_vlm],
    )

    assert route.provider == "openrouter"
    assert route.model == "nex-agi/nex-n2-pro:free"
    assert route.cost == "free"
    assert route.upload == "required"
    assert route.available is True


def test_auto_without_key_is_unavailable_for_remote_vlm() -> None:
    route = resolve_model_ref(
        "auto",
        capability=ModelCapability.VISION_DESCRIPTION,
        env={},
        openrouter_models=[
            OpenRouterModel(
                id="nex-agi/nex-n2-pro:free",
                architecture=OpenRouterModelArchitecture(
                    input_modalities=["text", "image"],
                    output_modalities=["text"],
                ),
                pricing=OpenRouterModelPricing(prompt="0", completion="0"),
            )
        ],
    )

    assert route.provider == "none"
    assert route.available is False
    assert "OPENROUTER_API_KEY" in route.reason


def test_choose_openrouter_model_filters_price_capability_and_ranks_best() -> None:
    text_free = OpenRouterModel(
        id="free/text-only:free",
        architecture=OpenRouterModelArchitecture(
            input_modalities=["text"],
            output_modalities=["text"],
        ),
        pricing=OpenRouterModelPricing(prompt="0", completion="0"),
    )
    unranked_vision_free = OpenRouterModel(
        id="free/vision:free",
        architecture=OpenRouterModelArchitecture(
            input_modalities=["text", "image"],
            output_modalities=["text"],
        ),
        pricing=OpenRouterModelPricing(prompt="0", completion="0"),
        context_length=999999,
    )
    audio_generation_free = OpenRouterModel(
        id="google/lyria-3-clip-preview",
        architecture=OpenRouterModelArchitecture(
            input_modalities=["text", "image"],
            output_modalities=["text", "audio"],
        ),
        pricing=OpenRouterModelPricing(prompt="0", completion="0"),
        context_length=1048576,
    )

    ranked_vision_free = OpenRouterModel(
        id="qwen/qwen2.5-vl-test:free",
        architecture=OpenRouterModelArchitecture(
            input_modalities=["text", "image"],
            output_modalities=["text"],
        ),
        pricing=OpenRouterModelPricing(prompt="0", completion="0"),
        context_length=1024,
    )

    selected = choose_openrouter_model(
        [text_free, unranked_vision_free, ranked_vision_free, audio_generation_free],
        capability=ModelCapability.VISION_DESCRIPTION,
    )

    assert selected == ranked_vision_free


def test_explicit_non_free_openrouter_ref_is_paid_and_available() -> None:
    route = resolve_model_ref(
        "openrouter:anthropic/claude-sonnet-4",
        capability=ModelCapability.ESSENTIAL_CASES,
        env={"OPENROUTER_API_KEY": "present"},
    )

    assert route.provider == "openrouter"
    assert route.model == "anthropic/claude-sonnet-4"
    assert route.cost == "paid"
    assert route.available is True
