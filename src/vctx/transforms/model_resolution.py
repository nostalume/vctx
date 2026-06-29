from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from vctx.net import NetRequest, NetRuntime, UrllibNetRuntime

ModelProvider = Literal["none", "openrouter", "local", "hf", "alias"]
ModelCost = Literal["free", "paid", "local", "unknown"]
ModelUpload = Literal["none", "required"]

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CACHE_KEY = "openrouter/models.json"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


class ModelCapability(StrEnum):
    ESSENTIAL_CASES = "essential_cases"
    VISION_DESCRIPTION = "vision_description"
    ASR = "asr"
    OCR = "ocr"


class ModelRef(BaseModel):
    prefix: Literal["auto", "none", "openrouter", "local", "hf", "alias"]
    value: str | None = None


class ResolvedModelRoute(BaseModel):
    ref: ModelRef
    provider: ModelProvider
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    cost: ModelCost = "unknown"
    upload: ModelUpload = "none"
    available: bool = False
    reason: str = ""


class OpenRouterModelArchitecture(BaseModel):
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)


class OpenRouterModelPricing(BaseModel):
    prompt: str = ""
    completion: str = ""


class OpenRouterModel(BaseModel):
    id: str
    architecture: OpenRouterModelArchitecture = Field(
        default_factory=OpenRouterModelArchitecture
    )
    pricing: OpenRouterModelPricing = Field(default_factory=OpenRouterModelPricing)
    context_length: int | None = None


class OpenRouterRegistryPayload(BaseModel):
    data: list[OpenRouterModel] = []


def parse_model_ref(value: str | None) -> ModelRef:
    if value is None or value == "auto":
        return ModelRef(prefix="auto")
    if value == "none":
        return ModelRef(prefix="none")
    prefix, separator, rest = value.partition(":")
    if separator:
        if prefix == "openrouter":
            return ModelRef(prefix="openrouter", value=rest)
        if prefix == "local":
            return ModelRef(prefix="local", value=rest)
        if prefix == "hf":
            return ModelRef(prefix="hf", value=rest)
        if prefix == "alias":
            return ModelRef(prefix="alias", value=rest)
    if _looks_like_path_value(value):
        return ModelRef(prefix="local", value=value)
    return ModelRef(prefix="alias", value=value)


def resolve_model_ref(
    value: str | None,
    *,
    capability: ModelCapability,
    env: Mapping[str, str],
    base_dir: Path | None = None,
    openrouter_models: list[OpenRouterModel] | None = None,
) -> ResolvedModelRoute:
    ref = parse_model_ref(value)
    if ref.prefix == "none":
        return ResolvedModelRoute(
            ref=ref,
            provider="none",
            available=False,
            reason="model-mediated transform disabled",
        )
    if ref.prefix == "auto":
        return _resolve_auto(
            capability=capability,
            env=env,
            openrouter_models=openrouter_models or [],
        )
    if ref.prefix == "openrouter":
        return _resolve_openrouter(ref, capability=capability, env=env)
    if ref.prefix == "local":
        return _resolve_local(ref, base_dir=base_dir)
    if ref.prefix == "hf":
        return ResolvedModelRoute(
            ref=ref,
            provider="hf",
            model=ref.value,
            cost="local",
            upload="none",
            available=True,
            reason="Hugging Face model id uses managed local cache when runtime exists",
        )
    return ResolvedModelRoute(
        ref=ref,
        provider="alias",
        model=ref.value,
        available=False,
        reason="model alias resolution is not implemented",
    )


OPENROUTER_MODEL_PREFERENCES: dict[ModelCapability, tuple[str, ...]] = {
    ModelCapability.VISION_DESCRIPTION: (
        "qwen/qwen2.5-vl",
        "qwen/qwen-vl",
        "google/gemini",
        "meta-llama/llama-3.2",
        "mistralai/pixtral",
        "nex-agi/nex-n2-pro",
    ),
    ModelCapability.ESSENTIAL_CASES: (
        "qwen/qwen3",
        "qwen/qwen2.5",
        "google/gemini",
        "meta-llama/llama-3.3",
        "meta-llama/llama-3.1",
        "mistralai/mistral",
    ),
}


def choose_openrouter_model(
    models: list[OpenRouterModel], *, capability: ModelCapability
) -> OpenRouterModel | None:
    candidates = [
        model
        for model in models
        if _is_free(model) and _supports_capability(model, capability)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda model: _openrouter_rank(model, capability))[0]


def _resolve_auto(
    *,
    capability: ModelCapability,
    env: Mapping[str, str],
    openrouter_models: list[OpenRouterModel],
) -> ResolvedModelRoute:
    if OPENROUTER_API_KEY_ENV not in env:
        return ResolvedModelRoute(
            ref=ModelRef(prefix="auto"),
            provider="none",
            available=False,
            reason=(
                f"{OPENROUTER_API_KEY_ENV} is not set; "
                "using deterministic fallback when available"
            ),
        )
    selected = choose_openrouter_model(openrouter_models, capability=capability)
    if selected is None:
        return ResolvedModelRoute(
            ref=ModelRef(prefix="auto"),
            provider="none",
            available=False,
            reason="no free OpenRouter model supports the requested capability",
        )
    return _openrouter_route(ModelRef(prefix="openrouter", value=selected.id), selected.id)


def _resolve_openrouter(
    ref: ModelRef, *, capability: ModelCapability, env: Mapping[str, str]
) -> ResolvedModelRoute:
    if ref.value is None or ref.value == "":
        return ResolvedModelRoute(
            ref=ref,
            provider="openrouter",
            available=False,
            reason="openrouter model reference is missing a model id",
        )
    if OPENROUTER_API_KEY_ENV not in env:
        return ResolvedModelRoute(
            ref=ref,
            provider="openrouter",
            model=ref.value,
            base_url=OPENROUTER_CHAT_COMPLETIONS_URL,
            api_key_env=OPENROUTER_API_KEY_ENV,
            cost=_openrouter_cost_from_id(ref.value),
            upload=_upload_for_capability(capability),
            available=False,
            reason=f"{OPENROUTER_API_KEY_ENV} is not set",
        )
    return _openrouter_route(ref, ref.value, upload=_upload_for_capability(capability))


def _openrouter_route(
    ref: ModelRef, model: str, *, upload: ModelUpload = "required"
) -> ResolvedModelRoute:
    return ResolvedModelRoute(
        ref=ref,
        provider="openrouter",
        model=model,
        base_url=OPENROUTER_CHAT_COMPLETIONS_URL,
        api_key_env=OPENROUTER_API_KEY_ENV,
        cost=_openrouter_cost_from_id(model),
        upload=upload,
        available=True,
        reason="resolved from OpenRouter model reference",
    )


def _resolve_local(ref: ModelRef, *, base_dir: Path | None) -> ResolvedModelRoute:
    model = ref.value or ""
    path = Path(model)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    normalized = str(path) if model else model
    return ResolvedModelRoute(
        ref=ModelRef(prefix="local", value=normalized),
        provider="local",
        model=normalized,
        cost="local",
        upload="none",
        available=True,
        reason="resolved from local model reference",
    )


def _is_free(model: OpenRouterModel) -> bool:
    return model.id.endswith(":free") or (
        _is_zero_price(model.pricing.prompt) and _is_zero_price(model.pricing.completion)
    )


def _is_zero_price(value: str) -> bool:
    try:
        return float(value) == 0.0
    except ValueError:
        return False


def _supports_capability(model: OpenRouterModel, capability: ModelCapability) -> bool:
    inputs = set(model.architecture.input_modalities)
    outputs = set(model.architecture.output_modalities)
    if "text" not in outputs:
        return False
    if capability == ModelCapability.VISION_DESCRIPTION:
        return {"text", "image"}.issubset(inputs) and outputs == {"text"}
    if capability == ModelCapability.ESSENTIAL_CASES:
        return "text" in inputs
    return False


def _upload_for_capability(capability: ModelCapability) -> ModelUpload:
    if capability in {ModelCapability.VISION_DESCRIPTION, ModelCapability.ASR}:
        return "required"
    if capability == ModelCapability.ESSENTIAL_CASES:
        return "required"
    return "none"


def _openrouter_rank(
    model: OpenRouterModel,
    capability: ModelCapability,
) -> tuple[int, int, str]:
    return (
        _preference_rank(model, capability),
        -(model.context_length or 0),
        model.id,
    )


def _preference_rank(model: OpenRouterModel, capability: ModelCapability) -> int:
    model_id = model.id.lower()
    for rank, prefix in enumerate(OPENROUTER_MODEL_PREFERENCES.get(capability, ()), start=0):
        if model_id.startswith(prefix):
            return rank
    return len(OPENROUTER_MODEL_PREFERENCES.get(capability, ()))


def _openrouter_cost_from_id(model: str) -> ModelCost:
    return "free" if model.endswith(":free") else "paid"


def _looks_like_path_value(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or value.startswith(".") or any(
        separator in value for separator in ("/", "\\")
    )


def load_openrouter_models(
    cache_root: Path, *, offline: bool, net: NetRuntime | None = None
) -> list[OpenRouterModel]:
    cache_path = cache_root / OPENROUTER_CACHE_KEY
    cached = _read_cached_openrouter_payload(cache_path)
    if cached is not None:
        return cached.data
    if offline:
        return []
    payload = _fetch_openrouter_payload(net or UrllibNetRuntime())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return payload.data


def _read_cached_openrouter_payload(cache_path: Path) -> OpenRouterRegistryPayload | None:
    if not cache_path.exists():
        return None
    return _openrouter_payload_from_json(cache_path.read_text(encoding="utf-8"))


def _fetch_openrouter_payload(net: NetRuntime) -> OpenRouterRegistryPayload:
    response = net.request(
        NetRequest(
            method="GET",
            url=OPENROUTER_MODELS_URL,
            headers={"Accept": "application/json", "User-Agent": "vctx"},
            timeout_s=20,
            purpose="model_registry",
        )
    )
    if response.status_code < 200 or response.status_code >= 300:
        return OpenRouterRegistryPayload()
    payload = _openrouter_payload_from_json(response.body.decode("utf-8"))
    return payload or OpenRouterRegistryPayload()


def _openrouter_payload_from_json(raw_payload: str) -> OpenRouterRegistryPayload | None:
    try:
        return OpenRouterRegistryPayload.model_validate_json(raw_payload)
    except ValidationError:
        return None
