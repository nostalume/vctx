from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self, assert_never

from pydantic import BaseModel, Field

from vctx.config import CapabilityEnabled, CapabilityPolicy
from vctx.models.manifest import CapabilityName, TransformEvidence
from vctx.transforms.model_resolution import (
    ModelCapability,
    ModelCost,
    ModelProvider,
    ModelUpload,
    OpenRouterModel,
    ResolvedModelRoute,
    load_openrouter_models,
    resolve_model_ref,
)

type AiTaskKind = Literal[
    "asr_transcription",
    "ocr_text",
    "vision_description",
    "essential_case_extraction",
    "knowledge_flow_extraction",
]
type AiSelectedRoute = Literal[
    "skipped",
    "deterministic",
    "local",
    "free-online",
    "configured-online",
    "unavailable",
]


class AiRoute(BaseModel):
    task: AiTaskKind
    selected: AiSelectedRoute
    provider: ModelProvider
    provider_id: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    cost: ModelCost = "unknown"
    upload: ModelUpload = "none"
    available: bool = False
    reason: str
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_model_route(
        cls,
        *,
        task: AiTaskKind,
        selected: AiSelectedRoute,
        model_route: ResolvedModelRoute,
        warnings: list[str] | None = None,
    ) -> Self:
        return cls(
            task=task,
            selected=selected,
            provider=model_route.provider,
            provider_id=_provider_id(model_route),
            model=model_route.model,
            base_url=model_route.base_url,
            api_key_env=model_route.api_key_env,
            cost=model_route.cost,
            upload=model_route.upload,
            available=model_route.available,
            reason=model_route.reason,
            warnings=warnings or [],
        )

    @classmethod
    def configured_alias(
        cls,
        *,
        task: AiTaskKind,
        selected: AiSelectedRoute,
        provider_id: str,
        reason: str,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        cost: ModelCost = "unknown",
        upload: ModelUpload = "none",
        warnings: list[str] | None = None,
    ) -> Self:
        return cls(
            task=task,
            selected=selected,
            provider="alias",
            provider_id=provider_id,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            cost=cost,
            upload=upload,
            available=True,
            reason=reason,
            warnings=warnings or [],
        )

    @classmethod
    def local(
        cls,
        *,
        task: AiTaskKind,
        provider_id: str,
        reason: str,
        model: str | None = None,
        warnings: list[str] | None = None,
    ) -> Self:
        return cls(
            task=task,
            selected="local",
            provider="local",
            provider_id=provider_id,
            model=model,
            cost="local",
            upload="none",
            available=True,
            reason=reason,
            warnings=warnings or [],
        )

    def transform_evidence(self, capability: CapabilityName) -> TransformEvidence:
        return TransformEvidence(
            capability=capability,
            selected_route=self.selected,
            provider_id=self.provider_id,
            model_id=self.model,
            requires_user_config=True,
            uploaded=self.upload == "required",
            cost_may_apply=self.cost == "paid",
            deterministic=False,
            reason=self.reason,
            warnings=self.warnings,
        )

    def detail(self) -> str:
        model = self.model or "unknown-model"
        provider = self.provider_id or self.provider
        return f"{provider}: {model}"


class AiResult[T](BaseModel):
    value: T
    route: AiRoute
    warnings: list[str] = Field(default_factory=list)


def model_capability_for_ai_task(task: AiTaskKind) -> ModelCapability:
    if task == "asr_transcription":
        return ModelCapability.ASR
    if task == "ocr_text":
        return ModelCapability.OCR
    if task == "vision_description":
        return ModelCapability.VISION_DESCRIPTION
    if task == "essential_case_extraction":
        return ModelCapability.ESSENTIAL_CASES
    if task == "knowledge_flow_extraction":
        return ModelCapability.ESSENTIAL_CASES
    assert_never(task)


def resolve_openrouter_ai_route(
    policy: CapabilityPolicy,
    *,
    task: AiTaskKind,
    capability: ModelCapability,
    cache_root: Path,
    env: Mapping[str, str],
    offline: bool,
    openrouter_models: list[OpenRouterModel] | None = None,
) -> AiRoute | None:
    if policy.enabled != CapabilityEnabled.TRUE:
        return None
    if not policy.allow_network or not policy.allow_upload:
        return None
    if policy.route not in {"auto", "default", "free-online", "configured-online"}:
        return None
    models = openrouter_models
    if policy.model in {None, "auto"} and models is None:
        models = load_openrouter_models(cache_root, offline=offline)
    model_route = resolve_model_ref(
        policy.model,
        capability=capability,
        env=env,
        openrouter_models=models,
    )
    if not model_route.available or model_route.provider != "openrouter":
        return None
    if policy.route == "free-online" and model_route.cost != "free":
        return None
    if model_route.upload != "required":
        return None
    selected: AiSelectedRoute = "free-online" if model_route.cost == "free" else "configured-online"
    return AiRoute.from_model_route(task=task, selected=selected, model_route=model_route)


def _provider_id(model_route: ResolvedModelRoute) -> str | None:
    if model_route.provider == "none":
        return None
    return model_route.provider
