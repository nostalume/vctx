from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping

from vctx.config import CapabilityEnabled, CapabilityPolicy, VisionInstanceConfig
from vctx.transforms.ai_routes import AiRoute
from vctx.transforms.model_resolution import (
    ModelCapability,
    OpenRouterModel,
    ResolvedModelRoute,
    resolve_model_ref,
)
from vctx.transforms.visual_planning import (
    ActionRoute,
    VisualAction,
    baseline_visual_actions,
)

RAPIDOCR_PROVIDER_ID = "rapidocr"


def discover_visual_actions(
    policy: CapabilityPolicy,
    *,
    vision_instance_configs: dict[str, VisionInstanceConfig] | None = None,
    ai_routes: list[AiRoute] | None = None,
    env: Mapping[str, str] | None = None,
    openrouter_models: list[OpenRouterModel] | None = None,
) -> list[VisualAction]:
    actions = baseline_visual_actions()
    if policy.enabled == CapabilityEnabled.FALSE:
        return actions
    if policy.route in {"auto", "default", "local"} and rapidocr_available():
        actions.append(VisualAction.ocr(provider_id=RAPIDOCR_PROVIDER_ID))
    selected_instance_config = _select_vision_instance_config(policy, vision_instance_configs or {})
    if selected_instance_config is not None:
        provider_id, instance_config = selected_instance_config
        route: ActionRoute = "free-online" if policy.route == "free-online" else "configured-online"
        actions.append(
            VisualAction.describe(
                AiRoute.configured_alias(
                    task="vision_description",
                    selected=route,
                    provider_id=provider_id,
                    model=instance_config.model,
                    base_url=instance_config.base_url,
                    api_key_env=instance_config.api_key_env,
                    cost="unknown",
                    upload="required",
                    reason="resolved from configured vision provider",
                )
            )
        )
        return actions

    ai_route = _visual_ai_route(policy, ai_routes or [])
    if ai_route is not None:
        actions.append(VisualAction.describe(ai_route))
        return actions

    resolved = _resolved_visual_model(policy, env=env, openrouter_models=openrouter_models)
    if resolved is not None:
        route: ActionRoute = "free-online" if resolved.cost == "free" else "configured-online"
        actions.append(
            VisualAction.describe(
                AiRoute.from_model_route(
                    task="vision_description",
                    selected=route,
                    model_route=resolved,
                )
            )
        )
    return actions


def _select_vision_instance_config(
    policy: CapabilityPolicy,
    instance_configs: dict[str, VisionInstanceConfig],
) -> tuple[str, VisionInstanceConfig] | None:
    if not policy.allow_network or not policy.allow_upload:
        return None
    if policy.route not in {"auto", "default", "free-online", "configured-online"}:
        return None
    selected_instance = policy.instance or policy.preferred_provider
    if selected_instance is not None:
        instance_config = instance_configs.get(selected_instance)
        if instance_config is not None and _vision_instance_config_allowed(policy, instance_config):
            return selected_instance, instance_config
        return None
    for provider_id, instance_config in instance_configs.items():
        if _vision_instance_config_allowed(policy, instance_config):
            return provider_id, instance_config
    return None


def _vision_instance_config_allowed(
    policy: CapabilityPolicy,
    instance_config: VisionInstanceConfig,
) -> bool:
    if instance_config.type != "openai-compatible-vision":
        return False
    return instance_config.base_url is not None and instance_config.model is not None


def _visual_ai_route(policy: CapabilityPolicy, routes: list[AiRoute]) -> AiRoute | None:
    if not policy.allow_network or not policy.allow_upload:
        return None
    if policy.route not in {"auto", "default", "free-online", "configured-online"}:
        return None
    for route in routes:
        if _ai_route_allowed(policy, route):
            return route
    return None


def _ai_route_allowed(policy: CapabilityPolicy, route: AiRoute) -> bool:
    if route.task != "vision_description" or not route.available:
        return False
    if route.upload != "required":
        return False
    if policy.route == "free-online" and route.cost != "free":
        return False
    return route.selected in {"free-online", "configured-online"}


def _resolved_visual_model(
    policy: CapabilityPolicy,
    *,
    env: Mapping[str, str] | None,
    openrouter_models: list[OpenRouterModel] | None,
) -> ResolvedModelRoute | None:
    if not policy.allow_network or not policy.allow_upload:
        return None
    if policy.route not in {"auto", "default", "free-online", "configured-online"}:
        return None
    if policy.model is None and openrouter_models is None:
        return None
    resolved = resolve_model_ref(
        policy.model,
        capability=ModelCapability.VISION_DESCRIPTION,
        env=env or os.environ,
        openrouter_models=openrouter_models,
    )
    if not resolved.available:
        return None
    if resolved.provider != "openrouter":
        return None
    if policy.route == "free-online" and resolved.cost != "free":
        return None
    if resolved.upload != "required":
        return None
    return resolved


def rapidocr_available() -> bool:
    return importlib.util.find_spec("rapidocr") is not None
