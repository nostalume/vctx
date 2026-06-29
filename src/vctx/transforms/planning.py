from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from vctx.config import CapabilityEnabled, CapabilityPolicy
from vctx.models.manifest import CapabilityName, SelectedRoute, TransformEvidence
from vctx.transforms.ai_routes import AiRoute

ProviderCostMode = Literal["free", "paid", "local", "unknown"]


class RoutePlan(BaseModel):
    capability: CapabilityName
    selected: SelectedRoute
    provider_id: str | None = None
    model_id: str | None = None
    reason: str
    requirements: list[str] = []
    warnings: list[str] = []
    ai_route: AiRoute | None = None
    evidence_seed: TransformEvidence


class SourceState(BaseModel):
    has_transcript: bool = False
    has_media: bool = False


class TransformEnvironment(BaseModel):
    installed_asr: bool = False
    network_available: bool = True
    offline: bool = False
    configured_asr: bool = False
    configured_asr_provider_id: str | None = None
    configured_asr_model_id: str | None = None
    configured_asr_cost_mode: ProviderCostMode = "unknown"
    free_online_asr: bool = False


def _plan(
    *,
    capability: CapabilityName,
    selected: SelectedRoute,
    reason: str,
    provider_id: str | None = None,
    model_id: str | None = None,
    requirements: list[str] | None = None,
    warnings: list[str] | None = None,
    requires_user_config: bool = False,
    uploaded: bool = False,
    cost_may_apply: bool = False,
    deterministic: bool = False,
    ai_route: AiRoute | None = None,
) -> RoutePlan:
    warnings = warnings or []
    return RoutePlan(
        capability=capability,
        selected=selected,
        provider_id=provider_id,
        model_id=model_id,
        reason=reason,
        requirements=requirements or [],
        warnings=warnings,
        ai_route=ai_route,
        evidence_seed=TransformEvidence(
            capability=capability,
            selected_route=selected,
            provider_id=provider_id,
            model_id=model_id,
            requires_user_config=requires_user_config,
            uploaded=uploaded,
            cost_may_apply=cost_may_apply,
            deterministic=deterministic,
            reason=reason,
            warnings=warnings,
        ),
    )


def _disabled(policy: CapabilityPolicy) -> bool:
    return policy.enabled == CapabilityEnabled.FALSE or policy.route == "disabled"


def _online_allowed(policy: CapabilityPolicy, environment: TransformEnvironment) -> bool:
    return policy.allow_network and environment.network_available and not environment.offline


def plan_asr(
    policy: CapabilityPolicy,
    environment: TransformEnvironment,
    source_state: SourceState,
) -> RoutePlan:
    if source_state.has_transcript:
        return _plan(
            capability="asr",
            selected="skipped",
            reason="transcript already available",
            deterministic=True,
        )
    if _disabled(policy):
        return _plan(capability="asr", selected="skipped", reason="ASR disabled by policy")
    if not source_state.has_media:
        return _plan(
            capability="asr",
            selected="unavailable",
            reason="No transcript found and no media asset is available for ASR.",
            requirements=["media asset"],
        )
    if environment.installed_asr:
        model_id = policy.model or environment.configured_asr_model_id or "base"
        return _plan(
            capability="asr",
            selected="local",
            provider_id="faster-whisper",
            model_id=model_id,
            reason="default local ASR route available",
            ai_route=AiRoute.local(
                task="asr_transcription",
                provider_id="faster-whisper",
                model=model_id,
                reason="default local ASR route available",
            ),
        )
    if _online_allowed(policy, environment) and environment.free_online_asr:
        return _plan(
            capability="asr",
            selected="free-online",
            provider_id="free-online-asr",
            reason="free zero-config online ASR route available",
            uploaded=True,
            ai_route=AiRoute.configured_alias(
                task="asr_transcription",
                selected="free-online",
                provider_id="free-online-asr",
                cost="free",
                upload="required",
                reason="free zero-config online ASR route available",
            ),
        )
    if _online_allowed(policy, environment) and policy.allow_upload and environment.configured_asr:
        provider_id = (
            environment.configured_asr_provider_id
            or policy.instance
            or policy.preferred_provider
            or "default-asr"
        )
        model_id = policy.model or environment.configured_asr_model_id
        reason = "configured online ASR route available"
        return _plan(
            capability="asr",
            selected="configured-online",
            provider_id=provider_id,
            model_id=model_id,
            reason=reason,
            requires_user_config=True,
            uploaded=True,
            cost_may_apply=environment.configured_asr_cost_mode == "paid",
            ai_route=AiRoute.configured_alias(
                task="asr_transcription",
                selected="configured-online",
                provider_id=provider_id,
                model=model_id,
                cost=environment.configured_asr_cost_mode,
                upload="required",
                reason=reason,
            ),
        )
    return _plan(
        capability="asr",
        selected="unavailable",
        reason=(
            "No transcript found, ASR extra not installed, no free-online ASR route registered, "
            "and no configured ASR provider."
        ),
        requirements=["install ASR extra", "configure ASR provider", "provide transcript file"],
    )
