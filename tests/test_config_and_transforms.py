from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from vctx.cli import app
from vctx.config import (
    CapabilityEnabled,
    PrepareRequest,
    WorkflowProfile,
    resolve_config,
)
from vctx.transforms.planning import (
    SourceState,
    TransformEnvironment,
    plan_asr,
)


def test_resolve_config_missing_fields_become_default_auto(tmp_path: Path) -> None:
    request = PrepareRequest(input="lecture.srt", out_dir=tmp_path / "out")

    resolved = resolve_config(request)

    assert resolved.runtime.workflow == WorkflowProfile.DEFAULT
    assert resolved.runtime.offline is False
    assert resolved.transforms.asr.enabled == CapabilityEnabled.AUTO
    assert resolved.transforms.asr.route == "auto"
    assert resolved.transforms.asr.allow_network is True
    assert resolved.transforms.asr.allow_upload is True
    assert resolved.transforms.visual_context.enabled == CapabilityEnabled.AUTO
    assert resolved.output.chunk_max_chars == 6000


def test_offline_request_disables_network_routes(tmp_path: Path) -> None:
    request = PrepareRequest(input="lecture.mp4", out_dir=tmp_path / "out", offline=True)

    resolved = resolve_config(request)

    assert resolved.runtime.offline is True
    assert resolved.transforms.asr.allow_network is False
    assert resolved.transforms.asr.allow_upload is False
    assert resolved.transforms.visual_context.allow_network is False


def test_plan_asr_missing_everything_returns_actionable_unavailable(tmp_path: Path) -> None:
    request = PrepareRequest(input="lecture.mp4", out_dir=tmp_path / "out")
    resolved = resolve_config(request)
    environment = TransformEnvironment()
    source = SourceState(has_transcript=False, has_media=True)

    plan = plan_asr(resolved.transforms.asr, environment, source)

    assert plan.selected == "unavailable"
    assert "ASR extra not installed" in plan.reason
    assert "no configured ASR provider" in plan.reason
    assert plan.evidence_seed.requires_user_config is False


def test_plan_asr_skips_when_transcript_exists(tmp_path: Path) -> None:
    request = PrepareRequest(input="lecture.srt", out_dir=tmp_path / "out")
    resolved = resolve_config(request)
    environment = TransformEnvironment(installed_asr=True)
    source = SourceState(has_transcript=True, has_media=False)

    plan = plan_asr(resolved.transforms.asr, environment, source)

    assert plan.selected == "skipped"
    assert plan.reason == "transcript already available"


def test_transcript_workflow_is_decisive_and_disables_enrichment(tmp_path: Path) -> None:
    request = PrepareRequest(
        input="lecture.srt",
        out_dir=tmp_path / "out",
        workflow=WorkflowProfile.TRANSCRIPT,
    )

    resolved = resolve_config(request)

    assert resolved.runtime.workflow == WorkflowProfile.TRANSCRIPT
    assert resolved.transforms.asr.enabled == CapabilityEnabled.AUTO
    assert resolved.transforms.visual_context.enabled == CapabilityEnabled.FALSE
    assert resolved.transforms.knowledge_flow.enabled == CapabilityEnabled.FALSE


def test_prepare_help_uses_decisive_flags_without_negation_pairs() -> None:
    result = CliRunner().invoke(app, ["prepare", "--help"])

    assert result.exit_code == 0
    assert "--workflow" in result.output
    assert "--config" in result.output
    assert "--offline" in result.output
    assert "--language" not in result.output
    assert "--no-auto" not in result.output
    assert "--no-offline" not in result.output
    assert "--no-overwrite" not in result.output
    assert "--no-keep-temp" not in result.output
    assert "--no-visual-context" not in result.output


def test_plan_asr_records_selected_local_model_id(tmp_path: Path) -> None:
    request = PrepareRequest(input="lecture.wav", out_dir=tmp_path / "out")
    resolved = resolve_config(request)
    policy = resolved.transforms.asr.model_copy(update={"model": "tiny"})
    environment = TransformEnvironment(installed_asr=True, configured_asr_model_id="tiny")
    source = SourceState(has_transcript=False, has_media=True)

    plan = plan_asr(policy, environment, source)

    assert plan.selected == "local"
    assert plan.provider_id == "faster-whisper"
    assert plan.model_id == "tiny"
    assert plan.evidence_seed.model_id == "tiny"
    assert plan.ai_route is not None
    assert plan.ai_route.task == "asr_transcription"
    assert plan.ai_route.selected == "local"
    assert plan.ai_route.provider == "local"
    assert plan.ai_route.provider_id == "faster-whisper"
    assert plan.ai_route.model == "tiny"
    assert plan.ai_route.cost == "local"
    assert plan.ai_route.upload == "none"


def test_plan_asr_uses_configured_provider_identity(tmp_path: Path) -> None:
    request = PrepareRequest(input="lecture.mp4", out_dir=tmp_path / "out")
    resolved = resolve_config(request)
    policy = resolved.transforms.asr.model_copy(
        update={
            "allow_upload": True,
            "preferred_provider": "openai-whisper",
            "model": "whisper-1",
        }
    )
    environment = TransformEnvironment(
        configured_asr=True,
        configured_asr_provider_id="openai-whisper",
        configured_asr_model_id="whisper-1",
        configured_asr_cost_mode="paid",
    )
    source = SourceState(has_transcript=False, has_media=True)

    plan = plan_asr(policy, environment, source)

    assert plan.selected == "configured-online"
    assert plan.provider_id == "openai-whisper"
    assert plan.model_id == "whisper-1"
    assert plan.evidence_seed.uploaded is True
    assert plan.evidence_seed.cost_may_apply is True
    assert plan.ai_route is not None
    assert plan.ai_route.task == "asr_transcription"
    assert plan.ai_route.selected == "configured-online"
    assert plan.ai_route.provider == "alias"
    assert plan.ai_route.provider_id == "openai-whisper"
    assert plan.ai_route.model == "whisper-1"
    assert plan.ai_route.cost == "paid"
    assert plan.ai_route.upload == "required"


def test_plan_asr_accepts_explicit_paid_provider_without_extra_gate(
    tmp_path: Path,
) -> None:
    request = PrepareRequest(input="lecture.mp4", out_dir=tmp_path / "out")
    resolved = resolve_config(request)
    policy = resolved.transforms.asr.model_copy(
        update={"allow_upload": True, "preferred_provider": "paid-asr"}
    )
    environment = TransformEnvironment(
        configured_asr=True,
        configured_asr_provider_id="paid-asr",
        configured_asr_cost_mode="paid",
    )
    source = SourceState(has_transcript=False, has_media=True)

    plan = plan_asr(policy, environment, source)

    assert plan.selected == "configured-online"
    assert plan.provider_id == "paid-asr"
    assert plan.evidence_seed.cost_may_apply is True
