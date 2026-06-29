from __future__ import annotations

from pathlib import Path

from vctx.models.visual import EssentialVisualCase, SourceAccess, VisualOperationMotive
from vctx.transforms.ai_routes import AiRoute
from vctx.transforms.model_resolution import ModelRef, ResolvedModelRoute
from vctx.transforms.visual_planning import (
    ActionName,
    VisualAction,
    VisualActionParams,
    VisualAssessment,
    aggregate_visual_motives,
    plan_visual_motives,
    visual_motives_from_cases,
)


def _action_names(assessment: VisualAssessment) -> list[str]:
    return [action.name for action in assessment.recipe]


def _available_actions(*names: ActionName) -> list[VisualAction]:
    action_by_name = {
        "sample": VisualAction.sample(),
        "ocr": VisualAction.ocr(provider_id="rapidocr"),
        "describe": VisualAction.describe(
            AiRoute.configured_alias(
                task="vision_description",
                selected="configured-online",
                provider_id="test-vlm",
                model="vision-test",
                cost="paid",
                upload="required",
                reason="test route",
            )
        ),
        "capture": VisualAction.capture(),
    }
    return [action_by_name[name] for name in names]


def _params(action: VisualAction) -> dict[str, object]:
    return action.params.model_dump(mode="json", exclude_none=True, exclude={"cases"})


def test_visual_planning_uses_typed_owned_params() -> None:
    source = Path("src/vctx/transforms/visual_planning.py").read_text(encoding="utf-8")

    assert "dict[str, Any]" not in source
    assert "from typing import Any" not in source


def test_visual_action_constructors_encode_variant_defaults() -> None:
    sample = VisualAction.sample(strategy="cover", budget=1)
    ocr = VisualAction.ocr(provider_id="rapidocr")
    capture = VisualAction.capture()

    assert sample.name == "sample"
    assert sample.route == "deterministic"
    assert sample.params == VisualActionParams(strategy="cover", budget=1)
    assert ocr.name == "ocr"
    assert ocr.route == "local"
    assert ocr.provider_id == "rapidocr"
    assert capture.name == "capture"
    assert capture.route == "deterministic"


def test_visual_describe_constructor_derives_route_and_provider_id_from_ai_route() -> None:
    action = VisualAction.describe(
        _openrouter_vision_route(reason="resolved from OpenRouter model reference")
    )

    assert action.name == "describe"
    assert action.route == "free-online"
    assert action.provider_id == "openrouter:nex-agi/nex-n2-pro:free"
    assert action.ai_route is not None


def _openrouter_vision_route(*, reason: str) -> AiRoute:
    return AiRoute.from_model_route(
        task="vision_description",
        selected="free-online",
        model_route=ResolvedModelRoute(
            ref=ModelRef(prefix="openrouter", value="nex-agi/nex-n2-pro:free"),
            provider="openrouter",
            model="nex-agi/nex-n2-pro:free",
            base_url="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="OPENROUTER_API_KEY",
            cost="free",
            upload="required",
            available=True,
            reason=reason,
        ),
    )



def test_motives_preserve_reasons_and_chart_defaults_to_capture_only() -> None:
    motives = visual_motives_from_cases(
        [
            EssentialVisualCase(
                segment_id="seg-chart",
                timestamp_seconds=10.0,
                case_type="other",
                priority=0.8,
                reason="chart visual reference",
                actions=["capture"],
            )
        ]
    )

    assert motives == [
        VisualOperationMotive(
            operation="capture",
            reason="chart_visual_reference",
            segment_id="seg-chart",
            timestamp_seconds=10.0,
            priority=0.8,
            explanation="chart visual reference",
        )
    ]

    assessment = aggregate_visual_motives(
        has_video=True,
        has_transcript=True,
        duration_seconds=120.0,
        motives=motives,
        available_actions=_available_actions("sample", "ocr", "describe", "capture"),
    )

    assert _action_names(assessment) == ["sample", "capture"]
    assert [m.reason for m in assessment.motives] == ["chart_visual_reference"]


def test_motive_aggregation_window_merges_close_timestamps_without_losing_ops() -> None:
    motives = visual_motives_from_cases(
        [
            EssentialVisualCase(
                segment_id="seg-formula",
                timestamp_seconds=10.0,
                case_type="formula",
                priority=0.9,
                reason="formula on slide",
                actions=["ocr", "describe", "capture"],
            ),
            EssentialVisualCase(
                segment_id="seg-demo",
                timestamp_seconds=10.4,
                case_type="screen_demo",
                priority=0.7,
                reason="nearby demo action",
                actions=["describe", "capture"],
            ),
        ]
    )

    assessment = aggregate_visual_motives(
        has_video=True,
        has_transcript=True,
        duration_seconds=120.0,
        motives=motives,
        available_actions=_available_actions("sample", "ocr", "describe", "capture"),
    )

    assert _action_names(assessment) == ["sample", "ocr", "describe", "capture"]
    sample = assessment.recipe[0]
    assert sample.params.strategy == "essential_cases"
    assert sample.params.budget == 1
    assert len(sample.params.cases) == 1
    assert sample.params.cases[0].segment_id == "seg-formula"
    assert {m.operation for m in assessment.motives} == {"capture", "ocr", "describe"}


def test_source_access_uses_3_bit_tav_notation() -> None:
    access = SourceAccess.TRANSCRIPT_VIDEO

    assert access.bits == "101"
    assert access.has_transcript is True
    assert access.has_audio is False
    assert access.has_video is True


def test_visual_motive_plan_is_skipped_when_3_bit_access_lacks_video() -> None:
    motives = [
        VisualOperationMotive(
            operation="capture",
            reason="chart_visual_reference",
            segment_id="seg-chart",
            timestamp_seconds=10.0,
            priority=0.8,
            explanation="chart visual reference",
        )
    ]

    plan = plan_visual_motives(
        source=SourceAccess.TRANSCRIPT_AUDIO,
        duration_seconds=120.0,
        motives=motives,
        available_actions=_available_actions("sample", "capture"),
    )

    assert plan.kind == "skipped"
    assert plan.reason == "no_video"


def test_visual_motive_plan_valid_only_for_transcript_video_with_motives() -> None:
    motives = [
        VisualOperationMotive(
            operation="capture",
            reason="chart_visual_reference",
            segment_id="seg-chart",
            timestamp_seconds=10.0,
            priority=0.8,
            explanation="chart visual reference",
        )
    ]

    plan = plan_visual_motives(
        source=SourceAccess.TRANSCRIPT_VIDEO,
        duration_seconds=120.0,
        motives=motives,
        available_actions=_available_actions("sample", "ocr", "describe", "capture"),
    )

    assert plan.kind == "valid"
    assert [action.name for action in plan.assessment.recipe] == ["sample", "capture"]
