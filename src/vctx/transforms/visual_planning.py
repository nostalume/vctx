from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field

from vctx.models.visual import (
    EssentialCaseType,
    EssentialVisualCase,
    Evidence,
    SourceAccess,
    VisualOperationMotive,
    VisualOperationName,
    VisualOperationReason,
)
from vctx.transforms.ai_routes import AiRoute

ActionName = Literal["sample", "ocr", "describe", "capture"]
ActionRoute = Literal["deterministic", "local", "free-online", "configured-online"]
SampleStrategy = Literal["cover", "changes", "changes+anchors", "essential_cases"]


class VisualActionParams(BaseModel):
    strategy: SampleStrategy | None = None
    budget: int | None = None
    min_gap_s: float | None = None
    cases: list[EssentialVisualCase] = Field(default_factory=list)


class VisualAction(BaseModel):
    name: ActionName
    route: ActionRoute = "deterministic"
    provider_id: str | None = None
    params: VisualActionParams = Field(default_factory=VisualActionParams)
    ai_route: AiRoute | None = None

    @classmethod
    def sample(
        cls,
        *,
        strategy: SampleStrategy | None = None,
        budget: int | None = None,
        min_gap_s: float | None = None,
        cases: list[EssentialVisualCase] | None = None,
        params: VisualActionParams | None = None,
    ) -> Self:
        return cls(
            name="sample",
            params=params
            or VisualActionParams(
                strategy=strategy,
                budget=budget,
                min_gap_s=min_gap_s,
                cases=cases or [],
            ),
        )

    @classmethod
    def ocr(cls, *, provider_id: str, route: ActionRoute = "local") -> Self:
        return cls(name="ocr", route=route, provider_id=provider_id)

    @classmethod
    def describe(cls, ai_route: AiRoute) -> Self:
        route: ActionRoute = (
            "free-online" if ai_route.selected == "free-online" else "configured-online"
        )
        provider_id = ai_route.provider_id
        if ai_route.provider == "alias":
            provider_id = ai_route.provider_id
        elif ai_route.provider != "none" and ai_route.model is not None:
            provider_id = f"{ai_route.provider}:{ai_route.model}"
        return cls(
            name="describe",
            route=route,
            provider_id=provider_id,
            ai_route=ai_route,
        )

    @classmethod
    def capture(cls) -> Self:
        return cls(name="capture")


def baseline_visual_actions() -> list[VisualAction]:
    return [VisualAction.sample(), VisualAction.capture()]



class VisualAssessment(BaseModel):
    visual_yield: float
    audio_sufficiency: float
    recipe: list[VisualAction] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: str
    cautions: list[str] = Field(default_factory=list)
    motives: list[VisualOperationMotive] = Field(default_factory=list)



VisualSkipReason = Literal[
    "no_transcript",
    "no_video",
    "no_visual_motives",
    "no_available_visual_actions",
]


class ValidVisualPlan(BaseModel):
    kind: Literal["valid"] = "valid"
    assessment: VisualAssessment


class SkippedVisualPlan(BaseModel):
    kind: Literal["skipped"] = "skipped"
    reason: VisualSkipReason
    rationale: str


VisualPlan = Annotated[ValidVisualPlan | SkippedVisualPlan, Field(discriminator="kind")]


def plan_visual_motives(
    *,
    source: SourceAccess,
    duration_seconds: float | None,
    motives: list[VisualOperationMotive],
    available_actions: list[VisualAction],
) -> VisualPlan:
    if not source.has_transcript:
        return SkippedVisualPlan(
            reason="no_transcript",
            rationale=f"source_access={source.bits}; no transcript-anchored visual motives",
        )
    if not source.has_video:
        return SkippedVisualPlan(
            reason="no_video",
            rationale=f"source_access={source.bits}; no video access",
        )
    if not motives:
        return SkippedVisualPlan(
            reason="no_visual_motives",
            rationale=f"source_access={source.bits}; no visual motives",
        )
    assessment = aggregate_visual_motives(
        has_video=True,
        has_transcript=True,
        duration_seconds=duration_seconds,
        motives=motives,
        available_actions=available_actions,
    )
    if not assessment.recipe:
        return SkippedVisualPlan(
            reason="no_available_visual_actions",
            rationale=f"source_access={source.bits}; no available visual actions",
        )
    return ValidVisualPlan(assessment=assessment)

def visual_motives_from_cases(
    cases: list[EssentialVisualCase],
) -> list[VisualOperationMotive]:
    motives: list[VisualOperationMotive] = []
    for case in cases:
        for operation in _case_operations(case):
            motives.append(
                VisualOperationMotive(
                    operation=operation,
                    reason=_case_operation_reason(case, operation),
                    segment_id=case.segment_id,
                    timestamp_seconds=case.timestamp_seconds,
                    priority=case.priority,
                    explanation=case.reason,
                )
            )
    return motives


def aggregate_visual_motives(
    *,
    has_video: bool,
    has_transcript: bool,
    duration_seconds: float | None,
    motives: list[VisualOperationMotive],
    available_actions: list[VisualAction],
) -> VisualAssessment:
    if not has_video:
        return VisualAssessment(
            visual_yield=0.0,
            audio_sufficiency=0.0,
            recipe=[],
            rationale="no video stream is available",
            motives=motives,
        )
    if not has_transcript or not motives:
        return VisualAssessment(
            visual_yield=0.0,
            audio_sufficiency=1.0 if has_transcript else 0.0,
            recipe=[],
            rationale="no transcript-anchored visual motives",
            motives=motives,
        )

    selected_motives = _dedupe_motives(motives)
    actions: list[VisualAction] = []
    sample = _available_action(available_actions, "sample")
    if sample is not None:
        cases = _sample_cases_from_motives(selected_motives)
        actions.append(
            VisualAction.sample(
                strategy="essential_cases",
                budget=len(cases),
                min_gap_s=1.0,
                cases=cases,
            )
        )
    for action_name in ("ocr", "describe", "capture"):
        if not any(motive.operation == action_name for motive in selected_motives):
            continue
        action = _available_action(available_actions, action_name)
        if action is not None:
            actions.append(action)
    return VisualAssessment(
        visual_yield=1.0,
        audio_sufficiency=0.0,
        recipe=actions,
        evidence=[
            Evidence(kind="transcript", name=motive.reason, weight=motive.priority)
            for motive in selected_motives
        ],
        rationale="visual operations are driven by transcript-anchored motives",
        cautions=(
            ["description is model output; keep source frames"]
            if any(action.name == "describe" for action in actions)
            else []
        ),
        motives=selected_motives,
    )


def _case_operations(case: EssentialVisualCase) -> list[VisualOperationName]:
    if case.actions:
        return [action for action in case.actions]
    return ["capture"]


def _case_operation_reason(
    case: EssentialVisualCase, operation: VisualOperationName
) -> VisualOperationReason:
    text = case.reason.lower()
    if "chart" in text or "graph" in text or "stats" in text:
        return "chart_visual_reference"
    if case.case_type == "formula":
        return "formula_or_equation"
    if case.case_type in {"table", "code"}:
        return "table_or_code"
    if case.case_type in {"slide_title", "visual_summary"}:
        return "shown_text" if operation == "ocr" else "visual_reference"
    if case.case_type == "screen_demo":
        return "shown_text" if operation == "ocr" else "action_demonstration"
    if case.case_type == "diagram":
        return "visual_reference"
    return "visual_reference"


def _dedupe_motives(
    motives: list[VisualOperationMotive], *, window_s: float = 1.0
) -> list[VisualOperationMotive]:
    selected: list[VisualOperationMotive] = []
    for motive in sorted(motives, key=lambda item: item.priority, reverse=True):
        duplicate = any(
            motive.operation == existing.operation
            and motive.reason == existing.reason
            and abs(motive.timestamp_seconds - existing.timestamp_seconds) <= window_s
            for existing in selected
        )
        if not duplicate:
            selected.append(motive)
    return sorted(selected, key=lambda item: (item.timestamp_seconds, item.operation))


def _sample_cases_from_motives(
    motives: list[VisualOperationMotive], *, window_s: float = 1.0
) -> list[EssentialVisualCase]:
    cases: list[EssentialVisualCase] = []
    for motive in sorted(motives, key=lambda item: item.timestamp_seconds):
        if any(
            abs(motive.timestamp_seconds - case.timestamp_seconds) <= window_s
            for case in cases
        ):
            continue
        window_motives = [
            item
            for item in motives
            if abs(item.timestamp_seconds - motive.timestamp_seconds) <= window_s
        ]
        anchor = max(window_motives, key=lambda item: item.priority)
        cases.append(
            EssentialVisualCase(
                segment_id=anchor.segment_id,
                timestamp_seconds=anchor.timestamp_seconds,
                case_type=_case_type_from_reason(anchor.reason),
                priority=anchor.priority,
                reason=anchor.explanation,
                actions=sorted(
                    {item.operation for item in window_motives},
                    key={"ocr": 0, "describe": 1, "capture": 2}.__getitem__,
                ),
            )
        )
    return cases


def _case_type_from_reason(reason: VisualOperationReason) -> EssentialCaseType:
    if reason == "formula_or_equation":
        return "formula"
    if reason == "table_or_code":
        return "table"
    if reason == "shown_text":
        return "slide_title"
    if reason == "action_demonstration":
        return "screen_demo"
    if reason == "visual_reference":
        return "diagram"
    return "other"


def _available_action(actions: list[VisualAction], name: ActionName) -> VisualAction | None:
    return next((action for action in actions if action.name == name), None)


def _score(evidence: list[Evidence], names: set[str]) -> float:
    return _clamp(sum(item.weight for item in evidence if item.name in names))


def _target_frames(
    duration_seconds: float | None, *, seconds_per_frame: int, minimum: int, maximum: int
) -> int:
    if duration_seconds is None or duration_seconds <= 0:
        return minimum
    estimated = round(duration_seconds / seconds_per_frame)
    return min(max(estimated, minimum), maximum)


def _clamp(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 3)
