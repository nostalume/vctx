from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vctx.ai import AiClient, AiMessage, AiReceipt, AiRoute
from vctx.transcript import Transcript, TranscriptSegment

_CORE_BYTES = 24 * 1024
_CORE_SECONDS = 10 * 60
_HALO_BYTES = 4 * 1024
_HALO_SECONDS = 30
_CAPTURE_BUDGET = 128

ClaimKind = Literal["fact", "definition", "process", "decision", "comparison", "instruction"]
RelationKind = Literal["supports", "causes", "precedes", "depends_on", "contrasts"]
FrameMode = Literal["still", "sequence"]
FrameProcessor = Literal["ocr", "describe"]
OmissionReason = Literal["invalid_anchor", "invalid_claim", "budget_exhausted"]
type ClaimKey = tuple[ClaimKind, tuple[str, ...], str]


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannerWindow(ClosedModel):
    id: str
    core: list[TranscriptSegment]
    before: list[TranscriptSegment] = Field(default_factory=list)
    after: list[TranscriptSegment] = Field(default_factory=list)

    @property
    def core_ids(self) -> list[str]:
        return [segment.id for segment in self.core]

    @property
    def before_ids(self) -> list[str]:
        return [segment.id for segment in self.before]

    @property
    def after_ids(self) -> list[str]:
        return [segment.id for segment in self.after]

    @property
    def scope_ids(self) -> set[str]:
        return {*self.before_ids, *self.core_ids, *self.after_ids}


class DraftClaim(ClosedModel):
    ref: str
    kind: ClaimKind
    text: str
    segment_ids: list[str] = Field(min_length=1)


class DraftRelation(ClosedModel):
    kind: RelationKind
    source_ref: str
    target_ref: str
    text: str | None = None


class DraftFrameRequest(ClosedModel):
    mode: FrameMode
    anchor_segment_id: str
    processors: list[FrameProcessor] = Field(default_factory=list)
    priority: float = Field(default=0.5, ge=0, le=1)
    claim_refs: list[str] = Field(default_factory=list)


class WindowResult(ClosedModel):
    claims: list[DraftClaim] = Field(default_factory=list)
    relations: list[DraftRelation] = Field(default_factory=list)
    frames: list[DraftFrameRequest] = Field(default_factory=list)


class EvidenceClaim(ClosedModel):
    id: str
    kind: ClaimKind
    text: str
    segment_ids: list[str]


class EvidenceRelation(ClosedModel):
    id: str
    kind: RelationKind
    source: str
    target: str
    text: str | None = None


class PlannedFrame(ClosedModel):
    id: str
    target_seconds: float = Field(ge=0)
    segment_ids: list[str]
    processors: list[FrameProcessor]
    request_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    priority: float


class PlanOmission(ClosedModel):
    request_id: str
    reason: OmissionReason
    detail: str


class PlanReceipt(ClosedModel):
    window_id: str
    status: Literal["ok", "empty", "unavailable", "failed"]
    detail: str
    ai: AiReceipt | None = None


class EvidencePlan(ClosedModel):
    video_id: str
    claims: list[EvidenceClaim] = Field(default_factory=list)
    relations: list[EvidenceRelation] = Field(default_factory=list)
    frames: list[PlannedFrame] = Field(default_factory=list)
    omissions: list[PlanOmission] = Field(default_factory=list)
    receipts: list[PlanReceipt] = Field(default_factory=list)


class VisualAction(ClosedModel):
    name: Literal["sample", "ocr", "describe", "capture"]
    frames: list[PlannedFrame] = Field(default_factory=list)
    provider_id: str | None = None
    ai_route: AiRoute | None = None


class VisualAssessment(ClosedModel):
    recipe: list[VisualAction]
    frames: list[PlannedFrame]
    rationale: str
    missing_processors: list[FrameProcessor] = Field(default_factory=list)


def execution_plan(
    plan: EvidencePlan, *, ocr_available: bool, vision_route: AiRoute | None
) -> VisualAssessment:
    if not plan.frames:
        return VisualAssessment(
            recipe=[], frames=[], rationale="evidence plan requested no captures"
        )
    requested = {processor for frame in plan.frames for processor in frame.processors}
    recipe = [VisualAction(name="sample", frames=plan.frames)]
    missing: list[FrameProcessor] = []
    if "ocr" in requested:
        if ocr_available:
            recipe.append(VisualAction(name="ocr", provider_id="rapidocr"))
        else:
            missing.append("ocr")
    if "describe" in requested:
        if vision_route is not None:
            recipe.append(
                VisualAction(
                    name="describe",
                    provider_id=vision_route.provider_id,
                    ai_route=vision_route,
                )
            )
        else:
            missing.append("describe")
    recipe.append(VisualAction(name="capture"))
    return VisualAssessment(
        recipe=recipe,
        frames=plan.frames,
        rationale="validated transcript-anchored evidence plan",
        missing_processors=missing,
    )


def planner_windows(transcript: Transcript) -> list[PlannerWindow]:
    cores: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    for segment in transcript.segments:
        candidate = [*current, segment]
        if current and (
            _segments_bytes(candidate) > _CORE_BYTES or _span(candidate) > _CORE_SECONDS
        ):
            cores.append(current)
            current = [segment]
        else:
            current = candidate
    if current:
        cores.append(current)
    positions = {segment.id: index for index, segment in enumerate(transcript.segments)}
    return [
        PlannerWindow(
            id=f"{core[0].id}--{core[-1].id}",
            core=core,
            before=_halo(transcript.segments, positions[core[0].id], -1),
            after=_halo(transcript.segments, positions[core[-1].id], 1),
        )
        for core in cores
    ]


def merge_window_results(
    transcript: Transcript,
    results: Iterable[tuple[PlannerWindow, WindowResult]],
) -> EvidencePlan:
    positions = {segment.id: index for index, segment in enumerate(transcript.segments)}
    segments = {segment.id: segment for segment in transcript.segments}
    ordered = sorted(results, key=lambda item: positions[item[0].core[0].id])
    claims, refs = _claims(ordered, positions)
    relations = _relations(ordered, refs, claims)
    frames, omissions = _frames(ordered, refs, claims, segments)
    return EvidencePlan(
        video_id=transcript.video_id,
        claims=claims,
        relations=relations,
        frames=frames,
        omissions=omissions,
    )


def plan_evidence(transcript: Transcript, client: AiClient) -> EvidencePlan:
    admitted: list[tuple[PlannerWindow, WindowResult]] = []
    receipts: list[PlanReceipt] = []
    for window in planner_windows(transcript):
        results, window_receipts = _plan_window(transcript, window, client)
        admitted.extend(results)
        receipts.extend(window_receipts)
    plan = merge_window_results(transcript, admitted)
    return plan.model_copy(update={"receipts": receipts})


def _plan_window(
    transcript: Transcript, window: PlannerWindow, client: AiClient
) -> tuple[list[tuple[PlannerWindow, WindowResult]], list[PlanReceipt]]:
    outcome = client.complete(
        task="evidence_plan",
        request_id=window.id,
        messages=[
            AiMessage(
                role="system",
                content=(
                    "Return claims, relations, and transcript-anchored frame requests. "
                    "Never invent timestamps. Anchor only to supplied segment ids. Context "
                    "segments explain core segments but do not own claims. Preserve source "
                    "language."
                ),
            ),
            AiMessage(role="user", content=_window_prompt(window)),
        ],
        result=WindowResult,
    )
    if outcome.kind == "ok":
        result = outcome.value
        status = "empty" if not (result.claims or result.relations or result.frames) else "ok"
        return [(window, result)], [
            PlanReceipt(
                window_id=window.id,
                status=status,
                detail="validated structured result",
                ai=outcome.receipt,
            )
        ]
    if outcome.failure == "context_length" and len(window.core) > 1:
        middle = len(window.core) // 2
        results: list[tuple[PlannerWindow, WindowResult]] = []
        receipts: list[PlanReceipt] = []
        for core in (window.core[:middle], window.core[middle:]):
            split = _window_for_core(transcript, core)
            split_results, split_receipts = _plan_window(transcript, split, client)
            results.extend(split_results)
            receipts.extend(split_receipts)
        return results, receipts
    status = (
        "unavailable" if outcome.failure in {"unavailable", "connection", "timeout"} else "failed"
    )
    return [], [
        PlanReceipt(
            window_id=window.id,
            status=status,
            detail=outcome.reason,
            ai=outcome.receipt,
        )
    ]


def _window_prompt(window: PlannerWindow) -> str:
    lines = [f"window={window.id}"]
    for role, segments in (
        ("context", window.before),
        ("core", window.core),
        ("context", window.after),
    ):
        lines.extend(
            f"{role} {segment.id} [{segment.start:.3f}-{(segment.end or segment.start):.3f}] "
            f"{segment.text}"
            for segment in segments
        )
    return "\n".join(lines)


def _window_for_core(transcript: Transcript, core: list[TranscriptSegment]) -> PlannerWindow:
    positions = {segment.id: index for index, segment in enumerate(transcript.segments)}
    return PlannerWindow(
        id=f"{core[0].id}--{core[-1].id}",
        core=core,
        before=_halo(transcript.segments, positions[core[0].id], -1),
        after=_halo(transcript.segments, positions[core[-1].id], 1),
    )


def _claims(
    results: list[tuple[PlannerWindow, WindowResult]], positions: dict[str, int]
) -> tuple[list[EvidenceClaim], dict[tuple[str, str], ClaimKey]]:
    admitted: list[tuple[ClaimKey, DraftClaim]] = []
    refs: dict[tuple[str, str], ClaimKey] = {}
    for window, result in results:
        for claim in result.claims:
            if not set(claim.segment_ids) <= window.scope_ids:
                continue
            anchors = sorted(set(claim.segment_ids), key=positions.__getitem__)
            if not anchors or anchors[0] not in window.core_ids:
                continue
            key = (claim.kind, tuple(anchors), _text(claim.text))
            admitted.append((key, claim.model_copy(update={"segment_ids": anchors})))
            refs[(window.id, claim.ref)] = key
    unique = {key: claim for key, claim in admitted}
    keys = sorted(unique, key=lambda key: (positions[key[1][0]], key))
    claims = [
        EvidenceClaim(
            id=f"claim-{index:04d}",
            kind=unique[key].kind,
            text=_text(unique[key].text),
            segment_ids=unique[key].segment_ids,
        )
        for index, key in enumerate(keys, 1)
    ]
    return claims, refs


def _relations(
    results: list[tuple[PlannerWindow, WindowResult]],
    refs: dict[tuple[str, str], ClaimKey],
    claims: list[EvidenceClaim],
) -> list[EvidenceRelation]:
    ids = {(claim.kind, tuple(claim.segment_ids), claim.text): claim.id for claim in claims}
    admitted: set[tuple[RelationKind, str, str, str | None]] = set()
    for window, result in results:
        for relation in result.relations:
            source_key = refs.get((window.id, relation.source_ref))
            target_key = refs.get((window.id, relation.target_ref))
            source = ids.get(source_key) if source_key is not None else None
            target = ids.get(target_key) if target_key is not None else None
            if source and target and source != target:
                admitted.add((relation.kind, source, target, _optional_text(relation.text)))
    return [
        EvidenceRelation(
            id=f"relation-{index:04d}",
            kind=kind,
            source=source,
            target=target,
            text=text,
        )
        for index, (kind, source, target, text) in enumerate(sorted(admitted), 1)
    ]


def _frames(
    results: list[tuple[PlannerWindow, WindowResult]],
    refs: dict[tuple[str, str], ClaimKey],
    claims: list[EvidenceClaim],
    segments: dict[str, TranscriptSegment],
) -> tuple[list[PlannedFrame], list[PlanOmission]]:
    claim_ids = {(claim.kind, tuple(claim.segment_ids), claim.text): claim.id for claim in claims}
    candidates: list[tuple[float, float, str, str, list[FrameProcessor], list[str]]] = []
    omissions: list[PlanOmission] = []
    request_index = 0
    for window, result in results:
        for request in result.frames:
            request_index += 1
            request_id = f"request-{request_index:04d}"
            if request.anchor_segment_id not in window.core_ids:
                omissions.append(
                    PlanOmission(
                        request_id=request_id,
                        reason="invalid_anchor",
                        detail=f"unknown or non-core segment {request.anchor_segment_id}",
                    )
                )
                continue
            linked = [
                claim_ids[key]
                for ref in request.claim_refs
                if (key := refs.get((window.id, ref))) in claim_ids
            ]
            if request.claim_refs and not linked:
                omissions.append(
                    PlanOmission(
                        request_id=request_id,
                        reason="invalid_claim",
                        detail="no referenced claim survived admission",
                    )
                )
                continue
            segment = segments[request.anchor_segment_id]
            for target in _targets(segment, request.mode):
                candidates.append(
                    (
                        target,
                        request.priority,
                        request_id,
                        request.anchor_segment_id,
                        _processors(request.processors),
                        sorted(set(linked)),
                    )
                )
    selected = sorted(candidates, key=lambda item: (-item[1], item[0], item[2]))[:_CAPTURE_BUDGET]
    selected_ids = {(target, request_id) for target, _, request_id, _, _, _ in selected}
    for target, _, request_id, _, _, _ in candidates:
        if (target, request_id) not in selected_ids:
            omissions.append(
                PlanOmission(
                    request_id=request_id,
                    reason="budget_exhausted",
                    detail=f"capture target {target:.3f}s exceeded source budget",
                )
            )
    merged: list[tuple[float, float, set[str], set[str], set[FrameProcessor], set[str]]] = []
    for target, priority, request_id, segment_id, processors, linked in sorted(selected):
        if merged and abs(target - merged[-1][0]) <= 0.5:
            (
                old_target,
                old_priority,
                request_ids,
                segment_ids,
                old_processors,
                old_claims,
            ) = merged[-1]
            request_ids.add(request_id)
            segment_ids.add(segment_id)
            old_processors.update(processors)
            old_claims.update(linked)
            merged[-1] = (
                old_target,
                max(priority, old_priority),
                request_ids,
                segment_ids,
                old_processors,
                old_claims,
            )
        else:
            merged.append(
                (target, priority, {request_id}, {segment_id}, set(processors), set(linked))
            )
    frames = [
        PlannedFrame(
            id=f"frame-{index:04d}",
            target_seconds=round(target, 3),
            segment_ids=sorted(segment_ids),
            processors=sorted(processors, key={"ocr": 0, "describe": 1}.__getitem__),
            request_ids=sorted(request_ids),
            claim_ids=sorted(linked),
            priority=priority,
        )
        for index, (target, priority, request_ids, segment_ids, processors, linked) in enumerate(
            merged, 1
        )
    ]
    return frames, omissions


def _targets(segment: TranscriptSegment, mode: FrameMode) -> list[float]:
    end = segment.end if segment.end is not None else segment.start
    duration = max(0.0, end - segment.start)
    fractions = (0.5,) if mode == "still" else (0.25, 0.5, 0.75)
    return [round(segment.start + duration * fraction, 3) for fraction in fractions]


def _processors(values: list[FrameProcessor]) -> list[FrameProcessor]:
    return sorted(set(values), key={"ocr": 0, "describe": 1}.__getitem__)


def _halo(
    segments: list[TranscriptSegment], anchor: int, direction: int
) -> list[TranscriptSegment]:
    chosen: list[TranscriptSegment] = []
    index = anchor + direction
    while 0 <= index < len(segments):
        candidate = segments[index]
        temporal = (
            segments[anchor].start - (candidate.end or candidate.start)
            if direction < 0
            else candidate.start - (segments[anchor].end or segments[anchor].start)
        )
        proposed = [candidate, *chosen] if direction < 0 else [*chosen, candidate]
        if temporal > _HALO_SECONDS or _segments_bytes(proposed) > _HALO_BYTES:
            break
        chosen = proposed
        index += direction
    return chosen


def _segments_bytes(segments: list[TranscriptSegment]) -> int:
    return sum(len(segment.text.encode("utf-8")) for segment in segments)


def _span(segments: list[TranscriptSegment]) -> float:
    end = segments[-1].end if segments[-1].end is not None else segments[-1].start
    return max(0.0, end - segments[0].start)


def _text(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split()))


def _optional_text(value: str | None) -> str | None:
    return _text(value) if value else None
