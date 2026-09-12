from __future__ import annotations

import json

from vctx.ai import AiBinding, AiClient, AiInstance, AiRoute
from vctx.net import NetRequest, NetResponse
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment
from vctx.visual.plan import (
    DraftClaim,
    DraftFrameRequest,
    DraftRelation,
    EvidencePlanner,
    PlanLinearizer,
    TranscriptIndex,
    WindowDraft,
    WindowResult,
)


def _transcript(segments: list[tuple[float, float, str]]) -> Transcript:
    return Transcript(
        source_id="video",
        provenance=TranscriptProvenance(method="local_file"),
        segments=[
            TranscriptSegment(id=f"seg_{index:06d}", start=start, end=end, text=text)
            for index, (start, end, text) in enumerate(segments, 1)
        ],
    )


def test_windows_keep_whole_segments_and_bounded_context_halos() -> None:
    transcript = _transcript(
        [
            (0, 10, "a" * 13_000),
            (10, 20, "b" * 13_000),
            (20, 30, "context"),
            (70, 80, "too far"),
        ]
    )

    windows = TranscriptIndex(transcript).windows()

    assert [window.id for window in windows] == ["seg_000001--seg_000001", "seg_000002--seg_000004"]
    assert windows[0].after_ids == []
    assert windows[1].before_ids == []  # 13 KiB exceeds the 4 KiB whole-segment halo


def test_merge_is_ordered_validated_and_derives_frame_times() -> None:
    transcript = _transcript([(0, 4, "α"), (4, 8, "β"), (8, 12, "γ")])
    index = TranscriptIndex(transcript)
    windows = index.windows()
    first = WindowResult(
        claims=[
            DraftClaim(ref="b", kind="fact", text="β", segment_ids=["seg_000002"]),
            DraftClaim(ref="a", kind="fact", text="α", segment_ids=["seg_000001"]),
        ],
        relations=[DraftRelation(kind="supports", source_ref="a", target_ref="b")],
        frames=[
            DraftFrameRequest(
                mode="sequence",
                anchor_segment_id="seg_000002",
                processors=["describe", "ocr"],
                priority=0.8,
                claim_refs=["b"],
            ),
            DraftFrameRequest(
                mode="still",
                anchor_segment_id="missing",
                processors=["ocr"],
                priority=1.0,
            ),
        ],
    )

    plan = PlanLinearizer(index, [WindowDraft(window=windows[0], result=first)]).build([])

    assert [(claim.id, claim.text) for claim in plan.claims] == [
        ("claim-0001", "α"),
        ("claim-0002", "β"),
    ]
    assert [(relation.source, relation.target) for relation in plan.relations] == [
        ("claim-0001", "claim-0002")
    ]
    assert [frame.target_seconds for frame in plan.frames] == [5.0, 6.0, 7.0]
    assert plan.frames[0].processors == ["ocr", "describe"]
    assert plan.frames[0].request_ids == ["request-0001"]
    assert plan.omissions[0].reason == "invalid_anchor"


def test_exact_duplicates_merge_and_capture_budget_is_receipted() -> None:
    transcript = _transcript([(float(i), float(i + 1), str(i)) for i in range(130)])
    index = TranscriptIndex(transcript)
    window = index.windows()[0]
    claims = [DraftClaim(ref="same", kind="fact", text="same", segment_ids=["seg_000001"])]
    frames = [
        DraftFrameRequest(
            mode="still",
            anchor_segment_id=f"seg_{index:06d}",
            processors=["ocr"],
            priority=float(index) / 130,
        )
        for index in range(1, 131)
    ]
    result = WindowResult(claims=claims + claims, frames=frames)

    plan = PlanLinearizer(index, [WindowDraft(window=window, result=result)]).build([])

    assert len(plan.claims) == 1
    assert len(plan.frames) == 128
    assert len(plan.omissions) == 2
    assert {item.reason for item in plan.omissions} == {"budget_exhausted"}


def test_merge_does_not_depend_on_window_completion_order() -> None:
    transcript = _transcript([(0, 5, "a" * 13_000), (5, 10, "b" * 13_000)])
    index = TranscriptIndex(transcript)
    windows = index.windows()
    results = [
        (
            window,
            WindowResult(
                claims=[
                    DraftClaim(
                        ref="claim",
                        kind="fact",
                        text=window.core[0].text,
                        segment_ids=window.core_ids,
                    )
                ],
            ),
        )
        for window in windows
    ]

    forward = (
        PlanLinearizer(
            index,
            [WindowDraft(window=window, result=result) for window, result in results],
        )
        .build([])
        .model_dump_json()
    )
    reverse = (
        PlanLinearizer(
            index,
            [WindowDraft(window=window, result=result) for window, result in reversed(results)],
        )
        .build([])
        .model_dump_json()
    )

    assert forward == reverse


class _Net:
    def __init__(self, responses: list[NetResponse]) -> None:
        self.responses = responses
        self.requests: list[NetRequest] = []

    def request(self, request: NetRequest) -> NetResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _chat(status: int, content: str) -> NetResponse:
    body: dict[str, object] = (
        {"error": {"message": content}}
        if status >= 400
        else {"choices": [{"message": {"content": content}}]}
    )
    return NetResponse(
        url="http://127.0.0.1:1/v1/chat/completions",
        status_code=status,
        body=json.dumps(body).encode(),
    )


def test_context_rejection_bisects_whole_segment_core() -> None:
    transcript = _transcript([(0, 2, "one"), (2, 4, "two")])
    net = _Net(
        [
            _chat(400, "maximum context length exceeded"),
            _chat(
                200,
                json.dumps(
                    {
                        "claims": [
                            {
                                "ref": "a",
                                "kind": "fact",
                                "text": "one",
                                "segment_ids": ["seg_000001"],
                            }
                        ]
                    }
                ),
            ),
            _chat(
                200,
                json.dumps(
                    {
                        "claims": [
                            {
                                "ref": "b",
                                "kind": "fact",
                                "text": "two",
                                "segment_ids": ["seg_000002"],
                            }
                        ]
                    }
                ),
            ),
        ]
    )
    instance = AiInstance(
        name="local",
        provider_id="local",
        base_url="http://127.0.0.1:1/v1",
        model="planner",
        format="json",
    )
    client = AiClient(
        AiBinding(
            AiRoute(
                task="evidence_plan",
                selected="local",
                instance=instance,
                reason="test binding",
            ),
            None,
        ),
        net=net,
    )

    plan = EvidencePlanner(client).plan(transcript)

    assert [claim.text for claim in plan.claims] == ["one", "two"]
    assert [receipt.window_id for receipt in plan.receipts] == [
        "seg_000001--seg_000001",
        "seg_000002--seg_000002",
    ]
    assert len(net.requests) == 3


def test_schedule_coalesces_before_budget_and_preserves_intents() -> None:
    transcript = _transcript([(0, 2, "one")])
    index = TranscriptIndex(transcript)
    window = index.windows()[0]
    result = WindowResult(
        frames=[
            DraftFrameRequest(
                mode="still",
                anchor_segment_id="seg_000001",
                processors=["ocr"],
                priority=index / 130,
            )
            for index in range(130)
        ]
    )

    plan = PlanLinearizer(index, [WindowDraft(window=window, result=result)]).build([])

    assert len(plan.intents) == 130
    assert len(plan.frames) == 1
    assert len(plan.frames[0].request_ids) == 130
    assert not [item for item in plan.omissions if item.reason == "budget_exhausted"]
    disabled = PlanLinearizer(
        index, [WindowDraft(window=window, result=result)], processors=()
    ).build([])
    assert all(not intent.processors for intent in disabled.intents)
    assert all(not frame.processors for frame in disabled.frames)


def test_coalesced_frame_uses_highest_priority_target() -> None:
    transcript = _transcript([(0, 0.2, "low"), (0.2, 0.6, "high")])
    index = TranscriptIndex(transcript)
    window = index.windows()[0]
    result = WindowResult(
        frames=[
            DraftFrameRequest(
                mode="still",
                anchor_segment_id="seg_000001",
                priority=0.2,
            ),
            DraftFrameRequest(
                mode="still",
                anchor_segment_id="seg_000002",
                priority=0.9,
            ),
        ]
    )

    plan = PlanLinearizer(index, [WindowDraft(window=window, result=result)]).build([])

    assert [frame.target_seconds for frame in plan.frames] == [0.4]
    assert plan.frames[0].priority == 0.9


def test_context_rejection_slices_one_oversized_segment() -> None:
    transcript = _transcript([(0, 2, "abcdefgh")])
    net = _Net(
        [
            _chat(400, "maximum context length exceeded"),
            _chat(200, json.dumps({"claims": []})),
            _chat(200, json.dumps({"claims": []})),
        ]
    )
    instance = AiInstance(
        name="local",
        provider_id="local",
        base_url="http://127.0.0.1:1/v1",
        model="planner",
        format="json",
    )
    client = AiClient(
        AiBinding(
            AiRoute(
                task="evidence_plan",
                selected="local",
                instance=instance,
                reason="test binding",
            ),
            None,
        ),
        net=net,
    )

    plan = EvidencePlanner(client).plan(transcript)

    assert len(net.requests) == 3
    assert len(plan.receipts) == 2
    prompts: list[str] = []
    for request in net.requests:
        assert request.body is not None
        body = json.loads(request.body)
        assert "untrusted" in body["messages"][0]["content"]
        prompts.append(json.loads(body["messages"][1]["content"])["core"][0]["text"])
    assert prompts == ["abcdefgh", "abcd", "efgh"]
