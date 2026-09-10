from __future__ import annotations

from typing import cast

import pytest

from tests.support import asr_ready_segments
from vctx.ai import AiClient, AiFailure, AiFailureCode, AiReceipt, AiSuccess
from vctx.summary import DraftPoint, SummaryDraft, SummaryPacket, SummaryWriter
from vctx.visual.plan import EvidencePlan


def _r(request: str) -> AiReceipt:
    return AiReceipt(
        task="summary",
        request_id=request,
        provider="fixture",
        configured_model="m",
        format="json",
        attempts=1,
        latency_ms=0,
        privacy="standard",
    )


class _Client:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, **kwargs: object) -> object:
        self.calls += 1
        self.prompts.append(str(kwargs["messages"]))
        return self.outcomes.pop(0)


def _ok(
    key: str, *points: DraftPoint, overview: str | None = "overview"
) -> AiSuccess[SummaryDraft]:
    return AiSuccess(value=SummaryDraft(overview=overview, points=list(points)), receipt=_r(key))


def _fail(request: str, failure: AiFailureCode = "context_length") -> AiFailure:
    return AiFailure(failure=failure, reason=failure, receipt=_r(request))


def _packet() -> SummaryPacket:
    transcript = asr_ready_segments("video", [(0, 1, "甲"), (1, 2, "乙")]).transcript
    return SummaryPacket.from_products(transcript)


def test_summary_rejects_unknown_ids_and_wrong_basis() -> None:
    packet = _packet()
    with pytest.raises(ValueError, match="unknown segment"):
        point = DraftPoint(text="x", basis="transcript", segment_ids=["other"])
        packet.admit(SummaryDraft(points=[point]), language="native")
    with pytest.raises(ValueError, match="basis"):
        DraftPoint(text="x", basis="visual", segment_ids=["seg_000001"])
    with pytest.raises(ValueError, match="one source"):
        transcript = asr_ready_segments("video", [(0, 1, "x")]).transcript
        SummaryPacket.from_products(transcript, EvidencePlan(source_id="other"))


def test_summary_normal_path_is_one_call_and_preserves_native_text() -> None:
    point = DraftPoint(text="要約", basis="transcript", segment_ids=["seg_000001"])
    client = _Client([_ok("summary", point)])
    outcome = SummaryWriter(cast(AiClient, client)).write(_packet())
    assert client.calls == 1 and outcome.status == "ready"
    assert outcome.summary is not None and outcome.summary.points[0].segment_ids == ["seg_000001"]
    assert all(text in client.prompts[0] for text in ("甲", "untrusted", "dominant transcript"))


@pytest.mark.parametrize("group_failure", [False, True])
def test_context_overflow_keeps_valid_groups_when_reduction_fails(group_failure: bool) -> None:
    one = DraftPoint(text="一", basis="transcript", segment_ids=["seg_000001"])
    two = DraftPoint(text="二", basis="transcript", segment_ids=["seg_000002"])
    tail = (
        [_fail("g2", "http"), _ok("reduce", two)]
        if group_failure
        else [_ok("g2", two), _fail("reduce", "http")]
    )
    client = _Client([_fail("all"), _ok("g1", one), *tail])
    outcome = SummaryWriter(cast(AiClient, client)).write(_packet())
    assert client.calls == 4 and outcome.status == "partial"
    assert outcome.summary is not None and outcome.summary.overview is None
    assert [point.segment_ids for point in outcome.summary.points] == [
        ["seg_000001"],
        *([] if group_failure else [["seg_000002"]]),
    ]


def test_empty_transcript_makes_no_call() -> None:
    client = _Client([])
    packet = SummaryPacket.from_products(asr_ready_segments("video", []).transcript)
    outcome = SummaryWriter(cast(AiClient, client)).write(packet)
    assert outcome.status == "unavailable" and outcome.summary is None and client.calls == 0
