from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.support import asr_ready_segments
from vctx.cli import app
from vctx.source.session import MediaAsset
from vctx.summary import (
    DraftPoint,
    SummaryDraft,
    SummaryOutcome,
    SummaryPacket,
    SummaryWriter,
)
from vctx.transcript import Transcript
from vctx.visual.frame import Frame, FrameBatch
from vctx.visual.ocr import OcrOutcome, OcrRuntimePool, RapidOcr
from vctx.visual.plan import EvidenceClaim, EvidencePlan, PlannedFrame

runner = CliRunner()
_PNG = b"\x89PNG\r\n\x1a\nfixture"


def test_prepare_publishes_validated_plan_and_planned_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.app.evidence as evidence_app
    import vctx.asr as asr_module

    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"video")
    config = tmp_path / "vctx.toml"
    config.write_text(
        """
[transforms.asr]
use = "instance:local-default"
[evidence]
planner = "instance:planner"
vision = "auto"
[instances.asr.local-default]
type = "local-faster-whisper"
model = "tiny"
cache = "persistent"
[instances.ai.planner]
base_url = "http://127.0.0.1:1234/v1"
model = "planner"
""".strip(),
        encoding="utf-8",
    )

    def fake_transcribe(self: object, asset: MediaAsset) -> object:
        del self
        return asr_ready_segments(asset.id, [(0, 4, "原生文本")])

    def fake_plan(*_args: object) -> EvidencePlan:
        return EvidencePlan(
            source_id="media-1",
            claims=[
                EvidenceClaim(
                    id="claim-0001",
                    kind="fact",
                    text="原生文本",
                    segment_ids=["seg_000001"],
                )
            ],
            frames=[
                PlannedFrame(
                    id="frame-0001",
                    target_seconds=2,
                    segment_ids=["seg_000001"],
                    processors=["ocr"],
                    request_ids=["request-0001"],
                    claim_ids=["claim-0001"],
                    priority=0.8,
                )
            ],
        )

    def fake_frames(_asset: object, _requests: object, out: Path) -> FrameBatch:
        frames = out / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        path = frames / "frame-0001.png"
        path.write_bytes(_PNG)
        return FrameBatch(
            (
                Frame(
                    id="frame-0001",
                    path=path,
                    requested_seconds=2,
                    actual_seconds=2,
                    original_size=(16, 9),
                    orientation=0,
                    size=(16, 9),
                    sha256=hashlib.sha256(_PNG).hexdigest(),
                    bytes=len(_PNG),
                    request_ids=("request-0001",),
                    segment_ids=("seg_000001",),
                    claim_ids=("claim-0001",),
                    processors=("ocr",),
                    priority=0.8,
                ),
            ),
            (),
        )

    class FailedOcr(RapidOcr):
        def __init__(self) -> None:
            pass

        def observe(self, frame: Frame) -> OcrOutcome:
            return OcrOutcome(status="failed", detail="fixture inference failure")

    monkeypatch.setattr(asr_module.FasterWhisperAsrAdapter, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        evidence_app.EvidencePlanner,
        "plan",
        lambda _self, _transcript: fake_plan(),
    )
    monkeypatch.setattr(evidence_app, "capture", fake_frames)
    monkeypatch.setattr(OcrRuntimePool, "load_rapid", lambda *_args: FailedOcr())
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "prepare",
            str(media),
            "--out",
            str(out),
            "--config",
            str(config),
            "--to",
            "evidence",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    lane = out / manifest["sources"][0]["path"]
    plan = EvidencePlan.model_validate_json(
        (lane / "evidence-plan.json").read_text(encoding="utf-8")
    )
    assert plan.claims[0].text == "原生文本"
    evidence = json.loads((lane / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["captures"][0]["ocr"] == {
        "status": "failed",
        "text": None,
        "detail": "fixture inference failure",
        "provider": "rapidocr",
    }
    assert (lane / "frames" / "frame-0001.png").read_bytes() == _PNG
    source = manifest["sources"][0]
    frame_ref = next(item for item in source["artifacts"] if item["kind"] == "visual_frame")
    assert frame_ref["path"] == "frames/frame-0001.png"
    outcome = next(item for item in source["outcomes"] if item["product"] == "evidence")
    assert outcome["status"] == "partial"
    assert "frames/frame-0001.png" in outcome["artifacts"]
    assert manifest["sources"][0]["status"] == "partial"


@pytest.mark.parametrize("target", ["evidence", "summary"])
def test_model_target_without_route_is_transcript_only_partial(
    tmp_path: Path, target: str
) -> None:
    subtitle = tmp_path / "lecture.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:02,000\nNo model inference.\n", encoding="utf-8")
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "prepare",
            str(subtitle),
            "--out",
            str(out),
            "--to",
            target,
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    lane = out / manifest["sources"][0]["path"]
    assert manifest["sources"][0]["status"] == "partial"
    outcome = next(
        item for item in manifest["sources"][0]["outcomes"] if item["product"] == target
    )
    assert outcome["status"] == "unavailable"
    assert not (lane / "evidence-plan.json").exists()
    assert not (lane / "evidence.json").exists()


def test_summary_target_publishes_cited_summary_with_earlier_products(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.app.evidence as evidence_app

    subtitle = tmp_path / "lecture.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nSource words.\n", encoding="utf-8"
    )
    config = tmp_path / "vctx.toml"
    config.write_text(
        """
[evidence]
planner = "instance:planner"
[summary]
use = "instance:writer"
language = "ja"
[instances.ai.planner]
base_url = "http://127.0.0.1:1234/v1"
model = "planner"
[instances.ai.writer]
base_url = "http://127.0.0.1:1234/v1"
model = "writer"
""".strip(),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_plan(_self: object, transcript: Transcript) -> EvidencePlan:
        return EvidencePlan(source_id=transcript.source_id)

    def fake_write(
        _self: object, packet: SummaryPacket, *, language: str = "native"
    ) -> SummaryOutcome:
        calls.append(language)
        summary = packet.admit(
            SummaryDraft(
                overview="要約",
                points=[
                    DraftPoint(
                        text="根拠付き要約",
                        basis="transcript",
                        segment_ids=["seg_000001"],
                    )
                ],
            ),
            language=language,
        )
        return SummaryOutcome(status="ready", summary=summary)

    monkeypatch.setattr(evidence_app.EvidencePlanner, "plan", fake_plan)
    monkeypatch.setattr(SummaryWriter, "write", fake_write)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "prepare",
            str(subtitle),
            "--out",
            str(out),
            "--config",
            str(config),
            "--to",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["ja"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    lane = out / source["path"]
    assert source["status"] == "ok"
    assert {item["product"] for item in source["outcomes"]} >= {
        "transcript",
        "evidence",
        "summary",
    }
    assert (lane / "transcript.json").exists()
    assert (lane / "evidence-plan.json").exists()
    assert json.loads((lane / "summary.json").read_text(encoding="utf-8"))["language"] == "ja"
