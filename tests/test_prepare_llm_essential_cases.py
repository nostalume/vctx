from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from vctx.cli import app
from vctx.models.knowledge_flow import (
    KnowledgeFlow,
    KnowledgeFlowSupplement,
    KnowledgeFlowSupplementEvidence,
)
from vctx.models.visual import (
    EssentialCaseSupplement,
    EssentialCaseSupplementEvidence,
    EssentialVisualCase,
)
from vctx.transcript import Transcript, TranscriptPayload, TranscriptProvenance

runner = CliRunner()


class FakeTextAdapter:
    essential_calls: list[Transcript] = []
    knowledge_calls: list[Transcript] = []

    def __init__(self, *, route: object, api_key: str, net: object | None = None) -> None:
        del route, net
        assert api_key == "from-dotenv"

    def essential_case_supplement(self, transcript: Transcript) -> EssentialCaseSupplement:
        self.essential_calls.append(transcript)
        return EssentialCaseSupplement(
            cases=[
                EssentialVisualCase(
                    segment_id="seg_000002",
                    timestamp_seconds=1.0,
                    case_type="screen_demo",
                    priority=0.8,
                    reason="model inferred screen demo cue",
                    actions=["ocr", "capture"],
                )
            ],
            evidence=EssentialCaseSupplementEvidence(
                route_provider_id="openrouter",
                route_model="test/text-model",
                source_segment_ids=["seg_000002"],
            ),
        )

    def knowledge_flow_supplement(self, transcript: Transcript) -> KnowledgeFlowSupplement:
        self.knowledge_calls.append(transcript)
        return KnowledgeFlowSupplement(
            flow=KnowledgeFlow(nodes=[], edges=[]),
            evidence=KnowledgeFlowSupplementEvidence(
                route_provider_id="openrouter",
                route_model="test/text-model",
                source_segment_ids=["seg_000002"],
            ),
        )


def test_prepare_visual_uses_llm_essential_cases_as_sampling_anchors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vctx.app.prepare as prepare_module
    import vctx.transforms.asr as asr_module
    import vctx.transforms.visual_frames as visual_frames_module
    from vctx.models.visual import FrameAsset
    from vctx.transforms.visual_planning import Evidence, VisualAction

    FakeTextAdapter.essential_calls = []
    FakeTextAdapter.knowledge_calls = []
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"fake mp4 bytes")
    config = tmp_path / "vctx.toml"
    env_path = tmp_path / ".env"
    env_path.write_text("OPENROUTER_API_KEY=from-dotenv", encoding="utf-8")
    config.write_text(
        """
[runtime]
env_files = [".env"]

[transforms.asr]
instance = "local-default"

[transforms.knowledge_flow]
enabled = "true"
route = "configured-online"
model = "openrouter:test/text-model"

[instances.asr.local-default]
type = "local-faster-whisper"
model = "tiny"
cache = "persistent"
""".strip(),
        encoding="utf-8",
    )
    seen_cases: list[dict[str, Any]] = []

    def fake_transcribe(self: object, media_asset: object) -> TranscriptPayload:
        del self, media_asset
        return TranscriptPayload(
            text=(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:02.000\n"
                "Plain setup narration.\n\n"
                "00:00:02.000 --> 00:00:04.000\n"
                "Look at this for the workflow.\n\n"
                "00:00:04.000 --> 00:00:06.000\n"
                "Plain follow-up narration.\n"
            ),
            format="vtt",
            provenance=TranscriptProvenance(
                method="asr",
                language="en",
                format="vtt",
                provider="faster-whisper",
            ),
        )

    def fake_extract_frames(
        media_asset: object,
        sample_action: VisualAction,
        frames_dir: Path,
    ) -> list[FrameAsset]:
        del media_asset
        seen_cases.extend(
            case.model_dump(mode="json") for case in sample_action.params.cases
        )
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / "frame-0001.png"
        frame_path.write_bytes(b"fake png bytes")
        return [
            FrameAsset(
                id="frame-0001",
                timestamp_seconds=1.0,
                path=frame_path,
                source="transcript_anchor",
                evidence=[Evidence(kind="transcript", name="screen-demo", weight=0.8)],
            )
        ]

    monkeypatch.setattr(asr_module.FasterWhisperAsrAdapter, "transcribe", fake_transcribe)
    monkeypatch.setattr(visual_frames_module, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(prepare_module, "OpenAiCompatibleTextAdapter", FakeTextAdapter)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "prepare",
            str(media),
            "--out",
            str(out_dir),
            "--config",
            str(config),
            "--workflow",
            "visual",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [(case["segment_id"], case["case_type"]) for case in seen_cases] == [
        ("seg_000002", "screen_demo")
    ]
    assert len(FakeTextAdapter.essential_calls) == 1
    assert [segment.id for segment in FakeTextAdapter.essential_calls[0].segments] == [
        "seg_000001",
        "seg_000002",
        "seg_000003",
    ]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert _step_status(manifest, "visual_cases.llm_extract") == "ok"
    assert any(
        evidence["capability"] == "essential_cases"
        for evidence in manifest["transform_evidence"]
    )


def _step_status(manifest: dict[str, Any], name: str) -> str:
    steps = manifest["steps"]
    assert isinstance(steps, list)
    for raw_step in steps:
        assert isinstance(raw_step, dict)
        step = cast(dict[str, Any], raw_step)
        if step["name"] == name:
            status = step["status"]
            assert isinstance(status, str)
            return status
    raise AssertionError(f"missing manifest step: {name}")
