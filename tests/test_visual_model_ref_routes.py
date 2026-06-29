from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from vctx.config import CapabilityEnabled, CapabilityPolicy, VisionInstanceConfig
from vctx.models import SourceRef
from vctx.models.media import LocalMediaAsset, MediaAsset
from vctx.models.visual import FrameAsset
from vctx.transforms.ai_routes import AiRoute
from vctx.transforms.model_resolution import (
    ModelRef,
    OpenRouterModel,
    OpenRouterModelArchitecture,
    OpenRouterModelPricing,
    ResolvedModelRoute,
)
from vctx.transforms.visual_execute import run_visual_context
from vctx.transforms.visual_planning import (
    Evidence,
    VisualAction,
    VisualAssessment,
)
from vctx.transforms.visual_routes import discover_visual_actions
from vctx.transforms.visual_vlm import VlmOutcome


def test_openrouter_prefix_model_discovers_describe_operation_without_provider_block() -> None:
    operations = discover_visual_actions(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            model="openrouter:nex-agi/nex-n2-pro:free",
            allow_network=True,
            allow_upload=True,
        ),
        env={"OPENROUTER_API_KEY": "present"},
    )

    describe = next(operation for operation in operations if operation.name == "describe")

    assert describe.route == "free-online"
    assert describe.provider_id == "openrouter:nex-agi/nex-n2-pro:free"
    assert describe.ai_route is not None
    assert describe.ai_route.task == "vision_description"
    assert describe.ai_route.selected == "free-online"
    assert describe.ai_route.provider == "openrouter"
    assert describe.ai_route.model == "nex-agi/nex-n2-pro:free"
    assert describe.ai_route.base_url == "https://openrouter.ai/api/v1/chat/completions"
    assert describe.ai_route.api_key_env == "OPENROUTER_API_KEY"


def test_configured_online_allows_explicit_paid_openrouter_model_without_extra_gate() -> None:
    operations = discover_visual_actions(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            route="configured-online",
            model="openrouter:anthropic/claude-sonnet-4",
            allow_network=True,
            allow_upload=True,
        ),
        env={"OPENROUTER_API_KEY": "present"},
    )

    describe = next(operation for operation in operations if operation.name == "describe")

    assert describe.route == "configured-online"
    assert describe.provider_id == "openrouter:anthropic/claude-sonnet-4"
    assert describe.ai_route is not None
    assert describe.ai_route.cost == "paid"


def test_free_online_rejects_explicit_paid_openrouter_model() -> None:
    operations = discover_visual_actions(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            route="free-online",
            model="openrouter:anthropic/claude-sonnet-4",
            allow_network=True,
            allow_upload=True,
        ),
        env={"OPENROUTER_API_KEY": "present"},
    )

    assert all(operation.name != "describe" for operation in operations)


def test_auto_model_discovers_free_openrouter_vlm_from_registry_metadata() -> None:
    operations = discover_visual_actions(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            model="auto",
            allow_network=True,
            allow_upload=True,
        ),
        env={"OPENROUTER_API_KEY": "present"},
        openrouter_models=[
            OpenRouterModel(
                id="text-only/free:free",
                architecture=OpenRouterModelArchitecture(
                    input_modalities=["text"], output_modalities=["text"]
                ),
                pricing=OpenRouterModelPricing(prompt="0", completion="0"),
            ),
            OpenRouterModel(
                id="nex-agi/nex-n2-pro:free",
                architecture=OpenRouterModelArchitecture(
                    input_modalities=["text", "image"], output_modalities=["text"]
                ),
                pricing=OpenRouterModelPricing(prompt="0", completion="0"),
            ),
        ],
    )

    describe = next(operation for operation in operations if operation.name == "describe")

    assert describe.provider_id == "openrouter:nex-agi/nex-n2-pro:free"
    assert describe.ai_route is not None
    assert describe.ai_route.cost == "free"
    assert describe.ai_route.task == "vision_description"
    assert describe.ai_route.model == "nex-agi/nex-n2-pro:free"


def test_vision_instance_still_takes_precedence_over_prefix_auto() -> None:
    operations = discover_visual_actions(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            route="configured-online",
            preferred_provider="test-vlm",
            model="auto",
            allow_network=True,
            allow_upload=True,
        ),
        vision_instance_configs={
            "test-vlm": VisionInstanceConfig(
                type="openai-compatible-vision",
                base_url="https://example.invalid/v1/chat/completions",
                api_key_env="VISION_KEY",
                model="vision-test",
            )
        },
        env={"OPENROUTER_API_KEY": "present"},
    )

    describe = next(operation for operation in operations if operation.name == "describe")

    assert describe.provider_id == "test-vlm"
    assert describe.ai_route is not None
    assert describe.ai_route.task == "vision_description"
    assert describe.ai_route.selected == "configured-online"
    assert describe.ai_route.provider == "alias"
    assert describe.ai_route.provider_id == "test-vlm"
    assert describe.ai_route.model == "vision-test"
    assert describe.ai_route.base_url == "https://example.invalid/v1/chat/completions"


def test_injected_ai_route_discovers_visual_describe_without_env_or_registry() -> None:
    operations = discover_visual_actions(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            model="auto",
            allow_network=True,
            allow_upload=True,
        ),
        ai_routes=[
            _openrouter_vision_route(reason="pre-resolved by app boundary")
        ],
    )

    describe = next(operation for operation in operations if operation.name == "describe")

    assert describe.route == "free-online"
    assert describe.provider_id == "openrouter:nex-agi/nex-n2-pro:free"
    assert describe.ai_route is not None
    assert describe.ai_route.reason == "pre-resolved by app boundary"


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


def test_visual_execution_materializes_prefix_resolved_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.transforms.visual_frames as visual_frames_module
    import vctx.transforms.visual_vlm as visual_vlm_module

    media_path = tmp_path / "lecture.mp4"
    media_path.write_bytes(b"fake video")
    frame_path = tmp_path / "visual" / "frames" / "frame-0001.png"
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-token\n", encoding="utf-8")

    def fake_extract_frames(
        media_asset: MediaAsset,
        action: VisualAction,
        frames_dir: Path,
    ) -> list[FrameAsset]:
        del media_asset, action
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"fake png")
        return [
            FrameAsset(
                id="frame-0001",
                timestamp_seconds=12.0,
                path=frame_path,
                source="transcript_anchor",
                evidence=[Evidence(kind="transcript", name="diagram", weight=0.9)],
            )
        ]

    seen: dict[str, str | None] = {}

    def fake_describe_many(self: object, frames: list[FrameAsset]) -> list[VlmOutcome]:
        del frames
        adapter = cast(Any, self)
        seen["base_url"] = adapter.route.base_url
        seen["api_key"] = adapter.api_key
        seen["model"] = adapter.route.model
        return [VlmOutcome(frame_id="frame-0001", text="A remote VLM description.")]

    monkeypatch.setattr(visual_frames_module, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        visual_vlm_module.OpenAiCompatibleVisionAdapter,
        "describe_many",
        fake_describe_many,
    )

    records = run_visual_context(
        VisualAssessment(
            visual_yield=0.8,
            audio_sufficiency=0.2,
            rationale="test",
            recipe=[
                VisualAction.sample(strategy="cover", budget=1),
                VisualAction.describe(
                    AiRoute.configured_alias(
                        task="vision_description",
                        selected="free-online",
                        provider_id="openrouter",
                        model="nex-agi/nex-n2-pro:free",
                        base_url="https://openrouter.ai/api/v1/chat/completions",
                        api_key_env="OPENROUTER_API_KEY",
                        cost="free",
                        upload="required",
                        reason="resolved from OpenRouter model reference",
                    )
                ),
            ],
        ),
        LocalMediaAsset(
            id="media-1",
            source=SourceRef(kind="file", value=str(media_path)),
            local_path=media_path,
            media_type="video",
        ),
        tmp_path,
        env_files=[env_file],
    )

    assert records.records[0].text == "A remote VLM description."
    assert seen == {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": "test-token",
        "model": "nex-agi/nex-n2-pro:free",
    }


def test_visual_execution_rejects_legacy_string_vlm_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.transforms.visual_frames as visual_frames_module
    import vctx.transforms.visual_vlm as visual_vlm_module

    media_path = tmp_path / "lecture.mp4"
    media_path.write_bytes(b"fake video")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-token\n", encoding="utf-8")

    def fake_extract_frames(
        media_asset: MediaAsset,
        action: VisualAction,
        frames_dir: Path,
    ) -> list[FrameAsset]:
        del media_asset, action
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / "frame-0001.png"
        frame_path.write_bytes(b"fake png")
        return [
            FrameAsset(
                id="frame-0001",
                timestamp_seconds=10.0,
                path=frame_path,
                source="transcript_anchor",
                evidence=[Evidence(kind="transcript", name="diagram", weight=0.9)],
            )
        ]

    def fake_describe_many(self: object, frames: list[FrameAsset]) -> list[str]:
        del self, frames
        return ["legacy raw string description"]

    monkeypatch.setattr(visual_frames_module, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        visual_vlm_module.OpenAiCompatibleVisionAdapter,
        "describe_many",
        fake_describe_many,
        raising=False,
    )

    with pytest.raises(AttributeError):
        run_visual_context(
            VisualAssessment(
                visual_yield=0.8,
                audio_sufficiency=0.2,
                rationale="test",
                recipe=[
                    VisualAction.sample(strategy="cover", budget=1),
                    VisualAction.describe(
                        AiRoute.configured_alias(
                            task="vision_description",
                            selected="free-online",
                            provider_id="openrouter",
                            model="nex-agi/nex-n2-pro:free",
                            base_url="https://openrouter.ai/api/v1/chat/completions",
                            api_key_env="OPENROUTER_API_KEY",
                            cost="free",
                            upload="required",
                            reason="resolved from OpenRouter model reference",
                        )
                    ),
                ],
            ),
            LocalMediaAsset(
                id="media-1",
                source=SourceRef(kind="file", value=str(media_path)),
                local_path=media_path,
                media_type="video",
            ),
            tmp_path,
            env_files=[env_file],
        )


def test_visual_execution_batches_descriptions_through_describe_many(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.transforms.visual_frames as visual_frames_module
    import vctx.transforms.visual_vlm as visual_vlm_module

    media_path = tmp_path / "lecture.mp4"
    media_path.write_bytes(b"fake video")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-token\n", encoding="utf-8")

    def fake_extract_frames(
        media_asset: MediaAsset,
        action: VisualAction,
        frames_dir: Path,
    ) -> list[FrameAsset]:
        del media_asset, action
        frames_dir.mkdir(parents=True, exist_ok=True)
        first = frames_dir / "frame-0001.png"
        second = frames_dir / "frame-0002.png"
        first.write_bytes(b"first png")
        second.write_bytes(b"second png")
        return [
            FrameAsset(
                id="frame-0001",
                timestamp_seconds=10.0,
                path=first,
                source="transcript_anchor",
                evidence=[Evidence(kind="transcript", name="diagram", weight=0.9)],
            ),
            FrameAsset(
                id="frame-0002",
                timestamp_seconds=20.0,
                path=second,
                source="transcript_anchor",
                evidence=[Evidence(kind="transcript", name="summary", weight=0.8)],
            ),
        ]

    seen_batches: list[list[str]] = []

    def fail_single_describe(self: object, frame: FrameAsset) -> str:
        del self, frame
        raise AssertionError("executor should call describe_many, not per-frame describe")

    def fake_describe_many(self: object, frames: list[FrameAsset]) -> list[VlmOutcome]:
        del self
        seen_batches.append([frame.id for frame in frames])
        return [
            VlmOutcome(frame_id="frame-0001", text="First batched description."),
            VlmOutcome(frame_id="frame-0002", text="Second batched description."),
        ]

    monkeypatch.setattr(visual_frames_module, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        visual_vlm_module.OpenAiCompatibleVisionAdapter,
        "describe",
        fail_single_describe,
    )
    monkeypatch.setattr(
        visual_vlm_module.OpenAiCompatibleVisionAdapter,
        "describe_many",
        fake_describe_many,
        raising=False,
    )

    records = run_visual_context(
        VisualAssessment(
            visual_yield=0.8,
            audio_sufficiency=0.2,
            rationale="test",
            recipe=[
                VisualAction.sample(strategy="cover", budget=2),
                VisualAction.describe(
                    AiRoute.configured_alias(
                        task="vision_description",
                        selected="free-online",
                        provider_id="openrouter",
                        model="nex-agi/nex-n2-pro:free",
                        base_url="https://openrouter.ai/api/v1/chat/completions",
                        api_key_env="OPENROUTER_API_KEY",
                        cost="free",
                        upload="required",
                        reason="resolved from OpenRouter model reference",
                    )
                ),
            ],
        ),
        LocalMediaAsset(
            id="media-1",
            source=SourceRef(kind="file", value=str(media_path)),
            local_path=media_path,
            media_type="video",
        ),
        tmp_path,
        env_files=[env_file],
    )

    assert seen_batches == [["frame-0001", "frame-0002"]]
    assert [record.text for record in records.records] == [
        "First batched description.",
        "Second batched description.",
    ]


def test_visual_execution_keeps_capture_for_failed_vlm_outcomes_without_single_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.transforms.visual_frames as visual_frames_module
    import vctx.transforms.visual_vlm as visual_vlm_module

    media_path = tmp_path / "lecture.mp4"
    media_path.write_bytes(b"fake video")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-token\n", encoding="utf-8")

    def fake_extract_frames(
        media_asset: MediaAsset,
        action: VisualAction,
        frames_dir: Path,
    ) -> list[FrameAsset]:
        del media_asset, action
        frames_dir.mkdir(parents=True, exist_ok=True)
        first = frames_dir / "frame-0001.png"
        second = frames_dir / "frame-0002.png"
        first.write_bytes(b"first png")
        second.write_bytes(b"second png")
        return [
            FrameAsset(
                id="frame-0001",
                timestamp_seconds=10.0,
                path=first,
                source="transcript_anchor",
                evidence=[Evidence(kind="transcript", name="diagram", weight=0.9)],
            ),
            FrameAsset(
                id="frame-0002",
                timestamp_seconds=20.0,
                path=second,
                source="transcript_anchor",
                evidence=[Evidence(kind="transcript", name="summary", weight=0.8)],
            ),
        ]

    seen_batches: list[list[str]] = []

    def fail_single_describe(self: object, frame: FrameAsset) -> str:
        del self, frame
        raise AssertionError("failed VLM outcomes must not call per-frame describe fallback")

    def fake_describe_many(self: object, frames: list[FrameAsset]) -> list[VlmOutcome]:
        del self
        seen_batches.append([frame.id for frame in frames])
        return [
            VlmOutcome(frame_id="frame-0001", text="First description."),
            VlmOutcome(frame_id="frame-0002", error="fixture VLM failure"),
        ]

    monkeypatch.setattr(visual_frames_module, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        visual_vlm_module.OpenAiCompatibleVisionAdapter,
        "describe",
        fail_single_describe,
    )
    monkeypatch.setattr(
        visual_vlm_module.OpenAiCompatibleVisionAdapter,
        "describe_many",
        fake_describe_many,
        raising=False,
    )

    records = run_visual_context(
        VisualAssessment(
            visual_yield=0.8,
            audio_sufficiency=0.2,
            rationale="test",
            recipe=[
                VisualAction.sample(strategy="cover", budget=2),
                VisualAction.describe(
                    AiRoute.configured_alias(
                        task="vision_description",
                        selected="free-online",
                        provider_id="openrouter",
                        model="nex-agi/nex-n2-pro:free",
                        base_url="https://openrouter.ai/api/v1/chat/completions",
                        api_key_env="OPENROUTER_API_KEY",
                        cost="free",
                        upload="required",
                        reason="resolved from OpenRouter model reference",
                    )
                ),
                VisualAction.capture(),
            ],
        ),
        LocalMediaAsset(
            id="media-1",
            source=SourceRef(kind="file", value=str(media_path)),
            local_path=media_path,
            media_type="video",
        ),
        tmp_path,
        env_files=[env_file],
    )

    assert seen_batches == [["frame-0001", "frame-0002"]]
    assert [(record.kind, record.frame_id, record.text) for record in records.records] == [
        ("description", "frame-0001", "First description."),
        ("capture", "frame-0001", None),
        ("capture", "frame-0002", None),
    ]
