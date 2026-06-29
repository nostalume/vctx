from __future__ import annotations

from pathlib import Path

from vctx.config import PrepareRequest, resolve_config
from vctx.transforms.planning import SourceState, TransformEnvironment, plan_asr


def test_cache_dir_defaults_to_platform_persistent_cache(tmp_path: Path) -> None:
    resolved = resolve_config(PrepareRequest(input="lecture.srt", out_dir=tmp_path / "out"))

    assert resolved.runtime.cache_dir is not None
    assert resolved.runtime.cache_dir.name == "Cache"
    assert resolved.runtime.cache_dir.parent.name == "vctx"


def test_instance_config_separates_local_and_online_asr(tmp_path: Path) -> None:
    config_path = tmp_path / "vctx.toml"
    config_path.write_text(
        """
[runtime]
env_files = [".env"]

[transforms.asr]
instance = "local-default"

[instances.asr.local-default]
type = "local-faster-whisper"
model = "models/faster-whisper-tiny"
cache = "persistent"

[instances.asr.openai-whisper]
type = "openai-compatible-audio"
base_url = "https://api.openai.com/v1/audio/transcriptions"
api_key_env = "OPENAI_API_KEY"
model = "whisper-1"
""".strip(),
        encoding="utf-8",
    )

    resolved = resolve_config(
        PrepareRequest(input="lecture.mp4", out_dir=tmp_path / "out", config_path=config_path)
    )

    assert resolved.runtime.env_files == [tmp_path / ".env"]
    assert resolved.transforms.asr.instance == "local-default"
    assert resolved.instances.asr["local-default"].type == "local-faster-whisper"
    assert resolved.instances.asr["local-default"].model == str(
        tmp_path / "models" / "faster-whisper-tiny"
    )
    assert resolved.instances.asr["local-default"].cache == "persistent"
    assert resolved.instances.asr["openai-whisper"].type == "openai-compatible-audio"
    assert resolved.instances.asr["openai-whisper"].api_key_env == "OPENAI_API_KEY"


def test_explicit_online_asr_instance_is_paid_and_upload_evidence(tmp_path: Path) -> None:
    request = PrepareRequest(input="lecture.mp4", out_dir=tmp_path / "out")
    resolved = resolve_config(request)
    policy = resolved.transforms.asr.model_copy(update={"instance": "openai-whisper"})
    environment = TransformEnvironment(
        configured_asr=True,
        configured_asr_provider_id="openai-whisper",
        configured_asr_model_id="whisper-1",
        configured_asr_cost_mode="paid",
    )

    plan = plan_asr(policy, environment, SourceState(has_transcript=False, has_media=True))

    assert plan.selected == "configured-online"
    assert plan.provider_id == "openai-whisper"
    assert plan.model_id == "whisper-1"
    assert plan.evidence_seed.uploaded is True
    assert plan.evidence_seed.cost_may_apply is True
