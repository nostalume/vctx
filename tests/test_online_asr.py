from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.app.credentials import CredentialError, resolve_env_credential
from vctx.cli import app
from vctx.config import AsrInstanceConfig, CapabilityEnabled, CapabilityPolicy
from vctx.models import SourceRef
from vctx.models.media import LocalMediaAsset, MediaAsset
from vctx.net import NetRequest, NetResponse
from vctx.transcript import TranscriptPayload, TranscriptProvenance
from vctx.transforms.asr import AsrExecutionError, OpenAiCompatibleAsrAdapter, run_asr
from vctx.transforms.planning import SourceState, TransformEnvironment, plan_asr

runner = CliRunner()


class _FakeAsrNetRuntime:
    def __init__(self) -> None:
        self.requests: list[NetRequest] = []

    def request(self, request: NetRequest) -> NetResponse:
        self.requests.append(request)
        return NetResponse(
            url=request.url,
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "text": "hello online asr",
                    "segments": [
                        {"start": 0.0, "end": 1.2, "text": "hello online asr"},
                    ],
                }
            ).encode("utf-8"),
        )


class _FailingAsrNetRuntime:
    def request(self, request: NetRequest) -> NetResponse:
        del request
        raise OSError("network down with secret-token Authorization header")


def test_resolve_env_credential_reads_selected_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("IGNORED=value\nOPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    assert resolve_env_credential("OPENAI_API_KEY", env_files=[env_file]) == "from-dotenv"


def test_resolve_env_credential_prefers_shell_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    assert resolve_env_credential("OPENAI_API_KEY", env_files=[env_file]) == "from-shell"


def test_resolve_env_credential_reports_missing_without_secret(tmp_path: Path) -> None:
    with pytest.raises(CredentialError, match="OPENAI_API_KEY"):
        resolve_env_credential("OPENAI_API_KEY", env_files=[tmp_path / "missing.env"])


def test_openai_compatible_asr_posts_audio_through_injected_net_runtime(
    tmp_path: Path,
) -> None:
    runtime = _FakeAsrNetRuntime()
    media = _media_asset(tmp_path / "lecture.wav")
    media.local_path.write_bytes(b"fake audio")
    adapter = OpenAiCompatibleAsrAdapter(
        instance=AsrInstanceConfig(
            type="openai-compatible-audio",
            base_url="https://api.example.test/v1/audio/transcriptions",
            model="whisper-test",
            api_key_env="OPENAI_API_KEY",
        ),
        api_key="secret-token",
        provider_id="example-asr",
        net=runtime,
    )

    payload = adapter.transcribe(media)

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.method == "POST"
    assert request.url == "https://api.example.test/v1/audio/transcriptions"
    assert request.timeout_s == 120
    assert request.purpose == "asr_transcription"
    assert request.provider_id == "example-asr"
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert request.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert request.body is not None
    assert b'name="model"' in request.body
    assert b"whisper-test" in request.body
    assert b'name="file"' in request.body
    assert b"fake audio" in request.body
    assert payload.format == "vtt"
    assert payload.provenance.provider == "example-asr"
    assert "hello online asr" in payload.text


def test_openai_compatible_asr_errors_do_not_include_api_key(tmp_path: Path) -> None:
    media = _media_asset(tmp_path / "lecture.wav")
    media.local_path.write_bytes(b"fake audio")
    adapter = OpenAiCompatibleAsrAdapter(
        instance=AsrInstanceConfig(
            type="openai-compatible-audio",
            base_url="https://api.example.test/v1/audio/transcriptions",
            model="whisper-test",
            api_key_env="OPENAI_API_KEY",
        ),
        api_key="secret-token",
        provider_id="example-asr",
        net=_FailingAsrNetRuntime(),
    )

    with pytest.raises(AsrExecutionError) as exc_info:
        adapter.transcribe(media)

    message = str(exc_info.value)
    assert "online ASR request failed for provider example-asr: OSError" in message
    assert "network down" not in message
    assert "Authorization" not in message
    assert "secret-token" not in message


def test_prepare_local_media_can_use_configured_online_asr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.transforms.asr as asr_module

    calls: dict[str, object] = {}
    adapter_kwargs: dict[str, object] = {}

    class FakeOnlineAdapter:
        def __init__(self, **kwargs: object) -> None:
            adapter_kwargs.update(kwargs)

        def transcribe(self, media_asset: MediaAsset) -> TranscriptPayload:
            calls["media"] = media_asset.local_path
            return TranscriptPayload(
                text="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nOnline CLI ASR.\n",
                format="vtt",
                provenance=TranscriptProvenance(
                    method="asr", language=None, format="vtt", provider="online-test"
                ),
            )

    monkeypatch.setattr(asr_module, "OpenAiCompatibleAsrAdapter", FakeOnlineAdapter)
    media = tmp_path / "lecture.wav"
    media.write_bytes(b"fake audio")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    config_path = tmp_path / "vctx.toml"
    config_path.write_text(
        f'''
[runtime]
env_files = ["{env_file.as_posix()}"]

[transforms.asr]
instance = "online-test"

[instances.asr.online-test]
type = "openai-compatible-audio"
base_url = "https://api.example.test/v1/audio/transcriptions"
api_key_env = "OPENAI_API_KEY"
model = "whisper-test"
'''.strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["prepare", str(media), "--out", str(out_dir), "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.output
    assert calls["media"] == media
    assert adapter_kwargs["api_key"] == "from-dotenv"
    assert adapter_kwargs["provider_id"] == "online-test"
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["transform_evidence"][0]["selected_route"] == "configured-online"
    assert manifest["transform_evidence"][0]["uploaded"] is True
    assert manifest["transform_evidence"][0]["cost_may_apply"] is True
    assert "from-dotenv" not in json.dumps(manifest)


def test_run_asr_uses_ai_route_provider_identity_when_legacy_field_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.transforms.asr as asr_module

    calls: dict[str, object] = {}

    class FakeOnlineAdapter:
        def __init__(
            self,
            *,
            instance: AsrInstanceConfig,
            api_key: str,
            provider_id: str,
        ) -> None:
            del instance, api_key
            calls["provider_id"] = provider_id

        def transcribe(self, media_asset: MediaAsset) -> TranscriptPayload:
            calls["media"] = media_asset.local_path
            return TranscriptPayload(
                text="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nOnline ASR.\n",
                format="vtt",
                provenance=TranscriptProvenance(
                    method="asr", language=None, format="vtt", provider="online-test"
                ),
            )

    monkeypatch.setattr(asr_module, "OpenAiCompatibleAsrAdapter", FakeOnlineAdapter)
    media = _media_asset(tmp_path / "lecture.wav")
    media.local_path.write_bytes(b"fake audio")
    instance = AsrInstanceConfig(
        type="openai-compatible-audio",
        base_url="https://api.example.test/v1/audio/transcriptions",
        api_key_env="OPENAI_API_KEY",
        model="whisper-test",
    )
    route_plan = plan_asr(
        CapabilityPolicy(
            enabled=CapabilityEnabled.TRUE,
            allow_upload=True,
            model="whisper-test",
        ),
        TransformEnvironment(
            configured_asr=True,
            configured_asr_provider_id="online-test",
            configured_asr_model_id="whisper-test",
            configured_asr_cost_mode="paid",
        ),
        SourceState(has_transcript=False, has_media=True),
    )
    route_plan = route_plan.model_copy(update={"provider_id": None})

    run_asr(
        route_plan,
        media,
        instance=instance,
        cache_root=tmp_path,
        api_key="secret-token",
    )

    assert calls["provider_id"] == "online-test"


def _media_asset(path: Path) -> MediaAsset:
    return LocalMediaAsset(
        id="local__lecture",
        source=SourceRef(kind="file", value=str(path)),
        local_path=path,
        media_type="audio",
        container="wav",
        provider="local-file",
    )
