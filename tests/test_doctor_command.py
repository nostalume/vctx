from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vctx.asr import AsrReadinessFacts, decide_asr_readiness
from vctx.cli import app
from vctx.config import AsrInstanceConfig, CapabilityPolicy, ModelRefUse

runner = CliRunner()


def test_config_selection_is_single_ordered_and_doctor_is_read_only(
    monkeypatch, tmp_path: Path
) -> None:
    import vctx.config as config_module

    workspace = tmp_path / "workspace"
    environment = tmp_path / "environment" / "chosen.toml"
    global_root = tmp_path / "global"
    explicit = tmp_path / "explicit.toml"
    for path in (workspace / "vctx.toml", environment, global_root / "config.toml", explicit):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[cache]\nsource_dir='source'\nmodel_dir='models'\n", encoding="utf-8")
    (workspace / ".vctx.toml").write_text("invalid=true\n", encoding="utf-8")
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("VCTX_CONFIG", str(environment))
    monkeypatch.setattr(config_module, "user_config_path", lambda *_args, **_kwargs: global_root)

    select = config_module.select_config_file
    assert (select(explicit).origin, select(None).origin) == ("explicit", "workspace")
    (workspace / "vctx.toml").unlink()
    assert select(None).origin == "environment"
    monkeypatch.delenv("VCTX_CONFIG")
    assert select(None).origin == "global"
    monkeypatch.setenv("VCTX_CONFIG", str(environment))

    result = runner.invoke(app, ["doctor", "--json"])

    report = json.loads(result.output)
    assert report["config"] == {"origin": "environment", "path": str(environment)}
    assert report["cache"]["source_dir"] == str(environment.parent / "source")
    assert report["cache"]["model_dir"] == str(environment.parent / "models")
    assert report["cache"]["source_state"] == "missing" and not any(
        (environment.parent / name).exists() for name in ("source", "models")
    )
    override = json.loads(runner.invoke(app, ["doctor", "--cache-dir", "cli", "--json"]).output)[
        "cache"
    ]
    assert {Path(override[key]) for key in ("source_dir", "model_dir")} == {
        workspace / "cli" / name for name in ("source", "models")
    }
    text = runner.invoke(app, ["doctor"])
    assert text.exit_code == 0 and "config:" in text.output and "cache.source:" in text.output


def test_prompt_is_static_terse_agent_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["prompt"])

    assert result.exit_code == 0, result.output
    assert all(word in result.output for word in ("prepare", "manifest.json", "render", "verify"))
    assert "doctor --json" in result.output and "--help" in result.output
    assert len([line for line in result.output.splitlines() if line.strip()]) <= 20
    assert len(result.output.encode()) <= 1024 and not list(tmp_path.iterdir())


def test_doctor_admits_explicit_asr_model_path_without_writes(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "explicit-model"
    model.mkdir()
    (model / "model.bin").write_bytes(b"model")
    (model / "config.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / "unused-cache"
    monkeypatch.setattr("vctx.app.doctor.bundled_cuda_state", lambda: "missing")
    monkeypatch.setattr("vctx.app.doctor.package_version", lambda _name: "missing")

    result = runner.invoke(
        app,
        ["doctor", "--asr", f"path:{model}", "--cache-dir", str(cache), "--json"],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["capabilities"]["asr"]["readiness"] == "explicit-ready"
    assert report["capabilities"]["asr"]["runtime"] == {
        "requested_device": "auto",
        "compute_type": "auto",
        "package_state": "missing",
        "cuda_libraries": "missing",
    }
    assert not cache.exists()


def test_asr_readiness_decision_consumes_only_observed_facts() -> None:
    readiness = decide_asr_readiness(
        CapabilityPolicy(enabled=True, use=ModelRefUse(ref="local:small")),
        AsrInstanceConfig(type="local-faster-whisper", model="small", device="cuda"),
        AsrReadinessFacts(
            model_kind="managed",
            model_reference="small",
            model_state="incomplete",
            package_state="missing",
        ),
    )

    assert readiness.state == "managed-incomplete"
    assert readiness.runtime.package_state == "missing"
    assert readiness.runtime.requested_device == "cuda"


def test_doctor_reduces_inaccessible_keyring_to_presence_state(monkeypatch, tmp_path: Path) -> None:
    class BrokenKeyring:
        priority = 1

        def get_password(self, _service: str, _account: str) -> str | None:
            raise RuntimeError("backend leaked-secret-detail")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("vctx.app.doctor.system_keyring", lambda: BrokenKeyring())

    result = runner.invoke(app, ["doctor", "--to", "evidence", "--json"])

    assert result.exit_code == 0, result.output
    assert "backend leaked-secret-detail" not in result.output
    report = json.loads(result.output)
    assert report["capabilities"]["planner"]["readiness"] == "unavailable-keyring"
