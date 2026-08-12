from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vctx.cli import app

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
    override = json.loads(
        runner.invoke(app, ["doctor", "--cache-dir", "cli", "--json"]).output
    )["cache"]
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
