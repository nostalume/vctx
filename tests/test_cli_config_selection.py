from __future__ import annotations

from pathlib import Path

import pytest

from vctx.cli import _select_config_path


def test_cli_config_selection_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.toml"

    assert _select_config_path(explicit) == explicit


def test_cli_config_selection_prefers_local_vctx_over_env_and_global(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local = tmp_path / "vctx.toml"
    env_config = tmp_path / "env.toml"
    global_config = tmp_path / "global" / "config.toml"
    for path in (local, env_config, global_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VCTX_CONFIG", str(env_config))
    monkeypatch.setattr("vctx.cli._global_config_path", lambda: global_config)

    assert _select_config_path(None) == local


def test_cli_config_selection_uses_dot_vctx_when_vctx_toml_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dot_local = tmp_path / ".vctx.toml"
    dot_local.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VCTX_CONFIG", raising=False)
    monkeypatch.setattr("vctx.cli._global_config_path", lambda: tmp_path / "missing.toml")

    assert _select_config_path(None) == dot_local


def test_cli_config_selection_uses_env_when_no_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_config = tmp_path / "env.toml"
    env_config.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VCTX_CONFIG", str(env_config))
    monkeypatch.setattr("vctx.cli._global_config_path", lambda: tmp_path / "missing.toml")

    assert _select_config_path(None) == env_config


def test_cli_config_selection_uses_global_when_no_local_or_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    global_config = tmp_path / "global" / "config.toml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VCTX_CONFIG", raising=False)
    monkeypatch.setattr("vctx.cli._global_config_path", lambda: global_config)

    assert _select_config_path(None) == global_config


def test_cli_config_selection_returns_none_when_no_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VCTX_CONFIG", raising=False)
    monkeypatch.setattr("vctx.cli._global_config_path", lambda: tmp_path / "missing.toml")

    assert _select_config_path(None) is None
