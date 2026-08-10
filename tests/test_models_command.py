from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


def test_models_status_is_network_free_and_machine_readable(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["models", "status", "--cache-dir", str(tmp_path), "--json"]
    )

    assert result.exit_code == 0, result.output
    records = json.loads(result.output)
    assert [(item["capability"], item["state"]) for item in records] == [
        ("asr", "missing"),
        ("ocr", "missing"),
    ]
    assert all(not Path(item["cache_path"]).is_absolute() for item in records)


def test_models_pull_uses_adapter_boundary_and_verify_detects_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.app.models as models_module

    def fake_pull(capability: str, model_id: str, cache_root: Path) -> Path:
        model_dir = cache_root / "models" / capability / model_id
        model_dir.mkdir(parents=True)
        (model_dir / "weights.bin").write_bytes(b"prepared")
        return model_dir

    monkeypatch.setattr(models_module, "_pull_model", fake_pull)
    pull = runner.invoke(
        app,
        ["models", "pull", "asr", "ocr", "--cache-dir", str(tmp_path), "--asr", "local:tiny"],
    )
    assert pull.exit_code == 0, pull.output
    assert "asr: ready" in pull.output
    assert "ocr: ready" in pull.output

    verify = runner.invoke(
        app,
        [
            "models", "verify", "asr", "ocr", "--cache-dir", str(tmp_path),
            "--asr", "local:tiny", "--json",
        ],
    )
    assert [item["state"] for item in json.loads(verify.output)] == ["ready", "ready"]

    (tmp_path / "models" / "asr" / "tiny" / "weights.bin").write_bytes(b"corrupt")
    corrupt = runner.invoke(
        app,
        [
            "models", "verify", "asr", "--cache-dir", str(tmp_path),
            "--asr", "local:tiny", "--json",
        ],
    )
    assert json.loads(corrupt.output)[0]["state"] == "corrupt"
