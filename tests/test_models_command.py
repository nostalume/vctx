from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import vctx.model_store as model_store
from vctx.cli import app

runner = CliRunner()


def test_models_status_is_network_free_and_machine_readable(tmp_path: Path) -> None:
    result = runner.invoke(app, ["models", "status", "--cache-dir", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    records = json.loads(result.output)
    assert [(item["capability"], item["state"]) for item in records] == [
        ("asr", "missing"),
        ("ocr", "missing"),
    ]
    assert records[0]["cache_path"] == "asr/small"
    assert all(not Path(item["cache_path"]).is_absolute() for item in records)


def test_models_pull_maps_options_and_renders_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def pull(capabilities: list[str] | None, **options: object) -> list[model_store.ModelReceipt]:
        observed.update(capabilities=capabilities, **options)
        return [
            model_store.ModelReceipt(
                capability="asr",
                provider="faster-whisper",
                model_id="tiny",
                state="ready",
                cache_path="generations/asr/model/generation",
                bytes=8,
                package_version="test",
            )
        ]

    monkeypatch.setattr(model_store, "pull_models", pull)
    result = runner.invoke(
        app,
        [
            "models",
            "pull",
            "asr",
            "--cache-dir",
            str(tmp_path),
            "--asr",
            "local:tiny",
            "--conservative",
            "--refresh",
            "--max-runtime",
            "90",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.startswith("asr: ready")
    assert observed["capabilities"] == ["asr"]
    assert observed["asr_model_id"] == "tiny"
    assert observed["conservative"] is True
    assert observed["refresh"] is True
    assert observed["max_runtime"] == 90


def test_models_verify_and_prune_render_public_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = model_store.ModelReceipt(
        capability="asr",
        provider="faster-whisper",
        model_id="small",
        state="corrupt",
        cache_path="generations/asr/model/generation",
        bytes=8,
        package_version="test",
    )
    monkeypatch.setattr(model_store, "verify_models", lambda *_args, **_kwargs: [receipt])
    monkeypatch.setattr(
        model_store,
        "prune_model_cache",
        lambda *_args, **_kwargs: model_store.ModelPruneReport(
            dry_run=True, incomplete=[".incomplete/asr/model"], generations=[], bytes=8
        ),
    )

    verify = runner.invoke(app, ["models", "verify", "asr", "--cache-dir", str(tmp_path), "--json"])
    prune = runner.invoke(
        app,
        ["models", "prune", "--incomplete", "--dry-run", "--cache-dir", str(tmp_path)],
    )
    missing_mode = runner.invoke(app, ["models", "prune", "--cache-dir", str(tmp_path)])

    assert verify.exit_code == 0
    assert json.loads(verify.output)[0]["state"] == "corrupt"
    assert prune.exit_code == 0
    assert prune.output == "would prune 1 model path(s), 8 bytes\n"
    assert missing_mode.exit_code == 2
    assert "choose --incomplete and/or --unreferenced" in missing_mode.output
