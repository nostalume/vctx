from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()

def test_cache_status_missing_is_empty_and_read_only(tmp_path: Path) -> None:
    cache = tmp_path / "cache"

    result = runner.invoke(app, ["cache", "status", "--cache-dir", str(cache), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "records": 0,
        "assets": 0,
        "blobs": 0,
        "temporary": 0,
        "bytes": 0,
    }
    assert not cache.exists()

def test_cache_prune_dry_run_matches_real_orphan_cleanup_and_ignores_models(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    blobs = cache / "source" / "blobs"
    temporary = cache / "source" / "tmp"
    blobs.mkdir(parents=True)
    temporary.mkdir()
    (blobs / ("a" * 64)).write_bytes(b"blob")
    (temporary / "fetch.part").write_bytes(b"tmp")
    model = cache / "models" / "asr.bin"
    model.parent.mkdir()
    model.write_bytes(b"model")

    dry = runner.invoke(
        app, ["cache", "prune", "--cache-dir", str(cache), "--dry-run", "--json"]
    )
    real = runner.invoke(app, ["cache", "prune", "--cache-dir", str(cache), "--json"])

    assert dry.exit_code == real.exit_code == 0
    assert json.loads(dry.output) == {
        "dry_run": True,
        "examined": 2,
        "selected": 2,
        "removed": 0,
        "reclaimed_bytes": 7,
        "failures": [],
    }
    assert json.loads(real.output) == {
        **json.loads(dry.output),
        "dry_run": False,
        "removed": 2,
    }
    assert model.read_bytes() == b"model"
    assert not list(blobs.iterdir()) and not list(temporary.iterdir())

def test_cache_prune_rejects_invalid_age_and_unowned_blob_name(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    blobs = cache / "source" / "blobs"
    blobs.mkdir(parents=True)
    suspect = blobs / "outside"
    suspect.write_bytes(b"keep")

    age = runner.invoke(
        app, ["cache", "prune", "--cache-dir", str(cache), "--age", "30", "--json"]
    )
    prune = runner.invoke(app, ["cache", "prune", "--cache-dir", str(cache), "--json"])

    assert age.exit_code == 2
    assert "positive duration" in age.output
    assert prune.exit_code == 5
    assert suspect.read_bytes() == b"keep"

def test_cache_status_reports_corrupt_catalog_without_replacing_it(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    catalog = cache / "source" / "index.sqlite3"
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(b"not sqlite")

    result = runner.invoke(app, ["cache", "status", "--cache-dir", str(cache), "--json"])

    assert result.exit_code == 5
    assert catalog.read_bytes() == b"not sqlite"

def test_cache_prune_reports_deletion_failure_and_keeps_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    blob = cache / "source" / "blobs" / ("b" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"keep")
    unlink = Path.unlink

    def fail(path: Path, missing_ok: bool = False) -> None:
        if path == blob:
            raise PermissionError("in use")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail)
    result = runner.invoke(app, ["cache", "prune", "--cache-dir", str(cache), "--json"])

    receipt = json.loads(result.output)
    assert result.exit_code == 0
    assert receipt["selected"] == 1 and receipt["removed"] == 0
    assert receipt["reclaimed_bytes"] == 0 and receipt["failures"]
    assert blob.read_bytes() == b"keep"

def test_cache_selection_prefers_cli_base_over_config_source_dir(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    cli_cache = tmp_path / "cli"
    for root, digest, body in (
        (configured, "c" * 64, b"x"),
        (cli_cache / "source", "d" * 64, b"yy"),
    ):
        blob = root / "blobs" / digest
        blob.parent.mkdir(parents=True)
        blob.write_bytes(body)
    config = tmp_path / "vctx.toml"
    config.write_text("[cache]\nsource_dir = 'configured'\n", encoding="utf-8")

    from_config = runner.invoke(app, ["cache", "status", "--config", str(config), "--json"])
    from_cli = runner.invoke(
        app,
        ["cache", "status", "--config", str(config), "--cache-dir", str(cli_cache), "--json"],
    )

    assert json.loads(from_config.output)["bytes"] == 1
    assert json.loads(from_cli.output)["bytes"] == 2
