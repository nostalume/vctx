from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from vctx.artifact.manifest import InputSourceAsset, Manifest, source_key
from vctx.cli import app

runner = CliRunner()


def test_prepare_retains_local_media_as_manifest_artifact(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"self-contained-media")
    out = tmp_path / "pack"

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out)])

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "2"
    UUID(manifest["pack_id"])
    UUID(manifest["updated_run_id"])
    source_entry = manifest["sources"][0]
    assert "input" not in manifest
    assert "source_media" not in manifest
    assert source_entry["id"].startswith("local__")
    assert str(source) not in json.dumps(manifest)
    receipt = source_entry["assets"][0]
    assert receipt["kind"] == "input"
    assert receipt["path"] == "input.mp4"
    assert not Path(receipt["path"]).is_absolute()
    assert ".." not in Path(receipt["path"]).parts
    source.unlink()
    relocated = tmp_path / "relocated-pack"
    out.rename(relocated)
    retained = relocated / source_entry["path"] / receipt["path"]
    assert retained.read_bytes() == b"self-contained-media"
    assert receipt["retained"] is True
    assert receipt["bytes"] == len(b"self-contained-media")
    assert len(receipt["sha256"]) == 64
    assert receipt["path"] not in {item["path"] for item in source_entry["artifacts"]}
    assert [(item["operation"], item["status"]) for item in source_entry["effects"]] == [
        ("observe", "succeeded"),
        ("subtitle", "failed"),
        ("media", "succeeded"),
    ]


def test_prepare_can_explicitly_omit_local_media(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"size-sensitive-media")
    out = tmp_path / "pack"

    result = runner.invoke(
        app, ["prepare", str(source), "--out", str(out), "--no-retain-media"]
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    source_entry = manifest["sources"][0]
    lane = out / source_entry["path"]
    receipt = source_entry["assets"][0]
    assert receipt["kind"] == "input"
    assert receipt["retained"] is False
    assert receipt["path"] is None
    assert not (lane / "input.mp4").exists()
    assert any(step["name"] == "source.asset_retention" for step in source_entry["steps"])


def test_source_asset_integrity_failure_writes_error_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    import vctx.app.media_retention as retention

    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"media")
    out = tmp_path / "pack"
    digests = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(retention, "_sha256", lambda _path: next(digests))

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out)])

    assert result.exit_code == 5
    manifest = Manifest.model_validate_json((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.status == "error"
    source_entry = manifest.sources[0]
    assert source_entry.assets == []
    assert not (out / source_entry.path / "input.mp4").exists()


@pytest.mark.parametrize("path", ["../input.mp4", "/media/input.mp4", "media/../input.mp4"])
def test_manifest_rejects_source_asset_traversal(path: str) -> None:
    with pytest.raises(ValidationError, match="direct source-lane child"):
        InputSourceAsset(
            retained=True,
            path=path,
            media_type="video/mp4",
            bytes=1,
            sha256="a" * 64,
            selected_format="mp4",
        )


def test_source_key_is_portable_and_disambiguates_truncation_collision() -> None:
    first_id = f"Provider__{'a' * 40}x"
    second_id = f"provider__{'a' * 40}y"
    first = source_key(first_id)
    second = source_key(second_id, {first.casefold(): first_id})

    assert first == first.lower()
    assert first != second
    assert len(second) <= 64
    assert source_key("CON__device") != "con"
