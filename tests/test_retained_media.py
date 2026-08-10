from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


def test_prepare_retains_local_media_as_manifest_artifact(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"self-contained-media")
    out = tmp_path / "pack"

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out)])

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    receipt = manifest["source_media"][0]
    assert not Path(receipt["path"]).is_absolute()
    assert ".." not in Path(receipt["path"]).parts
    source.unlink()
    relocated = tmp_path / "relocated-pack"
    out.rename(relocated)
    retained = relocated / receipt["path"]
    assert retained.read_bytes() == b"self-contained-media"
    assert receipt["retained"] is True
    assert receipt["bytes"] == len(b"self-contained-media")
    assert len(receipt["sha256"]) == 64
    assert receipt["path"] in {item["path"] for item in manifest["artifacts"]}


def test_prepare_can_explicitly_omit_local_media(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"size-sensitive-media")
    out = tmp_path / "pack"

    result = runner.invoke(
        app, ["prepare", str(source), "--out", str(out), "--no-retain-media"]
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    receipt = manifest["source_media"][0]
    assert receipt["retained"] is False
    assert receipt["path"] is None
    assert not (out / "media").exists()
    assert any(step["name"] == "source.media_retention" for step in manifest["steps"])
