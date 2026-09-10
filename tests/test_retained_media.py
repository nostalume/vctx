from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vctx.artifact.manifest import Manifest, source_key
from vctx.cli import app

runner = CliRunner()


def test_prepare_retains_relocatable_local_media_once(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"self-contained-media")
    out = tmp_path / "pack"
    read_bytes = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            (_ for _ in ()).throw(AssertionError("media was buffered"))
            if path == source
            else read_bytes(path)
        ),
    )

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out)])

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["sources"][0]
    retained = next(item for item in entry["artifacts"] if item["kind"] == "media")
    assert (
        retained["path"] == "media.mp4"
        and [item["path"] for item in entry["artifacts"]].count(retained["path"]) == 1
    )
    assert str(source) not in json.dumps(manifest)
    source.unlink()
    relocated = tmp_path / "relocated-pack"
    out.rename(relocated)
    assert (relocated / entry["path"] / retained["path"]).read_bytes() == b"self-contained-media"


def test_prepare_can_explicitly_omit_local_media(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"size-sensitive-media")
    out = tmp_path / "pack"

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out), "--no-retain-media"])

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["sources"][0]
    outcome = next(item for item in entry["outcomes"] if item["product"] == "retained-media")
    assert outcome["status"] == "unavailable"
    assert not any(item["kind"] == "media" for item in entry["artifacts"])


def test_retention_integrity_failure_publishes_no_mixed_lane(monkeypatch, tmp_path: Path) -> None:
    import vctx.artifact.bundle as retention
    from vctx.errors import CacheError

    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"media")
    out = tmp_path / "pack"
    monkeypatch.setattr(
        retention, "_copy_file", lambda *_args, **_kwargs: (_ for _ in ()).throw(CacheError())
    )

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out)])

    assert result.exit_code == 5
    manifest = Manifest.model_validate_json((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.status == "error"
    assert not any(item.kind == "media" for item in manifest.sources[0].artifacts)


def test_source_key_is_portable_and_disambiguates_truncation_collision() -> None:
    first_id = f"Provider__{'a' * 40}x"
    second_id = f"provider__{'a' * 40}y"
    first = source_key(first_id)
    second = source_key(second_id, {first.casefold(): first_id})

    assert first == first.lower()
    assert first != second
    assert len(second) <= 64
    assert source_key("CON__device") != "con"
