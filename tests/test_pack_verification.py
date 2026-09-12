from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from vctx.artifact.bundle import write_manifest
from vctx.artifact.manifest import ArtifactRef, Manifest
from vctx.artifact.publish import open_pack, verify_pack, verify_required
from vctx.cli import app
from vctx.errors import OutputExistsError

runner = CliRunner()


def _pack(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello.\n", encoding="utf-8")
    root = tmp_path / "pack"
    result = runner.invoke(app, ["prepare", str(source), "--out", str(root)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return root, manifest["sources"][0]["key"]


def test_schema_five_indexes_flat_source_lanes_once(
    tmp_path: Path,
) -> None:
    root, key = _pack(tmp_path)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    paths = [item["path"] for item in source["artifacts"]]

    assert manifest["schema_version"] == "5"
    assert len(paths) == len(set(paths))
    assert "subtitle.und.srt" in paths
    assert source["asset_scope"] == "consumed"
    assert source["source_capabilities"] == ["subtitle"]
    assert set(source).isdisjoint({"assets", "steps", "warnings", "transform_evidence"})
    assert source["outcomes"]
    assert (root / key / "subtitle.und.srt").is_file()


def test_required_verification_ignores_unrelated_media_and_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, key = _pack(tmp_path)
    manifest = open_pack(root)
    lane = root / key
    media = next(item for item in manifest.sources[0].artifacts if item.kind == "subtitle")
    projection = next(item for item in manifest.sources[0].artifacts if item.path == "context.md")
    (lane / media.path).write_bytes(b"unrelated corrupt retained input")
    (lane / projection.path).write_text("unrelated corrupt projection", encoding="utf-8")

    open_pack(root)
    products = verify_required(root, key, {"transcript"})

    assert products["transcript"].source_id


def test_required_verification_rejects_corrupt_canonical_product(tmp_path: Path) -> None:
    root, key = _pack(tmp_path)
    (root / key / "transcript.json").write_text("{}", encoding="utf-8")

    with pytest.raises(OutputExistsError, match="integrity"):
        verify_required(root, key, {"transcript"})


def test_full_verification_checks_unknown_files_but_does_not_reject_their_kind(
    tmp_path: Path,
) -> None:
    root, key = _pack(tmp_path)
    manifest = open_pack(root)
    lane = root / key
    body = b"opaque extension payload"
    (lane / "opaque.bin").write_bytes(body)
    source = manifest.sources[0]
    source.artifacts.append(
        ArtifactRef(
            kind="vendor-payload",
            path="opaque.bin",
            media_type="application/octet-stream",
            bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
        )
    )
    write_manifest(root, manifest)

    report = verify_pack(root)

    assert report.unchecked_kinds == ("subtitle", "vendor-payload")


def test_full_verification_rejects_unlisted_and_modified_files(tmp_path: Path) -> None:
    root, key = _pack(tmp_path)
    extra = root / key / "extra.txt"
    extra.write_text("not indexed", encoding="utf-8")
    with pytest.raises(OutputExistsError, match="unlisted"):
        verify_pack(root)
    extra.unlink()
    (root / key / "context.md").write_text("modified", encoding="utf-8")
    with pytest.raises(OutputExistsError, match="integrity"):
        verify_pack(root)


@pytest.mark.parametrize(
    "path",
    ["../media.bin", "/media.bin", "C:/media.bin", "media/../file.bin", "CON/file.bin"],
)
def test_artifact_paths_are_portable_and_contained(path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(
            kind="opaque",
            path=path,
            media_type="application/octet-stream",
            bytes=0,
            sha256=hashlib.sha256(b"").hexdigest(),
        )


def test_manifest_rejects_duplicate_artifact_path(tmp_path: Path) -> None:
    root, _key = _pack(tmp_path)
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    raw["sources"][0]["artifacts"].append(raw["sources"][0]["artifacts"][0])

    with pytest.raises(ValidationError, match="unique"):
        Manifest.model_validate(raw)


def test_manifest_rejects_false_complete_scope_and_prior_schema_fields(tmp_path: Path) -> None:
    root, _key = _pack(tmp_path)
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    raw["sources"][0]["asset_scope"] = "complete"
    raw["sources"][0]["source_capabilities"].append("audio")
    with pytest.raises(ValidationError, match="cover"):
        Manifest.model_validate(raw)
    raw["schema_version"] = "3"
    with pytest.raises(ValidationError, match="schema-3/4"):
        Manifest.model_validate(raw)


@pytest.mark.parametrize("version", ["3", "4"])
def test_reader_accepts_immutable_prior_schema(tmp_path: Path, version: str) -> None:
    root, key = _pack(tmp_path)
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source = raw["sources"][0]
    raw["schema_version"] = version
    source["path"] = key if version == "3" else f"sources/{key}"
    if version == "4":
        (root / "sources").mkdir()
        (root / key).replace(root / "sources" / key)
    source.pop("asset_scope")
    source.pop("source_capabilities")
    (root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")

    assert verify_pack(root).manifest.schema_version == version
    migrated = runner.invoke(app, ["prepare", str(tmp_path / "source.srt"), "--out", str(root)])
    assert migrated.exit_code == 0, migrated.output
    current = verify_pack(root).manifest
    assert current.schema_version == "5" and (root / key / "subtitle.und.srt").is_file()
    assert not (root / "sources").exists()


def test_open_rejects_linked_artifact(tmp_path: Path) -> None:
    root, key = _pack(tmp_path)
    linked = root / key / "context.md"
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    linked.unlink()
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("creating a symlink is not available in this Windows environment")

    with pytest.raises(OutputExistsError, match="linked"):
        open_pack(root)
