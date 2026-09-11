from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

import pytest
from typer.testing import CliRunner

from vctx.artifact.bundle import Artifact, write_artifact, write_manifest
from vctx.artifact.publish import open_pack
from vctx.cli import app
from vctx.transcript import Transcript
from vctx.visual.evidence import CaptureEvidence, Evidence, Observation
from vctx.visual.plan import EvidencePlan, PlannedFrame

runner = CliRunner()


def _pack(tmp_path: Path, *names: str) -> Path:
    pack = tmp_path / "pack"
    inputs: list[str] = []
    for name in names or ("lecture",):
        source = tmp_path / f"{name}.srt"
        source.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{name} words\n", encoding="utf-8")
        inputs.append(str(source))
    result = runner.invoke(app, ["prepare", *inputs, "--out", str(pack)])
    assert result.exit_code == 0, result.output
    return pack


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _add_evidence(pack: Path) -> Path:
    manifest = open_pack(pack)
    source = manifest.sources[0]
    lane = pack / source.path
    transcript = Transcript.model_validate_json(
        (lane / "transcript.json").read_text(encoding="utf-8")
    )
    frame = PlannedFrame(
        id="frame-0001",
        target_seconds=0.5,
        segment_ids=["seg_000001"],
        processors=[],
        request_ids=["request-0001"],
        priority=1,
    )
    plan = EvidencePlan(source_id=transcript.source_id, frames=[frame])
    evidence = Evidence(
        source_id=transcript.source_id,
        captures=[
            CaptureEvidence(
                id=frame.id,
                requested_seconds=0.5,
                actual_seconds=0.5,
                artifact_path="frames/frame-0001.png",
                request_ids=frame.request_ids,
                segment_ids=frame.segment_ids,
                ocr=Observation(status="not_requested"),
                vision=Observation(status="not_requested"),
            )
        ],
    )
    refs = [
        write_artifact(lane, Artifact.json("evidence-plan.json", "evidence_plan", plan)),
        write_artifact(lane, Artifact.json("evidence.json", "evidence", evidence)),
        write_artifact(
            lane,
            Artifact(
                "frames/frame-0001.png",
                "visual_frame",
                "image/png",
                b"\x89PNG\r\n\x1a\nfixture",
            ),
        ),
    ]
    source.artifacts.extend(refs)
    write_manifest(pack, manifest)
    return lane / "frames" / "frame-0001.png"


@pytest.mark.parametrize(
    ("format", "marker"),
    [("context", "# Agent Context Pack"), ("read", "# lecture"), ("transcript", "# Transcript")],
)
def test_render_single_source_pack_to_stdout(tmp_path: Path, format: str, marker: str) -> None:
    pack = _pack(tmp_path)
    before = _tree(pack)

    result = runner.invoke(app, ["render", str(pack), "--format", format])

    assert result.exit_code == 0, result.output
    assert marker in result.stdout
    if format == "context":
        assert "Summary is not available in this pack." in result.stdout
        assert "Visual evidence is not available in this pack." in result.stdout
    assert _tree(pack) == before


def test_render_external_file_ignores_stored_projection_bytes(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    lane = pack / manifest["sources"][0]["path"]
    (lane / "context.md").write_text("corrupt stored projection", encoding="utf-8")
    before = _tree(pack)
    out = tmp_path / "views" / "context.md"

    result = runner.invoke(app, ["render", str(pack), "--format", "context", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "# Agent Context Pack" in out.read_text(encoding="utf-8")
    assert "Wrote render:" in result.stdout
    assert _tree(pack) == before


def test_rendered_frame_link_is_relative_to_external_destination(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    frame = _add_evidence(pack)
    out = tmp_path / "export with spaces" / "read.md"

    result = runner.invoke(app, ["render", str(pack), "--format", "read", "--out", str(out)])

    assert result.exit_code == 0, result.output
    match = re.search(r"!\[[^]]*\]\(([^)]+)\)", out.read_text(encoding="utf-8"))
    assert match is not None
    link = unquote(match.group(1))
    assert not Path(link).is_absolute()
    assert (out.parent / link).resolve() == frame.resolve()


def test_stdout_frame_link_is_relative_to_current_directory(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    frame = _add_evidence(pack)

    result = runner.invoke(app, ["render", str(pack), "--format", "context"])

    assert result.exit_code == 0, result.output
    match = re.search(r'<image path="([^"]+)"', result.stdout)
    assert match is not None
    assert (Path.cwd() / unquote(match.group(1))).resolve() == frame.resolve()


def test_render_rejects_pack_destination_and_missing_referenced_frame(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    frame = _add_evidence(pack)
    before = _tree(pack)

    inside = runner.invoke(
        app,
        ["render", str(pack), "--format", "read", "--out", str(pack / "view.md")],
    )
    frame.unlink()
    missing = runner.invoke(app, ["render", str(pack), "--format", "read"])

    assert inside.exit_code == 1
    assert "outside" in inside.stderr
    assert _tree(pack) == {key: value for key, value in before.items() if not key.endswith(".png")}
    assert missing.exit_code == 5
    assert "verified" in missing.stderr


def test_render_rejects_corrupt_required_product_but_ignores_unrelated_asset(
    tmp_path: Path,
) -> None:
    pack = _pack(tmp_path)
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    lane = pack / manifest["sources"][0]["path"]
    (lane / "subtitle.und.srt").write_text("corrupt retained input", encoding="utf-8")

    unrelated = runner.invoke(app, ["render", str(pack), "--format", "transcript"])
    (lane / "transcript.json").write_text("{}", encoding="utf-8")
    required = runner.invoke(app, ["render", str(pack), "--format", "transcript"])

    assert unrelated.exit_code == 0, unrelated.output
    assert required.exit_code == 5
    assert "integrity" in required.stderr


def test_render_reports_unrepresentable_cross_volume_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.app.render as render_module

    pack = _pack(tmp_path)
    _add_evidence(pack)
    out = tmp_path / "view.md"

    def fail_relpath(*_args: object, **_kwargs: object) -> str:
        raise ValueError("different drives")

    monkeypatch.setattr(render_module.os.path, "relpath", fail_relpath)

    result = runner.invoke(app, ["render", str(pack), "--format", "read", "--out", str(out)])

    assert result.exit_code == 1
    assert "relative links" in result.stderr
    assert not out.exists()


def test_render_multi_source_pack_requires_stable_source_key(tmp_path: Path) -> None:
    pack = _pack(tmp_path, "first", "second")
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    key = manifest["sources"][1]["key"]

    ambiguous = runner.invoke(app, ["render", str(pack), "--format", "transcript"])
    selected = runner.invoke(
        app,
        ["render", str(pack), "--source", key, "--format", "transcript"],
    )

    assert ambiguous.exit_code == 4
    assert "--source" in ambiguous.stderr
    assert selected.exit_code == 0, selected.output
    assert "second words" in selected.stdout
    assert "first words" not in selected.stdout


def test_render_help_has_no_detached_product_inputs() -> None:
    result = runner.invoke(app, ["render", "--help"])

    assert result.exit_code == 0
    assert "--metadata" not in result.output
    assert "--transcript" not in result.output
    assert "--chunks" not in result.output
