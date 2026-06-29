from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_knowledge_flow.py"
spec = importlib.util.spec_from_file_location("smoke_knowledge_flow", SMOKE_SCRIPT)
assert spec is not None
assert spec.loader is not None
smoke_knowledge_flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke_knowledge_flow)
assert isinstance(smoke_knowledge_flow, ModuleType)


def test_smoke_knowledge_flow_requires_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VCTX_SMOKE_VIDEO_URL", raising=False)

    assert smoke_knowledge_flow.main() == 2

    assert "Set VCTX_SMOKE_VIDEO_URL" in capsys.readouterr().out


def test_smoke_knowledge_flow_requires_knowledge_flow_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, text: bool, check: bool) -> object:
        calls.append(command)
        assert text is True
        assert check is False
        out_dir.mkdir(parents=True)
        for name in smoke_knowledge_flow.REQUIRED_FULL_PACK_ARTIFACTS:
            (out_dir / name).write_text("{}", encoding="utf-8")
        (out_dir / "knowledge_flow.json").write_text(
            json.dumps({"nodes": [{"id": "a"}], "edges": [{"id": "a__b"}]}),
            encoding="utf-8",
        )
        (out_dir / "context.md").write_text(
            "# Context\n\n## Knowledge-flow summary\n\n- Workflow: A -> B.\n",
            encoding="utf-8",
        )

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setenv("VCTX_SMOKE_VIDEO_URL", "https://example.invalid/video")
    monkeypatch.setenv("VCTX_SMOKE_OUT", str(out_dir))
    monkeypatch.setattr(smoke_knowledge_flow.subprocess, "run", fake_run)

    assert smoke_knowledge_flow.main() == 0
    assert calls == [
        [
            "uv",
            "run",
            "vctx",
            "prepare",
            "https://example.invalid/video",
            "--out",
            str(out_dir),
        ]
    ]


def test_smoke_knowledge_flow_fails_when_summary_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "out"

    def fake_run(command: list[str], *, text: bool, check: bool) -> object:
        del command, text, check
        out_dir.mkdir(parents=True)
        for name in [*smoke_knowledge_flow.REQUIRED_FULL_PACK_ARTIFACTS, "knowledge_flow.json"]:
            (out_dir / name).write_text("{}", encoding="utf-8")

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setenv("VCTX_SMOKE_VIDEO_URL", "https://example.invalid/video")
    monkeypatch.setenv("VCTX_SMOKE_OUT", str(out_dir))
    monkeypatch.setattr(smoke_knowledge_flow.subprocess, "run", fake_run)

    assert smoke_knowledge_flow.main() == 1
    assert "missing knowledge-flow summary" in capsys.readouterr().out
