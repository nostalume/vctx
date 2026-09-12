from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


class FakeYoutubeDL:
    info: dict[str, object] = {}

    def __init__(self, params: dict[str, object]) -> None:
        self.params = params

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
        assert value == "https://video.example/watch?v=abc"
        assert download is False
        return self.info


def test_prepare_url_without_subtitles_writes_metadata_partial_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.source.ytdlp as module

    FakeYoutubeDL.info = {
        "id": "abc",
        "title": "Lecture",
        "webpage_url": "https://video.example/watch?v=abc",
        "extractor": "example",
        "subtitles": {},
        "automatic_captions": {},
    }
    monkeypatch.setattr(module._yt_dlp(), "YoutubeDL", FakeYoutubeDL)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "prepare",
            "https://video.example/watch?v=abc",
            "--out",
            str(out_dir),
            "--asr",
            "none",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote partial context pack" in result.output
    assert "Target: transcript" in result.output
    assert "Status: partial" in result.output
    assert "Routes:" in result.output
    assert "/metadata.json" in result.output
    assert "Context:" not in result.output
    assert (out_dir / "manifest.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    source_entry = manifest["sources"][0]
    lane = out_dir / source_entry["path"]
    assert (lane / "metadata.json").exists()
    assert not (lane / "transcript.json").exists()
    assert not (lane / "chunks.json").exists()
    metadata = json.loads((lane / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "example__abc"
    assert manifest["status"] == "partial"
    omissions = "\n".join(
        item for outcome in source_entry["outcomes"] for item in outcome["omissions"]
    )
    assert "No subtitles found" in omissions
    assert "vctx models pull asr" in omissions


def test_prepare_offline_url_cache_miss_has_no_effect_or_partial_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.source.ytdlp as module

    monkeypatch.setattr(
        module._yt_dlp(),
        "YoutubeDL",
        lambda _params: pytest.fail("offline source admission attempted network access"),
    )
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "prepare",
            "https://video.example/watch?v=offline",
            "--out",
            str(out_dir),
            "--offline",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 6
    assert "offline URL cache miss" in result.output
    assert not out_dir.exists()
    assert not cache_dir.exists()


def test_prepare_offline_accepts_local_input(tmp_path: Path) -> None:
    source = tmp_path / "local.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nlocal only\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    result = runner.invoke(app, ["prepare", str(source), "--out", str(out_dir), "--offline"])

    assert result.exit_code == 0, result.output
    assert (out_dir / "manifest.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (out_dir / manifest["sources"][0]["path"] / "transcript.json").exists()


@pytest.mark.parametrize("cache_inside_output", [False, True])
def test_prepare_rejects_cache_output_containment_before_writing(
    tmp_path: Path, cache_inside_output: bool
) -> None:
    source = tmp_path / "local.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nlocal\n", encoding="utf-8")
    if cache_inside_output:
        out_dir = tmp_path / "out"
        cache_dir = out_dir / "cache"
    else:
        cache_dir = tmp_path / "cache"
        out_dir = cache_dir / "source" / "out"

    result = runner.invoke(
        app,
        ["prepare", str(source), "--out", str(out_dir), "--cache-dir", str(cache_dir)],
    )

    assert result.exit_code == 2
    assert "must not contain each other" in result.output
    assert not out_dir.exists()
