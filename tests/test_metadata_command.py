from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vctx.cli import app

runner = CliRunner()


def test_metadata_command_prints_human_readable_local_metadata(tmp_path: Path) -> None:
    source = tmp_path / "lecture.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    result = runner.invoke(app, ["metadata", str(source)])

    assert result.exit_code == 0, result.output
    assert "title: lecture" in result.output
    assert "source_type: local-file" in result.output
    assert "raw_provider: local-file" in result.output


def test_metadata_command_prints_json_local_metadata(tmp_path: Path) -> None:
    source = tmp_path / "lecture.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    result = runner.invoke(app, ["metadata", str(source), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"].startswith("local__")
    assert payload["source_type"] == "local-file"
    assert payload["source"] == {"kind": "file", "value": str(source)}
    assert payload["raw_provider"] == "local-file"


def test_metadata_help_uses_decisive_json_flag_without_negation_pair() -> None:
    result = runner.invoke(app, ["metadata", "--help"])

    assert result.exit_code == 0
    assert "--no-json" not in result.output


def test_metadata_applies_source_policy_once_and_redacts_locator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.source.ytdlp as module

    seen: list[dict[str, object]] = []

    class FakeYoutubeDL:
        def __init__(self, params: dict[str, object]) -> None:
            seen.append(params)

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
            assert value.endswith("?v=abc&token=secret") and not download
            return {
                "id": "abc",
                "extractor": "example",
                "webpage_url": value,
                "duration": 10,
            }

    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    config = tmp_path / "vctx.toml"
    config.write_text(
        '[source]\nmedia_quality = "high"\n[source.yt_dlp]\n'
        'session = "browser:firefox"\nnetwork = "proxy:http://proxy.test"\n'
        'playlist = "items:1"\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "metadata", "https://video.example/watch?v=abc&token=secret",
            "--json", "--config", str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source"]["value"] == "https://video.example/watch"
    assert "secret" not in result.output
    assert len(seen) == 1
    assert seen[0]["cookiesfrombrowser"] == ("firefox",)
    assert seen[0]["proxy"] == "http://proxy.test"
    assert seen[0]["playlist_items"] == "1"
    assert seen[0]["socket_timeout"] == 30


def test_metadata_can_inspect_live_but_prepare_rejects_it_before_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.source.ytdlp as module

    class FakeYoutubeDL:
        def __init__(self, params: dict[str, object]) -> None:
            del params

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
            return {
                "id": "live", "extractor": "example", "webpage_url": value,
                "live_status": "is_live", "is_live": True,
            }

    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    url = "https://video.example/live"
    metadata = runner.invoke(app, ["metadata", url, "--json"])
    out = tmp_path / "out"
    prepare = runner.invoke(app, ["prepare", url, "--out", str(out)])

    assert metadata.exit_code == 0, metadata.output
    assert prepare.exit_code == 4
    assert "prepare requires finite or archived media" in prepare.output
    assert not out.exists()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("[runtime]\ncache_dir = 'old'", "runtime.cache_dir was removed"),
        ("[source.yt_dlp]\nmedia_profile = 'high'", "source.yt_dlp.media_profile"),
    ],
)
def test_removed_config_keys_have_migration_errors(
    tmp_path: Path, body: str, message: str
) -> None:
    source = tmp_path / "lecture.srt"
    source.write_text("", encoding="utf-8")
    config = tmp_path / "vctx.toml"
    config.write_text(body, encoding="utf-8")

    result = runner.invoke(app, ["metadata", str(source), "--config", str(config)])

    assert result.exit_code == 2
    assert message in result.output


def test_metadata_maps_provider_failure_to_exit_7(monkeypatch: pytest.MonkeyPatch) -> None:
    import vctx.source.ytdlp as module

    class FailedYoutubeDL:
        def __init__(self, params: dict[str, object]) -> None:
            del params

        def __enter__(self) -> FailedYoutubeDL:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
            raise module.yt_dlp.utils.DownloadError("provider unavailable")

    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FailedYoutubeDL)
    result = runner.invoke(app, ["metadata", "https://video.example/fail"])

    assert result.exit_code == 7
    assert "provider unavailable" in result.output


def test_metadata_uses_verified_offline_observation_without_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vctx.source.ytdlp as module

    url = "https://video.example/watch?v=cached"

    class SeedYoutubeDL:
        def __init__(self, params: dict[str, object]) -> None:
            del params

        def __enter__(self) -> SeedYoutubeDL:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, value: str, download: bool = False) -> dict[str, object]:
            assert value == url and not download
            return {"id": "cached", "extractor": "example", "title": "Cached metadata"}

    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", SeedYoutubeDL)
    cache = tmp_path / "cache"
    online = runner.invoke(app, ["metadata", url, "--json", "--cache-dir", str(cache)])
    assert online.exit_code == 0, online.output

    monkeypatch.setattr(
        module.yt_dlp,
        "YoutubeDL",
        lambda _params: pytest.fail("offline metadata attempted provider discovery"),
    )
    offline = runner.invoke(
        app, ["metadata", url, "--json", "--cache-dir", str(cache), "--offline"]
    )

    assert offline.exit_code == 0, offline.output
    assert json.loads(offline.output)["title"] == "Cached metadata"
