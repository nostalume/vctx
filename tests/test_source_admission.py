from __future__ import annotations

from pathlib import Path

import pytest

from vctx.config import YtDlpSourceOptions
from vctx.errors import UnsupportedSourceError
from vctx.net import NetRequest, NetResponse
from vctx.source.admission import open_source, select_source
from vctx.source.session import ObservePermit


class _NoNetwork:
    def request(self, request: NetRequest) -> NetResponse:
        pytest.fail(f"local source attempted network access: {request.url}")


def test_source_selection_is_syntax_only_and_opening_owns_filesystem_effect(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.mp4"

    selection = select_source(str(missing))

    assert selection == "local-file"
    with pytest.raises(UnsupportedSourceError, match="does not exist"):
        open_source(
            selection,
            str(missing),
            permit=ObservePermit(operation="prepare", network="denied"),
            options=YtDlpSourceOptions(),
            net=_NoNetwork(),
        )


def test_exact_public_bilibili_video_is_selected_before_ytdlp() -> None:
    assert select_source("https://www.bilibili.com/video/BV1Tpbj6eEDZ") == "bilibili"
    assert select_source("https://www.bilibili.example/video/BV1Tpbj6eEDZ") == "yt-dlp"
