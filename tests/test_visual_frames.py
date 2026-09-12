from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

from vctx.source.session import MediaAsset, SourceRef
from vctx.visual.frame import FrameError, capture
from vctx.visual.plan import PlannedFrame

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("av")

FIXTURE = Path(__file__).parent / "fixtures" / "frame.mp4"


def _media(path: Path = FIXTURE) -> MediaAsset:
    return MediaAsset(
        id="media-1",
        source=SourceRef(kind="file", value=str(path)),
        local_path=path,
        container="mp4",
        capabilities={"video"},
        purpose="visual",
    )


def _request(frame_id: str, target: float, segment: str) -> PlannedFrame:
    return PlannedFrame(
        id=frame_id,
        target_seconds=target,
        segment_ids=[segment],
        processors=["ocr", "describe"],
        request_ids=[f"request-{frame_id[-4:]}"],
        claim_ids=[f"claim-{frame_id[-4:]}"],
        priority=0.8,
    )


def test_capture_resolves_displayed_frames_and_publishes_provenance(tmp_path: Path) -> None:
    batch = capture(
        _media(),
        [_request("frame-0002", 1.25, "seg_000002"), _request("frame-0001", 0.25, "seg_000001")],
        tmp_path,
    )

    assert not batch.misses
    assert [frame.id for frame in batch.frames] == ["frame-0001", "frame-0002"]
    assert [frame.requested_seconds for frame in batch.frames] == [0.25, 1.25]
    assert [frame.actual_seconds for frame in batch.frames] == [0.0, 1.0]
    assert [frame.orientation for frame in batch.frames] == [90, 90]
    assert batch.frames[0].original_size == (3000, 120)
    assert batch.frames[0].size == (102, 2560)
    assert batch.frames[0].request_ids == ("request-0001",)
    assert batch.frames[0].segment_ids == ("seg_000001",)
    assert batch.frames[0].claim_ids == ("claim-0001",)
    assert batch.frames[0].processors == ("ocr", "describe")

    for frame in batch.frames:
        assert frame.path.parent == tmp_path / "frames"
        assert frame.sha256 == hashlib.sha256(frame.path.read_bytes()).hexdigest()
        with Image.open(frame.path) as image:
            assert image.size == frame.size
            assert image.format == "PNG"

    with Image.open(batch.frames[0].path) as image:
        top = cast(
            tuple[int, int, int],
            image.convert("RGB").getpixel((image.width // 2, image.height // 6)),
        )
        bottom = cast(
            tuple[int, int, int],
            image.convert("RGB").getpixel((image.width // 2, image.height * 5 // 6)),
        )
    assert top[0] > top[1]
    assert bottom[1] > bottom[0]


def test_capture_returns_a_typed_miss_without_losing_valid_frames(tmp_path: Path) -> None:
    batch = capture(
        _media(),
        [_request("frame-0001", 2.25, "seg_000001"), _request("frame-0002", 9.0, "seg_000002")],
        tmp_path,
    )

    assert [frame.id for frame in batch.frames] == ["frame-0001"]
    assert [(miss.id, miss.reason) for miss in batch.misses] == [
        ("frame-0002", "target_out_of_range")
    ]
    assert sorted(path.name for path in (tmp_path / "frames").iterdir()) == ["frame-0001.png"]

    miss_root = tmp_path / "only-miss"
    only_miss = capture(_media(), [_request("frame-0001", 9.0, "seg_000001")], miss_root)
    assert not only_miss.frames and len(only_miss.misses) == 1
    assert not (miss_root / "frames").exists()


def test_capture_rejects_a_source_that_is_not_decodable(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not media")

    with pytest.raises(FrameError, match="cannot open visual media"):
        capture(_media(corrupt), [_request("frame-0001", 0.0, "seg_000001")], tmp_path)
    assert not (tmp_path / "frames").exists()
