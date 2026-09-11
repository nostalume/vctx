from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from tests.support import asr_ready_segments
from vctx.asr import _restore_timeline
from vctx.asr.input import AsrInput, admit_interval, open_asr_input
from vctx.errors import InvalidTranscriptError
from vctx.source.session import MediaAsset, SourceRef


def _media(tmp_path: Path) -> MediaAsset:
    path = tmp_path / "source.wav"
    rate, duration = 16_000, 4
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, duration * rate, "NONE", "not compressed"))
        output.writeframes(
            b"".join(
                struct.pack("<h", round(8_000 * math.sin(2 * math.pi * 440 * index / rate)))
                for index in range(rate * duration)
            )
        )
    return MediaAsset(
        id="fixture",
        source=SourceRef(kind="file", value=str(path)),
        local_path=path,
        duration_seconds=duration,
        media_type="audio",
        capabilities={"audio"},
    )


def test_interval_input_decodes_only_bound_and_cleans_owned_file(tmp_path: Path) -> None:
    media = _media(tmp_path)
    temporary = None
    with open_asr_input(media, (1.0, 3.0), tmp_path / "tmp" / "asr") as item:
        temporary = item.temporary
        assert temporary is not None and temporary.is_file()
        with wave.open(str(temporary)) as audio:
            assert audio.getnframes() / audio.getframerate() == 2.0
    assert temporary is not None and not temporary.exists()


def test_interval_input_cleans_owned_file_after_consumer_failure(tmp_path: Path) -> None:
    media = _media(tmp_path)
    temporary = None
    with pytest.raises(RuntimeError), open_asr_input(media, (1.0, 2.0), tmp_path / "tmp") as item:
        temporary = item.temporary
        raise RuntimeError("consumer failed")
    assert temporary is not None and not temporary.exists()


def test_interval_admission_requires_finite_contained_bounds(tmp_path: Path) -> None:
    media = _media(tmp_path)
    assert admit_interval(media, (0.0, 4.0)) == (0.0, 4.0)
    with pytest.raises(InvalidTranscriptError, match="start < end"):
        admit_interval(media, (3.0, 5.0))


def test_interval_result_uses_absolute_timestamps_and_explicit_provenance(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path)
    outcome = asr_ready_segments(media.id, [(0.25, 1.5, "bounded")])
    item = AsrInput(media, origin=10.0, end=12.0, source_duration=100.0, temporary=tmp_path / "x")

    restored = _restore_timeline(outcome, item)

    assert restored.kind == "ready"
    assert (restored.transcript.segments[0].start, restored.transcript.segments[0].end) == (
        10.25,
        11.5,
    )
    receipt = restored.receipt.model_dump()
    assert {
        key: receipt[key] for key in ("interval_start", "interval_end", "processed_duration")
    } == {
        "interval_start": 10.0,
        "interval_end": 12.0,
        "processed_duration": 2.0,
    }
