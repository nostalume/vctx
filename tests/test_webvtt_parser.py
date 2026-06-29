from __future__ import annotations

import warnings

from vctx.subtitles import parse_webvtt
from vctx.transcript import TranscriptPayload, TranscriptProvenance


def test_parse_webvtt_does_not_use_deprecated_webvtt_api() -> None:
    payload = TranscriptPayload(
        text="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n",
        format="vtt",
        provenance=TranscriptProvenance(method="official_subtitles", provider="fixture"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        transcript = parse_webvtt(payload, video_id="video")

    assert transcript.segments[0].text == "Hello"
