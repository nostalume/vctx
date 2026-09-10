from __future__ import annotations

from pathlib import Path

from vctx.asr import AsrReady, AsrReceipt
from vctx.transcript import Transcript, TranscriptProvenance, TranscriptSegment


def local_asr_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir(exist_ok=True)
    (model / "model.bin").write_bytes(b"model")
    (model / "config.json").write_text("{}", encoding="utf-8")
    return model


def asr_ready(media_id: str, text: str, *, language: str = "en", model: str = "small") -> AsrReady:
    return asr_ready_segments(media_id, [(0.0, 1.0, text)], language=language, model=model)


def asr_ready_segments(
    media_id: str,
    values: list[tuple[float, float, str]],
    *,
    language: str = "en",
    model: str = "small",
) -> AsrReady:
    return AsrReady(
        transcript=Transcript(
            source_id=media_id,
            provenance=TranscriptProvenance(
                method="asr", language=language, format="json", provider="faster-whisper"
            ),
            segments=[
                TranscriptSegment(id=f"seg_{index:06d}", start=start, end=end, text=text)
                for index, (start, end, text) in enumerate(values, start=1)
            ],
        ),
        receipt=AsrReceipt(
            provider="faster-whisper", model=model, device="cpu", compute_type="auto", vad=True
        ),
    )
