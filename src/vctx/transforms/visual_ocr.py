from __future__ import annotations

import importlib
from collections.abc import Callable

from vctx.models.visual import FrameAsset


class OcrExecutionError(RuntimeError):
    pass


class RapidOcrAdapter:
    provider_id = "rapidocr"

    def __init__(self) -> None:
        self._engine: Callable[[str], object] | None = None

    def extract_text(self, frame: FrameAsset) -> str:
        try:
            result = self._rapidocr()(str(frame.path))
        except Exception as exc:  # pragma: no cover - adapter boundary
            raise OcrExecutionError(f"rapidocr failed for {frame.path}: {exc}") from exc
        return _rapidocr_text(result)

    def _rapidocr(self) -> Callable[[str], object]:
        if self._engine is None:
            try:
                module = importlib.import_module("rapidocr")
            except ImportError as exc:
                raise OcrExecutionError("rapidocr is not installed; install vctx[visual]") from exc
            rapid_ocr = getattr(module, "RapidOCR", None)
            if not callable(rapid_ocr):
                raise OcrExecutionError("rapidocr does not expose callable RapidOCR")
            engine = rapid_ocr()
            if not callable(engine):
                raise OcrExecutionError("rapidocr RapidOCR() did not return a callable engine")
            self._engine = engine
        return self._engine


def _rapidocr_text(result: object) -> str:
    documented_texts = _rapidocr_output_texts(result)
    if documented_texts:
        return "\n".join(documented_texts).strip()
    blocks = _rapidocr_blocks(result)
    texts: list[str] = []
    for block in blocks:
        text = _block_text(block)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def _rapidocr_output_texts(result: object) -> list[str]:
    txts = getattr(result, "txts", None)
    if isinstance(txts, tuple):
        return [text for text in txts if isinstance(text, str)]
    return []


def _rapidocr_blocks(result: object) -> list[object]:
    if result is None:
        return []
    if isinstance(result, tuple) and result:
        first = result[0]
        return [item for item in first] if isinstance(first, list) else []
    if isinstance(result, list):
        return [item for item in result]
    return []


def _block_text(block: object) -> str | None:
    match block:
        case {"text": str(text)}:
            return text
    if isinstance(block, (list, tuple)):
        for item in block:
            if isinstance(item, str):
                return item
    return None
