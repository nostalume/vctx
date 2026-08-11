from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from vctx.app.models import ModelLifecycleError, rapidocr_config_path, require_prepared_model
from vctx.visual.evidence import Observation
from vctx.visual.frame import Frame


class OcrOutcome(Observation):
    provider: str = "rapidocr"


class RapidOcr:
    def __init__(self, engine: Callable[[str], object]) -> None:
        self._engine = engine

    def observe(self, frame: Frame) -> OcrOutcome:
        try:
            result = self._engine(str(frame.path))
            text = _text(result)
        except Exception as exc:  # pragma: no cover - vendor boundary
            return OcrOutcome(status="failed", detail=f"{type(exc).__name__}: {exc}")
        return OcrOutcome(status="ok", text=text) if text else OcrOutcome(status="empty")


@dataclass(frozen=True)
class OcrUnavailable:
    detail: str


type OcrAdmission = RapidOcr | OcrUnavailable


@dataclass
class OcrRuntimePool:
    runtimes: dict[str, OcrAdmission] = field(default_factory=dict)

    def load_rapid(self, cache_root: Path) -> OcrAdmission:
        key = str(cache_root.resolve())
        runtime = self.runtimes.get(key)
        if runtime is None:
            runtime = _load_rapid(cache_root)
            self.runtimes[key] = runtime
        return runtime


def _load_rapid(cache_root: Path) -> OcrAdmission:
    try:
        module = importlib.import_module("rapidocr")
        constructor = getattr(module, "RapidOCR", None)
        if not callable(constructor):
            raise RuntimeError("rapidocr does not expose callable RapidOCR")
        require_prepared_model("ocr", cache_root)
        engine = constructor(config_path=str(rapidocr_config_path(cache_root)))
        if not callable(engine):
            raise RuntimeError("rapidocr RapidOCR() did not return a callable engine")
        return RapidOcr(engine)
    except ImportError:
        return OcrUnavailable("rapidocr is not installed; install vctx[visual]")
    except (ModelLifecycleError, OSError, RuntimeError) as exc:
        return OcrUnavailable(str(exc))


def _text(result: object) -> str:
    texts = getattr(result, "txts", None)
    if isinstance(texts, tuple):
        return "\n".join(text for text in texts if isinstance(text, str)).strip()
    blocks = result[0] if isinstance(result, tuple) and result else result
    if not isinstance(blocks, list):
        return ""
    admitted: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            match block:
                case {"text": str(text)}:
                    admitted.append(text)
        elif isinstance(block, (list, tuple)):
            text = next((item for item in block if isinstance(item, str)), None)
            if text:
                admitted.append(text)
    return "\n".join(admitted).strip()
