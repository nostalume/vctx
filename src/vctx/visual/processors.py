from __future__ import annotations

import base64
import importlib
import mimetypes
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from vctx.ai import AiClient, AiImagePart, AiImageUrl, AiMessage, AiTextPart
from vctx.model.store import ModelLifecycleError, ModelStore
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
        store = ModelStore(cache_root)
        store.require("ocr")
        engine = constructor(config_path=str(store.rapidocr_config_path()))
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


class VlmOutcome(Observation):
    pass


class _Description(BaseModel):
    text: str


class VisionProcessor:
    def __init__(self, *, client: AiClient) -> None:
        self.client = client

    def observe(self, frame: Frame, *, ocr_text: str | None = None) -> VlmOutcome:
        try:
            media_type = mimetypes.guess_type(frame.path.name)[0] or "image/png"
            image = base64.b64encode(frame.path.read_bytes()).decode("ascii")
            prompt = (
                "Describe source information visible in this frame that is not recoverable "
                "from transcript text. Focus on diagrams, layout, labels, equations, and "
                "actions. Be concise and factual."
            )
            if ocr_text:
                prompt += f"\nOCR observation for context:\n{ocr_text}"
            outcome = self.client.complete(
                task="vision_description",
                request_id=str(uuid.uuid4()),
                result=_Description,
                messages=[
                    AiMessage(
                        role="user",
                        content=[
                            AiTextPart(text=prompt),
                            AiImagePart(
                                image_url=AiImageUrl(url=f"data:{media_type};base64,{image}")
                            ),
                        ],
                    )
                ],
            )
        except (OSError, ValueError) as exc:
            return VlmOutcome(
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
                provider=self.client.instance.provider_id,
            )
        if outcome.kind == "failed":
            return VlmOutcome(
                status="failed",
                detail=outcome.reason,
                provider=outcome.receipt.provider,
            )
        text = outcome.value.text.strip()
        return VlmOutcome(
            status="ok" if text else "empty",
            text=text or None,
            provider=outcome.receipt.provider,
        )
