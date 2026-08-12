from __future__ import annotations

import base64
import mimetypes
import uuid

from pydantic import BaseModel

from vctx.ai import AiClient, AiImagePart, AiImageUrl, AiMessage, AiTextPart
from vctx.visual.evidence import Observation
from vctx.visual.frame import Frame


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
