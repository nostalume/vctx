from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vctx.models.visual import FrameAsset
from vctx.net import BatchNetRuntime, NetRequest, NetResponse, NetRuntime, UrllibNetRuntime
from vctx.transforms.ai_routes import AiRoute


class VisionExecutionError(RuntimeError):
    pass


class VlmOutcome(BaseModel):
    frame_id: str
    text: str | None = None
    error: str | None = None


class OpenAiVisionMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None


class OpenAiVisionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: OpenAiVisionMessage


class OpenAiVisionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    choices: list[OpenAiVisionChoice] = Field(default_factory=list)


class OpenAiCompatibleVisionAdapter:
    def __init__(
        self,
        *,
        route: AiRoute,
        api_key: str,
        net: NetRuntime | None = None,
    ) -> None:
        if route.base_url is None:
            raise VisionExecutionError("vision route is missing base_url")
        if route.model is None:
            raise VisionExecutionError("vision route is missing model")
        if route.provider_id is None:
            raise VisionExecutionError("vision route is missing provider_id")
        self.route = route
        self.api_key = api_key
        self.net = net or UrllibNetRuntime()

    def describe(self, frame: FrameAsset) -> str:
        return _description_from_response(self._request_one(self._request_for_frame(frame)))

    def describe_many(self, frames: list[FrameAsset]) -> list[VlmOutcome]:
        requests = [self._request_for_frame(frame) for frame in frames]
        if isinstance(self.net, BatchNetRuntime):
            try:
                responses = self.net.request_many(requests)
            except Exception as exc:
                return [
                    VlmOutcome(
                        frame_id=frame.id,
                        error=(
                            "vision request failed "
                            f"for provider {self.route.provider_id}: {type(exc).__name__}"
                        ),
                    )
                    for frame in frames
                ]
            return [
                _outcome_from_response(frame, response)
                for frame, response in zip(frames, responses, strict=True)
            ]
        outcomes: list[VlmOutcome] = []
        for frame in frames:
            try:
                outcomes.append(VlmOutcome(frame_id=frame.id, text=self.describe(frame)))
            except VisionExecutionError as exc:
                outcomes.append(VlmOutcome(frame_id=frame.id, error=str(exc)))
        return outcomes

    def _request_for_frame(self, frame: FrameAsset) -> NetRequest:
        return NetRequest(
            method="POST",
            url=self.route.base_url or "",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            body=json.dumps(_vision_payload(frame.path, model=self.route.model or "")).encode(
                "utf-8"
            ),
            timeout_s=120,
            purpose="vision_description",
            provider_id=self.route.provider_id,
        )

    def _request_one(self, request: NetRequest) -> NetResponse:
        try:
            return self.net.request(request)
        except Exception as exc:
            raise VisionExecutionError(
                "vision request failed "
                f"for provider {self.route.provider_id}: {type(exc).__name__}"
            ) from exc


def _vision_payload(frame_path: Path, *, model: str) -> dict[str, object]:
    media_type = mimetypes.guess_type(frame_path.name)[0] or "image/png"
    image_b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the visual source content in the frame. "
                            "Focus on diagrams, layout, labels, equations, and information "
                            "that is not recoverable from transcript text. Be concise and factual."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
    }


def _description_from_response(response: NetResponse) -> str:
    if response.status_code < 200 or response.status_code >= 300:
        raise VisionExecutionError(f"vision request failed: HTTP {response.status_code}")
    try:
        response_payload = OpenAiVisionResponse.model_validate_json(response.body)
    except ValueError as exc:
        raise VisionExecutionError("vision provider returned invalid JSON") from exc
    return _description_text(response_payload)


def _outcome_from_response(frame: FrameAsset, response: NetResponse) -> VlmOutcome:
    try:
        return VlmOutcome(frame_id=frame.id, text=_description_from_response(response))
    except VisionExecutionError as exc:
        return VlmOutcome(frame_id=frame.id, error=str(exc))


def _description_text(payload: OpenAiVisionResponse) -> str:
    if not payload.choices:
        raise VisionExecutionError("vision provider returned no choices")
    text = payload.choices[0].message.content
    if text is None or not text.strip():
        raise VisionExecutionError("vision provider returned empty description")
    return text.strip()
