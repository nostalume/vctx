from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from vctx.models.knowledge_flow import (
    KnowledgeFlow,
    KnowledgeFlowSupplement,
    KnowledgeFlowSupplementEvidence,
)
from vctx.models.visual import EssentialCaseSupplement, EssentialCaseSupplementEvidence
from vctx.net import NetPurpose, NetRequest, NetRuntime, UrllibNetRuntime
from vctx.transcript import Transcript
from vctx.transforms.ai_routes import AiRoute


class TextAiExecutionError(RuntimeError):
    pass


TextProductTask = Literal["knowledge_flow_extraction", "essential_case_extraction"]


class _ChatMessage(BaseModel):
    content: str


class _ChatChoice(BaseModel):
    message: _ChatMessage


class _ChatResponse(BaseModel):
    choices: list[_ChatChoice]


class OpenAiCompatibleTextAdapter:
    def __init__(
        self,
        *,
        route: AiRoute,
        api_key: str,
        net: NetRuntime | None = None,
    ) -> None:
        if route.base_url is None:
            raise TextAiExecutionError("text AI route is missing base_url")
        if route.model is None:
            raise TextAiExecutionError("text AI route is missing model")
        self.route = route
        self.api_key = api_key
        self.net = net or UrllibNetRuntime()

    def knowledge_flow_supplement(self, transcript: Transcript) -> KnowledgeFlowSupplement:
        content = self._request_json_content(
            transcript,
            task="knowledge_flow_extraction",
            system_prompt=(
                "Return only JSON matching KnowledgeFlow: nodes[{id,label,evidence}], "
                "edges[{id,source,target,evidence}]. Use only supplied segment ids "
                "as evidence. Add only process/causal/data-flow relations."
            ),
            user_intro="Extract a supplemental evidence-linked knowledge flow from these segments:",
        )
        return KnowledgeFlowSupplement(
            flow=KnowledgeFlow.model_validate_json(content),
            evidence=self._knowledge_flow_evidence(transcript),
        )

    def essential_case_supplement(self, transcript: Transcript) -> EssentialCaseSupplement:
        content = self._request_json_content(
            transcript,
            task="essential_case_extraction",
            system_prompt=(
                "Return only JSON matching {cases:[{segment_id,timestamp_seconds,"
                "case_type,priority,reason,actions}]}. Use only supplied segment ids. "
                "Cases should identify transcript-anchored visual sampling motives. "
                "Use capture for visual reference/action context. "
                "Use OCR only for literal shown text, formulas, equations, tables, code, or UI. "
                "Use describe only for action, environment, or layout explanation. "
                "Do not mark charts for OCR/VLM unless the transcript specifically needs "
                "shown labels/text or chart explanation. Return no cases when visual evidence "
                "is not needed."
            ),
            user_intro="Extract supplemental evidence-linked visual cases from these segments:",
        )
        return EssentialCaseSupplement.model_validate_json(content).model_copy(
            update={"evidence": self._essential_case_evidence(transcript)}
        )

    def _request_json_content(
        self,
        transcript: Transcript,
        *,
        task: TextProductTask,
        system_prompt: str,
        user_intro: str,
    ) -> str:
        try:
            response = self.net.request(
                NetRequest(
                    method="POST",
                    url=self.route.base_url or "",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    body=json.dumps(
                        _request_body(self.route, transcript, system_prompt, user_intro)
                    ).encode("utf-8"),
                    timeout_s=60,
                    purpose=_net_purpose(task),
                    provider_id=self.route.provider_id,
                )
            )
        except Exception as exc:
            raise TextAiExecutionError(
                "text AI request failed "
                f"for provider {self.route.provider_id}: {type(exc).__name__}"
            ) from exc
        if response.status_code >= 400:
            raise TextAiExecutionError(f"text AI request failed: HTTP {response.status_code}")
        try:
            chat = _ChatResponse.model_validate_json(response.body)
        except ValueError as exc:
            raise TextAiExecutionError("text AI provider returned invalid JSON") from exc
        if not chat.choices:
            raise TextAiExecutionError("text AI response has no choices")
        return chat.choices[0].message.content

    def _knowledge_flow_evidence(self, transcript: Transcript) -> KnowledgeFlowSupplementEvidence:
        return KnowledgeFlowSupplementEvidence(
            route_provider_id=self.route.provider_id,
            route_model=self.route.model,
            source_segment_ids=[segment.id for segment in transcript.segments],
        )

    def _essential_case_evidence(self, transcript: Transcript) -> EssentialCaseSupplementEvidence:
        return EssentialCaseSupplementEvidence(
            route_provider_id=self.route.provider_id,
            route_model=self.route.model,
            source_segment_ids=[segment.id for segment in transcript.segments],
        )


def _request_body(
    route: AiRoute,
    transcript: Transcript,
    system_prompt: str,
    user_intro: str,
) -> dict[str, object]:
    return {
        "model": route.model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _transcript_prompt(transcript, user_intro)},
        ],
    }


def _transcript_prompt(transcript: Transcript, user_intro: str) -> str:
    lines = [user_intro]
    for segment in transcript.segments:
        text = " ".join(segment.text.split())
        end = segment.end if segment.end is not None else segment.start
        lines.append(f"{segment.id} [{segment.start:.3f}-{end:.3f}]: {text}")
    return "\n".join(lines)


def _net_purpose(task: TextProductTask) -> NetPurpose:
    return task
