from __future__ import annotations

from pathlib import Path

from vctx.app.credentials import CredentialError, resolve_env_credential
from vctx.models.media import MediaAsset
from vctx.models.visual import FrameAsset, VisualRecord, VisualRecordSet
from vctx.transforms import visual_frames
from vctx.transforms.ai_routes import AiRoute
from vctx.transforms.visual_ocr import OcrExecutionError, RapidOcrAdapter
from vctx.transforms.visual_planning import VisualAction, VisualAssessment
from vctx.transforms.visual_vlm import OpenAiCompatibleVisionAdapter, VlmOutcome


class VisualExecutionError(RuntimeError):
    pass


def run_visual_context(
    assessment: VisualAssessment,
    media_asset: MediaAsset,
    out_dir: Path,
    *,
    env_files: list[Path] | None = None,
) -> VisualRecordSet:
    frames: list[FrameAsset] = []
    records: list[VisualRecord] = []
    frames_dir = out_dir / "visual" / "frames"
    ocr_adapter: RapidOcrAdapter | None = None

    for action in assessment.recipe:
        if action.name == "sample":
            frames = _extract_frames(media_asset, action, frames_dir)
        elif action.name == "ocr":
            if ocr_adapter is None:
                ocr_adapter = RapidOcrAdapter()
            records.extend(_ocr_records(frames, ocr_adapter))
        elif action.name == "capture":
            records.extend(_capture_records(frames, out_dir))
        elif action.name == "describe":
            records.extend(
                _description_records(
                    frames,
                    action,
                    env_files or [],
                )
            )
    return VisualRecordSet(records=records)


def _extract_frames(
    media_asset: MediaAsset, action: VisualAction, frames_dir: Path
) -> list[FrameAsset]:
    try:
        return visual_frames.extract_frames(media_asset, action, frames_dir)
    except RuntimeError as exc:
        raise VisualExecutionError(str(exc)) from exc


def _ocr_records(frames: list[FrameAsset], adapter: RapidOcrAdapter) -> list[VisualRecord]:
    records: list[VisualRecord] = []
    for index, frame in enumerate(frames, start=1):
        try:
            text = adapter.extract_text(frame)
        except OcrExecutionError:
            continue
        if not text:
            continue
        records.append(
            VisualRecord(
                id=f"ocr-{index:04d}",
                timestamp_seconds=frame.timestamp_seconds,
                frame_id=frame.id,
                kind="ocr",
                text=text,
                evidence=frame.evidence,
            )
        )
    return records


def _description_records(
    frames: list[FrameAsset],
    action: VisualAction,
    env_files: list[Path],
) -> list[VisualRecord]:
    route = _description_route(action)
    api_key = _description_api_key(route, env_files)
    adapter = OpenAiCompatibleVisionAdapter(
        route=route,
        api_key=api_key,
    )
    outcomes = _describe_frames(frames, adapter)
    records: list[VisualRecord] = []
    frame_by_id = {frame.id: frame for frame in frames}
    for index, outcome in enumerate(outcomes, start=1):
        frame = frame_by_id.get(outcome.frame_id)
        if frame is None or not outcome.text:
            continue
        records.append(
            VisualRecord(
                id=f"description-{index:04d}",
                timestamp_seconds=frame.timestamp_seconds,
                frame_id=frame.id,
                kind="description",
                text=outcome.text,
                evidence=frame.evidence,
            )
        )
    return records


def _describe_frames(
    frames: list[FrameAsset], adapter: OpenAiCompatibleVisionAdapter
) -> list[VlmOutcome]:
    return adapter.describe_many(frames)


def _description_route(action: VisualAction) -> AiRoute:
    ai_route = action.ai_route
    if ai_route is None:
        raise VisualExecutionError("describe action is missing ai_route")
    provider_id = action.provider_id or ai_route.provider_id
    if provider_id is None:
        raise VisualExecutionError("describe action is missing provider_id")
    return ai_route.model_copy(update={"provider_id": provider_id})


def _description_api_key(route: AiRoute, env_files: list[Path]) -> str:
    try:
        return resolve_env_credential(route.api_key_env, env_files=env_files)
    except CredentialError as exc:
        raise VisualExecutionError(str(exc)) from exc


def _capture_records(frames: list[FrameAsset], out_dir: Path) -> list[VisualRecord]:
    records: list[VisualRecord] = []
    for index, frame in enumerate(frames, start=1):
        artifact_path = frame.path.relative_to(out_dir).as_posix()
        records.append(
            VisualRecord(
                id=f"capture-{index:04d}",
                timestamp_seconds=frame.timestamp_seconds,
                frame_id=frame.id,
                kind="capture",
                artifact_path=artifact_path,
                evidence=frame.evidence,
            )
        )
    return records
