from __future__ import annotations

import json
from pathlib import Path

from vctx.config import ResolvedConfig
from vctx.model_store import ModelLifecycleError, ModelReceipt


def select_asr_model_id(resolved: ResolvedConfig) -> str:
    policy = resolved.asr
    model_ref = policy.model_ref()
    if model_ref is not None:
        if model_ref.startswith("local:"):
            return model_ref.removeprefix("local:")
        raise ModelLifecycleError("models pull manages named local ASR models, not path/HF refs")
    instance_name = policy.instance_name()
    if instance_name is None:
        return "small"
    instance = resolved.instances.asr[instance_name]
    if instance.type != "local-faster-whisper":
        raise ModelLifecycleError("selected ASR instance is online and has no local model to pull")
    model_id = instance.model or "small"
    if Path(model_id).is_absolute():
        raise ModelLifecycleError("selected ASR model is an explicit local path and needs no pull")
    return model_id


def render_model_receipts(receipts: list[ModelReceipt], *, json_output: bool) -> str:
    if json_output:
        return json.dumps([item.model_dump(mode="json") for item in receipts], indent=2) + "\n"
    return "".join(
        f"{item.capability}: {item.state} ({item.provider}/{item.model_id})\n" for item in receipts
    )
