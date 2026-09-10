from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from vctx.artifact.bundle import encode_json
from vctx.asr import AsrReady, AsrReceipt
from vctx.config import AsrInstanceConfig
from vctx.model_store import ModelLifecycleError, require_prepared_model
from vctx.source.session import MediaAsset
from vctx.transcript import Transcript

_SCHEMA = 1


class _Entry(BaseModel):
    key: str
    digest: str
    transcript: Transcript
    receipt: AsrReceipt


@dataclass(frozen=True)
class AsrTransformStore:
    root: Path

    def get(self, key: str) -> AsrReady | None:
        path = self.root / "asr" / f"{key}.json"
        try:
            entry = _Entry.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError, ValidationError, ValueError:
            return None
        if entry.key != key or entry.digest != _digest(entry.transcript, entry.receipt):
            return None
        return AsrReady(
            transcript=entry.transcript,
            receipt=entry.receipt.model_copy(update={"cache_hit": True}),
        )

    def put(self, key: str, outcome: AsrReady) -> None:
        entry = _Entry(
            key=key,
            digest=_digest(outcome.transcript, outcome.receipt),
            transcript=outcome.transcript,
            receipt=outcome.receipt,
        )
        path = self.root / "asr" / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(encode_json(entry))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def asr_transform_key(
    media: MediaAsset,
    instance: AsrInstanceConfig,
    *,
    model_root: Path,
    model_id: str,
    interval: tuple[float, float | None] | None = None,
) -> str | None:
    if media.sha256 is None or Path(model_id).is_absolute():
        return None
    try:
        model = require_prepared_model("asr", model_root, asr_model_id=model_id)
    except ModelLifecycleError:
        return None
    facts = {
        "schema": _SCHEMA,
        "media": media.sha256,
        "interval": interval,
        "model": [model.provider, model.model_id, model.integrity],
        "runtime": _version("faster-whisper"),
        "engine": _version("ctranslate2"),
        "execution": instance.model_dump(mode="json"),
    }
    return hashlib.sha256(
        json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _digest(transcript: Transcript, receipt: AsrReceipt) -> str:
    body = {
        "transcript": transcript.model_dump(mode="json"),
        "receipt": receipt.model_dump(mode="json"),
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"
