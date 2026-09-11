from __future__ import annotations

from pathlib import Path

import pytest

import vctx.model.download as model_download
import vctx.model.store as models
from tests.support import asr_ready
from vctx.asr.cache import AsrTransformStore
from vctx.config import AsrInstanceConfig
from vctx.source.local import LocalMediaAsset
from vctx.source.session import SourceRef


def test_complete_asr_result_reuses_exact_identity_and_rejects_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def download(
        _capability: str,
        _model_id: str,
        target: Path,
        **_kwargs: object,
    ) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"model")
        (target / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(model_download, "download_model", download)
    models.ModelStore(tmp_path / "models").pull(["asr"], asr_model_id="tiny")
    media = LocalMediaAsset(
        id="media",
        source=SourceRef(kind="file", value="media.wav"),
        local_path=tmp_path / "media.wav",
        media_type="audio",
        sha256="a" * 64,
    )
    instance = AsrInstanceConfig(type="local-faster-whisper", model="tiny")
    store = AsrTransformStore(tmp_path / "source" / "transforms")
    key = store.key(media, instance, model_root=tmp_path / "models", model_id="tiny")
    assert key is not None
    store.put(key, asr_ready("media", "cached", model="tiny"))

    hit = store.get(key)
    assert hit is not None and hit.receipt.cache_hit
    assert hit.transcript.segments[0].text == "cached"

    path = tmp_path / "source" / "transforms" / "asr" / f"{key}.json"
    path.write_text(path.read_text(encoding="utf-8").replace("cached", "changed"))
    assert store.get(key) is None
