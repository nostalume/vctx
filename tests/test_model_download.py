from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

import pytest

import vctx.model_download as model_download
from vctx.supervise import SupervisedResult


def test_hub_download_fast_mode_is_child_scoped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def run(command: list[str], payload: bytes, **kwargs: object) -> SupervisedResult:
        observed.update(command=command, payload=payload, **kwargs)
        return SupervisedResult(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(model_download, "run_supervised", run)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)

    model_download.download_model(
        "asr",
        "small",
        tmp_path,
        cache_root=tmp_path,
        conservative=False,
        max_runtime=3600,
    )

    assert os.environ.get("HF_XET_HIGH_PERFORMANCE") is None
    environment = cast(dict[str, str], observed["environment"])
    command = cast(list[str], observed["command"])
    assert environment["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert command[:3] == [sys.executable, "-m", "vctx.model_hub"]
