from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

import pytest

import vctx.model.download as model_download
import vctx.supervise as supervise
from vctx.errors import DeadlineExceededError
from vctx.supervise import SupervisedResult


def _download(path: Path) -> None:
    model_download.download_model(
        "asr", "small", path, cache_root=path, conservative=False, max_runtime=3600
    )


def test_hub_download_fast_mode_is_child_scoped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def run(command: list[str], payload: bytes, **kwargs: object) -> SupervisedResult:
        observed.update(command=command, payload=payload, **kwargs)
        return SupervisedResult(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(supervise, "run_supervised", run)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)

    _download(tmp_path)

    assert os.environ.get("HF_XET_HIGH_PERFORMANCE") is None
    environment = cast(dict[str, str], observed["environment"])
    command = cast(list[str], observed["command"])
    assert environment["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert command[:3] == [sys.executable, "-m", "vctx.model.download"]


def test_hub_download_preserves_child_failure_classes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codes = iter((1, 124))

    def run(*_args: object, **_kwargs: object) -> SupervisedResult:
        return SupervisedResult(returncode=next(codes), stdout=b"", stderr=b"")

    monkeypatch.setattr(supervise, "run_supervised", run)
    for error, message in ((RuntimeError, "exit code 1"), (DeadlineExceededError, "exceeded")):
        with pytest.raises(error, match=message):
            _download(tmp_path)
