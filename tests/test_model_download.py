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
