from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).parents[1]


def _project() -> dict[str, object]:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_distribution_profiles_have_the_declared_capability_closure() -> None:
    project = _project()
    extras = cast(dict[str, list[str]], project["optional-dependencies"])

    assert project["requires-python"] == ">=3.12,<3.15"
    assert set(extras) == {"asr", "visual", "full"}
    assert set(extras["asr"]) == {"faster-whisper>=1.2.1"}
    assert set(extras["visual"]) == {
        "av>=18.0.0",
        "onnxruntime>=1.20.0",
        "pillow>=12.0.0",
        "rapidocr>=3.4.2",
    }
    assert set(extras["full"]) == set(extras["asr"]) | set(extras["visual"])
