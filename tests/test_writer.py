from __future__ import annotations

from vctx.io import model_to_json
from vctx.models import SourceRef
from vctx.models.metadata import VideoMetadata


def test_model_to_json_uses_pydantic_dump_contract() -> None:
    metadata = VideoMetadata(
        id="video-1",
        source_type="file",
        source=SourceRef(kind="file", value="lecture.mp4"),
        title="Lecture",
    )

    assert model_to_json(metadata).endswith("\n")
    assert '"title": "Lecture"' in model_to_json(metadata)
