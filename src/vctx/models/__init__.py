from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SourceRef(BaseModel):
    kind: Literal["url", "file"]
    value: str
