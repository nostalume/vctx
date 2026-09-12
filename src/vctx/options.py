from enum import StrEnum
from typing import Literal

type SourceAssetScope = Literal["omitted", "consumed", "complete"]


class PrepareTarget(StrEnum):
    TRANSCRIPT = "transcript"
    EVIDENCE = "evidence"
    SUMMARY = "summary"


class MediaQuality(StrEnum):
    AUTO = "auto"
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"


class SourceAssets(StrEnum):
    CONSUMED = "consumed"
    COMPLETE = "complete"
