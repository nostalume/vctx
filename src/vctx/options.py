from enum import StrEnum


class PrepareTarget(StrEnum):
    TRANSCRIPT = "transcript"
    EVIDENCE = "evidence"
    SUMMARY = "summary"


class MediaQuality(StrEnum):
    AUTO = "auto"
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
