from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    whole_seconds = max(0, int(seconds))
    hours, rem = divmod(whole_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def vctx_version() -> str:
    try:
        return version("vctx")
    except PackageNotFoundError:
        return "0.0.0"
