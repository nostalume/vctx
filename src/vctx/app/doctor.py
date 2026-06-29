from __future__ import annotations

import importlib.metadata
import shutil
import sys
from pathlib import Path

from vctx.io import build_cache


def doctor_report() -> str:
    lines = [
        f"python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"vctx: {_package_version('vctx')}",
        f"yt-dlp: {_package_version('yt-dlp')}",
        f"cache: {_cache_status()}",
        f"ffmpeg: {_command_status('ffmpeg')}",
    ]
    return "\n".join(lines) + "\n"


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _cache_status() -> str:
    try:
        cache = build_cache(None)
        probe = Path(cache.root) / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return f"error: {exc}"
    return f"writable ({cache.root})"


def _command_status(command: str) -> str:
    path = shutil.which(command)
    return path if path else "missing"
