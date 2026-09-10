from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TextIO

_profile: TextIO | None = None
_profile_lock = Lock()


@contextmanager
def phase(logger: logging.Logger, name: str) -> Iterator[None]:
    start = perf_counter()
    _profile_event({"schema": 1, "event": "start", "phase": name})
    logger.debug("%s start", name)
    try:
        yield
    finally:
        duration_ms = int((perf_counter() - start) * 1000)
        _profile_event({"schema": 1, "event": "finish", "phase": name, "duration_ms": duration_ms})
        logger.info("%s duration_ms=%s", name, duration_ms)


def configure_logging(
    *,
    verbose: bool,
    debug: bool,
    log_file: Path | None,
    profile_json: Path | None = None,
) -> None:
    global _profile

    if _profile is not None:
        _profile.close()
        _profile = None
    if profile_json is not None:
        profile_json.parent.mkdir(parents=True, exist_ok=True)
        _profile = profile_json.open("w", encoding="utf-8", buffering=1)
    logger = logging.getLogger("vctx")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    if not verbose and not debug and log_file is None:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False
        return
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")
    if verbose or debug:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def _profile_event(event: dict[str, object]) -> None:
    if _profile is None:
        return
    with _profile_lock:
        _profile.write(json.dumps(event, separators=(",", ":")) + "\n")
        _profile.flush()
