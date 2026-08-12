from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter


@contextmanager
def phase(logger: logging.Logger, name: str) -> Iterator[None]:
    start = perf_counter()
    logger.debug("%s start", name)
    try:
        yield
    finally:
        logger.info("%s duration_ms=%s", name, int((perf_counter() - start) * 1000))


def configure_logging(*, verbose: bool, debug: bool, log_file: Path | None) -> None:
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
