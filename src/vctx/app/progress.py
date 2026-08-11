from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def phase(logger: logging.Logger, name: str) -> Iterator[None]:
    start = perf_counter()
    logger.debug("%s start", name)
    try:
        yield
    finally:
        logger.info("%s duration_ms=%s", name, int((perf_counter() - start) * 1000))
