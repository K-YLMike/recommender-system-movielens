"""Minimal structured logging + timing helpers shared by all stages.

Every stage prints a consistent, timestamped, grep-able log line, which matters
when reconstructing what happened across a chain of resubmitted 1-hour jobs from
the ``logs/*.out`` files alone.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a process-wide configured logger.

    Logs go to stdout so they land in the Slurm ``.out`` file. Configuration is
    idempotent, so importing this from many modules is safe.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[None]:
    """Context manager that logs the wall-clock duration of a block.

    Timing is part of the "statistics matter" requirement: we record how long
    each stage takes so the README can report realistic per-stage runtimes and
    the recall-latency numbers are grounded.
    """
    start = time.perf_counter()
    logger.info("START %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("END   %s (%.2fs)", label, elapsed)
