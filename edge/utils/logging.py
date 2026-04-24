"""Logging configuration for the edge processing pipeline.

Provides structured logging to both console and optional log file,
using a consistent format across all pipeline modules.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
) -> None:
    """Configure root logging for console and optional file output.

    Parameters
    ----------
    level:
        Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file:
        Optional path to a log file.  Parent directories are created
        automatically if they do not exist.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_path, encoding="utf-8")
        )

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    """
    return logging.getLogger(name)
