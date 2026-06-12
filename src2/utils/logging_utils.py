"""Structured logging utilities for the project pipeline."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "music_genre",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or retrieve) a logger with console and optional file handlers.

    Parameters
    ----------
    name : str
        Logger name; reused across modules for consistency.
    log_file : Path or None
        If provided, also write log lines to this file.
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_section(logger: logging.Logger, title: str, char: str = "=") -> None:
    """Print a visible section header to aid in log scanning."""
    logger.info(char * 70)
    logger.info(f"  {title}")
    logger.info(char * 70)


def timestamp() -> str:
    """Return an ISO-8601 timestamp string (UTC)."""
    return datetime.now(timezone.utc).isoformat()