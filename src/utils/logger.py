"""
src/utils/logger.py
────────────────────────────────────────────────────────────────
Logging configuration for the Clinical NLP Pipeline.

Every module gets its own named logger via get_logger(__name__).
The root logger is configured once at import time; subsequent
calls to get_logger() are cheap and return cached instances.

Log format
──────────
  2024-01-15 09:32:11 | INFO     | src.nlp.ner       | Loaded en_core_sci_lg
  2024-01-15 09:32:14 | WARNING  | src.etl.extract   | ICD-10 file not found

Usage
─────
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Pipeline started")
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


# Read desired level from environment; default to INFO.
# Set LOG_LEVEL=DEBUG in .env to see every internal step.
_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# Optional: write logs to a file alongside console output.
# Leave LOG_FILE unset to log to console only.
_LOG_FILE: str | None = os.getenv("LOG_FILE")

# Date/time format used across all handlers
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Column-aligned format that is easy to scan in a terminal
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"


def _configure_root_logger() -> None:
    """Set up the root logger with console (and optionally file) output.

    Called once when this module is first imported.  All loggers
    created afterwards automatically inherit this configuration.
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers if this is called more than once
    # (can happen in interactive notebooks that reimport modules).
    if root.handlers:
        return

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)

    # Always write to stdout so Docker / Streamlit Cloud captures logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional file handler — useful when running long ETL jobs
    if _LOG_FILE:
        log_path = Path(_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root logger on first call.

    Args:
        name: Typically ``__name__`` from the calling module.
              This produces loggers like ``src.nlp.ner`` which
              makes log lines easy to trace back to their source.

    Returns:
        A standard :class:`logging.Logger` instance.

    Example::

        logger = get_logger(__name__)
        logger.info("Processing %d notes", len(notes))
        logger.warning("Low confidence match: %s", entity)
        logger.error("Model load failed: %s", exc)
    """
    _configure_root_logger()
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """Change the log level at runtime without restarting.

    Useful in notebooks where you want to toggle verbosity
    interactively.

    Args:
        level: One of ``'DEBUG'``, ``'INFO'``, ``'WARNING'``,
               ``'ERROR'``, or ``'CRITICAL'``.

    Example::

        set_log_level('DEBUG')   # see every internal step
        set_log_level('WARNING') # suppress INFO messages
    """
    numeric = getattr(logging, level.upper(), None)
    if numeric is None:
        raise ValueError(
            f"Invalid log level '{level}'. "
            "Use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )
    logging.getLogger().setLevel(numeric)
