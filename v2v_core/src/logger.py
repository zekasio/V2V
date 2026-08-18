"""
Logging setup — console + rotating file output via Rich.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.config import settings

_INITIALIZED = False


def setup_logging() -> logging.Logger:
    """Configure root logger once and return the named 'v2v' logger."""
    global _INITIALIZED
    if _INITIALIZED:
        return logging.getLogger("v2v")

    log_dir: Path = settings.log_path
    log_file = log_dir / f"v2v_{datetime.now():%Y%m%d}.log"

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # ── Console handler (rich) ────────────────────────────────
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(level)

    # ── File handler (rotating) ───────────────────────────────
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # ── Root logger ───────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _INITIALIZED = True
    logger = logging.getLogger("v2v")
    logger.info("Logging initialized  →  %s", log_file)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the v2v namespace."""
    setup_logging()
    return logging.getLogger(f"v2v.{name}")
