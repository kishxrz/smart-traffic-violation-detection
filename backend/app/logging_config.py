"""
backend/app/logging_config.py
─────────────────────────────
Structured logging setup using Python's standard logging library.

Design decisions:
- Structured JSON format in production for log aggregators (ELK, CloudWatch).
- Human-readable format in development.
- Single call to configure_logging() at application startup.
- Every module uses: logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


class _ColorFormatter(logging.Formatter):
    """ANSI-colored console formatter for development."""

    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
) -> None:
    """
    Configure the root logger.

    Args:
        level:       Log level string ("DEBUG", "INFO", etc.).
        json_format: If True, emit JSON lines suitable for log aggregators.
                     In development, use False for colored human-readable logs.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if json_format:
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"name":"%(name)s","message":"%(message)s"}'
        )
        formatter = logging.Formatter(fmt)
    else:
        fmt = "%(asctime)s  %(levelname)-8s  %(name)-35s  %(message)s"
        formatter = _ColorFormatter(fmt, datefmt="%H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers = [handler]

    # Silence noisy third-party libraries
    for noisy in ("ultralytics", "urllib3", "httpx", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
