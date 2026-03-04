"""
Centralized logging configuration for AffectSync.

Usage in any module:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

import logging
import sys
from pathlib import Path

# Add project root to path so config is always importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger instance.

    Each module gets its own named logger, but they all share
    the same format and level defined in config.py.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt=config.LOG_FORMAT,
            datefmt=config.LOG_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    return logger
