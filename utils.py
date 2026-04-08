"""
Shared utilities.
"""

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Returns a logger with a clean format. Call once per module:

        logger = get_logger(__name__)
        logger.info("something happened")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s %(name)s %(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper()))
    return logger
