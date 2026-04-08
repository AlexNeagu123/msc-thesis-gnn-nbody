"""Shared utilities."""

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger with a clean format.

    Usage::

        logger = get_logger(__name__)
        logger.info("something happened")

    Args:
        name: logger name, typically ``__name__``.
        level: logging level string (e.g. "INFO", "DEBUG").

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s %(name)s %(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper()))
    return logger
