"""Shared utilities."""

import logging
import sys
from typing import Any

_MISSING = object()


def nested_get(d: dict, key: str, *, default: Any = _MISSING) -> Any:  # noqa: ANN401
    """Get a value from a nested dict using dot-separated keys.

    Args:
        d: the dictionary to traverse.
        key: dot-separated path (e.g. "training.lr", "scheduler.enabled").
        default: value to return if the path is missing. If not provided,
            a KeyError is raised on missing keys.

    Returns:
        The value at the given path, or *default* if the path is missing.

    Raises:
        KeyError: if the path is missing and no default was provided.
    """
    parts = key.split(".")
    current = d
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            if default is _MISSING:
                msg = f"Missing config key: {key}"
                raise KeyError(msg)
            return default
        current = current[part]
    return current


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
