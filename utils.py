"""Shared utilities."""

import logging
import sys
from typing import Any

_MISSING = object()


def nested_get(d: dict, key: str, *, default: Any = _MISSING) -> Any:  # noqa: ANN401
    """Get a value from a nested dict by dot-separated path; raises KeyError if missing and no default."""
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
    """Return a stdout logger with a clean format, configured once per name."""
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
