"""Structured logging configuration for ResearchLens."""

import json
import logging
import os
import re
import sys
from typing import Optional

SECRET_PATTERNS = [
    re.compile(r"(?i)(password|secret|token|api[_-]?key|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{6,})['\"]?"),
    re.compile(r"(?i)authorization:\s*bearer\s+([a-zA-Z0-9_\-\.]+)"),
]


class SecretMaskingFilter(logging.Filter):
    """Filter that masks sensitive tokens, passwords, and API keys in log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern in SECRET_PATTERNS:
                msg = pattern.sub(r"\1: [REDACTED]", msg)
            record.msg = msg
        return True


class JsonLogFormatter(logging.Formatter):
    """Formatter that outputs structured JSON log entries."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with standard formatting or structured JSON."""
    logger_name = f"researchlens.{name}" if name else "researchlens"
    logger = logging.getLogger(logger_name)

    log_level_str = os.getenv("RESEARCHLENS_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.addFilter(SecretMaskingFilter())

        log_format = os.getenv("RESEARCHLENS_LOG_FORMAT", "text").lower()
        if log_format == "json":
            formatter: logging.Formatter = JsonLogFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ")
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(log_level)
        logger.propagate = False
    else:
        logger.setLevel(log_level)

    return logger

